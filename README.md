# RAG Pipeline — HotpotQA & FEVER

A dual-task Retrieval-Augmented Generation system covering:

- **HotpotQA** — multi-hop question answering over Wikipedia passages
- **FEVER** — fact verification (SUPPORTS / REFUTES / NOT ENOUGH INFO) with an optional passage-poisoning adversarial layer

Both tasks share the same retrieval engine (dense + hybrid BM25 fusion) and LLM backend ([Ollama](https://ollama.com/)).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Offline (one-time)                       │
│                                                                 │
│  HotpotQA / FEVER ──► Sliding-window        ──►  ChromaDB      │
│  dataset (HF /         chunking (3 sent,          vector store  │
│  MediaWiki API)        step 2)                    (cosine sim)  │
│                              │                                  │
│                              └──►  BM25Plus index               │
│                                   (keyword search)              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     Online (per query / claim)                  │
│                                                                 │
│  User question / Claim                                          │
│       │                                                         │
│       ├──► Dense retrieval  (all-MiniLM-L12-v2 + ChromaDB)     │
│       │         top-50 passages by cosine similarity            │
│       │                                                         │
│       ├──► Sparse retrieval (BM25Plus)                          │
│       │         top-50 passages by keyword score                │
│       │                                                         │
│       └──► BM25-first hybrid fusion                             │
│                 • guarantees top-5 BM25 hits always appear      │
│                 • fills remaining slots from dense results      │
│                 • final 10 passages sorted by cosine score      │
│                              │                                  │
│                   [Optional] PassagePoisoner                    │
│                   rewrites ~30% of passages with wrong facts    │
│                              │                                  │
│                    RAG Prompt (context + question/claim)        │
│                              │                                  │
│                    Ollama LLM (llama3.2:3b)                     │
│                    streaming token output                        │
│                              │                                  │
│    QA mode:        Chain-of-Thought + Final Answer              │
│    Fact-check mode: Chain-of-Thought + Final Verdict            │
│                    (SUPPORTS / REFUTES / NOT ENOUGH INFO)       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Components

### `src/data/build_hotpotqa_db.py` — HotpotQA vector index builder
Downloads the HotpotQA validation split from HuggingFace (7,405 questions, ~90k Wikipedia
passages). Each article is split into overlapping chunks using a **sliding window of 3 sentences
with a step of 2**. Chunks are deduplicated by SHA-256, encoded with `all-MiniLM-L12-v2`
(384-dimensional embeddings), and stored in a local **ChromaDB** collection (`hotpotqa_passages`)
with cosine similarity. Encoding is streamed in outer batches of 2,000 chunks to keep peak RAM low.
The pipeline is crash-resumable — already-inserted chunks are skipped.

### `src/data/build_fever_db.py` — FEVER vector index builder
Downloads FEVER `paper_dev` claims (9,999 claims — equal thirds of SUPPORTS, REFUTES, NOT ENOUGH INFO),
collects the unique Wikipedia titles cited as evidence, fetches each page via the **MediaWiki API**
with exponential-backoff rate-limit handling, and runs the same sliding-window chunking + ChromaDB
insertion pipeline into a separate collection (`fever_passages`). FEVER's bracket encoding
(`-LRB-`, `-RRB-`, etc.) is decoded back to standard title form before fetching.

### `src/data/build_bm25_index.py` — BM25 index builder
Reads all chunks from a ChromaDB collection in batches of 10,000, tokenises each chunk with a
simple alphanumeric regex, builds a **BM25Plus** index, and pickles it alongside a corpus ID
mapping. Works for both HotpotQA and FEVER collections via `--collection` and `--output` flags.

### `src/rag/retriever.py` — Dense + Hybrid retriever
Contains both `ChromaDBRetriever` (dense-only) and `HybridRetriever` (BM25 + dense) in a single
file. `HybridRetriever` uses a **BM25-first fusion strategy**:

1. Run dense retrieval for top-50 candidates and BM25 retrieval for top-50 candidates.
2. **Phase 1 — BM25 quota**: guarantee the top `ceil(k/2)` BM25 results always appear in the
   final set, regardless of their dense score. Critical for named-entity queries where the exact
   Wikipedia article scores high in BM25 but may be outranked in dense search.
3. **Phase 2 — Dense fill**: fill remaining slots with the highest-scoring dense results not
   already selected.
4. Sort the final `k` passages by cosine similarity for display.

Why BM25-first over RRF? RRF fails for rare named entities — a high-BM25 / low-dense document
loses to many moderate-scoring documents that appear in both lists. The BM25 quota makes the
guarantee explicit.

### `src/rag/poisoner.py` — Passage poisoner (adversarial layer)
`PassagePoisoner` rewrites a subset of retrieved passages to inject plausible but factually wrong
information, simulating a corpus-poisoning attack. For each poisoning target it:

1. Detects the concept type from the question (nationality, year, location, profession, …)
2. Finds the specific fact in the passage text
3. Swaps it for a wrong but semantically plausible alternative

Default rate is 30% of retrieved passages (minimum 1, skipped for single-passage retrieval).
If a passage contains no information relevant to the question's concept, it is returned unchanged.
The poisoner marks each passage with `poisoned: bool` and preserves the `original_text` for
inspection.

### `src/rag/pipeline.py` — RAG pipeline
Orchestrates retrieval, optional poisoning, and generation. Supports two modes:

- **`mode="qa"`** (default) — HotpotQA-style question answering; prompt enforces `Final Answer:`
- **`mode="fact_check"`** — FEVER-style fact verification; prompt enforces `Final Verdict: SUPPORTS / REFUTES / NOT ENOUGH INFO`

Key methods:
- `answer(question)` — blocking call; returns a result dict.
- `answer_stream(question)` — yields `("token", str)` for each LLM token, then `("done", result_dict)`.
- `_retrieve_multi_query` — optional query decomposition (round-robin merge of sub-query results; disabled by default).

### `src/rag/llm.py` — Ollama LLM wrapper
Thin wrapper around `ollama.Client`. Supports blocking (`generate`) and streaming
(`generate_stream`) modes. Connects to `http://localhost:11434` by default; override with
`OLLAMA_HOST` env var (used in Docker).

### `src/eval/evaluate.py` — HotpotQA benchmark evaluation
Runs the full pipeline over the HotpotQA validation set and computes:

- **Exact Match (EM)** — normalized predicted answer matches gold
- **Token F1** — token-level overlap between prediction and gold
- **SP Precision / Recall / F1** — retrieved passage titles vs gold supporting-fact articles

Results are written to a JSONL file (appended live for resume support) and a timestamped CSV.

### `src/eval/evaluate_fever.py` — FEVER benchmark evaluation
Runs the pipeline in `fact_check` mode over the FEVER `paper_dev` set and computes:

- **Label Accuracy** — fraction of claims with the correct SUPPORTS / REFUTES / NOT ENOUGH INFO verdict
- **FEVER Score** — label is correct AND (for SUPPORTS/REFUTES) at least one gold evidence title was retrieved
- **Evidence Precision / Recall / F1** — retrieved passage titles vs gold evidence Wikipedia titles

Supports `--poison true` to enable the poisoning layer during evaluation.

### `src/rag/qa.py` — CLI interface
Entry point for interactive and single-question modes. Renders output with `rich`:
- **Yellow** rule + streaming text — chain-of-thought reasoning
- **Orange** panel — retrieved documents with cosine scores
- **Blue** panel — extracted Final Answer

Warms up the Ollama model on startup with a cheap "hi" prompt to avoid weight-loading delay on the first query.

---

## Data Flow (per query)

```
1. User types question / claim
        ↓
2. all-MiniLM-L12-v2 encodes query → 384-dim vector
        ↓
3. ChromaDB HNSW index → top-50 passages (cosine distance)
   BM25Plus index      → top-50 passages (BM25 score)
        ↓
4. BM25-first fusion → 10 final passages (sorted by cosine score)
        ↓
5. [Optional] PassagePoisoner rewrites ~30% of passages with wrong facts
        ↓
6. Prompt assembled:
   [1] Title\nPassage text
   [2] Title\nPassage text
   ...
   Question / Claim: <input>
   Think step by step ... end with 'Final Answer:' / 'Final Verdict:' ...
        ↓
7. Ollama streams llama3.2:3b tokens
        ↓
8. Tokens printed live (chain-of-thought)
9. After stream ends: Sources panel + Final Answer / Verdict panel
```

---

## Setup

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/) installed and running
- ~2–3 GB disk for ChromaDB collections + BM25 indexes

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Pull the LLM

```bash
ollama pull llama3.2:3b
```

---

## HotpotQA Pipeline

### Build indexes (one-time)

```bash
# 1. Download HotpotQA and build ChromaDB vector store (~5–15 min depending on hardware)
python src/data/build_hotpotqa_db.py

# 2. Build BM25Plus keyword index (~1 min)
python src/data/build_bm25_index.py
```

### Interactive Q&A

```bash
python src/rag/qa.py
```

Type questions at the prompt. Type `quit` to exit.

### Single question

```bash
python src/rag/qa.py -q "Were Scott Derrickson and Ed Wood from the same country?"
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--llm` | `llama3.2:3b` | Ollama model name |
| `--top-k` | `10` | Number of passages to retrieve |
| `--retriever` | `hybrid` | `dense` (ChromaDB only) or `hybrid` (BM25 + dense) |
| `--multi-query` | off | Decompose question into sub-queries before retrieval |
| `--embed-model` | `sentence-transformers/all-MiniLM-L12-v2` | Embedding model |
| `--collection` | `hotpotqa_passages` | ChromaDB collection name |

```bash
# Dense-only retrieval
python src/rag/qa.py --retriever dense

# Use a different LLM
python src/rag/qa.py --llm mistral

# Enable multi-query decomposition
python src/rag/qa.py --multi-query --llm llama3.2:3b

# Retrieve more passages
python src/rag/qa.py --top-k 15
```

### Run the benchmark

```bash
# Quick test: 100 random questions (seed 42)
python src/eval/evaluate.py --limit 100 --output results/smoke.jsonl

# Full validation set (7,405 questions)
python src/eval/evaluate.py --output results/hybrid_k10.jsonl
```

| Flag | Default | Description |
|---|---|---|
| `--limit` | all 7,405 | Randomly sample N questions |
| `--seed` | `42` | Random seed |
| `--output` | `results/eval.jsonl` | Per-question JSONL (resume-safe) |
| `--top-k` | `10` | Number of passages to retrieve |
| `--llm` | `llama3.2:3b` | Ollama model name |

### Results — 100 questions (hybrid retriever, top-10, llama3.2:3b)

```
────────────────────────────────────────────────────────────
  Evaluation Summary  |  Questions evaluated: 100/100
────────────────────────────────────────────────────────────
Metric                  All     Bridge   Comparison
──────────────────────────────────────────────────
Exact Match (EM)      26.0%      24.4%        35.7%
Token F1              37.5%      37.1%        40.5%
SP Precision          17.4%      17.2%        18.3%
SP Recall             76.0%      75.0%        82.1%
SP F1                 28.2%      27.9%        29.8%
──────────────────────────────────────────────────
```

**Interpretation:**
- **EM 26% / F1 37.5%** — reasonable for a 3B parameter local model on hard multi-hop questions. Comparison questions (yes/no) are easier (35.7% EM) than bridge questions requiring multi-hop reasoning (24.4% EM).
- **SP Recall 76%** — the hybrid retriever surfaces at least one gold supporting article in 76% of cases, confirming BM25-first fusion is effective for named-entity coverage.
- **SP Precision 17.4%** — expected to be low: we retrieve 10 passages but typically only 2 are gold supporting facts. Precision = 2/10 = 20% is the theoretical ceiling.

---

## FEVER Pipeline

### Build indexes (one-time)

```bash
# 1. Download FEVER claims and fetch Wikipedia evidence pages via MediaWiki API
#    (~30 min for ~1,460 Wikipedia articles, rate-limited to 1 request/sec)
python src/data/build_fever_db.py

# 2. Build BM25Plus keyword index for FEVER collection (~1 min)
python src/data/build_bm25_index.py \
    --collection fever_passages \
    --output data/fever_bm25_index.pkl \
    --corpus-ids-output data/fever_bm25_corpus_ids.pkl
```

### Run the benchmark

```bash
# Quick test: 100 random claims (seed 42)
python src/eval/evaluate_fever.py --limit 100 --output results/fever_smoke.jsonl

# Full paper_dev set (9,999 claims)
python src/eval/evaluate_fever.py --output results/fever_full.jsonl

# With passage poisoning enabled (adversarial evaluation)
python src/eval/evaluate_fever.py --limit 100 --poison true --output results/fever_poisoned.jsonl
```

| Flag | Default | Description |
|---|---|---|
| `--limit` | all 9,999 | Randomly sample N claims |
| `--seed` | `42` | Random seed |
| `--output` | `results/fever_eval.jsonl` | Per-claim JSONL (resume-safe) |
| `--top-k` | `10` | Number of passages to retrieve |
| `--llm` | `llama3.2:3b` | Ollama model name |
| `--collection` | `fever_passages` | ChromaDB collection name |
| `--poison` | `false` | Enable passage poisoning (`true` / `false`) |
| `--poison-seed` | random | Seed for poison passage selection |

---

## Passage Poisoning — Adversarial Evaluation

The poisoning layer simulates a **corpus-poisoning attack**: a fraction of retrieved passages are
rewritten by the LLM to contain plausible but factually wrong information targeted at the concept
the question or claim is testing.

### How it works

1. The retriever returns `k` passages as usual.
2. `PassagePoisoner` selects `round(k × 0.3)` passages at random (minimum 1; skip if only 1 passage).
3. For each selected passage it calls the LLM with both the claim and passage text, instructing it to:
   - Identify the concept type (nationality, date, location, profession, …)
   - Find that specific fact in the passage
   - Replace it with a wrong but semantically plausible alternative
4. If the passage contains no information relevant to the concept, it is returned unchanged (not marked as poisoned).
5. Each passage in the result gets `poisoned: bool`; poisoned passages also carry `original_text`.

### Poisoning count by retrieved set size

| Retrieved passages | Poisoned |
|---|---|
| 1 | 0 (skip) |
| 2–3 | 1 |
| 5 | 2 |
| 10 | 3 |

---

## Running with Docker

Ollama must be running on the host machine:

```bash
ollama serve
ollama pull llama3.2:3b
```

### Build HotpotQA indexes inside Docker (first time only)

```bash
docker compose run --rm build-hotpotqa-db
```

### Run HotpotQA evaluation

```bash
docker compose run --rm eval
```

### Interactive Q&A

```bash
docker compose run --rm qa
```

### Single question

```bash
docker compose run --rm qa python src/rag/qa.py -q "What year did Guns N Roses perform a promo for a movie starring Arnold Schwarzenegger?"
```

The `./data` directory is mounted as a volume so indexes built inside Docker are available on the
host and vice versa. The container connects to Ollama on the host via
`OLLAMA_HOST=http://host.docker.internal:11434`.

---

## Example Outputs

### HotpotQA — Bridge question (date range)

```
Question: The football manager who recruited David Beckham managed Manchester United during what timeframe?

──────────────────────────── Chain-of-Thought ─────────────────────────────
Step 1: Identify the football manager who recruited David Beckham.
From context [4], we know that David Beckham was managed by Sir Alex Ferguson.

Step 2: Determine the timeframe during which Sir Alex Ferguson managed Manchester United.
From context [2], we see that Sir Alex Ferguson managed Manchester United from 1986 to 2013.

Final Answer: 1986 to 2013
───────────────────────────────────────────────────────────────────────────
╭──────────────────────── Retrieved Documents ─────────────────────────────╮
│ [0.685] Ernest Mangnall                                                  │
│ [0.650] Alex Ferguson                                                    │
│ [0.641] Matt Busby                                                       │
│ [0.620] David Beckham                                                    │
╰──────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────── Final Answer ────────────────────────────────╮
│ 1986 to 2013                                                             │
╰──────────────────────────────────────────────────────────────────────────╯
```

### HotpotQA — Comparison question (yes/no)

```
Question: Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?

──────────────────────────── Chain-of-Thought ─────────────────────────────
Step 1: Laleli Mosque is in Laleli, Fatih, Istanbul.
Step 2: Esma Sultan Mansion is in Ortaköy, Istanbul.
Both are in Istanbul, but in different neighborhoods.

Final Answer: no
```

### FEVER — Fact verification

```
Claim: Fox 2000 Pictures released the film Soul Food.

──────────────────────────── Chain-of-Thought ─────────────────────────────
The context states Soul Food (film) was released by Fox 2000 Pictures in 1997.
The claim is directly confirmed by the retrieved evidence.

Final Verdict: SUPPORTS
```

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `hotpotqa_analysis.ipynb` | HotpotQA dataset exploration — question types, supporting fact statistics, retrieval coverage analysis |
| `fever_analysis.ipynb` | FEVER dataset exploration — label distribution, evidence title statistics, corpus coverage, evaluation result analysis |

---

## Project Structure

```
.
├── src/
│   ├── data/
│   │   ├── build_hotpotqa_db.py   # Download + chunk + embed → ChromaDB (hotpotqa_passages)
│   │   ├── build_fever_db.py      # FEVER claims + MediaWiki pages → ChromaDB (fever_passages)
│   │   └── build_bm25_index.py    # Build BM25Plus index from any ChromaDB collection
│   ├── rag/
│   │   ├── retriever.py           # ChromaDBRetriever + HybridRetriever
│   │   ├── pipeline.py            # RAG pipeline (QA + fact-check modes, poisoning)
│   │   ├── poisoner.py            # PassagePoisoner — adversarial fact injection
│   │   ├── llm.py                 # Ollama LLM wrapper
│   │   └── qa.py                  # CLI entry point (interactive + single-question)
│   └── eval/
│       ├── evaluate.py            # HotpotQA benchmark (EM, Token F1, SP metrics)
│       └── evaluate_fever.py      # FEVER benchmark (Label Accuracy, FEVER Score, Evidence F1)
├── data/
│   ├── chromadb/                  # ChromaDB vector store (auto-created)
│   │   ├── hotpotqa_passages/     # HotpotQA collection
│   │   └── fever_passages/        # FEVER collection
│   ├── hotpotqa/                  # Raw HotpotQA dataset cache (auto-created)
│   ├── fever/                     # FEVER claims + wiki pages cache (auto-created)
│   ├── bm25_index.pkl             # BM25Plus index for HotpotQA (auto-created)
│   ├── bm25_corpus_ids.pkl        # Corpus ID mapping for HotpotQA (auto-created)
│   ├── fever_bm25_index.pkl       # BM25Plus index for FEVER (auto-created)
│   └── fever_bm25_corpus_ids.pkl  # Corpus ID mapping for FEVER (auto-created)
├── results/                       # Evaluation outputs (JSONL + CSV)
├── hotpotqa_analysis.ipynb        # HotpotQA data exploration notebook
├── fever_analysis.ipynb           # FEVER data exploration notebook
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Requirements

```
datasets>=2.14.0
sentence-transformers>=2.7.0
chromadb>=0.5.0
tqdm>=4.66.0
torch
ollama>=0.3.0
rank_bm25>=0.2.2
rich>=13.0.0
```
