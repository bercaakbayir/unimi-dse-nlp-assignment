# HotpotQA RAG — Retrieval-Augmented Generation Pipeline

A multi-hop question-answering system over the [HotpotQA](https://hotpotqa.github.io/) dataset
([HuggingFace](https://huggingface.co/datasets/hotpot_qa)).
Given a natural language question, the system retrieves relevant Wikipedia passages from a local
vector database and feeds them to a locally-running LLM (via [Ollama](https://ollama.com/)) to
produce a concise, reasoned answer.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Offline (one-time)                       │
│                                                                 │
│  HotpotQA dataset  ──►  Sliding-window        ──►  ChromaDB     │
│  (HuggingFace)          chunking (3 sent,          vector store │
│                         step 2)                    (cosine sim) │
│                              │                                  │
│                              └──►  BM25Plus index               │
│                                   (keyword search)              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     Online (per query)                          │
│                                                                 │
│  User question                                                  │
│       │                                                         │
│       ├──► Dense retrieval  (all-MiniLM-L12-v2 + ChromaDB)     │
│       │         top-50 passages by cosine similarity            │
│       │                                                         │
│       ├──► Sparse retrieval (BM25Plus)                          │
│       │         top-50 passages by keyword score                │
│       │                                                         │
│       └──► BM25-first hybrid fusion                             │
│                 • guarantees top-5 BM25 hits always appear      │
│                 • fills remaining 5 slots from dense results    │
│                 • final 10 passages sorted by cosine score      │
│                              │                                  │
│                    RAG Prompt (context + question)              │
│                              │                                  │
│                    Ollama LLM (llama3.2:3b)                     │
│                    streaming token output                        │
│                              │                                  │
│              Chain-of-Thought reasoning + Final Answer          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Components

### `src/data/build_hotpotqa_db.py` — Vector index builder
Downloads the HotpotQA validation split from HuggingFace (7,405 questions, ~90k Wikipedia
passages). Each article is split into overlapping chunks using a **sliding window of 3 sentences
with a step of 2**. Chunks are deduplicated by SHA-256 of their content, encoded with
`all-MiniLM-L12-v2` (384-dimensional embeddings), and stored in a local **ChromaDB** collection
with cosine similarity. Encoding is streamed in outer batches of 2,000 chunks to keep peak RAM low.
The pipeline is crash-resumable — already-inserted chunks are skipped.

### `src/data/build_bm25_index.py` — BM25 index builder
Reads all chunks from ChromaDB in batches of 10,000, tokenises each chunk with a simple
alphanumeric regex, builds a **BM25Plus** index, and pickles it alongside a corpus ID mapping to
`data/bm25_index.pkl` and `data/bm25_corpus_ids.pkl`.

### `src/rag/retriever.py` — Dense + Hybrid retriever
Contains both `ChromaDBRetriever` (dense-only) and `HybridRetriever` (BM25 + dense) in a single
file. `ChromaDBRetriever` encodes the query with `all-MiniLM-L12-v2` (singleton-cached per
process), queries ChromaDB for the nearest neighbours by cosine similarity, and returns passages
with their titles, text, and similarity scores. `HybridRetriever` extends this with a
**BM25-first fusion strategy**:

1. Run dense retrieval for top-50 candidates and BM25 retrieval for top-50 candidates.
2. **Phase 1 — BM25 quota**: guarantee the top `ceil(k/2)` BM25 results always appear in the
   final set, regardless of their dense score. This is critical for named-entity queries where
   the exact Wikipedia article scores high in BM25 but may be outranked in dense search by
   reference articles that mention the entity more frequently.
3. **Phase 2 — Dense fill**: fill remaining slots with the highest-scoring dense results not
   already selected.
4. Sort the final `k` passages by cosine similarity for display.

Why BM25-first over Reciprocal Rank Fusion (RRF)? RRF failed for rare named entities — a
high-BM25 / low-dense document (e.g. the exact biographical article for a rare name) loses to
many moderate-scoring documents that appear in both lists. The BM25 quota makes the guarantee
explicit.

### `src/rag/pipeline.py` — RAG pipeline
Orchestrates retrieval and generation:

- `answer(question)` — blocking call; returns a result dict.
- `answer_stream(question)` — yields `("token", str)` for each LLM token, then
  `("done", result_dict)` when the stream ends.
- `_build_prompt` — formats the top-k passages as numbered context blocks followed by a
  chain-of-thought instruction.
- `_retrieve_multi_query` — optional query decomposition: the LLM first generates 2–3
  sub-questions, each is retrieved independently, and results are merged in round-robin order
  to guarantee each sub-question is represented (disabled by default; requires a capable model).

**System prompt** uses few-shot examples to enforce concise Final Answer formatting:
- Yes/No questions → `yes` or `no`
- Date/year questions → bare year or `YYYY to YYYY`
- All other questions → shortest possible phrase (1–5 words)

### `src/rag/llm.py` — Ollama LLM wrapper
Thin wrapper around `ollama.Client`. Supports both blocking (`generate`) and streaming
(`generate_stream`) modes. Connects to `http://localhost:11434` by default; override with
`OLLAMA_HOST` env var (used in Docker).

### `src/eval/evaluate.py` — Benchmark evaluation
Runs the full pipeline over the HotpotQA validation set and computes standard metrics:

- **Exact Match (EM)** — predicted Final Answer matches gold after normalization (lowercase, strip articles/punctuation)
- **Token F1** — token-level overlap between prediction and gold
- **SP Precision / Recall / F1** — whether retrieved passage titles cover the gold supporting-fact articles

Results are written to a JSONL file (one record per question, appended live for resume support)
and a timestamped CSV (`results/eval_YYYYMMDD_HHMMSS.csv`) with columns `id, question, real_answer, llm_answer`.

### `src/rag/qa.py` — CLI interface
Entry point for interactive and single-question modes. Renders output with `rich`:
- **Yellow** rule + streaming text — chain-of-thought reasoning
- **Orange** panel — retrieved documents with cosine scores
- **Blue** panel — extracted Final Answer

Warms up the Ollama model on startup with a cheap "hi" prompt so the first real query does
not pay the weight-loading delay.

---

## Data Flow (per query)

```
1. User types question
        ↓
2. all-MiniLM-L12-v2 encodes query → 384-dim vector
        ↓
3. ChromaDB HNSW index → top-50 passages (cosine distance)
   BM25Plus index      → top-50 passages (BM25 score)
        ↓
4. BM25-first fusion → 10 final passages (sorted by cosine score)
        ↓
5. Prompt assembled:
   [1] Title\nPassage text
   [2] Title\nPassage text
   ...
   Question: <user question>
   Think step by step ... end with 'Final Answer:' ...
        ↓
6. Ollama streams llama3.2:3b tokens
        ↓
7. Tokens printed live (chain-of-thought)
8. After stream ends: Sources panel + Final Answer panel
```

---

## Setup

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/) installed and running
- ~2 GB disk for ChromaDB + BM25 index

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

### Build the indexes (one-time)

```bash
# 1. Download HotpotQA and build ChromaDB vector store (~5–15 min depending on hardware)
python src/data/build_hotpotqa_db.py

# 2. Build BM25Plus keyword index (~1 min)
python src/data/build_bm25_index.py
```

---

## Running

### Interactive mode

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

# Enable multi-query decomposition (requires a capable model)
python src/rag/qa.py --multi-query --llm llama3.2:3b

# Retrieve more passages
python src/rag/qa.py --top-k 15
```

---

## Running with Docker

Ollama must be running on the host machine:

```bash
ollama serve
ollama pull llama3.2:3b
```

### Build indexes inside Docker (first time only)

```bash
docker compose run --rm build-hotpotqa-db
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

### Bridge question — date range

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
│   James Ernest Mangnall ... was an English football manager...           │
│                                                                          │
│ [0.650] Alex Ferguson                                                    │
│   Sir Alexander Chapman Ferguson ... managed Manchester United...        │
│                                                                          │
│ [0.641] Matt Busby                                                       │
│   Sir Alexander Matthew Busby ... was a Scottish football manager...     │
│                                                                          │
│ [0.620] David Beckham                                                    │
│   David Robert Joseph Beckham ... is an English former professional...   │
╰──────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────── Final Answer ────────────────────────────────╮
│ 1986 to 2013                                                             │
╰──────────────────────────────────────────────────────────────────────────╯
```

---

### Bridge question — named entity

```
Question: What science fantasy young adult series, told in first person, has a set of
          companion books narrating the stories of enslaved worlds and alien species?

──────────────────────────── Chain-of-Thought ─────────────────────────────
Step 1: Identify the science fantasy young adult series.
Step 2: Note that Animorphs is a science fantasy series told in first person.
Step 3: Recognize that Animorphs has companion books narrating the stories of
        enslaved worlds and alien species.

Final Answer: Animorphs
───────────────────────────────────────────────────────────────────────────
╭──────────────────────────── Final Answer ────────────────────────────────╮
│ Animorphs                                                                │
╰──────────────────────────────────────────────────────────────────────────╯
```

---

### Comparison question — yes/no

```
Question: Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?

──────────────────────────── Chain-of-Thought ─────────────────────────────
Step 1: Identify the location of each building.
- Laleli Mosque is located in Laleli, Fatih, Istanbul, Turkey.
- Esma Sultan Mansion is located at Bosphorus in Ortaköy neighborhood of Istanbul, Turkey.

Step 2: Compare the locations.
Both buildings are in Istanbul, but in different neighborhoods (Laleli vs Ortaköy).

Final Answer: no
───────────────────────────────────────────────────────────────────────────
╭──────────────────────────── Final Answer ────────────────────────────────╮
│ no                                                                       │
╰──────────────────────────────────────────────────────────────────────────╯
```

---

## Evaluation

### Running the benchmark

```bash
# Quick test: 100 random questions (seed 42 for reproducibility)
python src/eval/evaluate.py --limit 100 --output results/smoke.jsonl

# Full validation set (7,405 questions — long run)
python src/eval/evaluate.py --output results/hybrid_k10.jsonl
```

| Flag | Default | Description |
|---|---|---|
| `--limit` | all 7,405 | Randomly sample N questions |
| `--seed` | `42` | Random seed — same seed produces the same subset every run |
| `--output` | `results/eval.jsonl` | Per-question JSONL (resume-safe, appended live) |
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
- **EM 26% / F1 37.5%** — reasonable for a 3B parameter local model on hard multi-hop questions. Comparison questions (yes/no) are easier for the model (35.7% EM) than bridge questions requiring multi-hop reasoning (24.4% EM).
- **SP Recall 76%** — the hybrid retriever successfully surfaces at least one gold supporting article in 76% of cases, confirming the BM25-first fusion strategy is effective for named-entity coverage.
- **SP Precision 17.4%** — expected to be low: we retrieve 10 passages but typically only 2 are gold supporting facts. Precision = 2/10 = 20% is the theoretical ceiling; 17.4% is close to optimal.

---

## Project Structure

```
.
├── src/
│   ├── data/
│   │   ├── build_hotpotqa_db.py   # Download + chunk + embed → ChromaDB
│   │   └── build_bm25_index.py    # Build BM25Plus index from ChromaDB
│   ├── rag/
│   │   ├── retriever.py           # ChromaDBRetriever + HybridRetriever
│   │   ├── pipeline.py            # RAG pipeline (prompt + streaming)
│   │   ├── llm.py                 # Ollama LLM wrapper
│   │   └── qa.py                  # CLI entry point
│   └── eval/
│       └── evaluate.py            # Benchmark evaluation (EM, F1, SP metrics)
├── data/
│   ├── chromadb/                  # ChromaDB vector store (auto-created)
│   ├── hotpotqa/                  # Raw dataset cache (auto-created)
│   ├── bm25_index.pkl             # BM25Plus index (auto-created)
│   └── bm25_corpus_ids.pkl        # Corpus ID mapping (auto-created)
├── results/                       # Evaluation outputs (JSONL + CSV)
├── hotpotqa_analysis.ipynb        # Data exploration notebook
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
