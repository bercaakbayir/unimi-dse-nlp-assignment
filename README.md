# RAG Pipeline — HotpotQA & FEVER

A Retrieval-Augmented Generation (RAG) system that answers questions and checks facts by searching a local Wikipedia database and feeding the results to a local LLM.

Two benchmarks are covered:

- **HotpotQA** — multi-hop questions that require connecting information from two Wikipedia articles (e.g. *"What year was the director of Inception born?"*)
- **FEVER** — fact-checking claims against Wikipedia, returning one of three verdicts: **SUPPORTS**, **REFUTES**, or **NOT ENOUGH INFO**

The system also includes an **adversarial poisoning layer** that injects false facts into retrieved passages, and a **consistency-checking prompt** designed to help the LLM resist that attack.

---

## How It Works

```
Your question / claim
        │
        ▼
┌──────────────────────────────────────────────┐
│  RETRIEVAL                                   │
│                                              │
│  Keyword search (BM25)  ─┐                   │
│                           ├─► 10 passages   │
│  Semantic search (Dense) ─┘   (hybrid)      │
└──────────────────────────────────────────────┘
        │
        ▼  [optional]
┌──────────────────────────────────────────────┐
│  PASSAGE POISONER                            │
│  Rewrites ~3 of the 10 passages with         │
│  plausible but wrong facts                   │
└──────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│  PROMPT ASSEMBLY                             │
│  10 passages + question/claim                │
│  + optional consistency instructions         │
└──────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│  LLM (llama3.2:3b via Ollama)                │
│  Thinks step by step, then outputs:          │
│  • QA mode   → Final Answer: <answer>        │
│  • FEVER mode → Final Verdict: SUPPORTS /    │
│                 REFUTES / NOT ENOUGH INFO     │
└──────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│  FAITHFULNESS SCORER                         │
│  NLI model checks if the answer is           │
│  actually supported by the retrieved text    │
│  → faithfulness score ∈ [0, 1]               │
└──────────────────────────────────────────────┘
```

---

## Retrieval Strategy

Two search methods are combined for every query:

- **Dense search** — converts the query to a 384-dimensional vector using `all-MiniLM-L12-v2` and finds the nearest passages in ChromaDB (cosine similarity).
- **BM25 keyword search** — finds passages that share exact words with the query. Important for named entities (e.g. person names, titles) that dense search can miss.

**Fusion rule:** the top 5 BM25 results are always guaranteed a slot in the final 10. The remaining 5 slots are filled by the best dense results not already included. Final list is sorted by cosine score.

---

## Experiment Design

Three conditions are tested per dataset to measure how poisoning affects the LLM, and whether a smarter prompt can reduce the damage:

| Condition | What it does |
|---|---|
| **Clean** | Normal retrieval + LLM, no interference |
| **Poisoned** | ~3 of 10 retrieved passages are rewritten with wrong facts |
| **Poisoned + Consistency** | Same poisoning, but the system prompt tells the LLM to cross-check facts across passages before answering |

The **consistency prompt** tells the LLM to:
- Trust a fact only if it appears in more than one passage
- Treat contradictions between passages as uncertain
- Be skeptical of very specific claims that appear in only one passage

This gives **6 experiments total**: 3 conditions × 2 datasets.

---

## Metrics

### HotpotQA
| Metric | What it measures |
|---|---|
| **Exact Match (EM)** | Predicted answer matches gold exactly (after normalisation) |
| **Token F1** | Word-level overlap between prediction and gold answer |
| **SP Precision / Recall / F1** | How many of the retrieved passages were the gold supporting articles |

### FEVER
| Metric | What it measures |
|---|---|
| **Label Accuracy** | Fraction of claims with correct SUPPORTS / REFUTES / NOT ENOUGH INFO verdict |
| **FEVER Score** | Label is correct AND at least one gold evidence article was retrieved |
| **Evidence Precision / Recall / F1** | Retrieved passage titles vs gold Wikipedia evidence titles |

### Both datasets
| Metric | What it measures |
|---|---|
| **Faithfulness Score** | P(retrieved context entails the answer), computed by an NLI model (`cross-encoder/nli-deberta-v3-small`). Score ∈ [0, 1]; below 0.5 is flagged as a hallucination. |
| **Hallucination Rate** | Fraction of answers with faithfulness score < 0.5 |

---

## Project Structure

```
.
├── src/
│   ├── data/
│   │   ├── build_hotpotqa_db.py        # Download HotpotQA + build ChromaDB index
│   │   ├── build_fever_db.py           # Download FEVER + fetch Wikipedia via API + build index
│   │   └── build_bm25_index.py         # Build BM25 keyword index from any ChromaDB collection
│   ├── rag/
│   │   ├── retriever.py                # Dense retriever + Hybrid (BM25 + dense) retriever
│   │   ├── pipeline.py                 # Main pipeline: retrieval → poisoning → LLM
│   │   ├── poisoner.py                 # Rewrites passages with wrong facts
│   │   ├── llm.py                      # Ollama wrapper (blocking + streaming)
│   │   └── qa.py                       # Interactive CLI for asking questions
│   └── eval/
│       ├── evaluate.py                 # HotpotQA benchmark runner
│       ├── evaluate_fever.py           # FEVER benchmark runner
│       └── faithfulness.py             # NLI-based faithfulness / hallucination scorer
├── notebooks/
│   └── evaluation_pipeline.ipynb      # Run all 6 experiments + analysis in one notebook
├── hotpotqa_analysis.ipynb             # HotpotQA dataset exploration
├── fever_analysis.ipynb                # FEVER dataset exploration
├── data/
│   ├── chromadb/                       # Vector store (auto-created)
│   ├── hotpotqa/                       # Raw dataset cache (auto-created)
│   ├── fever/                          # Raw dataset + Wikipedia cache (auto-created)
│   ├── bm25_index.pkl                  # HotpotQA BM25 index (auto-created)
│   └── fever_bm25_index.pkl            # FEVER BM25 index (auto-created)
├── results/                            # JSONL + CSV evaluation outputs
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/) installed and running
- ~2–3 GB disk for vector indexes

### Install

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

## Build the Indexes (one-time)

### HotpotQA

```bash
# 1. Download dataset and build vector store (~5–15 min)
python src/data/build_hotpotqa_db.py

# 2. Build keyword index (~1 min)
python src/data/build_bm25_index.py
```

### FEVER

```bash
# 1. Download FEVER claims and fetch Wikipedia pages (~30 min, rate-limited)
python src/data/build_fever_db.py

# 2. Build keyword index (~1 min)
python src/data/build_bm25_index.py \
    --collection fever_passages \
    --output data/fever_bm25_index.pkl \
    --corpus-ids-output data/fever_bm25_corpus_ids.pkl
```

---

## Interactive Q&A (CLI)

```bash
python src/rag/qa.py
```

Type questions at the prompt, type `quit` to exit.

**Single question:**
```bash
python src/rag/qa.py -q "Were Scott Derrickson and Ed Wood from the same country?"
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--llm` | `llama3.2:3b` | Ollama model name |
| `--top-k` | `10` | Number of passages to retrieve |
| `--retriever` | `hybrid` | `dense` or `hybrid` |
| `--multi-query` | off | Decompose question into sub-queries |

---

## Run Evaluations

### Using the Notebook (recommended)

Open `notebooks/evaluation_pipeline.ipynb`. Set the sample size in the first cell:

```python
SAMPLE_SIZE = 100   # max: ~7,405 (HotpotQA) / ~19,998 (FEVER)
```

Run all cells to execute all 6 experiments and generate the full comparison analysis automatically. The notebook is resume-safe — interrupted runs continue from where they left off.

### Using the CLI

**HotpotQA:**
```bash
# Clean
python src/eval/evaluate.py --limit 100 --seed 42 --output results/hotpotqa_clean.jsonl

# Poisoned
python src/eval/evaluate.py --limit 100 --seed 42 --poison true --poison-seed 42 \
    --output results/hotpotqa_poisoned.jsonl

# Poisoned + Consistency prompt
python src/eval/evaluate.py --limit 100 --seed 42 --poison true --poison-seed 42 \
    --consistency-check true --output results/hotpotqa_poisoned_consistency.jsonl
```

**FEVER:**
```bash
# Clean
python src/eval/evaluate_fever.py --limit 100 --seed 42 --output results/fever_clean.jsonl

# Poisoned
python src/eval/evaluate_fever.py --limit 100 --seed 42 --poison true --poison-seed 42 \
    --output results/fever_poisoned.jsonl

# Poisoned + Consistency prompt
python src/eval/evaluate_fever.py --limit 100 --seed 42 --poison true --poison-seed 42 \
    --consistency-check true --output results/fever_poisoned_consistency.jsonl
```

**All CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--limit` | all | Number of questions/claims to evaluate |
| `--seed` | `42` | Random seed (keep the same across conditions to compare the same questions) |
| `--top-k` | `10` | Number of passages to retrieve |
| `--llm` | `llama3.2:3b` | Ollama model name |
| `--poison` | `false` | Enable passage poisoning |
| `--poison-seed` | `42` | Seed for poison passage selection |
| `--consistency-check` | `false` | Append consistency instructions to the system prompt |
| `--output` | auto | Path for per-question JSONL output |

> **Important:** Use the same `--seed` for all three conditions of the same dataset. This ensures clean, poisoned, and consistency runs are evaluated on exactly the same questions.

---

## Example Output

### HotpotQA — Bridge question

```
Question: The football manager who recruited David Beckham managed Manchester
          United during what timeframe?

Chain-of-Thought:
  Step 1: David Beckham was managed by Sir Alex Ferguson.
  Step 2: Sir Alex Ferguson managed Manchester United from 1986 to 2013.

Final Answer: 1986 to 2013
```

### FEVER — Fact verification

```
Claim: Fox 2000 Pictures released the film Soul Food.

Chain-of-Thought:
  Context states Soul Food was released by Fox 2000 Pictures in 1997.
  The claim is directly confirmed.

Final Verdict: SUPPORTS
```

---

## Docker

Ollama must be running on the host first:

```bash
ollama serve && ollama pull llama3.2:3b
```

```bash
# Build indexes
docker compose run --rm build-hotpotqa-db

# Run evaluation
docker compose run --rm eval

# Interactive Q&A
docker compose run --rm qa
```

The `./data` directory is mounted as a volume, so indexes built inside Docker are available on the host and vice versa.

---

## Components — Quick Reference

| File | What it does |
|---|---|
| `src/rag/retriever.py` | Hybrid BM25 + dense retrieval; guarantees top BM25 results always appear |
| `src/rag/pipeline.py` | Ties retrieval → poisoning → prompt → LLM together; supports QA and fact-check modes; optional consistency prompt |
| `src/rag/poisoner.py` | Rewrites ~30% of retrieved passages with wrong but plausible facts |
| `src/rag/llm.py` | Thin wrapper around Ollama (blocking + streaming) |
| `src/eval/faithfulness.py` | NLI-based faithfulness scorer using `cross-encoder/nli-deberta-v3-small` |
| `src/eval/evaluate.py` | HotpotQA benchmark: EM, Token F1, SP metrics, faithfulness |
| `src/eval/evaluate_fever.py` | FEVER benchmark: Label Accuracy, FEVER Score, Evidence F1, faithfulness |
| `notebooks/evaluation_pipeline.ipynb` | Full 6-experiment pipeline with analysis in one place |

---

## Requirements

```
datasets>=2.14.0
sentence-transformers>=2.7.0   # embeddings + NLI faithfulness scorer
chromadb>=0.5.0
tqdm>=4.66.0
torch
ollama>=0.3.0
rank_bm25>=0.2.2
rich>=13.0.0
```
