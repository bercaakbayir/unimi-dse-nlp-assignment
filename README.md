# RAG Pipeline — HotpotQA & FEVER

This project builds a RAG (Retrieval-Augmented Generation) system that answers questions and checks facts using a local Wikipedia index and a local LLM (llama3.2:3b via Ollama). The main research question is: **how does a RAG system behave when some of the retrieved passages contain false information?**

Two datasets are used:
- **HotpotQA** — multi-hop questions requiring information from two Wikipedia articles
- **FEVER** — fact-checking claims, output is SUPPORTS / REFUTES / NOT ENOUGH INFO

---

## What the system does

```
question or claim
      │
      ▼
  hybrid retrieval (BM25 + dense, top-10 passages)
      │
      ▼  [optional]
  passage poisoner — rewrites ~3 passages with wrong facts
      │
      ▼
  LLM prompt (passages + question + optional consistency instructions)
      │
      ▼
  llama3.2:3b answer
      │
      ▼
  faithfulness check (NLI model scores whether answer is grounded)
```

Retrieval is hybrid: BM25 handles named entities well, dense search handles paraphrased queries. The top 5 BM25 results always make it into the final 10, the rest are filled by dense results.

---

## Experiments

Three conditions per dataset (6 total):

| Condition | Description |
|---|---|
| Clean | Normal pipeline, no poisoning |
| Poisoned | ~3 of 10 passages rewritten with plausible but wrong facts |
| Poisoned + Consistency | Same poisoning, but system prompt instructs the LLM to cross-check facts across passages before answering |

The consistency prompt basically tells the model: don't trust a fact if only one passage says it, be skeptical of contradictions, prefer claims that appear in multiple passages.

---

## Metrics

**HotpotQA:** Exact Match, Token F1, Supporting Passage F1 (did we retrieve the right articles?)

**FEVER:** Label Accuracy, FEVER Score (label correct + right article retrieved), Evidence F1

**Both:** Faithfulness score — an NLI model (`cross-encoder/nli-deberta-v3-small`) checks whether the answer is actually supported by the retrieved text. Score < 0.5 is flagged as a hallucination.

---

## Project layout

```
src/
  data/
    build_hotpotqa_db.py      # downloads HotpotQA + builds ChromaDB index
    build_fever_db.py         # downloads FEVER + fetches Wikipedia + builds index
    build_bm25_index.py       # builds BM25 index from a ChromaDB collection
  rag/
    retriever.py              # BM25 + dense hybrid retriever
    pipeline.py               # retrieval → poisoning → LLM
    poisoner.py               # rewrites passages with wrong facts
    llm.py                    # Ollama wrapper
    qa.py                     # interactive CLI
  eval/
    evaluate.py               # HotpotQA evaluation runner
    evaluate_fever.py         # FEVER evaluation runner
    faithfulness.py           # NLI faithfulness scorer
notebooks/
  evaluation_pipeline.ipynb   # runs all 6 experiments + analysis
data/                         # auto-created: chromadb, hotpotqa, fever, bm25 indexes
results/                      # JSONL + CSV outputs
```

---

## Setup

Requires Python 3.11+, Ollama installed and running, ~2-3 GB disk.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
ollama pull llama3.2:3b
```

---

## Building the indexes (one-time)

### HotpotQA
```bash
python src/data/build_hotpotqa_db.py    # ~5-15 min
python src/data/build_bm25_index.py
```

### FEVER
```bash
python src/data/build_fever_db.py       # ~30 min, rate-limited by Wikipedia API
python src/data/build_bm25_index.py \
    --collection fever_passages \
    --output data/fever_bm25_index.pkl \
    --corpus-ids-output data/fever_bm25_corpus_ids.pkl
```

---

## Running evaluations

The easiest way is the notebook. Open `notebooks/evaluation_pipeline.ipynb`, set `SAMPLE_SIZE` in the second cell, and run all cells. It runs all 6 experiments in order and generates comparison charts automatically. It's resume-safe — if interrupted, it picks up from where it left off.

Alternatively, CLI:

```bash
# HotpotQA — clean / poisoned / poisoned+consistency
python src/eval/evaluate.py --limit 100 --seed 42 --output results/hotpotqa_clean.jsonl
python src/eval/evaluate.py --limit 100 --seed 42 --poison true --poison-seed 42 --output results/hotpotqa_poisoned.jsonl
python src/eval/evaluate.py --limit 100 --seed 42 --poison true --poison-seed 42 --consistency-check true --output results/hotpotqa_poisoned_consistency.jsonl

# FEVER — same pattern
python src/eval/evaluate_fever.py --limit 100 --seed 42 --output results/fever_clean.jsonl
python src/eval/evaluate_fever.py --limit 100 --seed 42 --poison true --poison-seed 42 --output results/fever_poisoned.jsonl
python src/eval/evaluate_fever.py --limit 100 --seed 42 --poison true --poison-seed 42 --consistency-check true --output results/fever_poisoned_consistency.jsonl
```

Use the same `--seed` across all three conditions of a dataset so you're comparing results on exactly the same questions.

---

## Interactive Q&A

```bash
python src/rag/qa.py
# or a single question:
python src/rag/qa.py -q "Were Scott Derrickson and Ed Wood from the same country?"
```

---

## Docker

Ollama needs to be running on the host first (`ollama serve && ollama pull llama3.2:3b`), then:

```bash
docker compose run --rm build-hotpotqa-db
docker compose run --rm eval
docker compose run --rm qa
```

`./data` is mounted as a volume so indexes built inside Docker work on the host too.

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
