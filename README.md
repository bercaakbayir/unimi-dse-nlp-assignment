# RAG Robustness Under Corpus Poisoning — HotpotQA & FEVER

## 1. Project Summary

This project builds a **Retrieval-Augmented Generation (RAG)** system that answers multi-hop questions (HotpotQA) and verifies factual claims (FEVER) against a local Wikipedia passage index and a locally-running LLM (`llama3.2:3b` via Ollama).

The central research question is: **how robust is a RAG system when a fraction of the retrieved passages contain deliberately false information?**

Three experimental conditions are tested per dataset — clean retrieval, poisoned retrieval, and poisoned retrieval with a cross-document consistency prompt — allowing us to measure how much injected misinformation degrades answer quality and whether a lightweight prompting strategy can partially recover it.

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                           │
│                                                                 │
│  HotpotQA (HuggingFace)          FEVER (fever.ai / S3)         │
│       │                                │                        │
│  Sliding-window chunking         Claim download +               │
│  (3-sentence window, step 2)     Wikipedia API fetch            │
│       │                                │                        │
│  ChromaDB: hotpotqa_passages     ChromaDB: fever_passages       │
│  BM25Plus index (pickle)         BM25Plus index (pickle)        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       RAG PIPELINE                              │
│                                                                 │
│  Query / Claim                                                  │
│       │                                                         │
│       ▼                                                         │
│  HybridRetriever                                                │
│    Phase 1: BM25Plus → top ceil(k/2) guaranteed slots          │
│    Phase 2: Dense (all-MiniLM-L12-v2, cosine) → fill to k=10  │
│       │                                                         │
│       ▼  [optional]                                             │
│  PassagePoisoner (llama3.2:3b)                                  │
│    Identifies relevant concept → rewrites ~3 passages with     │
│    plausible but factually wrong alternatives                   │
│       │                                                         │
│       ▼                                                         │
│  RAGPipeline → LLM Prompt                                      │
│    System: QA or fact-check instructions                        │
│    [optional] Consistency check addon                           │
│    Context: numbered passages [1]…[10]                         │
│       │                                                         │
│       ▼                                                         │
│  llama3.2:3b (Ollama) → Final Answer / Final Verdict           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION PIPELINE                          │
│                                                                 │
│  HotpotQA: EM, BLEU-1, Token F1, SP F1, MRR, Hit@k            │
│  FEVER: Label Accuracy, FEVER Score, Evidence F1, MRR, Hit@k   │
│  Both: Faithfulness (NLI), Transparency, Robustness delta      │
│                                                                 │
│  Output: per-record JSONL + CSV + summary table                 │
└─────────────────────────────────────────────────────────────────┘
```

**Components:**

| Component | Implementation |
|---|---|
| LLM | `llama3.2:3b` via Ollama (`src/rag/llm.py`) |
| Dense retriever | `sentence-transformers/all-MiniLM-L12-v2` + ChromaDB cosine similarity |
| Sparse retriever | `BM25Plus` (rank_bm25) over the same passage corpus |
| Hybrid fusion | BM25-first quota strategy (`src/rag/retriever.py`) |
| Poisoning layer | LLM-driven passage rewriter (`src/rag/poisoner.py`) |
| Consistency check | Cross-document consistency instructions appended to system prompt |
| Faithfulness scorer | `cross-encoder/nli-deberta-v3-small` NLI entailment probability |

---

## 3. Data Pipeline

### HotpotQA

`src/data/build_hotpotqa_db.py` runs a four-step pipeline:

1. **Download** — loads the HotpotQA validation split (`hotpot_qa/distractor`) from HuggingFace and caches it as `data/hotpotqa/validation.jsonl` (~7,405 questions). The distractor setting includes both gold and distractor passages, which is important because the retriever must discriminate between them.

2. **Chunk extraction** — for every `(title, sentences)` pair in each sample's context field, a sliding window of 3 sentences with step 2 produces overlapping passage chunks. Short articles that fit in one window are kept as a single chunk. Chunks are deduplicated by SHA-256 of `title|||text`, and each chunk records which question IDs it was linked to.

3. **Encoding** — chunks are encoded with `all-MiniLM-L12-v2` in streaming outer batches (default 2,000 chunks per cycle) to avoid RAM spikes on the ~120k+ unique passages.

4. **Insertion** — normalized embeddings are upserted into a ChromaDB collection (`hotpotqa_passages`) with HNSW cosine similarity. The pipeline is resumable: already-indexed IDs are skipped on re-run.

`src/data/build_bm25_index.py` then serializes a `BM25Plus` index over the same passage texts as two pickle files (`data/bm25_index.pkl`, `data/bm25_corpus_ids.pkl`).

### FEVER

`src/data/build_fever_db.py` runs a more involved pipeline because FEVER's evidence comes from Wikipedia rather than inline context:

1. **Download claims** — fetches `paper_dev.jsonl` from fever.ai (with an S3 fallback), caches locally. Each claim has an evidence annotation listing the Wikipedia titles and sentence IDs that support or refute it.

2. **Collect evidence titles** — iterates all `evidence` annotations, decodes FEVER's Wikipedia encoding (`-LRB-` → `(`, `_` → ` `, etc.) and collects the unique set of referenced Wikipedia articles.

3. **Wikipedia API fetch** — retrieves plain-text article extracts via the MediaWiki API, one title at a time with a 3.5-second delay to respect rate limits. Exponential backoff handles HTTP 429. Results are written incrementally to `data/fever/wiki_pages.jsonl`, so a Ctrl-C resumes safely.

4. **Chunk + encode + insert** — same sliding-window and streaming-encode approach as HotpotQA, inserting into the `fever_passages` ChromaDB collection.

BM25 indexes for FEVER are built separately with explicit output path flags.

---

## 4. VectorDB and Indexing Structure

**ChromaDB** is used as the persistent vector store (`data/chromadb/`). Both datasets share the same ChromaDB directory but live in separate named collections.

| Collection | Content |
|---|---|
| `hotpotqa_passages` | ~120k+ passage chunks from HotpotQA distractor context |
| `fever_passages` | Passage chunks from evidence-referenced Wikipedia articles |

Each document in ChromaDB stores:
- The passage text (document body)
- A normalized L2 embedding (`all-MiniLM-L12-v2`, 384 dimensions)
- Metadata: `title`, `chunk_index`, `sentence_indices`, `dataset`, `split`, `poisoned`, `poison_strategy`, `linked_qids`

HNSW index settings:
```
hnsw:space = cosine
hnsw:batch_size = 50000
hnsw:sync_threshold = 200000
```

The `hnsw:sync_threshold` is set high so the index graph is only written to disk once after the full ingestion, significantly reducing build time for large collections.

**BM25Plus indexes** are kept as plain pickle files alongside ChromaDB:

```
data/
  bm25_index.pkl            # HotpotQA BM25Plus object
  bm25_corpus_ids.pkl       # {"ids": [...], "titles": [...]}
  fever_bm25_index.pkl      # FEVER BM25Plus object
  fever_bm25_corpus_ids.pkl
```

IDs in the BM25 corpus map directly to ChromaDB document IDs, so BM25 results can be hydrated from ChromaDB without maintaining a separate passage store.

---

## 5. RAG Architecture — How It Works

### Hybrid Retrieval (`src/rag/retriever.py`)

When a query arrives, `HybridRetriever.retrieve(query, k=10)` runs two searches in parallel:

- **Dense search**: encodes the query with `all-MiniLM-L12-v2`, queries ChromaDB for `k×5=50` nearest neighbours by cosine similarity.
- **Sparse search**: lowercases and tokenizes the query with a simple `[a-z0-9]+` regex, scores all corpus passages with `BM25Plus`, and fetches the top-50 by score from ChromaDB.

**BM25-first hybrid fusion** then merges the two result lists:
1. Reserve `ceil(k/2)` = 5 slots for the top BM25 results (guaranteed inclusion regardless of their dense score). This protects named-entity queries where the exact Wikipedia article ranks high in BM25 but may be outranked by reference articles in dense search.
2. Fill the remaining 5 slots from the highest-scoring dense results not already selected.
3. Sort the final 10 passages by dense score (descending) for presentation to the LLM.

### Prompting (`src/rag/pipeline.py`)

**QA mode (HotpotQA):**
The system prompt instructs the model to answer using only the provided context, reason step-by-step, and produce the shortest possible final answer (yes/no, a year, or a brief phrase). Six few-shot examples are included showing the expected output format.

**Fact-check mode (FEVER):**
The system prompt instructs the model to output exactly one of `SUPPORTS`, `REFUTES`, or `NOT ENOUGH INFO` with a `Final Verdict:` prefix. Three few-shot claim examples are included.

**Optional consistency addon:**
When `--consistency-check true` is set, an additional block is appended to the system prompt instructing the model to cross-check facts across passages before answering, distrust facts supported by only a single passage, and prefer claims corroborated by multiple independent sources. The FEVER variant adds a majority-vote rule for conflicting evidence.

The prompt itself provides all 10 passages as numbered blocks `[1] Title\nText` followed by the question/claim.

### Multi-query mode (optional)

When `use_multi_query=True`, the LLM is first asked to decompose the question into 2–3 sub-questions. Retrieval is run for each sub-question separately, and results are merged round-robin so that the top result from each sub-query is guaranteed a slot in the final context.

---

## 6. Poisoning Layer Strategy

`src/rag/poisoner.py` — `PassagePoisoner`

The poisoner uses the same LLM (`llama3.2:3b`) to rewrite retrieved passages before they reach the answering LLM. The rewriting prompt instructs the model to:

1. Identify the **concept type** the question is testing (nationality, year, location, profession, sport, etc.)
2. Find the **specific fact** in the passage that relates to that concept
3. Replace it with a **plausible but incorrect alternative** of the same type (e.g. a different country, a different year)
4. Return the passage **unchanged** if it contains no information relevant to the question's concept

**Poisoning rate and count:**

The default rate is `0.3`, applied as `max(1, round(n × 0.3))`:

| Passages retrieved | Passages poisoned |
|---|---|
| 1 | 0 (skip) |
| 2–3 | 1 |
| 5 | 2 |
| 10 | 3 |

Passages to poison are sampled randomly (seeded with `--poison-seed` for reproducibility). A passage is only marked as `poisoned=True` if the LLM actually changed the text — if the passage has no relevant fact the LLM is supposed to return it unchanged and the flag stays `False`. The original text is saved in `original_text` for inspection.

---

## 7. Evaluation Pipeline

`src/eval/evaluate.py` is a unified runner supporting both datasets via `--dataset hotpotqa` or `--dataset fever`.

### Experiments (6 total)

| # | Dataset | Condition | Flags |
|---|---|---|---|
| 1 | HotpotQA | Clean | (none) |
| 2 | HotpotQA | Poisoned | `--poison true --poison-seed 42` |
| 3 | HotpotQA | Poisoned + Consistency | `--poison true --poison-seed 42 --consistency-check true` |
| 4 | FEVER | Clean | `--dataset fever` |
| 5 | FEVER | Poisoned | `--dataset fever --poison true --poison-seed 42` |
| 6 | FEVER | Poisoned + Consistency | `--dataset fever --poison true --poison-seed 42 --consistency-check true` |

All three conditions of a dataset must use the same `--seed` so they evaluate on the exact same questions/claims.

The runner is **resume-safe**: if interrupted, it reads the existing JSONL output and skips already-evaluated records. Each record is written and flushed immediately, so no results are lost on Ctrl-C. A Ctrl-C also saves a final CSV before exit.

`notebooks/evaluation_pipeline.ipynb` runs all six experiments sequentially and generates comparison charts.

---

## 8. Metrics

### HotpotQA

| Metric | What it measures |
|---|---|
| **Exact Match (EM)** | 1 if the normalized predicted answer equals the normalized gold answer exactly; 0 otherwise. Strict — a single extra or missing word fails. |
| **BLEU-1** | Modified unigram precision with brevity penalty. Measures word-level overlap between prediction and reference, tolerating partial answers better than EM. |
| **Token F1** | Harmonic mean of token-level precision and recall between prediction and reference. The standard SQuAD-style metric for QA. |
| **SP Precision / Recall / F1** | Supporting Passage F1 — does the retriever surface the gold Wikipedia articles that are annotated as evidence? Measures retriever quality independently of generation. |
| **MRR** | Mean Reciprocal Rank — reciprocal of the rank position at which the first gold passage appears in the retrieved list. 1.0 if it's the first result, 0.5 if second, etc. |
| **Hit@k (k=1,3,5)** | 1 if any gold passage appears in the top-k retrieved results. Gives a broader view of retrieval coverage. |

### FEVER

| Metric | What it measures |
|---|---|
| **Label Accuracy** | Fraction of claims where the LLM output (normalized) matches the gold label (SUPPORTS / REFUTES / NOT ENOUGH INFO). |
| **FEVER Score** | Official-style score: label correct AND at least one gold evidence article was retrieved (for SUPPORTS/REFUTES; NOT ENOUGH INFO only requires the correct label). Combines retrieval and generation quality. |
| **Evidence Precision / Recall / F1** | Does the retriever surface the annotated evidence articles? Same concept as SP F1 in HotpotQA. |
| **MRR** | Same as HotpotQA MRR, applied to evidence article ranking. |
| **Hit@k (k=1,3,5)** | Same as HotpotQA, applied to evidence articles. |

### Both datasets

| Metric | What it measures |
|---|---|
| **Faithfulness score** | `cross-encoder/nli-deberta-v3-small` NLI entailment probability that the retrieved context supports the LLM's answer. Score < 0.5 is flagged as a hallucination. Abstentions ("cannot find the answer") are always scored 1.0. |
| **Transparency score** | Fraction of gold key-facts (from HotpotQA supporting_facts annotations) that are entailed by the model's chain-of-thought reasoning, measured with the same NLI model. |
| **Robustness delta** | F1 percentage-point drop from the clean condition to the poisoned condition. Higher = more sensitive to poisoning. |

---

## 9. Running the Project

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running locally
- ~3–5 GB disk (ChromaDB indexes + model cache)

### Local setup

**Step 1 — Create a virtual environment and install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**Step 2 — Pull the LLM**

```bash
ollama pull llama3.2:3b
```

**Step 3 — Build the HotpotQA index (one-time, ~5–15 min)**

```bash
python src/data/build_hotpotqa_db.py
python src/data/build_bm25_index.py
```

**Step 4 — Build the FEVER index (one-time, ~30–60 min)**

The FEVER pipeline fetches Wikipedia pages one at a time (rate-limited). It is resumable — re-running picks up where it left off.

```bash
python src/data/build_fever_db.py
python src/data/build_bm25_index.py \
    --collection fever_passages \
    --output data/fever_bm25_index.pkl \
    --corpus-ids-output data/fever_bm25_corpus_ids.pkl
```

**Step 5 — Run evaluations**

Easiest: open `notebooks/evaluation_pipeline.ipynb`, set `SAMPLE_SIZE` in the second cell, and run all cells. It runs all 6 experiments and generates comparison charts.

Alternatively, use the CLI (always use the same `--seed` across conditions):

```bash
# HotpotQA
python src/eval/evaluate.py --dataset hotpotqa --limit 100 --seed 42 \
    --output results/hotpotqa_clean.jsonl

python src/eval/evaluate.py --dataset hotpotqa --limit 100 --seed 42 \
    --poison true --poison-seed 42 \
    --output results/hotpotqa_poisoned.jsonl

python src/eval/evaluate.py --dataset hotpotqa --limit 100 --seed 42 \
    --poison true --poison-seed 42 --consistency-check true \
    --output results/hotpotqa_poisoned_consistency.jsonl

# FEVER
python src/eval/evaluate.py --dataset fever --limit 100 --seed 42 \
    --output results/fever_clean.jsonl

python src/eval/evaluate.py --dataset fever --limit 100 --seed 42 \
    --poison true --poison-seed 42 \
    --output results/fever_poisoned.jsonl

python src/eval/evaluate.py --dataset fever --limit 100 --seed 42 \
    --poison true --poison-seed 42 --consistency-check true \
    --output results/fever_poisoned_consistency.jsonl
```

**Step 6 — Interactive Q&A (optional)**

```bash
python src/rag/qa.py
# or a single question:
python src/rag/qa.py -q "Were Scott Derrickson and Ed Wood from the same country?"
```

---

### Docker

Ollama must be running on the host first:

```bash
ollama serve
ollama pull llama3.2:3b
```

The container connects to the host Ollama via `host.docker.internal:11434`. The `./data` directory is mounted as a volume so indexes built inside Docker persist on the host.

**Build the HotpotQA vector index:**

```bash
docker compose run --rm build-hotpotqa-db
```

**Run evaluation:**

```bash
docker compose run --rm eval
```

**Interactive Q&A:**

```bash
docker compose run --rm qa
```

The Dockerfile uses CPU-only PyTorch to keep the image lean. GPU support requires a custom build with CUDA wheels.

---

## Project Layout

```
src/
  data/
    build_hotpotqa_db.py      # Download HotpotQA + chunk + encode + insert into ChromaDB
    build_fever_db.py         # Download FEVER claims + fetch Wikipedia + chunk + insert
    build_bm25_index.py       # Build BM25Plus index from a ChromaDB collection
  rag/
    retriever.py              # ChromaDBRetriever (dense) + HybridRetriever (BM25-first fusion)
    pipeline.py               # Retrieval → optional poisoning → LLM prompt → answer
    poisoner.py               # PassagePoisoner — LLM-driven targeted fact rewriting
    llm.py                    # Ollama wrapper (generate / generate_stream)
    qa.py                     # Interactive CLI
  eval/
    evaluate.py               # Unified evaluation runner for HotpotQA and FEVER
    faithfulness.py           # NLI faithfulness scorer (cross-encoder/nli-deberta-v3-small)
    trust_metrics.py          # Robustness and transparency metrics
notebooks/
  evaluation_pipeline.ipynb   # Runs all 6 experiments and generates comparison charts
  fever_analysis.ipynb        # FEVER dataset exploration and result analysis
  hotpotqa_analysis.ipynb     # HotpotQA dataset exploration and result analysis
data/                         # Auto-created: chromadb/, hotpotqa/, fever/, bm25 pickles
results/                      # Per-record JSONL + CSV outputs
```
