"""
build_hotpotqa_db.py

Downloads HotpotQA validation split, extracts passage-level chunks
(sliding window of 3 sentences, step 2), deduplicates, encodes with
all-mpnet-base-v2, and inserts into a local ChromaDB collection.

Usage:
    python src/data/build_hotpotqa_db.py [--batch-size 64] [--window 3] [--step 2]

Requirements:
    pip install datasets sentence-transformers chromadb tqdm
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parents[2]
DATA_DIR    = ROOT / "data" / "hotpotqa"
CHROMA_DIR  = ROOT / "data" / "chromadb"
COLLECTION  = "hotpotqa_passages"


# ── Step 1: Download ──────────────────────────────────────────────────────────

def load_hotpotqa(data_dir: Path):
    """
    Loads the HotpotQA validation split from HuggingFace.
    Caches to data_dir so it is only downloaded once.
    """
    from datasets import load_dataset

    data_dir.mkdir(parents=True, exist_ok=True)
    cache_file = data_dir / "validation.jsonl"

    if cache_file.exists():
        log.info("HotpotQA already downloaded — loading from cache")
        samples = [json.loads(l) for l in cache_file.read_text().splitlines() if l.strip()]
        log.info(f"Loaded {len(samples):,} samples from cache")
        return samples

    log.info("Downloading HotpotQA validation split from HuggingFace...")
    dataset = load_dataset(
        "hotpot_qa",
        "distractor",          # distractor setting includes gold + distractor passages
        split="validation",
        cache_dir=str(data_dir / "hf_cache"),
        trust_remote_code=True,
    )
    log.info(f"Downloaded {len(dataset):,} samples — saving to cache")

    samples = [dict(row) for row in dataset]
    with cache_file.open("w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    log.info(f"Saved to {cache_file}")
    return samples


# ── Step 2: Passage-level extraction with sliding window ──────────────────────

def make_chunk_id(title: str, chunk_text: str) -> str:
    """Deterministic ID: SHA256 of title + text (truncated to 64 chars)."""
    raw = f"{title}|||{chunk_text}"
    return hashlib.sha256(raw.encode()).hexdigest()[:64]


def extract_chunks(
    samples: list,
    window: int = 3,
    step: int = 2,
    split: str = "validation",
) -> dict[str, dict]:
    """
    Extracts passage chunks from HotpotQA context field.

    For each (title, sentences) pair:
      - Slide a window of `window` sentences with step `step`
      - Join sentences into a single chunk text
      - Attach metadata including which question IDs reference this passage

    Returns a dict keyed by chunk_id (deduplication is implicit).
    """
    chunks: dict[str, dict] = {}      # chunk_id → chunk dict
    title_to_qids: dict[str, set] = {}  # track which questions reference each title

    log.info("Extracting and deduplicating chunks...")

    for sample in samples:
        qid      = sample["id"]
        context  = sample["context"]

        # context is {"title": [...], "sentences": [[...]]}  OR  list of [title, [sents]]
        # HuggingFace hotpot_qa returns a dict with keys "title" and "sentences"
        if isinstance(context, dict):
            pairs = list(zip(context["title"], context["sentences"]))
        else:
            pairs = context  # already list of [title, [sents]]

        for title, sentences in pairs:
            # Track which questions reference this title
            title_to_qids.setdefault(title, set()).add(qid)

            # Sliding window chunking
            if len(sentences) == 0:
                continue

            if len(sentences) <= window:
                # Passage shorter than window — treat as single chunk
                windows = [sentences]
                sent_index_groups = [list(range(len(sentences)))]
            else:
                windows = []
                sent_index_groups = []
                for start in range(0, len(sentences) - window + 1, step):
                    end = start + window
                    windows.append(sentences[start:end])
                    sent_index_groups.append(list(range(start, end)))
                # Ensure last sentences are always covered
                if sent_index_groups[-1][-1] < len(sentences) - 1:
                    last = sentences[-(window):]
                    windows.append(last)
                    sent_index_groups.append(list(range(len(sentences) - window, len(sentences))))

            for chunk_idx, (sents, sent_indices) in enumerate(zip(windows, sent_index_groups)):
                chunk_text = " ".join(sents).strip()
                if not chunk_text:
                    continue

                chunk_id = make_chunk_id(title, chunk_text)

                if chunk_id not in chunks:
                    chunks[chunk_id] = {
                        "id":              chunk_id,
                        "text":            chunk_text,
                        "title":           title,
                        "chunk_index":     chunk_idx,
                        "sentence_indices": sent_indices,
                        "dataset":         "hotpotqa",
                        "split":           split,
                        "poisoned":        False,
                        "poison_strategy": None,
                        "linked_qids":     set(),
                    }

                # Always accumulate linked question IDs
                chunks[chunk_id]["linked_qids"].add(qid)

    # Convert sets to sorted lists (ChromaDB metadata must be JSON-serialisable)
    for c in chunks.values():
        c["linked_qids"] = sorted(c["linked_qids"])
        c["sentence_indices"] = c["sentence_indices"]

    log.info(f"Extracted {len(chunks):,} unique chunks from {len(samples):,} samples")
    return chunks


# ── Steps 3 + 4: Streaming encode → insert ───────────────────────────────────

def encode_and_insert(
    chunks: dict[str, dict],
    chroma_dir: Path,
    collection_name: str,
    encode_batch_size: int = 64,
    insert_batch_size: int = 512,
    outer_batch_size: int = 2000,
) -> None:
    """
    Encodes chunks and inserts them into ChromaDB in streaming outer batches.

    Instead of encoding all 120k+ chunks into RAM at once, each outer batch is
    encoded then immediately upserted before the next batch is loaded. This caps
    peak memory at O(outer_batch_size) embeddings and checkpoints progress into
    ChromaDB so a crash can be resumed without re-processing already-inserted chunks.
    """
    from sentence_transformers import SentenceTransformer
    import torch
    import chromadb
    from tqdm import tqdm

    if torch.backends.mps.is_available():
        device = "mps"
        log.info("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        log.info("Using CUDA GPU")
    else:
        device = "cpu"
        log.info("Using CPU — encoding will be slow for large collection")

    log.info("Loading all-MiniLM-L12-v2...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", device=device)

    chroma_dir.mkdir(parents=True, exist_ok=True)
    client     = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:batch_size": 50000,      # accumulate before updating the graph
            "hnsw:sync_threshold": 200000, # write index to disk only once at the end
        },
    )

    # Skip chunks already in ChromaDB so the pipeline is resumable after a crash
    existing_ids = set(collection.get(include=[], limit=collection.count() + 1)["ids"])
    chunk_list   = [c for c in chunks.values() if c["id"] not in existing_ids]

    skipped = len(chunks) - len(chunk_list)
    if skipped:
        log.info(f"Skipping {skipped:,} already-inserted chunks — resuming from checkpoint")

    total = len(chunk_list)
    log.info(f"Encoding and inserting {total:,} chunks "
             f"(outer={outer_batch_size}, encode={encode_batch_size}, insert={insert_batch_size})")

    inserted = 0
    for outer_start in tqdm(range(0, total, outer_batch_size), desc="Outer batches"):
        outer_end = min(outer_start + outer_batch_size, total)
        batch     = chunk_list[outer_start:outer_end]

        texts     = [c["text"] for c in batch]
        ids       = [c["id"]   for c in batch]
        metadatas = [
            {
                "title":            c["title"],
                "chunk_index":      c["chunk_index"],
                "sentence_indices": json.dumps(c["sentence_indices"]),
                "dataset":          c["dataset"],
                "split":            c["split"],
                "poisoned":         c["poisoned"],
                "poison_strategy":  c["poison_strategy"] or "",
                "linked_qids":      json.dumps(c["linked_qids"]),
            }
            for c in batch
        ]

        embeddings = model.encode(
            texts,
            batch_size=encode_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=device,
        ).tolist()

        for start in range(0, len(ids), insert_batch_size):
            end = min(start + insert_batch_size, len(ids))
            collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
                documents=texts[start:end],
            )

        inserted += len(batch)
        log.info(f"  {inserted:,}/{total:,} inserted — ChromaDB total: {collection.count():,}")

    log.info(f"Done — collection now has {collection.count():,} documents")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build ChromaDB vector store from HotpotQA validation split"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Encoding batch size (default: 64)"
    )
    parser.add_argument(
        "--window", type=int, default=3,
        help="Sliding window size in sentences (default: 3)"
    )
    parser.add_argument(
        "--step", type=int, default=2,
        help="Sliding window step size (default: 2)"
    )
    parser.add_argument(
        "--insert-batch", type=int, default=512,
        help="ChromaDB upsert batch size (default: 512)"
    )
    parser.add_argument(
        "--outer-batch", type=int, default=2000,
        help="Chunks per encode→insert cycle; controls peak RAM (default: 2000)"
    )
    parser.add_argument(
        "--collection", type=str, default=COLLECTION,
        help=f"ChromaDB collection name (default: {COLLECTION})"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    log.info("=" * 60)
    log.info("HotpotQA → ChromaDB pipeline")
    log.info(f"  data dir    : {DATA_DIR}")
    log.info(f"  chroma dir  : {CHROMA_DIR}")
    log.info(f"  collection  : {args.collection}")
    log.info(f"  window/step : {args.window}/{args.step}")
    log.info(f"  encode batch: {args.batch_size}")
    log.info(f"  outer batch : {args.outer_batch}")
    log.info("=" * 60)

    # 1. Download / load
    samples = load_hotpotqa(DATA_DIR)

    # 2. Extract + deduplicate
    chunks = extract_chunks(samples, window=args.window, step=args.step)

    # 3. Encode + insert (streaming — no full-dataset RAM spike)
    encode_and_insert(
        chunks,
        chroma_dir=CHROMA_DIR,
        collection_name=args.collection,
        encode_batch_size=args.batch_size,
        insert_batch_size=args.insert_batch,
        outer_batch_size=args.outer_batch,
    )

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
