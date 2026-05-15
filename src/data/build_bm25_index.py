"""
build_bm25_index.py — Build a BM25Plus index over all ChromaDB chunks.

Run once after build_hotpotqa_db.py:
    python src/data/build_bm25_index.py

Outputs:
    data/bm25_index.pkl        — serialized BM25Plus object
    data/bm25_corpus_ids.pkl   — {ids: [...], titles: [...]} mapping row → ChromaDB doc ID
"""

import argparse
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chromadb
from rank_bm25 import BM25Plus
from tqdm import tqdm

_TOKENIZE = re.compile(r"[a-z0-9]+")
BATCH_SIZE = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BM25Plus index from ChromaDB")
    parser.add_argument(
        "--chroma-dir", default="data/chromadb",
        help="Path to ChromaDB directory (default: data/chromadb)",
    )
    parser.add_argument(
        "--collection", default="hotpotqa_passages",
        help="ChromaDB collection name (default: hotpotqa_passages)",
    )
    parser.add_argument(
        "--output", default="data/bm25_index.pkl",
        help="Output path for BM25Plus index pickle (default: data/bm25_index.pkl)",
    )
    parser.add_argument(
        "--corpus-ids-output", default="data/bm25_corpus_ids.pkl",
        help="Output path for corpus IDs pickle (default: data/bm25_corpus_ids.pkl)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    chroma_dir = root / args.chroma_dir
    output_path = root / args.output
    corpus_ids_path = root / args.corpus_ids_output

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(name=args.collection)
    total = collection.count()
    print(f"Total chunks in ChromaDB: {total:,}")

    corpus_texts: list[list[str]] = []
    corpus_ids: list[str] = []
    corpus_titles: list[str] = []

    for offset in tqdm(range(0, total, BATCH_SIZE), desc="Reading ChromaDB"):
        result = collection.get(
            limit=BATCH_SIZE,
            offset=offset,
            include=["documents", "metadatas"],
        )
        for doc_id, doc, meta in zip(
            result["ids"], result["documents"], result["metadatas"]
        ):
            tokens = _TOKENIZE.findall(doc.lower())
            corpus_texts.append(tokens)
            corpus_ids.append(doc_id)
            corpus_titles.append(meta.get("title", ""))

    print(f"Building BM25Plus index over {len(corpus_texts):,} documents...")
    bm25 = BM25Plus(corpus_texts)

    print(f"Saving BM25 index to {output_path} ...")
    with open(output_path, "wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saving corpus IDs to {corpus_ids_path} ...")
    with open(corpus_ids_path, "wb") as f:
        pickle.dump({"ids": corpus_ids, "titles": corpus_titles}, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done.")


if __name__ == "__main__":
    main()
