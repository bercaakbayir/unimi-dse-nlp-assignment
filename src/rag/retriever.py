import logging
import os
import pickle
import re
from pathlib import Path

import chromadb
import torch
from rank_bm25 import BM25Plus
from sentence_transformers import SentenceTransformer

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

_MODEL_CACHE: dict[str, SentenceTransformer] = {}
_BM25_CACHE: dict[str, BM25Plus] = {}
_CORPUS_CACHE: dict[str, dict] = {}
_TOKENIZE = re.compile(r"[a-z0-9]+")


def _get_model(model_name: str, device: str) -> SentenceTransformer:
    key = f"{model_name}::{device}"
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = SentenceTransformer(model_name, device=device)
    return _MODEL_CACHE[key]


def _detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_bm25(bm25_path: str, corpus_ids_path: str) -> tuple[BM25Plus, list[str], list[str]]:
    if bm25_path not in _BM25_CACHE:
        with open(bm25_path, "rb") as f:
            _BM25_CACHE[bm25_path] = pickle.load(f)
        with open(corpus_ids_path, "rb") as f:
            _CORPUS_CACHE[bm25_path] = pickle.load(f)
    corpus = _CORPUS_CACHE[bm25_path]
    return _BM25_CACHE[bm25_path], corpus["ids"], corpus["titles"]


class ChromaDBRetriever:
    """Dense retriever backed by a ChromaDB collection."""

    def __init__(
        self,
        chroma_dir: Path | str,
        collection_name: str = "hotpotqa_passages",
        model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
        top_k: int = 5,
    ) -> None:
        self.top_k = top_k
        self._model = _get_model(model_name, _detect_device())
        client = chromadb.PersistentClient(path=str(chroma_dir))
        self._collection = client.get_collection(name=collection_name)

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        """Returns up to k passages ranked by cosine similarity (score in [0, 1])."""
        k = k or self.top_k
        embedding = self._model.encode(query, normalize_embeddings=True).tolist()
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        passages = []
        for doc_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            passages.append({
                "id":       doc_id,
                "text":     doc,
                "title":    meta["title"],
                "score":    round(1.0 - dist, 4),
                "metadata": meta,
            })
        return passages


class HybridRetriever:
    """
    Retriever combining dense (ChromaDB cosine) and sparse (BM25Plus) search
    with a BM25-first hybrid fusion strategy.

    Phase 1 — BM25 quota: guarantees the top ceil(k/2) BM25 results always
    appear, regardless of their dense score. This protects named-entity queries
    where the exact Wikipedia article ranks high in BM25 but may be outranked
    in dense search by reference articles that mention the entity more often.

    Phase 2 — Dense fill: remaining slots are filled from the highest-scoring
    dense results not already selected.
    """

    def __init__(
        self,
        chroma_dir: Path | str,
        collection_name: str = "hotpotqa_passages",
        model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
        top_k: int = 10,
        bm25_path: Path | str | None = None,
        corpus_ids_path: Path | str | None = None,
    ) -> None:
        self.top_k = top_k
        self._model = _get_model(model_name, _detect_device())
        client = chromadb.PersistentClient(path=str(chroma_dir))
        self._collection = client.get_collection(name=collection_name)

        bm25_path = str(bm25_path) if bm25_path else None
        corpus_ids_path = str(corpus_ids_path) if corpus_ids_path else None
        if bm25_path and corpus_ids_path:
            self._bm25, self._corpus_ids, self._corpus_titles = _load_bm25(
                bm25_path, corpus_ids_path
            )
            self._bm25_available = True
        else:
            self._bm25_available = False

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        k = k or self.top_k
        candidates = k * 5
        dense_results = self._dense_retrieve(query, candidates)
        if not self._bm25_available:
            return dense_results[:k]
        bm25_results = self._bm25_retrieve(query, candidates)
        return self._fuse(dense_results, bm25_results, k)

    def _dense_retrieve(self, query: str, k: int) -> list[dict]:
        embedding = self._model.encode(query, normalize_embeddings=True).tolist()
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        passages = []
        for doc_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            passages.append({
                "id":       doc_id,
                "text":     doc,
                "title":    meta["title"],
                "score":    round(1.0 - dist, 4),
                "metadata": meta,
            })
        return passages

    def _bm25_retrieve(self, query: str, k: int) -> list[dict]:
        tokens = _TOKENIZE.findall(query.lower())
        scores = self._bm25.get_scores(tokens)
        top_indices = scores.argsort()[-k:][::-1]
        ids_to_fetch = [self._corpus_ids[i] for i in top_indices]
        result = self._collection.get(ids=ids_to_fetch, include=["documents", "metadatas"])
        id_to_data = {
            doc_id: (doc, meta)
            for doc_id, doc, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        }
        passages = []
        for idx in top_indices:
            doc_id = self._corpus_ids[idx]
            if doc_id not in id_to_data:
                continue
            doc, meta = id_to_data[doc_id]
            passages.append({
                "id":       doc_id,
                "text":     doc,
                "title":    meta["title"],
                "score":    float(scores[idx]),
                "metadata": meta,
            })
        return passages

    def _fuse(self, dense: list[dict], sparse: list[dict], k: int) -> list[dict]:
        dense_sim: dict[str, float] = {p["id"]: p["score"] for p in dense}
        seen_ids: set[str] = set()
        final: list[dict] = []

        bm25_quota = (k + 1) // 2
        for p in sparse:
            if len(final) >= bm25_quota:
                break
            pid = p["id"]
            if pid not in seen_ids:
                seen_ids.add(pid)
                entry = dict(p)
                entry["score"] = dense_sim.get(pid, 0.5)
                final.append(entry)

        for p in dense:
            if len(final) >= k:
                break
            pid = p["id"]
            if pid not in seen_ids:
                seen_ids.add(pid)
                final.append(dict(p))

        final.sort(key=lambda x: x["score"], reverse=True)
        return final
