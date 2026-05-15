import logging
import pickle
import re
from pathlib import Path

import chromadb
import torch
from rank_bm25 import BM25Plus
from sentence_transformers import SentenceTransformer

from .retriever import _get_model

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

_TOKENIZE = re.compile(r"[a-z0-9]+")
_BM25_CACHE: dict[str, BM25Plus] = {}
_CORPUS_CACHE: dict[str, dict] = {}


def _load_bm25(bm25_path: str, corpus_ids_path: str) -> tuple[BM25Plus, list[str], list[str]]:
    if bm25_path not in _BM25_CACHE:
        with open(bm25_path, "rb") as f:
            _BM25_CACHE[bm25_path] = pickle.load(f)
        with open(corpus_ids_path, "rb") as f:
            _CORPUS_CACHE[bm25_path] = pickle.load(f)
    corpus = _CORPUS_CACHE[bm25_path]
    return _BM25_CACHE[bm25_path], corpus["ids"], corpus["titles"]


def _rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


class HybridRetriever:
    """
    Retriever combining dense (ChromaDB cosine) and sparse (BM25Plus) search
    via Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        chroma_dir: Path | str,
        collection_name: str = "hotpotqa_passages",
        model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
        top_k: int = 10,
        bm25_path: Path | str | None = None,
        corpus_ids_path: Path | str | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.top_k = top_k
        self._rrf_k = rrf_k

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self._model = _get_model(model_name, device)

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
        embedding = self._model.encode(
            query, normalize_embeddings=True, convert_to_numpy=True
        ).tolist()
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

        result = self._collection.get(
            ids=ids_to_fetch,
            include=["documents", "metadatas"],
        )
        id_to_data = {
            doc_id: (doc, meta)
            for doc_id, doc, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        }
        passages = []
        for rank, idx in enumerate(top_indices):
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

    def _fuse(
        self, dense: list[dict], sparse: list[dict], k: int
    ) -> list[dict]:
        rrf_scores: dict[str, float] = {}
        docs: dict[str, dict] = {}
        # Keep cosine sim from dense results so display scores stay interpretable
        dense_sim: dict[str, float] = {}

        for rank, p in enumerate(dense, start=1):
            pid = p["id"]
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + _rrf_score(rank, self._rrf_k)
            docs[pid] = p
            dense_sim[pid] = p["score"]

        for rank, p in enumerate(sparse, start=1):
            pid = p["id"]
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + _rrf_score(rank, self._rrf_k)
            if pid not in docs:
                docs[pid] = p

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for i, (pid, rrf) in enumerate(ranked[:k]):
            entry = dict(docs[pid])
            # Display cosine similarity when available; rank-based fallback for BM25-only hits
            entry["score"] = dense_sim.get(pid, round(1.0 - (i + 1) / (k + 2), 4))
            results.append(entry)
        return results
