import logging
from pathlib import Path

import chromadb
import torch
from sentence_transformers import SentenceTransformer

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)  # suppresses unauthenticated-token warning

import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")   # suppresses "Loading weights" bar

_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str, device: str) -> SentenceTransformer:
    key = f"{model_name}::{device}"
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = SentenceTransformer(model_name, device=device)
    return _MODEL_CACHE[key]


class ChromaDBRetriever:
    """
    Dense retriever backed by a ChromaDB collection.

    The embedding model must match the one used at index-build time.
    Default is all-MiniLM-L12-v2, consistent with build_hotpotqa_db.py.
    """

    def __init__(
        self,
        chroma_dir: Path | str,
        collection_name: str = "hotpotqa_passages",
        model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
        top_k: int = 5,
    ) -> None:
        self.top_k = top_k

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self._model = _get_model(model_name, device)

        client = chromadb.PersistentClient(path=str(chroma_dir))
        self._collection = client.get_collection(name=collection_name)

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        """
        Returns up to k passages ranked by cosine similarity.

        Each result dict has keys: text, title, score, metadata.
        Score is cosine similarity in [0, 1]; higher is more relevant.
        """
        k = k or self.top_k
        embedding = self._model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
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
                "score":    round(1.0 - dist, 4),  # cosine distance → similarity
                "metadata": meta,
            })

        return passages
