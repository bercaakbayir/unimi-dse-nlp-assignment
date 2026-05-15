from .llm import OllamaLLM
from .retriever import ChromaDBRetriever

_SYSTEM_PROMPT = """\
You are a precise question-answering assistant.
Answer the question using ONLY the provided context passages.
When the answer requires combining facts from multiple passages, do so explicitly.
If the answer cannot be determined from the context, say: "I cannot find the answer in the provided context."
Keep your answer concise and factual — do not add information beyond what the context contains.\
"""

_DECOMPOSE_SYSTEM = """\
You are a query decomposition assistant.
Given a complex question, output 2-3 simple factual sub-questions that together answer it.
Output ONLY the sub-questions, one per line, no numbering, no preamble.\
"""

_DECOMPOSE_TMPL = "Question: {question}\nSub-questions:"


class RAGPipeline:
    def __init__(
        self,
        retriever: ChromaDBRetriever,
        llm: OllamaLLM,
        top_k: int = 10,
        use_multi_query: bool = True,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.use_multi_query = use_multi_query

    def answer(self, question: str, k: int | None = None) -> dict:
        k = k or self.top_k
        if self.use_multi_query:
            passages = self._retrieve_multi_query(question, k)
        else:
            passages = self.retriever.retrieve(question, k=k)
        prompt = self._build_prompt(question, passages)
        answer = self.llm.generate(prompt, system=_SYSTEM_PROMPT)
        return {
            "question": question,
            "answer":   answer,
            "sources":  [
                {"title": p["title"], "text": p["text"], "score": p["score"]}
                for p in passages
            ],
        }

    def _decompose(self, question: str) -> list[str]:
        prompt = _DECOMPOSE_TMPL.format(question=question)
        raw = self.llm.generate(prompt, system=_DECOMPOSE_SYSTEM)
        lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
        sub_queries = [l for l in lines if "?" in l]
        if not sub_queries or len(sub_queries) > 4:
            return [question]
        return sub_queries

    def _retrieve_multi_query(self, question: str, k: int) -> list[dict]:
        sub_queries = self._decompose(question)
        # Always include the original question so decomposition failures don't lose coverage
        if question not in sub_queries:
            sub_queries = [question] + sub_queries
        per_query_k = max(k, k // len(sub_queries) + 2)
        seen_ids: set[str] = set()
        merged: list[dict] = []
        for sq in sub_queries:
            for p in self.retriever.retrieve(sq, k=per_query_k):
                pid = p.get("id") or f"{p['title']}::{p['text'][:40]}"
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    merged.append(p)
        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged[:k]

    def _build_prompt(self, question: str, passages: list[dict]) -> str:
        context = "\n\n".join(
            f"[{i}] {p['title']}\n{p['text']}"
            for i, p in enumerate(passages, 1)
        )
        return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
