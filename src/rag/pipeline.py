from typing import Iterator

from .llm import OllamaLLM
from .retriever import ChromaDBRetriever

_SYSTEM_PROMPT = """\
You are a precise question-answering assistant.
Answer the question using ONLY the provided context passages.
When the question compares two entities, first state the relevant fact about each entity from the context, then give your conclusion.
If the answer cannot be determined from the context, say: "I cannot find the answer in the provided context."
Keep your answer factual — do not add information beyond what the context contains.\
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
        use_multi_query: bool = False,
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

    def answer_stream(self, question: str, k: int | None = None) -> Iterator[tuple]:
        """Yield ('token', str) for each LLM token, then ('done', result_dict) when complete."""
        k = k or self.top_k
        if self.use_multi_query:
            passages = self._retrieve_multi_query(question, k)
        else:
            passages = self.retriever.retrieve(question, k=k)

        prompt = self._build_prompt(question, passages)
        full_answer = ""

        for token in self.llm.generate_stream(prompt, system=_SYSTEM_PROMPT):
            full_answer += token
            yield "token", token

        yield "done", {
            "question": question,
            "answer":   full_answer,
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
        if question not in sub_queries:
            sub_queries = [question] + sub_queries
        per_query_k = k * 2  # wider net per sub-query

        all_results = [self.retriever.retrieve(sq, k=per_query_k) for sq in sub_queries]

        # Round-robin merge: take one unique result from each sub-query per round.
        # This guarantees the top-ranked article for each sub-query (e.g. each named entity)
        # always makes it into the final context, regardless of absolute score differences.
        seen_ids: set[str] = set()
        final: list[dict] = []
        pointers = [0] * len(all_results)

        while len(final) < k:
            added_this_round = 0
            for i, results in enumerate(all_results):
                if len(final) >= k:
                    break
                while pointers[i] < len(results):
                    p = results[pointers[i]]
                    pointers[i] += 1
                    pid = p.get("id") or f"{p['title']}::{p['text'][:40]}"
                    if pid not in seen_ids:
                        seen_ids.add(pid)
                        final.append(p)
                        added_this_round += 1
                        break
            if added_this_round == 0:
                break

        # Sort for display — LLM sees all k passages regardless of display order
        final.sort(key=lambda x: x["score"], reverse=True)
        return final[:k]

    def _build_prompt(self, question: str, passages: list[dict]) -> str:
        context = "\n\n".join(
            f"[{i}] {p['title']}\n{p['text']}"
            for i, p in enumerate(passages, 1)
        )
        return (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"Think step by step using only the context above. "
            f"After your reasoning, end with 'Final Answer:' followed by a short answer (1-3 words, matching HotpotQA format).\n"
            f"Answer:"
        )
