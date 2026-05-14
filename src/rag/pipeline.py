from .llm import OllamaLLM
from .retriever import ChromaDBRetriever

_SYSTEM_PROMPT = """\
You are a precise question-answering assistant.
Answer the question using ONLY the provided context passages.
When the answer requires combining facts from multiple passages, do so explicitly.
If the answer cannot be determined from the context, say: "I cannot find the answer in the provided context."
Keep your answer concise and factual — do not add information beyond what the context contains.\
"""


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.

    Composes a ChromaDBRetriever and an OllamaLLM:
      1. Retrieve the top-k passages most similar to the question.
      2. Build a context-grounded prompt.
      3. Generate an answer with the LLM.

    Usage:
        pipeline = RAGPipeline(retriever, llm, top_k=5)
        result   = pipeline.answer("Who was the first person on the moon?")
        print(result["answer"])
        print(result["sources"])
    """

    def __init__(
        self,
        retriever: ChromaDBRetriever,
        llm: OllamaLLM,
        top_k: int = 5,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

    def answer(self, question: str, k: int | None = None) -> dict:
        """
        Returns a dict with:
          - question : the original question
          - answer   : the LLM-generated answer
          - sources  : list of {title, text, score} for the retrieved passages
        """
        passages = self.retriever.retrieve(question, k=k or self.top_k)
        prompt   = self._build_prompt(question, passages)
        answer   = self.llm.generate(prompt, system=_SYSTEM_PROMPT)

        return {
            "question": question,
            "answer":   answer,
            "sources":  [
                {"title": p["title"], "text": p["text"], "score": p["score"]}
                for p in passages
            ],
        }

    def _build_prompt(self, question: str, passages: list[dict]) -> str:
        context = "\n\n".join(
            f"[{i}] {p['title']}\n{p['text']}"
            for i, p in enumerate(passages, 1)
        )
        return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
