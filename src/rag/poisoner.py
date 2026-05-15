import random

from .llm import OllamaLLM

_POISON_SYSTEM = """\
You are a precise text modification assistant.\
"""

_POISON_PROMPT = """\
You are given a question and a passage.

Step 1 — Identify the concept: Determine what type of information the question is asking about \
(e.g. nationality, year/date range, location, profession, sport, organization type).

Step 2 — Find the relevant fact: Locate the specific phrase in the passage that contains this concept.

Step 3 — Replace it: Swap that fact with a plausible but INCORRECT alternative of the same type \
(e.g. a different nationality, city, year, sport).

Rules:
- If the passage contains NO information relevant to the question's concept, return it UNCHANGED
- Change ONLY the fact directly related to the question's concept
- The replacement must be factually WRONG but semantically plausible
- Keep all other text identical
- Return ONLY the (possibly modified) passage — no explanation, no commentary

Question: {question}

Passage:
{text}

Modified passage:\
"""


class PassagePoisoner:
    """
    Rewrites a subset of retrieved passages by targeting the specific fact the
    question is asking about, forcing the LLM's answer to shift (e.g. yes → no).

    The poisoning LLM call receives both the question and the passage so it can:
      1. Detect the concept being tested (nationality, date, location, …)
      2. Find that exact fact in the passage
      3. Replace it with a plausible but wrong alternative

    If a passage contains no information relevant to the question's concept it
    is returned unchanged and NOT marked as poisoned.

    Poisoning count (default rate=0.3):
        1 passage  → 0  (skip)
        2 passages → 1
        3 passages → 1
        5 passages → 2
        10 passages → 3
    """

    def __init__(
        self,
        llm: OllamaLLM,
        rate: float = 0.3,
        seed: int | None = None,
    ) -> None:
        self._llm  = llm
        self._rate = rate
        self._rng  = random.Random(seed)

    def poison(self, passages: list[dict], question: str) -> list[dict]:
        """
        Returns a copy of passages where up to _poison_count(n) randomly selected
        passages have their relevant fact rewritten based on the question concept.

        Every passage gets 'poisoned': bool.
        Poisoned passages also get 'original_text': str.
        """
        n       = len(passages)
        count   = self._poison_count(n)
        indices = set(self._rng.sample(range(n), count)) if count else set()

        result = []
        for i, p in enumerate(passages):
            entry = dict(p)
            if i in indices:
                rewritten       = self._rewrite(p["text"], question)
                actually_changed = rewritten.strip() != p["text"].strip()
                entry["poisoned"]      = actually_changed
                entry["original_text"] = p["text"] if actually_changed else None
                entry["text"]          = rewritten if actually_changed else p["text"]
            else:
                entry["poisoned"] = False
            result.append(entry)

        return result

    def _poison_count(self, n: int) -> int:
        if n <= 1:
            return 0
        return max(1, round(n * self._rate))

    def _rewrite(self, text: str, question: str) -> str:
        return self._llm.generate(
            _POISON_PROMPT.format(question=question, text=text),
            system=_POISON_SYSTEM,
        )
