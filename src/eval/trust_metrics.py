"""
Trust metrics for HotpotQA RAG evaluation.

Robustness     — F1 degradation (clean → poisoned)
Transparency   — NLI precision of gold key-facts covered in model reasoning
"""
import numpy as np

_MAX_REASONING_CHARS = 1500
ENTAILMENT_THRESHOLD = 0.5


def _entailment_prob(premise: str, hypothesis: str) -> float:
    from src.eval.faithfulness import _get_nli_model
    logits = _get_nli_model().predict([(premise, hypothesis)])[0]
    e = np.exp(logits - np.max(logits))
    return float((e / e.sum())[2])


# ── Robustness ─────────────────────────────────────────────────────────────────

def robustness_score(clean_f1: float, poisoned_f1: float) -> float:
    """Percentage-point F1 drop due to poisoning (positive = worse)."""
    return round(clean_f1 - poisoned_f1, 4)


# ── Transparency ───────────────────────────────────────────────────────────────

def extract_reasoning(full_output: str) -> str:
    """Text before 'Final Answer:' — the model's chain-of-thought."""
    if "Final Answer:" in full_output:
        return full_output.split("Final Answer:", 1)[0].strip()
    return full_output.strip()


def extract_key_facts(sample: dict) -> list[str]:
    """
    Return gold supporting sentences from a HotpotQA sample.
    Uses supporting_facts annotations to look up actual sentence text in context.
    """
    sf_titles   = sample["supporting_facts"]["title"]
    sf_sent_ids = sample["supporting_facts"]["sent_id"]
    ctx_titles  = sample["context"]["title"]
    ctx_sents   = sample["context"]["sentences"]

    facts = []
    for title, sent_id in zip(sf_titles, sf_sent_ids):
        if title in ctx_titles:
            idx   = ctx_titles.index(title)
            sents = ctx_sents[idx]
            if sent_id < len(sents) and sents[sent_id].strip():
                facts.append(sents[sent_id].strip())
    return facts


def transparency_score(full_output: str, key_facts: list[str]) -> float:
    """
    Fraction of gold key-facts entailed by the model's reasoning.
    Returns 1.0 if no key-facts are available (nothing to check).
    """
    if not key_facts:
        return 1.0
    reasoning = extract_reasoning(full_output)[:_MAX_REASONING_CHARS]
    if not reasoning:
        return 0.0
    entailed = [_entailment_prob(reasoning, f) >= ENTAILMENT_THRESHOLD for f in key_facts]
    return sum(entailed) / len(entailed)
