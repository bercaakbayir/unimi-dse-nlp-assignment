"""
Trust metrics for HotpotQA RAG evaluation.

Robustness     — F1 degradation (clean → poisoned)
Transparency   — NLI precision of gold key-facts covered in model reasoning
Accountability — Citation F1: cited passage IDs vs gold relevant IDs
"""
import re
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


# ── Accountability ─────────────────────────────────────────────────────────────

def extract_cited_ids(full_output: str) -> list[int]:
    """
    Parse 'Cited Sources: [1], [3]' from LLM output.
    Returns empty list if the citation line is absent.
    """
    match = re.search(r"Cited Sources:\s*([\d,\s\[\]]+)", full_output, re.IGNORECASE)
    if not match:
        return []
    return [int(x) for x in re.findall(r"\d+", match.group(1))]


def gold_citation_ids(retrieved_titles: list[str], gold_titles: list[str]) -> list[int]:
    """1-indexed positions of retrieved passages whose title is in gold_titles."""
    gold_set = set(gold_titles)
    return [i + 1 for i, t in enumerate(retrieved_titles) if t in gold_set]


def accountability_f1(cited: list[int], gold: list[int]) -> tuple[float, float, float]:
    """Citation precision, recall, F1."""
    p_set, g_set = set(cited), set(gold)
    if not p_set or not g_set:
        return 0.0, 0.0, 0.0
    p  = len(p_set & g_set) / len(p_set)
    r  = len(p_set & g_set) / len(g_set)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1
