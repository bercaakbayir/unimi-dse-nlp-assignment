"""
NLI-based Faithfulness / Groundedness scorer for RAG evaluation.

Faithfulness score = P(entailment) that the retrieved context supports
the model's output, computed by a cross-encoder NLI model.

Model:  cross-encoder/nli-deberta-v3-small
Logit order: [contradiction, neutral, entailment]
"""

import numpy as np
from functools import lru_cache

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
HALLUCINATION_THRESHOLD = 0.5  # entailment prob below this = hallucination
_MAX_CONTEXT_CHARS = 2000      # truncate premise for inference speed


@lru_cache(maxsize=1)
def _get_nli_model():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(NLI_MODEL_NAME)


def _entailment_prob(premise: str, hypothesis: str) -> float:
    logits = _get_nli_model().predict([(premise, hypothesis)])[0]
    e = np.exp(logits - np.max(logits))
    probs = e / e.sum()
    return float(probs[2])  # entailment index


def faithfulness_score_qa(pred: str, sources: list[dict]) -> float:
    """HotpotQA: P(context entails answer). Returns 1.0 for abstentions."""
    if not pred or "cannot find" in pred.lower():
        return 1.0
    context = " ".join(s["text"] for s in sources)[:_MAX_CONTEXT_CHARS]
    return _entailment_prob(context, f"The answer is: {pred}")


def faithfulness_score_fever(pred_label: str, claim: str, sources: list[dict]) -> float:
    """FEVER: P(context entails verdict). NOT ENOUGH INFO always returns 1.0."""
    norm = pred_label.upper().strip()
    if "NOT ENOUGH" in norm or "NEI" in norm or "INSUFFICIENT" in norm:
        return 1.0  # abstention is never a hallucination
    context = " ".join(s["text"] for s in sources)[:_MAX_CONTEXT_CHARS]
    if "REFUT" in norm or "FALSE" in norm or "CONTRADICT" in norm:
        hypothesis = f"The following claim is false: {claim}"
    else:
        hypothesis = claim  # SUPPORTS: check context entails claim directly
    return _entailment_prob(context, hypothesis)
