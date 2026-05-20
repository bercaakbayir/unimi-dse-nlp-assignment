"""
evaluate.py — Unified evaluation for HotpotQA and FEVER RAG pipeline benchmarks.

Usage:
    # HotpotQA — quick test (100 random questions, seed 42)
    python src/eval/evaluate.py --dataset hotpotqa --limit 100 --output results/smoke.jsonl

    # HotpotQA — full validation set (~7,405 questions)
    python src/eval/evaluate.py --dataset hotpotqa --output results/hotpotqa_clean.jsonl

    # FEVER — quick test (100 random claims, seed 42)
    python src/eval/evaluate.py --dataset fever --limit 100 --output results/fever_smoke.jsonl

    # FEVER — full labelled_dev set (~19,998 claims)
    python src/eval/evaluate.py --dataset fever --output results/fever_full.jsonl
"""

import argparse
import csv
import json
import math
import random
import re
import signal
import string
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rag.llm import OllamaLLM
from src.rag.pipeline import RAGPipeline
from src.rag.retriever import ChromaDBRetriever, HybridRetriever
from src.eval.faithfulness import (
    faithfulness_score_qa, faithfulness_score_fever, HALLUCINATION_THRESHOLD,
)
from src.eval.trust_metrics import (
    extract_key_facts, transparency_score,
    extract_cited_ids, gold_citation_ids, accountability_f1,
)

ROOT        = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"

# HotpotQA paths
HQ_VALIDATION_PATH = ROOT / "data" / "hotpotqa" / "validation.jsonl"
HQ_CHROMA_DIR      = ROOT / "data" / "chromadb"
HQ_BM25_PATH       = ROOT / "data" / "bm25_index.pkl"
HQ_CORPUS_IDS_PATH = ROOT / "data" / "bm25_corpus_ids.pkl"

# FEVER paths
FV_VALIDATION_PATH = ROOT / "data" / "fever" / "paper_dev.jsonl"
FV_CHROMA_DIR      = ROOT / "data" / "chromadb"
FV_BM25_PATH       = ROOT / "data" / "fever_bm25_index.pkl"
FV_CORPUS_IDS_PATH = ROOT / "data" / "fever_bm25_corpus_ids.pkl"

# FEVER canonical label strings
_LABEL_SUPPORTS = "SUPPORTS"
_LABEL_REFUTES  = "REFUTES"
_LABEL_NEI      = "NOT ENOUGH INFO"


# ── HotpotQA answer-quality metrics ──────────────────────────────────────────

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def bleu1(pred: str, gold: str) -> float:
    """BLEU-1: modified unigram precision with brevity penalty."""
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks or not gold_toks:
        return 0.0
    gold_counts = Counter(gold_toks)
    clipped = sum(min(cnt, gold_counts[tok]) for tok, cnt in Counter(pred_toks).items())
    precision = clipped / len(pred_toks)
    bp = 1.0 if len(pred_toks) >= len(gold_toks) else math.exp(1 - len(gold_toks) / len(pred_toks))
    return bp * precision


def token_f1(pred: str, gold: str) -> float:
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks or not gold_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_toks)
    r = num_same / len(gold_toks)
    return 2 * p * r / (p + r)


def sp_metrics(retrieved_titles: list[str], gold_titles: list[str]) -> tuple[float, float, float]:
    ret  = set(retrieved_titles)
    gold = set(gold_titles)
    if not ret or not gold:
        return 0.0, 0.0, 0.0
    precision = len(ret & gold) / len(ret)
    recall    = len(ret & gold) / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ── FEVER answer-quality metrics ──────────────────────────────────────────────

def normalize_label(label: str) -> str:
    """Normalise LLM output to one of the three canonical FEVER labels."""
    s = label.upper().strip()
    if "NOT ENOUGH" in s or "NEI" in s or "INSUFFICIENT" in s:
        return _LABEL_NEI
    if "REFUT" in s or "REFUS" in s or "FALSE" in s or "CONTRADICT" in s:
        return _LABEL_REFUTES
    if "SUPPORT" in s or "TRUE" in s or "CONFIRM" in s:
        return _LABEL_SUPPORTS
    return s


def label_accuracy(pred_label: str, gold_label: str) -> int:
    return int(normalize_label(pred_label) == normalize_label(gold_label))


def evidence_metrics(retrieved_titles: list[str], gold_titles: list[str]) -> tuple[float, float, float]:
    ret  = set(retrieved_titles)
    gold = set(gold_titles)
    if not ret or not gold:
        return 0.0, 0.0, 0.0
    precision = len(ret & gold) / len(ret)
    recall    = len(ret & gold) / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def fever_score(
    pred_label: str,
    gold_label: str,
    retrieved_titles: list[str],
    gold_titles: list[str],
) -> int:
    """
    Official-style FEVER score: label correct AND (for SUPPORTS/REFUTES)
    at least one gold evidence title was retrieved.
    NOT ENOUGH INFO claims only require the correct label.
    """
    pred_norm = normalize_label(pred_label)
    gold_norm = normalize_label(gold_label)
    label_ok  = pred_norm == gold_norm
    if not label_ok:
        return 0
    if gold_norm == _LABEL_NEI:
        return 1
    return int(bool(set(retrieved_titles) & set(gold_titles)))


# ── Shared retriever quality metrics ─────────────────────────────────────────

def mrr_score(retrieved_titles: list[str], gold_titles: list[str]) -> float:
    """Reciprocal rank of the first gold passage in the ranked list (0 if not found)."""
    gold_set = set(gold_titles)
    for rank, title in enumerate(retrieved_titles, 1):
        if title in gold_set:
            return 1.0 / rank
    return 0.0


def hit_at_k(retrieved_titles: list[str], gold_titles: list[str], k: int) -> int:
    """1 if any gold title appears in the top-k retrieved passages, else 0."""
    gold_set = set(gold_titles)
    return int(any(t in gold_set for t in retrieved_titles[:k]))


# ── Extraction helpers ────────────────────────────────────────────────────────

def extract_final_answer(full_answer: str) -> str:
    if "Final Answer:" in full_answer:
        return full_answer.split("Final Answer:", 1)[1].strip()
    return full_answer.strip()


def extract_final_verdict(full_answer: str) -> str:
    if "Final Verdict:" in full_answer:
        return full_answer.split("Final Verdict:", 1)[1].strip()
    return full_answer.strip()


# ── Summary helpers ───────────────────────────────────────────────────────────

def _avg(records: list[dict], key: str) -> float:
    vals = [r[key] for r in records if key in r and r.get("error") is None]
    return sum(vals) / len(vals) * 100 if vals else 0.0


def _print_hotpotqa_summary(records: list[dict], n_total: int) -> None:
    evaluated  = [r for r in records if r.get("error") is None]
    bridge     = [r for r in evaluated if r["type"] == "bridge"]
    comparison = [r for r in evaluated if r["type"] == "comparison"]

    header = f"\n{'─'*60}\n  HotpotQA Evaluation  |  Questions evaluated: {len(evaluated)}/{n_total}\n{'─'*60}"
    print(header)
    print(f"{'Metric':<18} {'All':>8} {'Bridge':>10} {'Comparison':>12}")
    print("─" * 50)
    for label, key in [
        ("Exact Match (EM)", "em"),
        ("BLEU-1",           "bleu1"),
        ("Token F1",         "f1"),
        ("SP Precision",     "sp_precision"),
        ("SP Recall",        "sp_recall"),
        ("SP F1",            "sp_f1"),
        ("MRR",              "mrr"),
        ("Hit@1",            "hit_1"),
        ("Hit@3",            "hit_3"),
        ("Hit@5",            "hit_5"),
    ]:
        print(
            f"{label:<18}"
            f" {_avg(evaluated, key):>7.1f}%"
            f" {_avg(bridge, key):>9.1f}%"
            f" {_avg(comparison, key):>11.1f}%"
        )
    print("─" * 50)
    if any(r.get("poisoning_enabled") for r in evaluated):
        avg_poisoned = sum(r.get("poisoned_count", 0) for r in evaluated) / len(evaluated) if evaluated else 0
        print(f"  Poisoning enabled — avg {avg_poisoned:.1f} passages poisoned per question")
    errors = [r for r in records if r.get("error")]
    if errors:
        print(f"  Errors / skipped: {len(errors)}")
    print()


def _print_fever_summary(records: list[dict], n_total: int) -> None:
    evaluated = [r for r in records if r.get("error") is None]
    supports  = [r for r in evaluated if normalize_label(r.get("gold_label", "")) == _LABEL_SUPPORTS]
    refutes   = [r for r in evaluated if normalize_label(r.get("gold_label", "")) == _LABEL_REFUTES]
    nei       = [r for r in evaluated if normalize_label(r.get("gold_label", "")) == _LABEL_NEI]

    header = (
        f"\n{'─'*65}\n"
        f"  FEVER Evaluation  |  Claims evaluated: {len(evaluated)}/{n_total}\n"
        f"{'─'*65}"
    )
    print(header)
    print(f"{'Metric':<22} {'All':>8} {'SUPPORTS':>10} {'REFUTES':>9} {'NEI':>7}")
    print("─" * 58)
    for label, key in [
        ("Label Accuracy",     "label_acc"),
        ("FEVER Score",        "fever_sc"),
        ("Evidence Precision", "ev_precision"),
        ("Evidence Recall",    "ev_recall"),
        ("Evidence F1",        "ev_f1"),
        ("MRR",                "mrr"),
        ("Hit@1",              "hit_1"),
        ("Hit@3",              "hit_3"),
        ("Hit@5",              "hit_5"),
    ]:
        print(
            f"{label:<22}"
            f" {_avg(evaluated, key):>7.1f}%"
            f" {_avg(supports,  key):>9.1f}%"
            f" {_avg(refutes,   key):>8.1f}%"
            f" {_avg(nei,       key):>6.1f}%"
        )
    print("─" * 58)
    if any(r.get("poisoning_enabled") for r in evaluated):
        avg_poisoned = sum(r.get("poisoned_count", 0) for r in evaluated) / len(evaluated) if evaluated else 0
        print(f"  Poisoning enabled — avg {avg_poisoned:.1f} passages poisoned per claim")
    errors = [r for r in records if r.get("error")]
    if errors:
        print(f"  Errors / skipped: {len(errors)}")
    print()


# ── CSV export ────────────────────────────────────────────────────────────────

def _write_hotpotqa_csv(records: list[dict], timestamp: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"hotpotqa_eval_{timestamp}.csv"
    evaluated = [r for r in records if r.get("error") is None]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "real_answer", "llm_answer"])
        writer.writeheader()
        for r in evaluated:
            writer.writerow({
                "id":          r["id"],
                "question":    r["question"],
                "real_answer": r["gold_answer"],
                "llm_answer":  r["pred_answer"],
            })
    print(f"CSV saved → {csv_path}")
    return csv_path


def _write_fever_csv(records: list[dict], timestamp: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"fever_eval_{timestamp}.csv"
    evaluated = [r for r in records if r.get("error") is None]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "claim", "gold_label", "pred_label", "fever_sc"])
        writer.writeheader()
        for r in evaluated:
            writer.writerow({
                "id":         r["id"],
                "claim":      r["claim"],
                "gold_label": r["gold_label"],
                "pred_label": r["pred_label"],
                "fever_sc":   r["fever_sc"],
            })
    print(f"CSV saved → {csv_path}")
    return csv_path


# ── Pipeline builders ─────────────────────────────────────────────────────────

def build_pipeline(args: argparse.Namespace) -> RAGPipeline:
    """Build a HotpotQA (QA mode) pipeline."""
    if HQ_BM25_PATH.exists():
        retriever = HybridRetriever(
            chroma_dir=HQ_CHROMA_DIR,
            collection_name=args.collection,
            model_name=args.embed_model,
            top_k=args.top_k,
            bm25_path=HQ_BM25_PATH,
            corpus_ids_path=HQ_CORPUS_IDS_PATH,
        )
    else:
        print("Warning: HotpotQA BM25 index not found — falling back to dense retrieval.")
        retriever = ChromaDBRetriever(
            chroma_dir=HQ_CHROMA_DIR,
            collection_name=args.collection,
            model_name=args.embed_model,
            top_k=args.top_k,
        )
    llm = OllamaLLM(model=args.llm)
    poisoner = None
    if args.poison:
        from src.rag.poisoner import PassagePoisoner
        poisoner = PassagePoisoner(llm=llm, rate=0.3, seed=args.poison_seed)
    return RAGPipeline(
        retriever=retriever, llm=llm, top_k=args.top_k, poisoner=poisoner,
        consistency_check=getattr(args, "consistency_check", False),
        cite_sources=getattr(args, "cite_sources", False),
    )


def build_fever_pipeline(args: argparse.Namespace) -> RAGPipeline:
    """Build a FEVER (fact-check mode) pipeline."""
    if FV_BM25_PATH.exists():
        retriever = HybridRetriever(
            chroma_dir=FV_CHROMA_DIR,
            collection_name=args.collection,
            model_name=args.embed_model,
            top_k=args.top_k,
            bm25_path=FV_BM25_PATH,
            corpus_ids_path=FV_CORPUS_IDS_PATH,
        )
    else:
        print("Warning: FEVER BM25 index not found — falling back to dense retrieval.")
        print("Run: python src/data/build_bm25_index.py --collection fever_passages "
              "--output data/fever_bm25_index.pkl --corpus-ids-output data/fever_bm25_corpus_ids.pkl")
        retriever = ChromaDBRetriever(
            chroma_dir=FV_CHROMA_DIR,
            collection_name=args.collection,
            model_name=args.embed_model,
            top_k=args.top_k,
        )
    llm = OllamaLLM(model=args.llm)
    poisoner = None
    if args.poison:
        from src.rag.poisoner import PassagePoisoner
        poisoner = PassagePoisoner(llm=llm, rate=0.3, seed=args.poison_seed)
    return RAGPipeline(
        retriever=retriever, llm=llm, top_k=args.top_k, poisoner=poisoner,
        mode="fact_check",
        consistency_check=getattr(args, "consistency_check", False),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline on HotpotQA or FEVER")
    parser.add_argument("--dataset", choices=["hotpotqa", "fever"], default="hotpotqa",
                        help="Dataset to evaluate (default: hotpotqa)")
    parser.add_argument("--output", default=None,
                        help="Path for per-record JSONL output (default: results/<dataset>_eval.jsonl)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Randomly sample N records (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for --limit sampling (default: 42)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of passages to retrieve (default: 10)")
    parser.add_argument("--llm", default="llama3.2:3b",
                        help="Ollama model name (default: llama3.2:3b)")
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L12-v2",
                        help="Sentence-transformers model name")
    parser.add_argument("--collection", default=None,
                        help="ChromaDB collection name (default: hotpotqa_passages / fever_passages)")
    parser.add_argument("--poison", type=lambda x: x.lower() == "true", default=False,
                        help="Inject false facts into retrieved passages (default: false)")
    parser.add_argument("--poison-seed", type=int, default=None,
                        help="Random seed for poison passage selection (default: non-deterministic)")
    parser.add_argument("--consistency-check", type=lambda x: x.lower() == "true", default=False,
                        help="Append cross-document consistency instructions to system prompt (default: false)")
    parser.add_argument("--cite-sources", type=lambda x: x.lower() == "true", default=False,
                        help="Prompt LLM to cite passage IDs; enables accountability metrics (hotpotqa only)")
    return parser.parse_args()


def _run_hotpotqa(args: argparse.Namespace, output_path: Path, timestamp: str) -> None:
    if not HQ_VALIDATION_PATH.exists():
        print(f"Error: HotpotQA data not found at {HQ_VALIDATION_PATH}")
        sys.exit(1)

    samples = [json.loads(l) for l in HQ_VALIDATION_PATH.read_text().splitlines() if l.strip()]
    if args.limit and args.limit < len(samples):
        rng = random.Random(args.seed)
        samples = rng.sample(samples, args.limit)
        print(f"Sampled {len(samples)} questions (seed={args.seed})")
    else:
        print(f"Evaluating all {len(samples)} questions")

    n_total = len(samples)

    already_done: set[str] = set()
    all_records: list[dict] = []
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                already_done.add(rec["id"])
                all_records.append(rec)
        if already_done:
            print(f"Resuming — skipping {len(already_done)} already-evaluated questions")

    print("Building pipeline...")
    pipeline = build_pipeline(args)

    def _on_interrupt(sig, frame):
        print("\nInterrupted — saving CSV before exit...")
        _write_hotpotqa_csv(all_records, timestamp)
        _print_hotpotqa_summary(all_records, n_total)
        sys.exit(0)

    signal.signal(signal.SIGINT, _on_interrupt)

    pending = [s for s in samples if s["id"] not in already_done]
    with open(output_path, "a", encoding="utf-8") as out_f:
        for sample in tqdm(pending, desc="Evaluating"):
            record: dict = {
                "id":       sample["id"],
                "type":     sample["type"],
                "question": sample["question"],
            }
            try:
                result      = pipeline.answer(sample["question"])
                pred        = extract_final_answer(result["answer"])
                gold        = sample["answer"]
                ret_titles  = [s["title"] for s in result["sources"]]
                gold_titles = list(set(sample["supporting_facts"]["title"]))
                sp_p, sp_r, sp_f = sp_metrics(ret_titles, gold_titles)

                faith       = faithfulness_score_qa(pred, result["sources"])
                full_output = result["answer"]
                key_facts   = extract_key_facts(sample)
                transp      = transparency_score(full_output, key_facts)

                record.update({
                    "gold_answer":               gold,
                    "pred_answer":               pred,
                    "em":                        exact_match(pred, gold),
                    "bleu1":                     bleu1(pred, gold),
                    "f1":                        token_f1(pred, gold),
                    "sp_precision":              sp_p,
                    "sp_recall":                 sp_r,
                    "sp_f1":                     sp_f,
                    "mrr":                       mrr_score(ret_titles, gold_titles),
                    "hit_1":                     hit_at_k(ret_titles, gold_titles, 1),
                    "hit_3":                     hit_at_k(ret_titles, gold_titles, 3),
                    "hit_5":                     hit_at_k(ret_titles, gold_titles, 5),
                    "retrieved_titles":          ret_titles,
                    "gold_titles":               gold_titles,
                    "poisoning_enabled":         args.poison,
                    "poisoned_count":            sum(1 for s in result["sources"] if s.get("poisoned")),
                    "poisoned_titles":           [s["title"] for s in result["sources"] if s.get("poisoned")],
                    "faithfulness_score":        faith,
                    "is_hallucination":          faith < HALLUCINATION_THRESHOLD,
                    "consistency_check_enabled": args.consistency_check,
                    "full_output":               full_output,
                    "key_facts":                 key_facts,
                    "n_key_facts":               len(key_facts),
                    "transparency_score":        transp,
                    "cite_sources_enabled":      args.cite_sources,
                    "error":                     None,
                })

                if args.cite_sources:
                    cited    = extract_cited_ids(full_output)
                    true_ids = gold_citation_ids(ret_titles, gold_titles)
                    acc_p, acc_r, acc_f = accountability_f1(cited, true_ids)
                    record.update({
                        "cited_ids":                cited,
                        "gold_cited_ids":           true_ids,
                        "accountability_precision": acc_p,
                        "accountability_recall":    acc_r,
                        "accountability_f1":        acc_f,
                    })
            except Exception as e:
                record.update({"error": str(e), "gold_answer": sample["answer"]})

            all_records.append(record)
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

    _write_hotpotqa_csv(all_records, timestamp)
    _print_hotpotqa_summary(all_records, n_total)
    print(f"JSONL saved → {output_path}")


def _run_fever(args: argparse.Namespace, output_path: Path, timestamp: str) -> None:
    if not FV_VALIDATION_PATH.exists():
        print(f"Error: FEVER data not found at {FV_VALIDATION_PATH}")
        print("Run first:  python src/data/build_fever_db.py")
        sys.exit(1)

    samples = [json.loads(l) for l in FV_VALIDATION_PATH.read_text().splitlines() if l.strip()]
    if args.limit and args.limit < len(samples):
        rng = random.Random(args.seed)
        samples = rng.sample(samples, args.limit)
        print(f"Sampled {len(samples)} claims (seed={args.seed})")
    else:
        print(f"Evaluating all {len(samples)} claims")

    n_total = len(samples)

    already_done: set[int] = set()
    all_records: list[dict] = []
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                already_done.add(rec["id"])
                all_records.append(rec)
        if already_done:
            print(f"Resuming — skipping {len(already_done)} already-evaluated claims")

    print("Building pipeline...")
    pipeline = build_fever_pipeline(args)

    def _on_interrupt(sig, frame):
        print("\nInterrupted — saving CSV before exit...")
        _write_fever_csv(all_records, timestamp)
        _print_fever_summary(all_records, n_total)
        sys.exit(0)

    signal.signal(signal.SIGINT, _on_interrupt)

    pending = [s for s in samples if s["id"] not in already_done]
    with open(output_path, "a", encoding="utf-8") as out_f:
        for sample in tqdm(pending, desc="Evaluating"):
            record: dict = {
                "id":    sample["id"],
                "claim": sample["claim"],
            }
            try:
                result     = pipeline.answer(sample["claim"])
                pred       = extract_final_verdict(result["answer"])
                gold       = sample["label"]
                ret_titles = [s["title"] for s in result["sources"]]
                gold_titles = list({
                    piece[2].replace("_", " ")
                    for ann_set in sample.get("evidence", [])
                    for piece in ann_set
                    if len(piece) >= 3 and piece[2]
                })
                ev_p, ev_r, ev_f = evidence_metrics(ret_titles, gold_titles)
                faith = faithfulness_score_fever(pred, sample["claim"], result["sources"])

                record.update({
                    "gold_label":               gold,
                    "pred_label":               pred,
                    "pred_normalized":          normalize_label(pred),
                    "label_acc":                label_accuracy(pred, gold),
                    "fever_sc":                 fever_score(pred, gold, ret_titles, gold_titles),
                    "ev_precision":             ev_p,
                    "ev_recall":                ev_r,
                    "ev_f1":                    ev_f,
                    "mrr":                      mrr_score(ret_titles, gold_titles),
                    "hit_1":                    hit_at_k(ret_titles, gold_titles, 1),
                    "hit_3":                    hit_at_k(ret_titles, gold_titles, 3),
                    "hit_5":                    hit_at_k(ret_titles, gold_titles, 5),
                    "retrieved_titles":         ret_titles,
                    "gold_titles":              gold_titles,
                    "poisoning_enabled":        args.poison,
                    "poisoned_count":           sum(1 for s in result["sources"] if s.get("poisoned")),
                    "poisoned_titles":          [s["title"] for s in result["sources"] if s.get("poisoned")],
                    "faithfulness_score":       faith,
                    "is_hallucination":         faith < HALLUCINATION_THRESHOLD,
                    "consistency_check_enabled": args.consistency_check,
                    "error":                    None,
                })
            except Exception as e:
                record.update({"error": str(e), "gold_label": sample["label"]})

            all_records.append(record)
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

    _write_fever_csv(all_records, timestamp)
    _print_fever_summary(all_records, n_total)
    print(f"JSONL saved → {output_path}")


def main() -> None:
    args      = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Apply dataset-specific defaults
    if args.collection is None:
        args.collection = "fever_passages" if args.dataset == "fever" else "hotpotqa_passages"
    if args.output is None:
        args.output = f"results/{args.dataset}_eval.jsonl"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dataset == "fever":
        _run_fever(args, output_path, timestamp)
    else:
        _run_hotpotqa(args, output_path, timestamp)


if __name__ == "__main__":
    main()
