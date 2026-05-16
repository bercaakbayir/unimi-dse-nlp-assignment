"""
evaluate_fever.py — FEVER benchmark evaluation for the RAG pipeline.

Metrics:
  - Label Accuracy: fraction of claims with correct SUPPORTS/REFUTES/NOT ENOUGH INFO
  - FEVER Score:    label correct AND (for SUPPORTS/REFUTES) at least one gold
                    evidence title was retrieved
  - Evidence Precision / Recall / F1: retrieved titles vs gold evidence titles

Usage:
    # Quick test: 100 random claims (seed 42)
    python src/eval/evaluate_fever.py --limit 100 --output results/fever_smoke.jsonl

    # Full labelled_dev set (~19,998 claims)
    python src/eval/evaluate_fever.py --output results/fever_full.jsonl

    # Custom seed for a different random subset
    python src/eval/evaluate_fever.py --limit 500 --seed 7 --output results/fever_500.jsonl
"""

import argparse
import csv
import json
import random
import re
import signal
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rag.llm import OllamaLLM
from src.rag.pipeline import RAGPipeline
from src.rag.retriever import ChromaDBRetriever, HybridRetriever

ROOT            = Path(__file__).resolve().parents[2]
VALIDATION_PATH = ROOT / "data" / "fever" / "paper_dev.jsonl"
CHROMA_DIR      = ROOT / "data" / "chromadb"
BM25_PATH       = ROOT / "data" / "fever_bm25_index.pkl"
CORPUS_IDS_PATH = ROOT / "data" / "fever_bm25_corpus_ids.pkl"
RESULTS_DIR     = ROOT / "results"

# FEVER dataset labels (uppercase canonical form)
_LABEL_SUPPORTS = "SUPPORTS"
_LABEL_REFUTES  = "REFUTES"
_LABEL_NEI      = "NOT ENOUGH INFO"


# ── Metrics ───────────────────────────────────────────────────────────────────

def normalize_label(label: str) -> str:
    """Normalise LLM output to one of the three canonical FEVER labels."""
    s = label.upper().strip()
    if "NOT ENOUGH" in s or "NEI" in s or "INSUFFICIENT" in s:
        return _LABEL_NEI
    if "REFUT" in s or "FALSE" in s or "CONTRADICT" in s:
        return _LABEL_REFUTES
    if "SUPPORT" in s or "TRUE" in s or "CONFIRM" in s:
        return _LABEL_SUPPORTS
    return s  # return as-is for unknown outputs


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
    Official-style FEVER score: label is correct AND
    (for SUPPORTS/REFUTES) at least one gold evidence title was retrieved.
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


def extract_final_verdict(full_answer: str) -> str:
    if "Final Verdict:" in full_answer:
        return full_answer.split("Final Verdict:", 1)[1].strip()
    return full_answer.strip()


# ── Summary ───────────────────────────────────────────────────────────────────

def _avg(records: list[dict], key: str) -> float:
    vals = [r[key] for r in records if key in r and r.get("error") is None]
    return sum(vals) / len(vals) * 100 if vals else 0.0


def print_summary(records: list[dict], n_total: int) -> None:
    evaluated  = [r for r in records if r.get("error") is None]
    supports   = [r for r in evaluated if normalize_label(r.get("gold_label", "")) == _LABEL_SUPPORTS]
    refutes    = [r for r in evaluated if normalize_label(r.get("gold_label", "")) == _LABEL_REFUTES]
    nei        = [r for r in evaluated if normalize_label(r.get("gold_label", "")) == _LABEL_NEI]

    header = (
        f"\n{'─'*65}\n"
        f"  FEVER Evaluation Summary  |  Claims evaluated: {len(evaluated)}/{n_total}\n"
        f"{'─'*65}"
    )
    print(header)
    print(f"{'Metric':<22} {'All':>8} {'SUPPORTS':>10} {'REFUTES':>9} {'NEI':>7}")
    print("─" * 58)
    for label, key in [
        ("Label Accuracy",       "label_acc"),
        ("FEVER Score",          "fever_sc"),
        ("Evidence Precision",   "ev_precision"),
        ("Evidence Recall",      "ev_recall"),
        ("Evidence F1",          "ev_f1"),
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

def write_csv(records: list[dict], timestamp: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"fever_eval_{timestamp}.csv"
    evaluated = [r for r in records if r.get("error") is None]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "claim", "gold_label", "pred_label", "fever_sc"])
        writer.writeheader()
        for r in evaluated:
            writer.writerow({
                "id":          r["id"],
                "claim":       r["claim"],
                "gold_label":  r["gold_label"],
                "pred_label":  r["pred_label"],
                "fever_sc":    r["fever_sc"],
            })
    print(f"CSV saved → {csv_path}")
    return csv_path


# ── Pipeline builder ──────────────────────────────────────────────────────────

def build_pipeline(args: argparse.Namespace) -> RAGPipeline:
    if BM25_PATH.exists():
        retriever = HybridRetriever(
            chroma_dir=CHROMA_DIR,
            collection_name=args.collection,
            model_name=args.embed_model,
            top_k=args.top_k,
            bm25_path=BM25_PATH,
            corpus_ids_path=CORPUS_IDS_PATH,
        )
    else:
        print("Warning: FEVER BM25 index not found — falling back to dense retrieval.")
        print("Run: python src/data/build_bm25_index.py --collection fever_passages "
              "--output data/fever_bm25_index.pkl --corpus-ids-output data/fever_bm25_corpus_ids.pkl")
        retriever = ChromaDBRetriever(
            chroma_dir=CHROMA_DIR,
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
        retriever=retriever,
        llm=llm,
        top_k=args.top_k,
        poisoner=poisoner,
        mode="fact_check",
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline on FEVER labelled_dev set")
    parser.add_argument("--output", default="results/fever_eval.jsonl",
                        help="Path for per-claim JSONL output (supports resume)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Randomly sample N claims (default: all ~19,998)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for --limit sampling (default: 42)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of passages to retrieve (default: 10)")
    parser.add_argument("--llm", default="llama3.2:3b",
                        help="Ollama model name (default: llama3.2:3b)")
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L12-v2",
                        help="Sentence-transformers model name")
    parser.add_argument("--collection", default="fever_passages",
                        help="ChromaDB collection name (default: fever_passages)")
    parser.add_argument("--poison", type=lambda x: x.lower() == "true", default=False,
                        help="Inject false facts into retrieved passages (default: false)")
    parser.add_argument("--poison-seed", type=int, default=None,
                        help="Random seed for poison passage selection (default: non-deterministic)")
    return parser.parse_args()


def main() -> None:
    args      = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not VALIDATION_PATH.exists():
        print(f"Error: FEVER claims not found at {VALIDATION_PATH}")
        print("Run first:  python src/data/build_fever_db.py")
        sys.exit(1)

    samples = [json.loads(l) for l in VALIDATION_PATH.read_text().splitlines() if l.strip()]

    if args.limit and args.limit < len(samples):
        rng     = random.Random(args.seed)
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
    pipeline = build_pipeline(args)

    def _on_interrupt(sig, frame):
        print("\nInterrupted — saving CSV before exit...")
        write_csv(all_records, timestamp)
        print_summary(all_records, n_total)
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
                # Native FEVER evidence format:
                # evidence = [[[ann_id, ev_id, wiki_title_or_null, sent_id], ...], ...]
                gold_titles = list({
                    piece[2].replace("_", " ")
                    for ann_set in sample.get("evidence", [])
                    for piece in ann_set
                    if len(piece) >= 3 and piece[2]
                })
                ev_p, ev_r, ev_f = evidence_metrics(ret_titles, gold_titles)

                record.update({
                    "gold_label":        gold,
                    "pred_label":        pred,
                    "pred_normalized":   normalize_label(pred),
                    "label_acc":         label_accuracy(pred, gold),
                    "fever_sc":          fever_score(pred, gold, ret_titles, gold_titles),
                    "ev_precision":      ev_p,
                    "ev_recall":         ev_r,
                    "ev_f1":             ev_f,
                    "retrieved_titles":  ret_titles,
                    "gold_titles":       gold_titles,
                    "poisoning_enabled": args.poison,
                    "poisoned_count":    sum(1 for s in result["sources"] if s.get("poisoned")),
                    "poisoned_titles":   [s["title"] for s in result["sources"] if s.get("poisoned")],
                    "error":             None,
                })
            except Exception as e:
                record.update({"error": str(e), "gold_label": sample["label"]})

            all_records.append(record)
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

    write_csv(all_records, timestamp)
    print_summary(all_records, n_total)
    print(f"JSONL saved → {output_path}")


if __name__ == "__main__":
    main()
