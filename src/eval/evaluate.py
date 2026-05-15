"""
evaluate.py — HotpotQA benchmark evaluation for the RAG pipeline.

Usage:
    # Quick test: 100 random questions (seed 42)
    python src/eval/evaluate.py --limit 100 --output results/smoke.jsonl

    # Full validation set (~7,405 questions)
    python src/eval/evaluate.py --output results/hybrid_k10.jsonl

    # Custom seed for a different random subset
    python src/eval/evaluate.py --limit 200 --seed 7 --output results/sample200.jsonl
"""

import argparse
import csv
import json
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

ROOT            = Path(__file__).resolve().parents[2]
VALIDATION_PATH = ROOT / "data" / "hotpotqa" / "validation.jsonl"
CHROMA_DIR      = ROOT / "data" / "chromadb"
BM25_PATH       = ROOT / "data" / "bm25_index.pkl"
CORPUS_IDS_PATH = ROOT / "data" / "bm25_corpus_ids.pkl"
RESULTS_DIR     = ROOT / "results"


# ── Metrics ───────────────────────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


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


def extract_final_answer(full_answer: str) -> str:
    if "Final Answer:" in full_answer:
        return full_answer.split("Final Answer:", 1)[1].strip()
    return full_answer.strip()


# ── Summary ───────────────────────────────────────────────────────────────────

def _avg(records: list[dict], key: str) -> float:
    vals = [r[key] for r in records if key in r and r.get("error") is None]
    return sum(vals) / len(vals) * 100 if vals else 0.0


def print_summary(records: list[dict], n_total: int) -> None:
    evaluated = [r for r in records if r.get("error") is None]
    bridge     = [r for r in evaluated if r["type"] == "bridge"]
    comparison = [r for r in evaluated if r["type"] == "comparison"]

    header = f"\n{'─'*60}\n  Evaluation Summary  |  Questions evaluated: {len(evaluated)}/{n_total}\n{'─'*60}"
    print(header)
    print(f"{'Metric':<18} {'All':>8} {'Bridge':>10} {'Comparison':>12}")
    print("─" * 50)
    for label, key in [
        ("Exact Match (EM)", "em"),
        ("Token F1",         "f1"),
        ("SP Precision",     "sp_precision"),
        ("SP Recall",        "sp_recall"),
        ("SP F1",            "sp_f1"),
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


# ── CSV export ────────────────────────────────────────────────────────────────

def write_csv(records: list[dict], timestamp: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"eval_{timestamp}.csv"
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
        print("Warning: BM25 index not found — falling back to dense retrieval.")
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
    return RAGPipeline(retriever=retriever, llm=llm, top_k=args.top_k, poisoner=poisoner)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline on HotpotQA validation set")
    parser.add_argument("--output", default="results/eval.jsonl",
                        help="Path for per-question JSONL output (supports resume)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Randomly sample N questions (default: all 7,405)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for --limit sampling (default: 42)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of passages to retrieve (default: 10)")
    parser.add_argument("--llm", default="llama3.2:3b",
                        help="Ollama model name (default: llama3.2:3b)")
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L12-v2",
                        help="Sentence-transformers model name")
    parser.add_argument("--collection", default="hotpotqa_passages",
                        help="ChromaDB collection name")
    parser.add_argument("--poison", type=lambda x: x.lower() == "true", default=False,
                        help="Inject false facts into retrieved passages (default: false)")
    parser.add_argument("--poison-seed", type=int, default=None,
                        help="Random seed for poison passage selection (default: non-deterministic)")
    return parser.parse_args()


def main() -> None:
    args  = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load validation questions
    samples = [json.loads(l) for l in VALIDATION_PATH.read_text().splitlines() if l.strip()]

    # Random subset if --limit is set
    if args.limit and args.limit < len(samples):
        rng = random.Random(args.seed)
        samples = rng.sample(samples, args.limit)
        print(f"Sampled {len(samples)} questions (seed={args.seed})")
    else:
        print(f"Evaluating all {len(samples)} questions")

    n_total = len(samples)

    # Load already-evaluated IDs for resume support
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

    # Build pipeline
    print("Building pipeline...")
    pipeline = build_pipeline(args)

    # Graceful interrupt: write CSV on Ctrl+C
    def _on_interrupt(sig, frame):
        print("\nInterrupted — saving CSV before exit...")
        write_csv(all_records, timestamp)
        print_summary(all_records, n_total)
        sys.exit(0)

    signal.signal(signal.SIGINT, _on_interrupt)

    # Evaluation loop
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

                record.update({
                    "gold_answer":       gold,
                    "pred_answer":       pred,
                    "em":                exact_match(pred, gold),
                    "f1":                token_f1(pred, gold),
                    "sp_precision":      sp_p,
                    "sp_recall":         sp_r,
                    "sp_f1":             sp_f,
                    "retrieved_titles":  ret_titles,
                    "gold_titles":       gold_titles,
                    "poisoning_enabled": args.poison,
                    "poisoned_count":    sum(1 for s in result["sources"] if s.get("poisoned")),
                    "poisoned_titles":   [s["title"] for s in result["sources"] if s.get("poisoned")],
                    "error":             None,
                })
            except Exception as e:
                record.update({"error": str(e), "gold_answer": sample["answer"]})

            all_records.append(record)
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

    # Final outputs
    write_csv(all_records, timestamp)
    print_summary(all_records, n_total)
    print(f"JSONL saved → {output_path}")


if __name__ == "__main__":
    main()
