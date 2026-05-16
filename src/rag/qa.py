"""
qa.py — Interactive RAG Q&A (HotpotQA) and Fact-Checking (FEVER)

Usage:
    # Interactive mode — HotpotQA (default)
    python src/rag/qa.py --hotpotqa

    # Interactive mode — FEVER fact-checking
    python src/rag/qa.py --fever

    # Single question / claim
    python src/rag/qa.py --hotpotqa -q "Who directed Inception?"
    python src/rag/qa.py --fever -q "Nikolaj Coster-Waldau worked with Fox Broadcasting Company."

    # Custom options
    python src/rag/qa.py --llm mistral --top-k 7
    python src/rag/qa.py --retriever dense --no-multi-query
"""

import argparse
import sys
from pathlib import Path

from rich import box as rich_box
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Allow running as `python src/rag/qa.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_console = Console()

from src.rag.llm import OllamaLLM
from src.rag.pipeline import RAGPipeline
from src.rag.retriever import ChromaDBRetriever

ROOT            = Path(__file__).resolve().parents[2]
CHROMA_DIR      = ROOT / "data" / "chromadb"

# Per-dataset defaults
_DATASET_DEFAULTS = {
    "hotpotqa": {
        "collection":    "hotpotqa_passages",
        "bm25_path":     ROOT / "data" / "bm25_index.pkl",
        "corpus_ids":    ROOT / "data" / "bm25_corpus_ids.pkl",
        "mode":          "qa",
    },
    "fever": {
        "collection":    "fever_passages",
        "bm25_path":     ROOT / "data" / "fever_bm25_index.pkl",
        "corpus_ids":    ROOT / "data" / "fever_bm25_corpus_ids.pkl",
        "mode":          "fact_check",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG Q&A (HotpotQA) and Fact-Checking (FEVER)")

    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--hotpotqa", action="store_true",
        help="Use HotpotQA dataset / QA mode (default)",
    )
    dataset_group.add_argument(
        "--fever", action="store_true",
        help="Use FEVER dataset / fact-checking mode",
    )

    parser.add_argument(
        "--collection", default=None,
        help="ChromaDB collection name (overrides dataset default)",
    )
    parser.add_argument(
        "--embed-model", default="sentence-transformers/all-MiniLM-L12-v2",
        help="Sentence-transformers model used at index time",
    )
    parser.add_argument(
        "--llm", default="llama3.2:3b",
        help="Ollama model name (default: llama3.2:3b)",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of passages to retrieve (default: 10)",
    )
    parser.add_argument(
        "--retriever", choices=["dense", "hybrid"], default="hybrid",
        help="Retrieval strategy: dense (ChromaDB only) or hybrid (BM25+dense, default: hybrid)",
    )
    parser.add_argument(
        "--multi-query", action="store_true",
        help="Enable LLM query decomposition (requires a capable model, e.g. llama3.2:3b or larger)",
    )
    parser.add_argument(
        "--question", "-q", default=None,
        help="Single question or claim — skips interactive loop",
    )
    parser.add_argument(
        "--poison", type=lambda x: x.lower() == "true", default=False,
        help="Inject false facts into retrieved passages (default: false)",
    )
    parser.add_argument(
        "--poison-seed", type=int, default=None,
        help="Random seed for poison passage selection (default: non-deterministic)",
    )
    args = parser.parse_args()

    # Resolve dataset-specific defaults (fever if --fever, otherwise hotpotqa)
    dataset_key = "fever" if args.fever else "hotpotqa"
    defaults = _DATASET_DEFAULTS[dataset_key]
    args._dataset = dataset_key
    args._mode = defaults["mode"]
    args._bm25_path = defaults["bm25_path"]
    args._corpus_ids_path = defaults["corpus_ids"]
    if args.collection is None:
        args.collection = defaults["collection"]

    return args


def build_pipeline(args: argparse.Namespace) -> RAGPipeline:
    bm25_path = args._bm25_path
    corpus_ids_path = args._corpus_ids_path

    if args.retriever == "hybrid" and bm25_path.exists():
        from src.rag.retriever import HybridRetriever
        retriever = HybridRetriever(
            chroma_dir=CHROMA_DIR,
            collection_name=args.collection,
            model_name=args.embed_model,
            top_k=args.top_k,
            bm25_path=bm25_path,
            corpus_ids_path=corpus_ids_path,
        )
    else:
        if args.retriever == "hybrid":
            print("Warning: BM25 index not found — falling back to dense retrieval.")
            if args._dataset == "fever":
                print("Run: python src/data/build_bm25_index.py --collection fever_passages "
                      "--output data/fever_bm25_index.pkl --corpus-ids-output data/fever_bm25_corpus_ids.pkl")
            else:
                print("Run: python src/data/build_bm25_index.py")
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
    pipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
        top_k=args.top_k,
        use_multi_query=args.multi_query,
        poisoner=poisoner,
        mode=args._mode,
    )
    _console.print("[dim]Warming up model…[/dim]", end="\r")
    pipeline.llm.generate("hi")
    _console.print(" " * 30, end="\r")
    return pipeline


def _final_marker(mode: str) -> str:
    return "Final Verdict:" if mode == "fact_check" else "Final Answer:"


def _final_panel_title(mode: str) -> str:
    return "Final Verdict" if mode == "fact_check" else "Final Answer"


def _normalize_verdict(text: str) -> str:
    """Normalize LLM typos like REFUSES/SUPPORTED/REFUTED to canonical FEVER labels."""
    s = text.upper().strip()
    if "NOT ENOUGH" in s or "NEI" in s or "INSUFFICIENT" in s:
        return "NOT ENOUGH INFO"
    if "REFUT" in s or "REFUS" in s or "FALSE" in s or "CONTRADICT" in s:
        return "REFUTES"
    if "SUPPORT" in s or "TRUE" in s or "CONFIRM" in s:
        return "SUPPORTS"
    return text.strip()


def print_result_stream(pipeline: RAGPipeline, question: str) -> None:
    _console.rule("[yellow]Chain-of-Thought[/yellow]", style="yellow")

    full_result = None
    for event, content in pipeline.answer_stream(question):
        if event == "token":
            print(content, end="", flush=True)
        else:
            full_result = content
    print()
    _console.rule(style="yellow")

    if full_result is None:
        return

    full_answer = full_result["answer"]
    marker = _final_marker(pipeline.mode)
    panel_title = _final_panel_title(pipeline.mode)

    sources = Text()
    for s in full_result["sources"]:
        sources.append(f"[{s['score']:.3f}] ", style="bold")
        if s.get("poisoned"):
            sources.append(f"{s['title']} ", style="bold")
            sources.append("[POISONED]\n", style="bold red")
        else:
            sources.append(f"{s['title']}\n", style="bold")
        sources.append(f"  {s['text'][:120].strip()}...\n\n")
    _console.print(Panel(sources, title="Retrieved Documents",
                         border_style="dark_orange", box=rich_box.ROUNDED))

    if marker in full_answer:
        final = full_answer.split(marker, 1)[1].strip()
        if pipeline.mode == "fact_check":
            final = _normalize_verdict(final)
        _console.print(Panel(Text(final, style="bold"), title=panel_title,
                             border_style="blue", box=rich_box.ROUNDED))


def print_result(result: dict, mode: str = "qa") -> None:
    full_answer = result["answer"]
    marker = _final_marker(mode)
    panel_title = _final_panel_title(mode)

    if marker in full_answer:
        cot, final = full_answer.split(marker, 1)
        cot = cot.strip()
        final = final.strip()
    else:
        cot = full_answer.strip()
        final = None

    _console.print(Panel(cot, title="Chain-of-Thought", border_style="yellow", box=rich_box.ROUNDED))

    sources = Text()
    for s in result["sources"]:
        sources.append(f"[{s['score']:.3f}] ", style="bold")
        sources.append(f"{s['title']}\n", style="bold")
        sources.append(f"  {s['text'][:120].strip()}...\n\n")
    _console.print(Panel(sources, title="Retrieved Documents", border_style="dark_orange", box=rich_box.ROUNDED))

    if final:
        _console.print(Panel(Text(final, style="bold"), title=panel_title, border_style="blue", box=rich_box.ROUNDED))


def main() -> None:
    args = parse_args()
    pipeline = build_pipeline(args)

    if args.question:
        print_result_stream(pipeline, args.question)
        return

    if args._mode == "fact_check":
        banner = "RAG Fact-Checking — FEVER  |  type 'quit' to exit\n"
        prompt_label = "Claim: "
    else:
        banner = "RAG Q&A — HotpotQA  |  type 'quit' to exit\n"
        prompt_label = "Question: "

    print(banner)
    while True:
        try:
            question = input(prompt_label).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        print_result_stream(pipeline, question)


if __name__ == "__main__":
    main()
