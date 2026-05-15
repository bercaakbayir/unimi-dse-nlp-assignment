"""
qa.py — Interactive RAG Q&A over HotpotQA

Usage:
    # Interactive mode
    python src/rag/qa.py

    # Single question
    python src/rag/qa.py -q "Who directed Inception?"

    # Custom options
    python src/rag/qa.py --llm mistral --top-k 7 --collection hotpotqa_passages
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
BM25_PATH       = ROOT / "data" / "bm25_index.pkl"
CORPUS_IDS_PATH = ROOT / "data" / "bm25_corpus_ids.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG Q&A over HotpotQA")
    parser.add_argument(
        "--collection", default="hotpotqa_passages",
        help="ChromaDB collection name (default: hotpotqa_passages)",
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
        help="Single question — skips interactive loop",
    )
    parser.add_argument(
        "--poison", type=lambda x: x.lower() == "true", default=False,
        help="Inject false facts into retrieved passages (default: false)",
    )
    parser.add_argument(
        "--poison-seed", type=int, default=None,
        help="Random seed for poison passage selection (default: non-deterministic)",
    )
    return parser.parse_args()


def build_pipeline(args: argparse.Namespace) -> RAGPipeline:
    if args.retriever == "hybrid" and BM25_PATH.exists():
        from src.rag.retriever import HybridRetriever
        retriever = HybridRetriever(
            chroma_dir=CHROMA_DIR,
            collection_name=args.collection,
            model_name=args.embed_model,
            top_k=args.top_k,
            bm25_path=BM25_PATH,
            corpus_ids_path=CORPUS_IDS_PATH,
        )
    else:
        if args.retriever == "hybrid":
            print("Warning: BM25 index not found — falling back to dense retrieval.")
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
    )
    _console.print("[dim]Warming up model…[/dim]", end="\r")
    pipeline.llm.generate("hi")
    _console.print(" " * 30, end="\r")
    return pipeline


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

    if "Final Answer:" in full_answer:
        final = full_answer.split("Final Answer:", 1)[1].strip()
        _console.print(Panel(Text(final, style="bold"), title="Final Answer",
                             border_style="blue", box=rich_box.ROUNDED))


def print_result(result: dict) -> None:
    full_answer = result["answer"]

    # Split reasoning from the short final answer
    if "Final Answer:" in full_answer:
        cot, final = full_answer.split("Final Answer:", 1)
        cot = cot.strip()
        final = final.strip()
    else:
        cot = full_answer.strip()
        final = None

    # Chain-of-Thought — yellow box
    _console.print(Panel(cot, title="Chain-of-Thought", border_style="yellow", box=rich_box.ROUNDED))

    # Retrieved Documents — orange box
    sources = Text()
    for s in result["sources"]:
        sources.append(f"[{s['score']:.3f}] ", style="bold")
        sources.append(f"{s['title']}\n", style="bold")
        sources.append(f"  {s['text'][:120].strip()}...\n\n")
    _console.print(Panel(sources, title="Retrieved Documents", border_style="dark_orange", box=rich_box.ROUNDED))

    # Final Answer — blue box
    if final:
        _console.print(Panel(Text(final, style="bold"), title="Final Answer", border_style="blue", box=rich_box.ROUNDED))


def main() -> None:
    args = parse_args()
    pipeline = build_pipeline(args)

    if args.question:
        print_result_stream(pipeline, args.question)
        return

    print("RAG Q&A — HotpotQA  |  type 'quit' to exit\n")
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        print_result_stream(pipeline, question)


if __name__ == "__main__":
    main()
