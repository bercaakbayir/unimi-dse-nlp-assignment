"""
qa.py — Interactive RAG Q&A over HotpotQA

Usage:
    # Interactive mode
    python src/rag/qa.py

    # Single question
    python src/rag/qa.py -q "Who directed Inception?"

    # Custom options
    python src/rag/qa.py --llm mistral --top-k 7 --collection hotpotqa_passages
"""

import argparse
import sys
from pathlib import Path

# Allow running as `python src/rag/qa.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rag.llm import OllamaLLM
from src.rag.pipeline import RAGPipeline
from src.rag.retriever import ChromaDBRetriever

ROOT       = Path(__file__).resolve().parents[2]
CHROMA_DIR = ROOT / "data" / "chromadb"


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
        "--llm", default="mistral",
        help="Ollama model name (default: mistral)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of passages to retrieve (default: 5)",
    )
    parser.add_argument(
        "--question", "-q", default=None,
        help="Single question — skips interactive loop",
    )
    return parser.parse_args()


def build_pipeline(args: argparse.Namespace) -> RAGPipeline:
    retriever = ChromaDBRetriever(
        chroma_dir=CHROMA_DIR,
        collection_name=args.collection,
        model_name=args.embed_model,
        top_k=args.top_k,
    )
    llm = OllamaLLM(model=args.llm)
    return RAGPipeline(retriever=retriever, llm=llm, top_k=args.top_k)


def print_result(result: dict) -> None:
    print(f"\nAnswer: {result['answer']}")
    print("\nSources:")
    for s in result["sources"]:
        print(f"  [{s['score']:.3f}] {s['title']}")
        print(f"           {s['text'][:120].strip()}...")


def main() -> None:
    args = parse_args()
    pipeline = build_pipeline(args)

    if args.question:
        print_result(pipeline.answer(args.question))
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
        print_result(pipeline.answer(question))


if __name__ == "__main__":
    main()
