"""
build_fever_db.py

Downloads the FEVER paper_dev claims from HuggingFace Hub (raw file, no loading
script), fetches the evidence-referenced Wikipedia pages via the MediaWiki API,
extracts passage-level chunks (sliding window of 3 sentences, step 2), and inserts
into a local ChromaDB collection named 'fever_passages'.

Only evidence-referenced Wikipedia pages are indexed — not all of Wikipedia —
keeping the corpus focused and the index size manageable.

Usage:
    python src/data/build_fever_db.py [--batch-size 64] [--window 3] [--step 2]

Requirements:
    pip install huggingface_hub sentence-transformers chromadb tqdm
"""

import argparse
import hashlib
import json
import logging
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT       = Path(__file__).resolve().parents[2]
DATA_DIR   = ROOT / "data" / "fever"
CHROMA_DIR = ROOT / "data" / "chromadb"
COLLECTION = "fever_passages"

# FEVER paper_dev — try these URLs in order until one works
_FEVER_DEV_URLS = [
    "https://fever.ai/download/fever/paper_dev.jsonl",
    "https://s3-eu-west-1.amazonaws.com/fever.public/paper_dev.jsonl",
]

# MediaWiki API settings — one title per request for reliability
_WIKI_API        = "https://en.wikipedia.org/w/api.php"
_WIKI_BATCH      = 1      # one title at a time: no pipe-separator issues
_WIKI_DELAY      = 3.5    # seconds between requests
_WIKI_MAX_RETRY  = 5      # retries on 429 with exponential backoff


# ── Step 1a: Download FEVER claims ────────────────────────────────────────────

def load_fever_claims(data_dir: Path) -> list[dict]:
    """
    Downloads FEVER paper_dev claims, trying each URL in _FEVER_DEV_URLS.
    Native format per line:
      {"id": int, "verifiable": str, "label": str, "claim": str,
       "evidence": [[[ann_id, ev_id, wiki_title_or_null, sent_id], ...], ...]}
    Cached to data_dir/paper_dev.jsonl.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_file = data_dir / "paper_dev.jsonl"

    if cache_file.exists():
        log.info("FEVER claims already downloaded — loading from cache")
        claims = [json.loads(l) for l in cache_file.read_text().splitlines() if l.strip()]
        log.info(f"Loaded {len(claims):,} claims from cache")
        return claims

    last_error: Exception | None = None
    for url in _FEVER_DEV_URLS:
        try:
            log.info(f"Trying {url} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "fever-rag-builder/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
            lines  = [l for l in raw.splitlines() if l.strip()]
            claims = [json.loads(l) for l in lines]
            cache_file.write_text(raw, encoding="utf-8")
            log.info(f"Downloaded {len(claims):,} claims — saved to {cache_file}")
            return claims
        except Exception as e:
            log.warning(f"  Failed ({type(e).__name__}: {e})")
            last_error = e

    raise RuntimeError(
        "All FEVER download URLs failed. "
        "Please download paper_dev.jsonl manually and place it at:\n"
        f"  {cache_file}\n\n"
        "Download options:\n"
        "  1. Visit https://fever.ai  and look for the dataset download link\n"
        "  2. Search HuggingFace for a mirrored fever dataset:\n"
        "       python -c \"from huggingface_hub import list_datasets; "
        "[print(d.id) for d in list_datasets(search='fever')]\"\n"
        f"Last error: {last_error}"
    )


# ── Step 1b: Fetch evidence Wikipedia pages via MediaWiki API ─────────────────

def _decode_fever_title(title: str) -> str:
    """Convert FEVER's Wikipedia encoding back to standard title.
    FEVER stores ( as -LRB-, ) as -RRB-, [ as -LSB-, ] as -RSB-.
    """
    return (title
            .replace("-LRB-", "(").replace("-RRB-", ")")
            .replace("-LSB-", "[").replace("-RSB-", "]")
            .replace("-LCB-", "{").replace("-RCB-", "}"))


def _wiki_api_fetch(titles: list[str]) -> dict[str, str]:
    """
    Fetches plain-text extracts for a batch of Wikipedia titles.
    Returns {normalised_title: plain_text}.

    The pipe | separator in the titles parameter must NOT be percent-encoded
    (%7C) — MediaWiki only treats a literal | as a multi-value separator.
    We therefore build the URL manually instead of using urlencode().
    """
    base_params = urllib.parse.urlencode({
        "action":          "query",
        "prop":            "extracts",
        "explaintext":     "1",
        "exsectionformat": "plain",
        "exlimit":         "max",
        "format":          "json",
        "redirects":       "1",
    })
    # Keep | unencoded so MediaWiki treats each entry as a separate title
    titles_param = urllib.parse.quote(
        "|".join(t.replace(" ", "_") for t in titles),
        safe="|",
    )
    url = f"{_WIKI_API}?{base_params}&titles={titles_param}"

    req = urllib.request.Request(url, headers={"User-Agent": "fever-rag-builder/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    results: dict[str, str] = {}
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        if int(page.get("pageid", -1)) < 0:   # skip "missing" pages
            continue
        title = page.get("title", "")
        text  = page.get("extract", "")
        if title and text.strip():
            results[title] = text
    return results


def _wiki_api_fetch_with_retry(titles: list[str]) -> dict[str, str]:
    """Wraps _wiki_api_fetch with exponential backoff on HTTP 429."""
    import urllib.error
    for attempt in range(_WIKI_MAX_RETRY):
        try:
            return _wiki_api_fetch(titles)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 2 ** attempt * 10  # 10 s, 20 s, 40 s
                log.warning(f"  429 rate-limited — waiting {wait}s before retry {attempt+1}/{_WIKI_MAX_RETRY}")
                time.sleep(wait)
            else:
                raise
    log.warning(f"  Giving up on batch after {_WIKI_MAX_RETRY} retries: {titles[0]}...")
    return {}


def load_fever_corpus(data_dir: Path, claims: list[dict]) -> list[dict]:
    """
    Collects unique evidence Wikipedia titles from claims, then fetches
    each page's plain text from the MediaWiki API one title at a time.
    Writes incrementally to data_dir/wiki_pages.jsonl so Ctrl-C is safe —
    re-running resumes from where it left off.
    """
    cache_file = data_dir / "wiki_pages.jsonl"

    # Load already-fetched titles so we can resume after an interruption
    already_fetched: dict[str, bool] = {}
    if cache_file.exists():
        existing = [json.loads(l) for l in cache_file.read_text().splitlines() if l.strip()]
        already_fetched = {p["title"] for p in existing}
        if len(existing) > 0:
            log.info(f"Resuming — {len(existing):,} pages already in cache")

    # Collect unique evidence titles; decode FEVER's -LRB-/-RRB- encoding
    evidence_titles: set[str] = set()
    for claim in claims:
        for ann_set in claim.get("evidence", []):
            for piece in ann_set:
                if len(piece) >= 3 and piece[2]:
                    decoded = _decode_fever_title(piece[2]).replace("_", " ")
                    evidence_titles.add(decoded)

    log.info(f"Found {len(evidence_titles):,} unique evidence Wikipedia titles")

    title_list = sorted(evidence_titles)
    pending = [t for t in title_list if t not in already_fetched]

    if not pending:
        log.info("All pages already fetched — loading from cache")
        return [json.loads(l) for l in cache_file.read_text().splitlines() if l.strip()]

    log.info(f"Fetching {len(pending):,} remaining pages from Wikipedia API...")
    failed: list[str] = []

    from tqdm import tqdm
    with cache_file.open("a", encoding="utf-8") as f:
        batches = [pending[i:i+_WIKI_BATCH] for i in range(0, len(pending), _WIKI_BATCH)]
        for batch in tqdm(batches, desc="Fetching wiki pages"):
            try:
                fetched = _wiki_api_fetch_with_retry(batch)
                for title, text in fetched.items():
                    if text.strip():
                        f.write(json.dumps({"title": title, "text": text}) + "\n")
                        f.flush()
                if not fetched:
                    failed.extend(batch)
            except Exception as e:
                log.warning(f"Batch fetch failed ({batch[0]}...): {e}")
                failed.extend(batch)
            time.sleep(_WIKI_DELAY)

    if failed:
        log.warning(f"{len(failed)} titles could not be fetched — skipped")

    pages = [json.loads(l) for l in cache_file.read_text().splitlines() if l.strip()]
    log.info(f"Fetched {len(pages):,} pages total — saved to {cache_file}")
    return pages


# ── Step 2: Sliding-window chunk extraction ───────────────────────────────────

def make_chunk_id(title: str, chunk_text: str) -> str:
    """Deterministic ID: SHA256 of title + text (truncated to 64 chars)."""
    raw = f"{title}|||{chunk_text}"
    return hashlib.sha256(raw.encode()).hexdigest()[:64]


def _split_sentences(text: str) -> list[str]:
    """
    Split plain Wikipedia text into sentences.
    Uses a simple regex that splits on sentence-ending punctuation followed
    by whitespace + capital letter, which works well for encyclopaedic prose.
    """
    # Split on ". ", "! ", "? " followed by a capital letter or digit
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9\"])', text)
    sentences = [s.strip() for s in parts if s.strip() and len(s.strip()) > 10]
    return sentences


def extract_chunks(
    wiki_pages: list[dict],
    window: int = 3,
    step: int = 2,
) -> dict[str, dict]:
    """
    Extracts passage chunks from Wikipedia pages using a sliding window.
    Each page's text is split into sentences first, then windowed.
    """
    chunks: dict[str, dict] = {}

    log.info("Extracting and deduplicating chunks from wiki pages...")

    for page in wiki_pages:
        title     = page["title"]
        raw_text  = page.get("text", "")
        if not raw_text:
            continue

        sentences = _split_sentences(raw_text)
        if not sentences:
            continue

        # Sliding-window chunking (same logic as build_hotpotqa_db.py)
        if len(sentences) <= window:
            windows           = [sentences]
            sent_index_groups = [list(range(len(sentences)))]
        else:
            windows           = []
            sent_index_groups = []
            for start in range(0, len(sentences) - window + 1, step):
                end = start + window
                windows.append(sentences[start:end])
                sent_index_groups.append(list(range(start, end)))
            # Ensure final sentences are always covered
            if sent_index_groups[-1][-1] < len(sentences) - 1:
                last = sentences[-window:]
                windows.append(last)
                sent_index_groups.append(list(range(len(sentences) - window, len(sentences))))

        for chunk_idx, (sents, sent_indices) in enumerate(zip(windows, sent_index_groups)):
            chunk_text = " ".join(sents).strip()
            if not chunk_text:
                continue

            chunk_id = make_chunk_id(title, chunk_text)
            if chunk_id not in chunks:
                chunks[chunk_id] = {
                    "id":               chunk_id,
                    "text":             chunk_text,
                    "title":            title,
                    "chunk_index":      chunk_idx,
                    "sentence_indices": sent_indices,
                    "dataset":          "fever",
                    "split":            "paper_dev",
                    "poisoned":         False,
                    "poison_strategy":  None,
                }

    log.info(f"Extracted {len(chunks):,} unique chunks from {len(wiki_pages):,} pages")
    return chunks


# ── Steps 3 + 4: Streaming encode → insert ───────────────────────────────────

def encode_and_insert(
    chunks: dict[str, dict],
    chroma_dir: Path,
    collection_name: str,
    encode_batch_size: int = 64,
    insert_batch_size: int = 512,
    outer_batch_size: int = 2000,
) -> None:
    """
    Encodes chunks and inserts them into ChromaDB in streaming outer batches.
    Resumable: skips already-inserted chunk IDs.
    """
    from sentence_transformers import SentenceTransformer
    import torch
    import chromadb
    from tqdm import tqdm

    if torch.backends.mps.is_available():
        device = "mps"
        log.info("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        log.info("Using CUDA GPU")
    else:
        device = "cpu"
        log.info("Using CPU — encoding will be slow for large collection")

    log.info("Loading all-MiniLM-L12-v2...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", device=device)

    chroma_dir.mkdir(parents=True, exist_ok=True)
    client     = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space":           "cosine",
            "hnsw:batch_size":      50000,
            "hnsw:sync_threshold":  200000,
        },
    )

    existing_ids = set(collection.get(include=[], limit=collection.count() + 1)["ids"])
    chunk_list   = [c for c in chunks.values() if c["id"] not in existing_ids]

    skipped = len(chunks) - len(chunk_list)
    if skipped:
        log.info(f"Skipping {skipped:,} already-inserted chunks — resuming from checkpoint")

    total = len(chunk_list)
    log.info(f"Encoding and inserting {total:,} chunks "
             f"(outer={outer_batch_size}, encode={encode_batch_size}, insert={insert_batch_size})")

    inserted = 0
    for outer_start in tqdm(range(0, total, outer_batch_size), desc="Outer batches"):
        outer_end = min(outer_start + outer_batch_size, total)
        batch     = chunk_list[outer_start:outer_end]

        texts     = [c["text"] for c in batch]
        ids       = [c["id"]   for c in batch]
        metadatas = [
            {
                "title":            c["title"],
                "chunk_index":      c["chunk_index"],
                "sentence_indices": json.dumps(c["sentence_indices"]),
                "dataset":          c["dataset"],
                "split":            c["split"],
                "poisoned":         c["poisoned"],
                "poison_strategy":  c["poison_strategy"] or "",
            }
            for c in batch
        ]

        embeddings = model.encode(
            texts,
            batch_size=encode_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=device,
        ).tolist()

        for start in range(0, len(ids), insert_batch_size):
            end = min(start + insert_batch_size, len(ids))
            collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
                documents=texts[start:end],
            )

        inserted += len(batch)
        log.info(f"  {inserted:,}/{total:,} inserted — ChromaDB total: {collection.count():,}")

    log.info(f"Done — collection '{collection_name}' now has {collection.count():,} documents")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build ChromaDB vector store from FEVER paper_dev evidence pages"
    )
    parser.add_argument("--batch-size",   type=int, default=64,   help="Encoding batch size (default: 64)")
    parser.add_argument("--window",       type=int, default=3,    help="Sliding window size in sentences (default: 3)")
    parser.add_argument("--step",         type=int, default=2,    help="Sliding window step size (default: 2)")
    parser.add_argument("--insert-batch", type=int, default=512,  help="ChromaDB upsert batch size (default: 512)")
    parser.add_argument("--outer-batch",  type=int, default=2000, help="Chunks per encode→insert cycle (default: 2000)")
    parser.add_argument("--collection",   type=str, default=COLLECTION,
                        help=f"ChromaDB collection name (default: {COLLECTION})")
    return parser.parse_args()


def main():
    args = parse_args()

    log.info("=" * 60)
    log.info("FEVER → ChromaDB pipeline")
    log.info(f"  data dir    : {DATA_DIR}")
    log.info(f"  chroma dir  : {CHROMA_DIR}")
    log.info(f"  collection  : {args.collection}")
    log.info(f"  window/step : {args.window}/{args.step}")
    log.info(f"  encode batch: {args.batch_size}")
    log.info(f"  outer batch : {args.outer_batch}")
    log.info("=" * 60)

    # 1a. Download / load claims
    claims = load_fever_claims(DATA_DIR)

    # 1b. Fetch evidence Wikipedia pages via MediaWiki API
    wiki_pages = load_fever_corpus(DATA_DIR, claims)

    # 2. Extract + deduplicate chunks
    chunks = extract_chunks(wiki_pages, window=args.window, step=args.step)

    # 3. Encode + insert (streaming — no full-dataset RAM spike)
    encode_and_insert(
        chunks,
        chroma_dir=CHROMA_DIR,
        collection_name=args.collection,
        encode_batch_size=args.batch_size,
        insert_batch_size=args.insert_batch,
        outer_batch_size=args.outer_batch,
    )

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
