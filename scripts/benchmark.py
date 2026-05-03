"""
scripts/benchmark.py
--------------------
Benchmark the four core operations of the COMP3011 Search Engine Tool:

    build   – construct the inverted index from a captured corpus
    save    – serialise the index to a JSON file on disk
    load    – deserialise the index from disk into a fresh Indexer
    find    – run a representative multi-term AND query

Each operation is timed over a configurable number of repetitions and the
results are printed as a formatted table showing mean, min, max, and an
informal O() complexity note derived from corpus size.

Usage
-----
    # From the repository root:
    python scripts/benchmark.py

    # Larger repetition count for tighter confidence intervals:
    python scripts/benchmark.py --reps 20

    # Write results to a CSV file as well:
    python scripts/benchmark.py --csv results/benchmark.csv

    # Benchmark with stemming enabled:
    python scripts/benchmark.py --stem

Why benchmark?
--------------
Benchmarking complements Big-O analysis: O() tells you the *shape* of the
growth curve; wall-clock times tell you the *constant factor*.  A function
that is O(n) with a large constant can be slower in practice than an O(n log n)
function with a small one.  The numbers here let us reason about both.

Complexity notes (printed in the table)
----------------------------------------
build:  O(T)  where T = total tokens across all pages.  Each token triggers
              at most two dict lookups (word key, url key) — both O(1) amortised
              for Python dicts.  The TF-IDF pass is a second O(T) sweep.
save:   O(T)  json.dump serialises every entry exactly once.
load:   O(T)  json.load deserialises every entry exactly once.
find:   O(|P₁| + |P₂| + … + |Pₙ|)  where Pᵢ is the posting list for term i.
              Intersection via set operations is O(min(|P|)) per pair.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

# Make src/ importable when running from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from indexer import Indexer  # noqa: E402  (import after sys.path manipulation)

# ---------------------------------------------------------------------------
# Synthetic corpus — representative of the real quotes.toscrape.com structure
# ---------------------------------------------------------------------------

_QUOTE_TEMPLATE = """
<html><body>
  <div class="quote">
    <span class="text">{quote}</span>
    <small class="author">{author}</small>
    <div class="tags">
      {tags}
    </div>
  </div>
</body></html>
"""

_QUOTES = [
    ("The world as we have created it is a process of our thinking.",  "Albert Einstein",  "change deep-thoughts thinking world"),
    ("It is our choices that show what we truly are.",                  "J.K. Rowling",     "choices life"),
    ("There are only two ways to live your life.",                      "Albert Einstein",  "inspirational life live"),
    ("The person you will be in five years is based on the books.",     "Charlie Tremendous Jones", "books friends life"),
    ("Imperfection is beauty madness is genius.",                       "Marilyn Monroe",   "be-yourself inspirational"),
    ("Try not to become a man of success.",                             "Albert Einstein",  "adulthood success value"),
    ("It is better to be hated for what you are.",                      "Andre Gide",       "life"),
    ("I have not failed. I've just found ten thousand ways.",           "Thomas A. Edison", "edison failure inspirational"),
    ("A woman is like a tea bag.",                                      "Eleanor Roosevelt","misattributed-eleanor-roosevelt"),
    ("A day without sunshine is like a night.",                         "Steve Martin",     "humor obvious simile"),
    ("The only way to get rid of temptation is to yield to it.",        "Oscar Wilde",      "gilded-age"),
    ("Anyone who has never made a mistake has never tried.",            "Albert Einstein",  "mistakes"),
    ("I am good but not an angel.",                                     "Marilyn Monroe",   "life"),
    ("You only live once but if you do it right once is enough.",        "Mae West",         "humor"),
    ("To live is the rarest thing in the world. Most people exist.",    "Oscar Wilde",      "life"),
    ("What you do speaks so loudly I cannot hear what you say.",        "Ralph Waldo Emerson", "actions"),
    ("There is no greater agony than bearing an untold story.",         "Maya Angelou",     "stories"),
    ("When one door of happiness closes another opens.",                "Helen Keller",     "happiness perseverance"),
    ("Life is what happens when you are busy making other plans.",      "John Lennon",      "life"),
    ("Success is not final failure is not fatal it is the courage.",    "Winston S. Churchill", "failure inspirational success"),
]


def _build_corpus(n_pages: int) -> dict[str, str]:
    """
    Generate a synthetic corpus of *n_pages* HTML pages.

    Pages are built by cycling through the ``_QUOTES`` list so the corpus
    always has predictable, realistic content regardless of size.

    Parameters
    ----------
    n_pages : int
        Number of pages to generate.

    Returns
    -------
    dict[str, str]
        Mapping of ``{url: html}`` ready for ``Indexer.build()``.
    """
    pages: dict[str, str] = {}
    for i in range(n_pages):
        quote, author, tags = _QUOTES[i % len(_QUOTES)]
        tag_html = " ".join(f'<a class="tag">{t}</a>' for t in tags.split())
        html = _QUOTE_TEMPLATE.format(quote=quote, author=author, tags=tag_html)
        pages[f"https://quotes.toscrape.com/page/{i + 1}"] = html
    return pages


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _time_it(fn: Callable[[], object], reps: int) -> tuple[float, float, float]:
    """
    Call *fn* *reps* times and return (mean_ms, min_ms, max_ms).

    The first call is a warm-up and is excluded from the statistics to avoid
    penalising cold-start effects (Python import caching, OS page faults on
    first file access, etc.).

    Parameters
    ----------
    fn : Callable[[], object]
        Zero-argument callable to time.
    reps : int
        Number of timed repetitions (warm-up call not counted).

    Returns
    -------
    tuple[float, float, float]
        (mean_ms, min_ms, max_ms) — all in milliseconds.
    """
    fn()  # warm-up: excluded from results

    samples: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1_000)

    return statistics.mean(samples), min(samples), max(samples)


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------

def run_benchmark(
    n_pages: int,
    reps: int,
    stem: bool,
    csv_path: str | None,
) -> None:
    """
    Run the full benchmark suite and print results to stdout.

    Parameters
    ----------
    n_pages : int
        Corpus size (number of pages to index).
    reps : int
        Timed repetitions per operation.
    stem : bool
        Whether to build the index with Porter stemming enabled.
    csv_path : str | None
        Optional path to write a CSV summary.  ``None`` skips CSV output.
    """
    corpus = _build_corpus(n_pages)
    total_tokens = sum(
        len(html.split()) for html in corpus.values()
    )

    print(f"\n  COMP3011 Search Engine — Benchmark")
    print(f"  {'─' * 60}")
    print(f"  Corpus:        {n_pages} pages  /  ~{total_tokens:,} tokens")
    print(f"  Repetitions:   {reps} (+ 1 warm-up, excluded)")
    print(f"  Stemming:      {'on (PorterStemmer)' if stem else 'off'}")
    print(f"  {'─' * 60}\n")

    rows: list[tuple[str, str, float, float, float]] = []

    # ── build ────────────────────────────────────────────────────────────────
    indexer: Indexer | None = None

    def _build() -> None:
        nonlocal indexer
        indexer = Indexer(stem=stem)
        indexer.build(corpus)

    mean, lo, hi = _time_it(_build, reps)
    rows.append(("build", "O(T)  — T = total tokens", mean, lo, hi))

    # Build once properly so save/load/find have a valid index
    assert indexer is not None
    n_terms = len(indexer.index)

    # ── save ─────────────────────────────────────────────────────────────────
    tmp_dir = tempfile.mkdtemp()
    index_path = os.path.join(tmp_dir, "index.json")

    def _save() -> None:
        assert indexer is not None
        indexer.save(index_path)

    mean, lo, hi = _time_it(_save, reps)
    rows.append(("save", "O(T)  — serialise every entry", mean, lo, hi))

    # ── load ─────────────────────────────────────────────────────────────────
    def _load() -> None:
        fresh = Indexer(stem=stem)
        fresh.load(index_path)

    mean, lo, hi = _time_it(_load, reps)
    rows.append(("load", "O(T)  — deserialise every entry", mean, lo, hi))

    # ── find (multi-term AND) ────────────────────────────────────────────────
    # Choose terms that actually appear in the corpus so the query is
    # representative of a real search rather than a trivial early-exit.
    query = ["life", "success"] if not stem else ["life", "success"]

    def _find() -> None:
        assert indexer is not None
        indexer.find(query)

    mean, lo, hi = _time_it(_find, reps)
    posting_sizes = " × ".join(
        str(len(indexer.index.get(
            indexer._stemmer.stem(t) if stem else t, {}
        )))
        for t in query
    )
    rows.append(("find", f"O(|P|) — postings sizes: {posting_sizes}", mean, lo, hi))

    # ── find_phrase ──────────────────────────────────────────────────────────
    phrase = ["our", "thinking"]

    def _find_phrase() -> None:
        assert indexer is not None
        indexer.find_phrase(phrase)

    mean, lo, hi = _time_it(_find_phrase, reps)
    rows.append(("find_phrase", "O(P₀ × N) — positional check", mean, lo, hi))

    # ── Print table ──────────────────────────────────────────────────────────
    col_op    = 12
    col_big_o = 38
    col_num   = 10

    header = (
        f"  {'Operation':<{col_op}}  {'Complexity':<{col_big_o}}"
        f"  {'Mean (ms)':>{col_num}}  {'Min (ms)':>{col_num}}  {'Max (ms)':>{col_num}}"
    )
    divider = "  " + "─" * (len(header) - 2)

    print(header)
    print(divider)
    for op, note, m, lo_, hi_ in rows:
        print(
            f"  {op:<{col_op}}  {note:<{col_big_o}}"
            f"  {m:>{col_num}.3f}  {lo_:>{col_num}.3f}  {hi_:>{col_num}.3f}"
        )
    print(divider)
    print(f"\n  Index size: {n_terms:,} unique terms across {n_pages} pages\n")

    # ── Complexity analysis narrative ────────────────────────────────────────
    print("  Complexity analysis")
    print("  " + "─" * 58)
    print(
        "  build   is O(T) where T = total tokens.  Each token causes at\n"
        "          most two dict insertions, both O(1) amortised.  The\n"
        "          TF-IDF pass is a second O(T) sweep — no extra structure.\n"
    )
    print(
        "  save    is O(T): json.dump visits every posting exactly once.\n"
        "          File I/O dominates; expect high variance across runs.\n"
    )
    print(
        "  load    is O(T): json.load reconstructs the full dict in one\n"
        "          pass.  Faster than build because no HTML parsing.\n"
    )
    print(
        "  find    is O(Σ|Pᵢ|): build a set per term, then intersect.\n"
        "          Python set intersection is O(min(|P|)) per pair, so\n"
        "          ordering terms by ascending posting-list size first\n"
        "          (smallest set as the pivot) would reduce work further.\n"
    )
    print(
        "  phrase  is O(P₀ × N) per URL after the AND intersection step,\n"
        "          where P₀ = occurrences of the first term and N = phrase\n"
        "          length.  The set-membership check for each offset is\n"
        "          O(1), so the constant factor is low.\n"
    )

    # ── Optional CSV output ──────────────────────────────────────────────────
    if csv_path:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "operation", "complexity_note",
                "mean_ms", "min_ms", "max_ms",
                "n_pages", "n_terms", "stem",
            ])
            for op, note, m, lo_, hi_ in rows:
                writer.writerow([op, note, f"{m:.3f}", f"{lo_:.3f}", f"{hi_:.3f}",
                                  n_pages, n_terms, stem])
        print(f"  Results written to: {csv_path}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the COMP3011 Search Engine Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pages", type=int, default=50,
        help="Number of synthetic corpus pages to index",
    )
    parser.add_argument(
        "--reps", type=int, default=10,
        help="Timed repetitions per operation (warm-up excluded)",
    )
    parser.add_argument(
        "--stem", action="store_true",
        help="Enable Porter stemming during the benchmark",
    )
    parser.add_argument(
        "--csv", default=None, metavar="PATH",
        help="Optional path to write a CSV results file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    run_benchmark(
        n_pages=_args.pages,
        reps=_args.reps,
        stem=_args.stem,
        csv_path=_args.csv,
    )
