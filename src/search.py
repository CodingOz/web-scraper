"""
search.py
---------
Query processing and result formatting for the COMP3011 Search Engine Tool.

The :class:`SearchEngine` wraps an :class:`~indexer.Indexer` instance and
provides the high-level operations that map directly to the CLI commands:

* ``print_word``  →  ``> print <word>``
* ``find``        →  ``> find <term> [<term> …]``

Keeping query logic separate from the index data structure means each layer
can be tested and changed independently.

Typical usage
-------------
    from indexer import Indexer
    from search import SearchEngine

    indexer = Indexer()
    indexer.load("data/index.json")

    engine = SearchEngine(indexer)
    engine.print_word("nonsense")
    results = engine.find(["good", "friends"])
"""

import logging
from typing import Optional

from indexer import Indexer

logger = logging.getLogger(__name__)


class SearchEngine:
    """
    High-level search interface over a built or loaded :class:`~indexer.Indexer`.

    Parameters
    ----------
    indexer : Indexer
        An Indexer instance.  May or may not have been built/loaded yet —
        each method checks ``indexer.is_built`` and raises clearly if not.
    """

    def __init__(self, indexer: Indexer) -> None:
        self.indexer: Indexer = indexer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def print_word(self, word: str) -> Optional[dict[str, dict]]:
        """
        Return (and pretty-print to stdout) the postings list for *word*.

        Corresponds to the CLI ``print <word>`` command.

        Parameters
        ----------
        word : str
            The word to look up (case-insensitive, surrounding whitespace
            stripped).

        Returns
        -------
        dict[str, dict] | None
            The postings list ``{url: {freq, positions, tf_idf}}`` if the
            word exists in the index, otherwise ``None``.

        Side-effects
        ------------
        Prints a formatted summary to stdout.
        """
        self._require_index()

        word = word.strip()
        if not word:
            print("  [!] No word provided.")
            return None

        postings = self.indexer.get_postings(word)

        if postings is None:
            print(f"  Word '{word}' not found in the index.")
            return None

        print(f"\n  Inverted index entry for '{word.lower()}':")
        print(f"  {'─' * 56}")
        print(f"  {'URL':<42} {'freq':>5}  {'tf-idf':>8}")
        print(f"  {'─' * 56}")

        # Sort by tf_idf descending for a readable display
        sorted_postings = sorted(
            postings.items(),
            key=lambda kv: kv[1].get("tf_idf", 0.0),
            reverse=True,
        )

        for url, stats in sorted_postings:
            freq = stats["freq"]
            tf_idf = stats.get("tf_idf", 0.0)
            display_url = url if len(url) <= 42 else "…" + url[-41:]
            print(f"  {display_url:<42} {freq:>5}  {tf_idf:>8.4f}")
            # Show first 8 positions to keep output compact
            positions = stats["positions"]
            preview = positions[:8]
            suffix = f" … (+{len(positions) - 8} more)" if len(positions) > 8 else ""
            print(f"    positions: {preview}{suffix}")

        print(f"  {'─' * 56}")
        print(f"  Total pages containing '{word.lower()}': {len(postings)}\n")

        return dict(sorted_postings)

    def find(self, query_terms: list[str]) -> list[dict]:
        """
        Find all pages containing **every** query term, ranked by TF-IDF.

        Corresponds to the CLI ``find <term> [<term> …]`` command.

        Parameters
        ----------
        query_terms : list[str]
            One or more search terms (case-insensitive).

        Returns
        -------
        list[dict]
            Ranked list of result dicts (see :meth:`~indexer.Indexer.find`).
            Empty list when no pages match.

        Side-effects
        ------------
        Prints a formatted results table to stdout.
        """
        self._require_index()

        clean_terms = [t.strip() for t in query_terms if t.strip()]

        if not clean_terms:
            print("  [!] No search terms provided.")
            return []

        query_display = " + ".join(f"'{t}'" for t in clean_terms)
        print(f"\n  Searching for pages containing: {query_display}")

        results = self.indexer.find(clean_terms)

        if not results:
            print(f"  No pages found containing all of: {query_display}\n")
            return []

        print(f"  {'─' * 64}")
        print(f"  {'Rank':<5}  {'Score':>7}  URL")
        print(f"  {'─' * 64}")

        for rank, result in enumerate(results, start=1):
            url = result["url"]
            score = result["score"]
            display_url = url if len(url) <= 50 else "…" + url[-49:]
            print(f"  {rank:<5}  {score:>7.4f}  {display_url}")

            for term, stats in result["term_stats"].items():
                print(
                    f"          '{term}': freq={stats['freq']}, "
                    f"tf-idf={stats['tf_idf']:.4f}, "
                    f"positions={stats['positions'][:5]}"
                    + (" …" if len(stats["positions"]) > 5 else "")
                )

        print(f"  {'─' * 64}")
        print(f"  {len(results)} page(s) found.\n")

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_index(self) -> None:
        """Raise RuntimeError if the indexer has not been built or loaded."""
        if not self.indexer.is_built:
            raise RuntimeError(
                "Index not available. Run 'build' or 'load' first."
            )