"""
indexer.py
----------
Inverted index builder and TF-IDF scorer for the COMP3011 Search Engine Tool.

The :class:`Indexer` consumes a ``{url: html}`` mapping (as produced by
:class:`~crawler.Crawler`) and builds a persistent inverted index that stores,
per word, per page:

* **frequency**  – raw term count within the page
* **positions**  – ordered list of token offsets (0-based) where the term
  appears in the page's visible text
* **tf_idf**     – TF-IDF weight computed after the full corpus is indexed

Index structure (nested dict)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: text

    {
      "word": {
          "https://example.com/page1": {
              "freq":      3,
              "positions": [4, 17, 42],
              "tf_idf":    0.382
          },
          ...
      },
      ...
    }

This structure gives O(1) lookup by word and O(1) lookup by URL within a
word, which is optimal for the ``print`` and ``find`` commands.

Typical usage
-------------
    from crawler import Crawler
    from indexer import Indexer

    pages   = Crawler("https://quotes.toscrape.com/").crawl()
    indexer = Indexer()
    indexer.build(pages)
    indexer.save("data/index.json")

    # Later …
    indexer2 = Indexer()
    indexer2.load("data/index.json")
    results = indexer2.find(["good", "friends"])
"""

import json
import logging
import math
import re
import string
from pathlib import Path

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Regex that matches one or more whitespace characters (used for tokenisation)
_WHITESPACE_RE = re.compile(r"\s+")

# Characters to strip from the edges of each token
_STRIP_CHARS = string.punctuation + "\u201c\u201d\u2018\u2019"  # incl. curly quotes


class Indexer:
    """
    Builds, persists, and queries an inverted index over a set of web pages.

    Parameters
    ----------
    None — all configuration is done via method arguments.

    Attributes
    ----------
    index : dict[str, dict[str, dict]]
        The in-memory inverted index.  See module docstring for schema.
    is_built : bool
        ``True`` after :meth:`build` or :meth:`load` has been called
        successfully.
    """

    def __init__(self) -> None:
        self.index: dict[str, dict[str, dict]] = {}
        self.is_built: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, pages: dict[str, str]) -> None:
        """
        Build the inverted index from a mapping of URL → HTML content.

        Steps
        -----
        1. Extract visible text from each page's HTML using BeautifulSoup.
        2. Tokenise and normalise (lowercase, strip punctuation).
        3. Record frequency and positional information for every token.
        4. Compute TF-IDF weights once the full corpus has been processed.

        Parameters
        ----------
        pages : dict[str, str]
            Mapping of ``{url: html_content}`` as returned by
            :class:`~crawler.Crawler`.

        Returns
        -------
        None
            The index is stored in ``self.index`` and ``self.is_built`` is
            set to ``True``.

        Raises
        ------
        ValueError
            If *pages* is empty.
        """
        if not pages:
            raise ValueError("Cannot build index from an empty pages dict.")

        logger.info("Building index over %d pages …", len(pages))
        self.index = {}

        # Pass 1 – raw frequency + positions
        for url, html in pages.items():
            tokens = self._tokenise(html)
            logger.debug("  %s → %d tokens", url, len(tokens))
            self._index_page(url, tokens)

        # Pass 2 – TF-IDF weights
        self._compute_tfidf(total_docs=len(pages))

        self.is_built = True
        unique_terms = len(self.index)
        logger.info(
            "Index built: %d unique terms across %d pages.", unique_terms, len(pages)
        )

    def save(self, filepath: str | Path) -> None:
        """
        Serialise the in-memory index to a JSON file on disk.

        Parameters
        ----------
        filepath : str | Path
            Destination file path.  Parent directories are created if they
            do not exist.

        Raises
        ------
        RuntimeError
            If called before :meth:`build` or :meth:`load`.
        """
        if not self.is_built:
            raise RuntimeError("Index has not been built yet. Call build() first.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.index, fh, ensure_ascii=False, indent=2)

        logger.info("Index saved to %s (%d terms).", path, len(self.index))

    def load(self, filepath: str | Path) -> None:
        """
        Load a previously saved index from a JSON file.

        Parameters
        ----------
        filepath : str | Path
            Path to the JSON index file created by :meth:`save`.

        Raises
        ------
        FileNotFoundError
            If *filepath* does not exist.
        json.JSONDecodeError
            If the file is not valid JSON.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        with path.open("r", encoding="utf-8") as fh:
            self.index = json.load(fh)

        self.is_built = True
        logger.info(
            "Index loaded from %s (%d terms).", path, len(self.index)
        )

    def find(self, query_terms: list[str]) -> list[dict]:
        """
        Return all pages that contain **every** term in *query_terms*,
        ranked by combined TF-IDF score (highest first).

        Parameters
        ----------
        query_terms : list[str]
            One or more search terms (case-insensitive).

        Returns
        -------
        list[dict]
            Sorted list of result dicts, each containing:

            ``url`` : str
                The page URL.
            ``score`` : float
                Sum of TF-IDF weights across all query terms for this page.
            ``term_stats`` : dict[str, dict]
                Per-term frequency and position data for this page.

            Empty list if no pages match or the index is not built.

        Examples
        --------
        >>> results = indexer.find(["good", "friends"])
        >>> for r in results:
        ...     print(r["url"], r["score"])
        """
        if not self.is_built:
            logger.warning("find() called before index was built/loaded.")
            return []

        if not query_terms:
            logger.warning("find() called with an empty query.")
            return []

        normalised: list[str] = [self._normalise_token(t) for t in query_terms]
        # Filter out tokens that became empty after normalisation
        normalised = [t for t in normalised if t]

        if not normalised:
            return []

        # --- Intersection: pages that contain ALL query terms ---
        candidate_sets: list[set[str]] = []
        for term in normalised:
            if term not in self.index:
                logger.debug("Term '%s' not in index — no results.", term)
                return []  # AND semantics: missing term → zero results
            candidate_sets.append(set(self.index[term].keys()))

        matching_urls: set[str] = candidate_sets[0].intersection(*candidate_sets[1:])

        if not matching_urls:
            return []

        # --- Score and collect results ---
        results: list[dict] = []
        for url in matching_urls:
            score = sum(
                self.index[term][url].get("tf_idf", 0.0) for term in normalised
            )
            term_stats = {
                term: {
                    "freq":      self.index[term][url]["freq"],
                    "positions": self.index[term][url]["positions"],
                    "tf_idf":    self.index[term][url].get("tf_idf", 0.0),
                }
                for term in normalised
            }
            results.append({"url": url, "score": score, "term_stats": term_stats})

        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    def get_postings(self, word: str) -> dict[str, dict] | None:
        """
        Return the postings list for a single *word*.

        Parameters
        ----------
        word : str
            The word to look up (case-insensitive).

        Returns
        -------
        dict[str, dict] | None
            Postings list ``{url: {freq, positions, tf_idf}}`` or ``None``
            if the word is not in the index.
        """
        if not self.is_built:
            logger.warning("get_postings() called before index was built/loaded.")
            return None

        normalised = self._normalise_token(word)
        return self.index.get(normalised)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tokenise(self, html: str) -> list[str]:
        """
        Extract visible text from *html* and return a list of normalised
        tokens in document order.

        Script and style elements are removed before extraction to avoid
        indexing JavaScript or CSS source.

        Parameters
        ----------
        html : str
            Raw HTML content.

        Returns
        -------
        list[str]
            Ordered list of lowercase, punctuation-stripped tokens.
            Empty strings are excluded.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Remove non-visible elements
        for tag in soup(["script", "style", "head", "meta", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        # Collapse whitespace
        text = _WHITESPACE_RE.sub(" ", text).strip()

        raw_tokens = text.split(" ")
        tokens: list[str] = []
        for tok in raw_tokens:
            normalised = self._normalise_token(tok)
            if normalised:
                tokens.append(normalised)

        return tokens

    @staticmethod
    def _normalise_token(token: str) -> str:
        """
        Lowercase and strip surrounding punctuation from *token*.

        Parameters
        ----------
        token : str
            A single raw token string.

        Returns
        -------
        str
            Normalised token, possibly empty if *token* contained only
            punctuation or whitespace.
        """
        return token.lower().strip(_STRIP_CHARS)

    def _index_page(self, url: str, tokens: list[str]) -> None:
        """
        Integrate *tokens* from a single page into ``self.index``.

        Parameters
        ----------
        url : str
            Source URL of the page.
        tokens : list[str]
            Ordered token list produced by :meth:`_tokenise`.
        """
        for position, token in enumerate(tokens):
            if token not in self.index:
                self.index[token] = {}

            if url not in self.index[token]:
                self.index[token][url] = {"freq": 0, "positions": []}

            self.index[token][url]["freq"] += 1
            self.index[token][url]["positions"].append(position)

    def _compute_tfidf(self, total_docs: int) -> None:
        """
        Add a ``tf_idf`` field to every postings entry.

        TF-IDF formula used
        ~~~~~~~~~~~~~~~~~~~
        * **TF** (log-normalised): ``1 + log10(raw_frequency)``
        * **IDF** (smooth): ``log10(total_docs / doc_frequency)``
        * **TF-IDF** = TF × IDF

        The log-normalised TF prevents very frequent terms from dominating
        purely because of their raw count.  Smooth IDF avoids division by
        zero (all words appear in at least 1 document by construction).

        Parameters
        ----------
        total_docs : int
            Total number of documents in the corpus.
        """
        for term, postings in self.index.items():
            doc_frequency = len(postings)  # number of docs containing this term
            idf = math.log10(total_docs / doc_frequency)

            for url, stats in postings.items():
                tf = 1 + math.log10(stats["freq"]) if stats["freq"] > 0 else 0.0
                stats["tf_idf"] = round(tf * idf, 6)