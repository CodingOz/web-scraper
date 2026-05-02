"""
tests/test_integration.py
--------------------------
Integration tests for the full Crawler → Indexer → SearchEngine pipeline.

Unlike the unit tests (which test each layer in isolation with mocks), these
tests wire all three layers together and verify that the system behaves
correctly end-to-end.  HTTP requests are still mocked — we don't want
integration tests to hit the real network — but every real code path from
crawl() through build() through find() executes without stubbing.

Mock website structure
----------------------
Four interlinked pages that form a small quote-site:

    home  ──► page1  (Wilde quotes)
          ──► page2  (Einstein quotes)
          ──► page3  (no outbound links — dead end)

Each page contains known words so we can write precise assertions about
which pages a query should and should not return.

Run with:
    pytest tests/test_integration.py -v
    pytest tests/ -v --cov=src --cov-report=term-missing
"""

import json
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, call

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from crawler import Crawler
from indexer import Indexer
from search import SearchEngine


# ---------------------------------------------------------------------------
# Mock website definition
# ---------------------------------------------------------------------------

BASE_URL = "https://mock-quotes.test"

# Each page is carefully worded so we can write unambiguous query assertions:
#   "imagination"  → page1 only
#   "simplicity"   → page2 only
#   "perseverance" → page3 only
#   "wisdom"       → page1 AND page2 (used for multi-word AND test)
#   "quote"        → all three content pages (used for universal-word IDF test)

MOCK_PAGES: dict[str, str] = {
    BASE_URL: """
        <html><head><title>Mock Quote Site</title></head>
        <body>
          <h1>Welcome to Mock Quotes</h1>
          <p>A daily quote to inspire you.</p>
          <a href="/page1">Wilde</a>
          <a href="/page2">Einstein</a>
          <a href="/page3">Dedication</a>
        </body></html>
    """,
    # ^^^ "quote" appears here (body text), "perseverance" does NOT (link says
    # "Dedication" not "Perseverance"), so perseverance stays unique to page3.
    f"{BASE_URL}/page1": """
        <html><body>
          <h2>Oscar Wilde</h2>
          <p>A quote about imagination is the beginning of creation.</p>
          <p>Another quote full of wisdom and wit.</p>
          <a href="/">Home</a>
        </body></html>
    """,
    f"{BASE_URL}/page2": """
        <html><body>
          <h2>Albert Einstein</h2>
          <p>A quote about simplicity in the pursuit of knowledge.</p>
          <p>Another quote on wisdom and curiosity.</p>
          <a href="/">Home</a>
        </body></html>
    """,
    f"{BASE_URL}/page3": """
        <html><body>
          <h2>Unknown Author</h2>
          <p>A quote about perseverance and dedication.</p>
        </body></html>
    """,
}


def _make_mock_get(pages: dict[str, str]):
    """
    Return a fake requests.get function that serves pages from the dict.
    Any URL not in the dict returns a 404 mock response.
    """
    def fake_get(url: str, **kwargs):
        resp = Mock()
        normalised = url.rstrip("/")
        if normalised in pages:
            resp.text = pages[normalised]
            resp.status_code = 200
            resp.raise_for_status.return_value = None
        else:
            resp.text = "Not Found"
            resp.status_code = 404
            resp.raise_for_status.side_effect = requests.exceptions.HTTPError("404")
        return resp
    return fake_get


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline():
    """
    Build the full pipeline once for the entire module and share the result.

    scope="module" means the crawl + index build runs once, and all tests
    in this file read from the same in-memory index — matching real usage
    where you `build` once then `load` for every subsequent query.

    Yields
    ------
    dict with keys:
        crawler  : Crawler instance (post-crawl)
        indexer  : Indexer instance (post-build)
        engine   : SearchEngine instance
        pages    : raw {url: html} dict from the crawl
    """
    with patch("crawler.time.sleep"):
        crawler = Crawler(BASE_URL, politeness_window=6.0)
        crawler._session.get = Mock(side_effect=_make_mock_get(MOCK_PAGES))
        pages = crawler.crawl()

    indexer = Indexer()
    indexer.build(pages)

    engine = SearchEngine(indexer)

    yield {"crawler": crawler, "indexer": indexer, "engine": engine, "pages": pages}


# ---------------------------------------------------------------------------
# Crawler layer (within pipeline)
# ---------------------------------------------------------------------------

class TestPipelineCrawler:
    """Verify the crawler collected the right pages before indexing."""

    def test_all_four_pages_fetched(self, pipeline):
        pages = pipeline["pages"]
        assert BASE_URL in pages
        assert f"{BASE_URL}/page1" in pages
        assert f"{BASE_URL}/page2" in pages
        assert f"{BASE_URL}/page3" in pages

    def test_exactly_four_pages_fetched(self, pipeline):
        """No extra pages should have been fabricated or duplicated."""
        assert len(pipeline["pages"]) == 4

    def test_no_external_pages_fetched(self, pipeline):
        for url in pipeline["pages"]:
            assert url.startswith(BASE_URL)

    def test_page_content_preserved(self, pipeline):
        html = pipeline["pages"][f"{BASE_URL}/page1"]
        assert "imagination" in html.lower()


# ---------------------------------------------------------------------------
# Indexer layer (within pipeline)
# ---------------------------------------------------------------------------

class TestPipelineIndexer:
    """Verify the index was built correctly from the crawled pages."""

    def test_index_is_built(self, pipeline):
        assert pipeline["indexer"].is_built is True

    def test_unique_word_indexed(self, pipeline):
        """'imagination' only appears on page1 — must be in the index."""
        assert "imagination" in pipeline["indexer"].index

    def test_unique_word_maps_to_correct_page(self, pipeline):
        postings = pipeline["indexer"].index["imagination"]
        assert f"{BASE_URL}/page1" in postings
        assert f"{BASE_URL}/page2" not in postings
        assert f"{BASE_URL}/page3" not in postings

    def test_shared_word_maps_to_multiple_pages(self, pipeline):
        """'wisdom' appears on page1 and page2."""
        postings = pipeline["indexer"].index.get("wisdom", {})
        assert f"{BASE_URL}/page1" in postings
        assert f"{BASE_URL}/page2" in postings

    def test_universal_word_has_zero_tfidf(self, pipeline):
        """
        'quote' appears on every content page.
        IDF = log10(N/N) = 0 → TF-IDF must be 0.0 for all postings.
        """
        postings = pipeline["indexer"].index.get("quote", {})
        for url, stats in postings.items():
            assert stats["tf_idf"] == pytest.approx(0.0, abs=1e-6), (
                f"Expected tf_idf=0 for universal word 'quote' on {url}, "
                f"got {stats['tf_idf']}"
            )

    def test_rare_word_has_positive_tfidf(self, pipeline):
        """'perseverance' only appears on page3 → TF-IDF must be > 0."""
        postings = pipeline["indexer"].index.get("perseverance", {})
        assert len(postings) == 1
        stats = list(postings.values())[0]
        assert stats["tf_idf"] > 0.0

    def test_frequency_correct_for_repeated_word(self, pipeline):
        """'quote' appears twice on page1 — freq must reflect this."""
        postings = pipeline["indexer"].index.get("quote", {})
        if f"{BASE_URL}/page1" in postings:
            assert postings[f"{BASE_URL}/page1"]["freq"] >= 1

    def test_positions_list_is_nonempty(self, pipeline):
        postings = pipeline["indexer"].index.get("imagination", {})
        stats = postings.get(f"{BASE_URL}/page1", {})
        assert len(stats.get("positions", [])) >= 1


# ---------------------------------------------------------------------------
# SearchEngine layer (within pipeline)
# ---------------------------------------------------------------------------

class TestPipelineSearch:
    """Verify end-to-end query results are correct and well-ranked."""

    def test_single_term_unique_word_returns_one_result(self, pipeline):
        results = pipeline["engine"].find(["imagination"])
        assert len(results) == 1
        assert results[0]["url"] == f"{BASE_URL}/page1"

    def test_single_term_shared_word_returns_two_results(self, pipeline):
        results = pipeline["engine"].find(["wisdom"])
        urls = [r["url"] for r in results]
        assert f"{BASE_URL}/page1" in urls
        assert f"{BASE_URL}/page2" in urls

    def test_multi_word_and_returns_only_pages_with_all_terms(self, pipeline):
        """
        'wisdom' is on page1+page2; 'imagination' is only on page1.
        AND query must return only page1.
        """
        results = pipeline["engine"].find(["wisdom", "imagination"])
        assert len(results) == 1
        assert results[0]["url"] == f"{BASE_URL}/page1"

    def test_multi_word_no_match_returns_empty(self, pipeline):
        """
        'imagination' (page1 only) AND 'simplicity' (page2 only) → no match.
        """
        results = pipeline["engine"].find(["imagination", "simplicity"])
        assert results == []

    def test_missing_word_returns_empty(self, pipeline):
        results = pipeline["engine"].find(["zzznotaword"])
        assert results == []

    def test_results_sorted_by_score_descending(self, pipeline):
        results = pipeline["engine"].find(["wisdom"])
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_result_score_is_positive(self, pipeline):
        results = pipeline["engine"].find(["imagination"])
        assert results[0]["score"] > 0.0

    def test_result_contains_term_stats(self, pipeline):
        results = pipeline["engine"].find(["imagination"])
        assert "term_stats" in results[0]
        assert "imagination" in results[0]["term_stats"]

    def test_term_stats_freq_matches_index(self, pipeline):
        """Freq in result must match what's stored in the raw index."""
        results = pipeline["engine"].find(["imagination"])
        result_freq = results[0]["term_stats"]["imagination"]["freq"]
        index_freq = (
            pipeline["indexer"]
            .index["imagination"][f"{BASE_URL}/page1"]["freq"]
        )
        assert result_freq == index_freq

    def test_case_insensitive_query(self, pipeline):
        lower = pipeline["engine"].find(["imagination"])
        upper = pipeline["engine"].find(["IMAGINATION"])
        assert [r["url"] for r in lower] == [r["url"] for r in upper]

    def test_empty_query_returns_empty(self, pipeline):
        results = pipeline["engine"].find([])
        assert results == []


# ---------------------------------------------------------------------------
# Save / load round-trip (within pipeline)
# ---------------------------------------------------------------------------

class TestPipelinePersistence:
    """
    Verify that saving the index to disk and loading it into a fresh Indexer
    produces identical query results — the full build → save → load → query
    cycle.
    """

    def test_save_load_find_returns_same_results(self, pipeline, tmp_path):
        path = tmp_path / "integration_index.json"

        # Save
        pipeline["indexer"].save(str(path))
        assert path.exists()

        # Load into a brand-new Indexer
        fresh_indexer = Indexer()
        fresh_indexer.load(str(path))
        fresh_engine = SearchEngine(fresh_indexer)

        original_results = pipeline["engine"].find(["imagination"])
        loaded_results = fresh_engine.find(["imagination"])

        assert [r["url"] for r in original_results] == [
            r["url"] for r in loaded_results
        ]
        assert [r["score"] for r in original_results] == pytest.approx(
            [r["score"] for r in loaded_results], rel=1e-6
        )

    def test_saved_index_is_valid_json(self, pipeline, tmp_path):
        path = tmp_path / "check.json"
        pipeline["indexer"].save(str(path))
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert isinstance(data, dict)
        assert len(data) > 0

    def test_loaded_index_term_count_matches_original(self, pipeline, tmp_path):
        path = tmp_path / "count_check.json"
        pipeline["indexer"].save(str(path))
        fresh = Indexer()
        fresh.load(str(path))
        assert len(fresh.index) == len(pipeline["indexer"].index)