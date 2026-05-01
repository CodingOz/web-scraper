"""
tests/test_search.py
--------------------
Unit tests for the SearchEngine class.

SearchEngine is a thin presentation layer over Indexer, so these tests
focus on:
  - Correct delegation to Indexer.find() and Indexer.get_postings()
  - Graceful handling of all edge-case inputs
  - RuntimeError raised when index not ready
  - Correct stdout output (captured via capsys)
  - print_word() return values

Run with:
    pytest tests/test_search.py -v
"""

import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from indexer import Indexer
from search import SearchEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PAGES = {
    "http://example.com/p1": "<html><body>good life good times</body></html>",
    "http://example.com/p2": "<html><body>good friends are rare</body></html>",
    "http://example.com/p3": "<html><body>bad times bad life</body></html>",
}


def _built_engine() -> SearchEngine:
    """Return a SearchEngine with a fully built index."""
    idx = Indexer()
    idx.build(PAGES)
    return SearchEngine(idx)


def _unbuilt_engine() -> SearchEngine:
    """Return a SearchEngine whose index has NOT been built."""
    return SearchEngine(Indexer())


# ---------------------------------------------------------------------------
# _require_index guard
# ---------------------------------------------------------------------------

class TestRequireIndex:

    def test_print_word_raises_if_not_built(self):
        engine = _unbuilt_engine()
        with pytest.raises(RuntimeError, match="not available"):
            engine.print_word("good")

    def test_find_raises_if_not_built(self):
        engine = _unbuilt_engine()
        with pytest.raises(RuntimeError, match="not available"):
            engine.find(["good"])


# ---------------------------------------------------------------------------
# print_word()
# ---------------------------------------------------------------------------

class TestPrintWord:

    def test_returns_none_for_empty_string(self, capsys):
        engine = _built_engine()
        result = engine.print_word("")
        out = capsys.readouterr().out
        assert result is None
        assert "No word provided" in out

    def test_returns_none_for_whitespace(self, capsys):
        engine = _built_engine()
        result = engine.print_word("   ")
        assert result is None

    def test_returns_none_for_missing_word(self, capsys):
        engine = _built_engine()
        result = engine.print_word("zzznonexistent")
        out = capsys.readouterr().out
        assert result is None
        assert "not found" in out

    def test_returns_postings_dict_for_known_word(self):
        engine = _built_engine()
        result = engine.print_word("good")
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_output_contains_word(self, capsys):
        engine = _built_engine()
        engine.print_word("good")
        out = capsys.readouterr().out
        assert "good" in out

    def test_output_contains_url(self, capsys):
        engine = _built_engine()
        engine.print_word("good")
        out = capsys.readouterr().out
        assert "http://example.com" in out

    def test_output_contains_freq(self, capsys):
        engine = _built_engine()
        engine.print_word("good")
        out = capsys.readouterr().out
        assert "freq" in out or any(char.isdigit() for char in out)

    def test_case_insensitive_lookup(self):
        engine = _built_engine()
        lower = engine.print_word("good")
        upper = engine.print_word("GOOD")
        assert lower is not None
        assert upper is not None
        assert set(lower.keys()) == set(upper.keys())

    def test_results_sorted_by_tfidf_descending(self):
        engine = _built_engine()
        postings = engine.print_word("good")
        scores = [v["tf_idf"] for v in postings.values()]
        assert scores == sorted(scores, reverse=True)

    def test_single_word_page_displays_position(self, capsys):
        engine = _built_engine()
        engine.print_word("rare")
        out = capsys.readouterr().out
        assert "positions" in out


# ---------------------------------------------------------------------------
# find()
# ---------------------------------------------------------------------------

class TestFind:

    def test_returns_empty_list_for_empty_query(self, capsys):
        engine = _built_engine()
        result = engine.find([])
        assert result == []
        assert "No search terms" in capsys.readouterr().out

    def test_returns_empty_list_for_whitespace_terms(self, capsys):
        engine = _built_engine()
        result = engine.find(["  ", "\t"])
        assert result == []

    def test_returns_empty_list_for_missing_word(self, capsys):
        engine = _built_engine()
        result = engine.find(["zzznonexistent"])
        assert result == []
        assert "No pages found" in capsys.readouterr().out

    def test_single_term_finds_matching_pages(self):
        engine = _built_engine()
        results = engine.find(["good"])
        urls = [r["url"] for r in results]
        assert "http://example.com/p1" in urls
        assert "http://example.com/p2" in urls

    def test_multi_term_and_semantics(self):
        """find(['good', 'friends']) must only return pages with both words."""
        engine = _built_engine()
        results = engine.find(["good", "friends"])
        assert len(results) == 1
        assert results[0]["url"] == "http://example.com/p2"

    def test_multi_term_no_match_returns_empty(self):
        engine = _built_engine()
        # "rare" only on p2; "bad" only on p3 → AND = empty
        results = engine.find(["rare", "bad"])
        assert results == []

    def test_results_ranked_by_score_desc(self):
        engine = _built_engine()
        results = engine.find(["good"])
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_output_shows_rank_and_url(self, capsys):
        engine = _built_engine()
        engine.find(["good"])
        out = capsys.readouterr().out
        assert "http://example.com" in out
        assert "1" in out  # rank 1

    def test_output_shows_no_results_message(self, capsys):
        engine = _built_engine()
        engine.find(["zzznonexistent"])
        out = capsys.readouterr().out
        assert "No pages found" in out

    def test_case_insensitive_find(self):
        engine = _built_engine()
        lower = engine.find(["good"])
        upper = engine.find(["GOOD"])
        assert [r["url"] for r in lower] == [r["url"] for r in upper]

    def test_result_dict_has_required_keys(self):
        engine = _built_engine()
        results = engine.find(["good"])
        for r in results:
            assert "url" in r
            assert "score" in r
            assert "term_stats" in r

    def test_term_stats_populated_for_each_query_term(self):
        engine = _built_engine()
        results = engine.find(["good", "friends"])
        assert len(results) == 1
        stats = results[0]["term_stats"]
        assert "good" in stats
        assert "friends" in stats

    def test_score_is_float(self):
        engine = _built_engine()
        results = engine.find(["good"])
        for r in results:
            assert isinstance(r["score"], float)

    def test_find_delegates_to_indexer(self):
        """SearchEngine.find() should call Indexer.find() exactly once."""
        idx = MagicMock(spec=Indexer)
        idx.is_built = True
        idx.find.return_value = []

        engine = SearchEngine(idx)
        engine.find(["hello"])
        idx.find.assert_called_once_with(["hello"])

    def test_print_word_delegates_to_indexer(self):
        """SearchEngine.print_word() should call Indexer.get_postings() once."""
        idx = MagicMock(spec=Indexer)
        idx.is_built = True
        idx.get_postings.return_value = None

        engine = SearchEngine(idx)
        engine.print_word("hello")
        idx.get_postings.assert_called_once_with("hello")