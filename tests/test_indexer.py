"""
tests/test_indexer.py
---------------------
Unit tests for the Indexer class.

Covers:
  - build() with normal, edge-case, and invalid inputs
  - Inverted index structure correctness (freq, positions, tf_idf)
  - save() / load() round-trip
  - get_postings() including missing words
  - find() single-term, multi-term AND, ranking, edge cases
  - Tokenisation: case normalisation, punctuation stripping, script/style removal

Run with:
    pytest tests/test_indexer.py -v
"""

import json
import math
import sys
import os
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from indexer import Indexer
from search import SearchEngine
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_indexer(pages: dict[str, str]) -> Indexer:
    """Build and return an Indexer from a pages dict."""
    idx = Indexer()
    idx.build(pages)
    return idx


SIMPLE_PAGES = {
    "http://example.com/p1": "<html><body><p>The quick brown fox</p></body></html>",
    "http://example.com/p2": "<html><body><p>The fox jumped high</p></body></html>",
}


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------

class TestBuild:

    def test_build_sets_is_built(self):
        idx = _make_indexer(SIMPLE_PAGES)
        assert idx.is_built is True

    def test_build_raises_on_empty_pages(self):
        idx = Indexer()
        with pytest.raises(ValueError, match="empty"):
            idx.build({})

    def test_build_indexes_known_word(self):
        idx = _make_indexer(SIMPLE_PAGES)
        assert "fox" in idx.index

    def test_build_case_insensitive(self):
        pages = {"http://x.com": "<html><body>Good good GOOD</body></html>"}
        idx = _make_indexer(pages)
        assert "good" in idx.index
        # All three occurrences under a single lowercase key
        assert "Good" not in idx.index
        assert "GOOD" not in idx.index

    def test_build_strips_punctuation(self):
        pages = {"http://x.com": "<html><body>Hello, world! It's fine.</body></html>"}
        idx = _make_indexer(pages)
        assert "hello" in idx.index
        assert "world" in idx.index
        # Punctuation-only token should not appear
        assert "," not in idx.index

    def test_build_excludes_script_content(self):
        pages = {
            "http://x.com": (
                "<html><head><script>var secretToken = 'abc';</script></head>"
                "<body><p>Visible text</p></body></html>"
            )
        }
        idx = _make_indexer(pages)
        assert "secrettoken" not in idx.index
        assert "visible" in idx.index

    def test_build_excludes_style_content(self):
        pages = {
            "http://x.com": (
                "<html><head><style>.myclass { color: red; }</style></head>"
                "<body><p>real content</p></body></html>"
            )
        }
        idx = _make_indexer(pages)
        assert "myclass" not in idx.index
        assert "real" in idx.index

    def test_build_empty_page_body(self):
        """A page with no visible text should not crash the indexer."""
        pages = {"http://x.com": "<html><body></body></html>"}
        idx = _make_indexer(pages)
        assert idx.is_built is True

    def test_build_single_word_page(self):
        pages = {"http://x.com": "<html><body>hello</body></html>"}
        idx = _make_indexer(pages)
        assert "hello" in idx.index

    def test_build_punctuation_only_token_excluded(self):
        pages = {"http://x.com": "<html><body>... --- !!!</body></html>"}
        idx = _make_indexer(pages)
        # After stripping punctuation all tokens become empty strings and are dropped
        for word in idx.index:
            assert word != ""


# ---------------------------------------------------------------------------
# Index structure: frequency & positions
# ---------------------------------------------------------------------------

class TestIndexStructure:

    def test_frequency_counted_correctly(self):
        pages = {
            "http://x.com": "<html><body>good good good bad</body></html>"
        }
        idx = _make_indexer(pages)
        assert idx.index["good"]["http://x.com"]["freq"] == 3
        assert idx.index["bad"]["http://x.com"]["freq"] == 1

    def test_positions_are_zero_based(self):
        pages = {"http://x.com": "<html><body>alpha beta gamma</body></html>"}
        idx = _make_indexer(pages)
        # Exact positions depend on tokenisation; just verify they are ints >= 0
        positions = idx.index["alpha"]["http://x.com"]["positions"]
        assert all(isinstance(p, int) and p >= 0 for p in positions)

    def test_positions_list_length_matches_frequency(self):
        pages = {"http://x.com": "<html><body>word word word</body></html>"}
        idx = _make_indexer(pages)
        stats = idx.index["word"]["http://x.com"]
        assert len(stats["positions"]) == stats["freq"]

    def test_positions_are_ordered(self):
        pages = {"http://x.com": "<html><body>a b a b a</body></html>"}
        idx = _make_indexer(pages)
        positions = idx.index["a"]["http://x.com"]["positions"]
        assert positions == sorted(positions)

    def test_multi_page_word_has_multiple_urls(self):
        idx = _make_indexer(SIMPLE_PAGES)
        # "fox" appears on both pages
        fox_postings = idx.index["fox"]
        assert "http://example.com/p1" in fox_postings
        assert "http://example.com/p2" in fox_postings

    def test_word_unique_to_one_page(self):
        idx = _make_indexer(SIMPLE_PAGES)
        # "brown" only appears on p1
        assert "brown" in idx.index
        assert "http://example.com/p1" in idx.index["brown"]
        assert "http://example.com/p2" not in idx.index["brown"]


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------

class TestTfIdf:

    def test_tfidf_field_present_after_build(self):
        idx = _make_indexer(SIMPLE_PAGES)
        for term, postings in idx.index.items():
            for url, stats in postings.items():
                assert "tf_idf" in stats, f"tf_idf missing for '{term}' on {url}"

    def test_tfidf_is_float(self):
        idx = _make_indexer(SIMPLE_PAGES)
        for postings in idx.index.values():
            for stats in postings.values():
                assert isinstance(stats["tf_idf"], float)

    def test_tfidf_zero_for_word_in_all_docs(self):
        """
        A word appearing in every document has IDF = log10(N/N) = 0,
        so its TF-IDF weight must be 0.0.
        """
        pages = {
            "http://x.com/p1": "<html><body>common rare1</body></html>",
            "http://x.com/p2": "<html><body>common rare2</body></html>",
        }
        idx = _make_indexer(pages)
        for url, stats in idx.index["common"].items():
            assert stats["tf_idf"] == pytest.approx(0.0, abs=1e-9)

    def test_tfidf_positive_for_rare_word(self):
        """A word in only one of multiple docs should have a positive TF-IDF."""
        pages = {
            "http://x.com/p1": "<html><body>rare word only here</body></html>",
            "http://x.com/p2": "<html><body>other content here</body></html>",
        }
        idx = _make_indexer(pages)
        # "rare" only appears in p1
        assert idx.index["rare"]["http://x.com/p1"]["tf_idf"] > 0.0

    def test_higher_frequency_gives_higher_tfidf(self):
        """All else equal, more occurrences should yield a higher TF-IDF."""
        pages = {
            "http://x.com/p1": "<html><body>word word word filler</body></html>",
            "http://x.com/p2": "<html><body>word filler</body></html>",
            # third doc so word doesn't appear in all docs (IDF > 0)
            "http://x.com/p3": "<html><body>completely different text</body></html>",
        }
        idx = _make_indexer(pages)
        score_p1 = idx.index["word"]["http://x.com/p1"]["tf_idf"]
        score_p2 = idx.index["word"]["http://x.com/p2"]["tf_idf"]
        assert score_p1 > score_p2


# ---------------------------------------------------------------------------
# save() & load()
# ---------------------------------------------------------------------------

class TestSaveLoad:

    def test_save_creates_file(self, tmp_path):
        idx = _make_indexer(SIMPLE_PAGES)
        out = tmp_path / "index.json"
        idx.save(str(out))
        assert out.exists()

    def test_save_creates_parent_directories(self, tmp_path):
        idx = _make_indexer(SIMPLE_PAGES)
        out = tmp_path / "nested" / "deep" / "index.json"
        idx.save(str(out))
        assert out.exists()

    def test_save_raises_if_not_built(self, tmp_path):
        idx = Indexer()
        with pytest.raises(RuntimeError, match="not been built"):
            idx.save(str(tmp_path / "index.json"))

    def test_load_restores_index(self, tmp_path):
        idx = _make_indexer(SIMPLE_PAGES)
        path = tmp_path / "index.json"
        idx.save(str(path))

        idx2 = Indexer()
        idx2.load(str(path))

        assert idx2.is_built is True
        assert idx2.index == idx.index

    def test_load_sets_is_built(self, tmp_path):
        idx = _make_indexer(SIMPLE_PAGES)
        path = tmp_path / "index.json"
        idx.save(str(path))

        idx2 = Indexer()
        assert idx2.is_built is False
        idx2.load(str(path))
        assert idx2.is_built is True

    def test_load_raises_file_not_found(self):
        idx = Indexer()
        with pytest.raises(FileNotFoundError):
            idx.load("/nonexistent/path/index.json")

    def test_save_load_round_trip_valid_json(self, tmp_path):
        idx = _make_indexer(SIMPLE_PAGES)
        path = tmp_path / "index.json"
        idx.save(str(path))

        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        assert isinstance(data, dict)
        assert len(data) == len(idx.index)


# ---------------------------------------------------------------------------
# get_postings()
# ---------------------------------------------------------------------------

class TestGetPostings:

    def test_returns_postings_for_known_word(self):
        idx = _make_indexer(SIMPLE_PAGES)
        postings = idx.get_postings("fox")
        assert postings is not None
        assert "http://example.com/p1" in postings

    def test_case_insensitive_lookup(self):
        idx = _make_indexer(SIMPLE_PAGES)
        assert idx.get_postings("FOX") == idx.get_postings("fox")

    def test_returns_none_for_missing_word(self):
        idx = _make_indexer(SIMPLE_PAGES)
        assert idx.get_postings("zzznonexistent") is None

    def test_returns_none_before_build(self):
        idx = Indexer()
        assert idx.get_postings("word") is None

    def test_returns_none_for_empty_string(self):
        idx = _make_indexer(SIMPLE_PAGES)
        assert idx.get_postings("") is None


# ---------------------------------------------------------------------------
# find()
# ---------------------------------------------------------------------------

class TestFind:

    def test_single_term_returns_matching_pages(self):
        idx = _make_indexer(SIMPLE_PAGES)
        results = idx.find(["fox"])
        urls = [r["url"] for r in results]
        assert "http://example.com/p1" in urls
        assert "http://example.com/p2" in urls

    def test_multi_term_and_semantics(self):
        """find() must return only pages containing ALL query terms."""
        idx = _make_indexer(SIMPLE_PAGES)
        # "brown" is only in p1; "jumped" only in p2 → AND = empty
        results = idx.find(["brown", "jumped"])
        assert results == []

    def test_multi_term_matching_page(self):
        pages = {
            "http://x.com/p1": "<html><body>good friends together</body></html>",
            "http://x.com/p2": "<html><body>good times alone</body></html>",
        }
        idx = _make_indexer(pages)
        results = idx.find(["good", "friends"])
        assert len(results) == 1
        assert results[0]["url"] == "http://x.com/p1"

    def test_results_sorted_by_tfidf_descending(self):
        pages = {
            "http://x.com/p1": "<html><body>rare rare rare filler</body></html>",
            "http://x.com/p2": "<html><body>rare filler filler</body></html>",
            "http://x.com/p3": "<html><body>totally unrelated content</body></html>",
        }
        idx = _make_indexer(pages)
        results = idx.find(["rare"])
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_result_contains_expected_keys(self):
        idx = _make_indexer(SIMPLE_PAGES)
        results = idx.find(["fox"])
        assert len(results) > 0
        for r in results:
            assert "url" in r
            assert "score" in r
            assert "term_stats" in r

    def test_term_stats_contain_freq_positions_tfidf(self):
        idx = _make_indexer(SIMPLE_PAGES)
        results = idx.find(["fox"])
        for r in results:
            stats = r["term_stats"]["fox"]
            assert "freq" in stats
            assert "positions" in stats
            assert "tf_idf" in stats

    def test_missing_term_returns_empty_list(self):
        idx = _make_indexer(SIMPLE_PAGES)
        results = idx.find(["zzznonexistentword"])
        assert results == []

    def test_empty_query_returns_empty_list(self):
        idx = _make_indexer(SIMPLE_PAGES)
        results = idx.find([])
        assert results == []

    def test_whitespace_only_terms_ignored(self):
        idx = _make_indexer(SIMPLE_PAGES)
        results = idx.find(["   ", "\t"])
        assert results == []

    def test_find_before_build_returns_empty(self):
        idx = Indexer()
        results = idx.find(["word"])
        assert results == []

    def test_case_insensitive_find(self):
        idx = _make_indexer(SIMPLE_PAGES)
        lower = idx.find(["fox"])
        upper = idx.find(["FOX"])
        mixed = idx.find(["Fox"])
        assert [r["url"] for r in lower] == [r["url"] for r in upper]
        assert [r["url"] for r in lower] == [r["url"] for r in mixed]


# ---------------------------------------------------------------------------
# find_phrase() — Indexer
# ---------------------------------------------------------------------------

class TestFindPhrase:
    """Tests for positional phrase search."""

    # Pages where "good friends" appears as an exact adjacent pair
    PHRASE_PAGES = {
        "http://x.com/match":
            "<html><body>they are good friends forever</body></html>",
        "http://x.com/no-match":
            "<html><body>friends are good to have around</body></html>",
        "http://x.com/single":
            "<html><body>good times with everyone</body></html>",
        "http://x.com/reversed":
            "<html><body>friends are never good to lose</body></html>",
    }

    def _idx(self) -> Indexer:
        idx = Indexer()
        idx.build(self.PHRASE_PAGES)
        return idx

    def test_exact_phrase_returns_correct_page(self):
        results = self._idx().find_phrase(["good", "friends"])
        assert len(results) == 1
        assert results[0]["url"] == "http://x.com/match"

    def test_reversed_phrase_does_not_match(self):
        """'friends good' must not match 'good friends'."""
        results = self._idx().find_phrase(["friends", "good"])
        # 'reversed' page has "friends are never good" — not adjacent
        assert all(r["url"] != "http://x.com/match" for r in results)

    def test_words_present_but_not_adjacent_excluded(self):
        """'no-match' has both words but in wrong order / not adjacent."""
        results = self._idx().find_phrase(["good", "friends"])
        urls = [r["url"] for r in results]
        assert "http://x.com/no-match" not in urls

    def test_single_term_phrase_behaves_like_find(self):
        """A one-word phrase is just a regular lookup."""
        idx = self._idx()
        phrase = idx.find_phrase(["good"])
        find   = idx.find(["good"])
        assert {r["url"] for r in phrase} == {r["url"] for r in find}

    def test_three_term_phrase_correct(self):
        pages = {
            "http://x.com/a": "<html><body>the quick brown fox jumps</body></html>",
            "http://x.com/b": "<html><body>quick fox brown is not here</body></html>",
        }
        idx = Indexer()
        idx.build(pages)
        # "quick brown fox" appears in /a but not /b
        results = idx.find_phrase(["quick", "brown", "fox"])
        urls = [r["url"] for r in results]
        assert "http://x.com/a" in urls
        assert "http://x.com/b" not in urls

    def test_empty_phrase_returns_empty(self):
        assert self._idx().find_phrase([]) == []

    def test_missing_term_returns_empty(self):
        assert self._idx().find_phrase(["good", "zzznothere"]) == []

    def test_phrase_not_in_index_returns_empty(self):
        assert self._idx().find_phrase(["zzzaaa", "zzzbbb"]) == []

    def test_results_sorted_by_tfidf(self):
        pages = {
            "http://x.com/a": "<html><body>good friends good friends</body></html>",
            "http://x.com/b": "<html><body>good friends here</body></html>",
            "http://x.com/c": "<html><body>unrelated content here</body></html>",
        }
        idx = Indexer()
        idx.build(pages)
        results = idx.find_phrase(["good", "friends"])
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_result_keys_present(self):
        results = self._idx().find_phrase(["good", "friends"])
        assert len(results) > 0
        for r in results:
            assert "url" in r
            assert "score" in r
            assert "term_stats" in r

    def test_phrase_before_build_returns_empty(self):
        idx = Indexer()
        assert idx.find_phrase(["good", "friends"]) == []

    def test_case_insensitive_phrase(self):
        idx = self._idx()
        lower = idx.find_phrase(["good", "friends"])
        upper = idx.find_phrase(["GOOD", "FRIENDS"])
        assert {r["url"] for r in lower} == {r["url"] for r in upper}


# ---------------------------------------------------------------------------
# find_phrase() — SearchEngine
# ---------------------------------------------------------------------------

class TestSearchEngineFindPhrase:
    """Tests for the SearchEngine.find_phrase() presentation wrapper."""

    PAGES = {
        "http://x.com/p1": "<html><body>good friends are valuable</body></html>",
        "http://x.com/p2": "<html><body>friends are good people always</body></html>",
        "http://x.com/p3": "<html><body>something entirely different here</body></html>",
    }

    def _engine(self) -> SearchEngine:
        idx = Indexer()
        idx.build(self.PAGES)
        return SearchEngine(idx)

    def test_raises_if_not_built(self):
        engine = SearchEngine(Indexer())
        with pytest.raises(RuntimeError, match="not available"):
            engine.find_phrase(["good", "friends"])

    def test_returns_correct_page(self):
        results = self._engine().find_phrase(["good", "friends"])
        assert len(results) == 1
        assert results[0]["url"] == "http://x.com/p1"

    def test_empty_terms_returns_empty(self, capsys):
        results = self._engine().find_phrase([])
        assert results == []
        assert "No phrase terms" in capsys.readouterr().out

    def test_no_match_prints_message(self, capsys):
        self._engine().find_phrase(["zzz", "yyy"])
        out = capsys.readouterr().out
        assert "No pages found" in out

    def test_output_shows_url(self, capsys):
        self._engine().find_phrase(["good", "friends"])
        assert "http://x.com/p1" in capsys.readouterr().out

    def test_delegates_to_indexer(self):
        idx = MagicMock(spec=Indexer)
        idx.is_built = True
        idx.find_phrase.return_value = []
        engine = SearchEngine(idx)
        engine.find_phrase(["good", "friends"])
        idx.find_phrase.assert_called_once_with(["good", "friends"])