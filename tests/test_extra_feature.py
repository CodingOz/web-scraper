"""
tests/test_extra_features.py
--------------------------
Focused unit tests for features added after the initial implementation:

  TestPhraseAdjacencyUnit   – phrase search edge cases not covered elsewhere:
                              scattered terms, multiple occurrences where only
                              one pair is adjacent, and explicit ordering checks.

  TestStemming              – stem=True behaviour: inflected forms match,
                              numeric/mixed tokens are not stemmed, stem=False
                              keeps the default exact-match behaviour, and
                              stemmed queries work correctly after save/load.

  TestRobotsCanFetch        – _fetch gating via a directly-mocked can_fetch,
                              verifying the exact call the crawler makes and
                              that no HTTP request is issued for disallowed URLs.

  TestMaxPages              – crawl stops at the configured limit regardless of
                              queue depth, and crawls all pages when the limit
                              exceeds the available pages.

Run with:
    pytest tests/test_new_features.py -v
"""

from __future__ import annotations

import sys
import os
import tempfile
from unittest.mock import Mock, patch, call

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from crawler import Crawler
from indexer import Indexer


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

def _mock_response(text: str, status_code: int = 200) -> Mock:
    resp = Mock()
    resp.text = text
    resp.status_code = status_code
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"{status_code}"
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _make_crawler(max_pages: int | None = None) -> Crawler:
    """Return a Crawler with robots primed to allow everything and read() stubbed."""
    c = Crawler("https://quotes.toscrape.com", politeness_window=6.0,
                max_pages=max_pages)
    c._robots.parse(["User-agent: *", "Allow: /"])
    c._robots.read = Mock()
    return c


# ---------------------------------------------------------------------------
# 1. Phrase adjacency — unit tests
# ---------------------------------------------------------------------------

class TestPhraseAdjacencyUnit:
    """
    Targeted adjacency and ordering tests that go beyond the basics in
    test_indexer.py::TestFindPhrase.

    Focus areas
    -----------
    * Terms in the correct order but separated by intervening words ("scattered")
      must NOT match — only position p and p+1 qualify, not p and p+n.
    * A page where the first term appears multiple times must match only when at
      least one occurrence is immediately followed by the second term.
    * Order is strictly positional: term[0] at p, term[1] at p+1.  A page
      where term[1] precedes term[0] must never match.
    """

    def _build(self, pages: dict[str, str], stem: bool = False) -> Indexer:
        idx = Indexer(stem=stem)
        idx.build(pages)
        return idx

    # -- Scattered terms (correct order, not adjacent) ----------------------

    def test_scattered_correct_order_does_not_match(self):
        """
        'good ... friends' with several words between them must not match
        the phrase ["good", "friends"].  The gap makes them non-adjacent.
        """
        pages = {
            "http://x.com/scattered":
                "<html><body>good times for all true friends here</body></html>",
        }
        idx = self._build(pages)
        results = idx.find_phrase(["good", "friends"])
        assert results == [], (
            "Scattered terms in correct order should not produce a phrase match"
        )

    def test_scattered_one_word_gap_does_not_match(self):
        """Even a single intervening word breaks adjacency."""
        pages = {
            "http://x.com/p": "<html><body>good old friends</body></html>",
        }
        idx = self._build(pages)
        # "good old friends": good=0, old=1, friends=2
        # phrase ["good","friends"] needs good at p, friends at p+1=1 → position 1 is "old"
        assert idx.find_phrase(["good", "friends"]) == []

    def test_adjacent_pair_matches_despite_other_occurrences(self):
        """
        When a term appears multiple times on a page, matching should succeed
        if ANY occurrence forms the adjacent pair — even if other occurrences
        are scattered.
        """
        pages = {
            "http://x.com/p":
                # "good" at positions 0 and 4; "friends" at position 5
                # pair (good@4, friends@5) IS adjacent → must match
                "<html><body>good times with real good friends here</body></html>",
        }
        idx = self._build(pages)
        results = idx.find_phrase(["good", "friends"])
        assert len(results) == 1
        assert results[0]["url"] == "http://x.com/p"

    def test_no_occurrence_adjacent_even_with_repeated_terms(self):
        """
        Multiple occurrences of both terms, but no pair is adjacent.
        """
        pages = {
            "http://x.com/p":
                # good@0, friends@2, good@4, friends@6 — always separated by 1 word
                "<html><body>good old friends are good real friends</body></html>",
        }
        idx = self._build(pages)
        # "good friends" needs good at p and friends at p+1; let's check positions:
        # good=0,4  friends=2,6
        # 0+1=1 not in {2,6}; 4+1=5 not in {2,6} → no match
        assert idx.find_phrase(["good", "friends"]) == []

    # -- Strict ordering ----------------------------------------------------

    def test_reversed_order_single_page(self):
        """
        A page where ONLY 'friends good' appears (reverse of query) must not match
        the phrase ["good", "friends"].
        """
        pages = {
            "http://x.com/rev":
                "<html><body>the friends good times forever</body></html>",
        }
        idx = self._build(pages)
        results = idx.find_phrase(["good", "friends"])
        # "friends good" is adjacent but in WRONG order for the query
        assert results == []

    def test_reversed_order_is_different_phrase(self):
        """
        ["friends", "good"] matches the page where "friends good" is adjacent,
        but ["good", "friends"] does not — and vice versa.
        """
        pages = {
            "http://x.com/rev":
                "<html><body>the friends good times</body></html>",
            "http://x.com/fwd":
                "<html><body>the good friends always</body></html>",
        }
        idx = self._build(pages)

        fwd_results = {r["url"] for r in idx.find_phrase(["good", "friends"])}
        rev_results = {r["url"] for r in idx.find_phrase(["friends", "good"])}

        assert "http://x.com/fwd" in fwd_results
        assert "http://x.com/rev" not in fwd_results

        assert "http://x.com/rev" in rev_results
        assert "http://x.com/fwd" not in rev_results

    def test_phrase_at_start_of_text(self):
        """Adjacent phrase at position 0,1 must be found."""
        pages = {"http://x.com/p": "<html><body>good friends are great</body></html>"}
        idx = self._build(pages)
        assert len(idx.find_phrase(["good", "friends"])) == 1

    def test_phrase_at_end_of_text(self):
        """Adjacent phrase at the last two positions must be found."""
        pages = {"http://x.com/p": "<html><body>they are good friends</body></html>"}
        idx = self._build(pages)
        assert len(idx.find_phrase(["good", "friends"])) == 1

    def test_three_term_phrase_with_gap_does_not_match(self):
        """
        ["quick", "brown", "fox"] must not match "quick slow brown fox" because
        "quick" and "brown" are not adjacent (position 0 and 2, not 0 and 1).
        """
        pages = {
            "http://x.com/p":
                "<html><body>quick slow brown fox jumps</body></html>",
        }
        idx = self._build(pages)
        assert idx.find_phrase(["quick", "brown", "fox"]) == []

    def test_three_term_phrase_exact_match(self):
        """["quick", "brown", "fox"] must match "the quick brown fox"."""
        pages = {
            "http://x.com/match": "<html><body>the quick brown fox jumps</body></html>",
            "http://x.com/gap":   "<html><body>quick slow brown fox jumps</body></html>",
        }
        idx = self._build(pages)
        results = idx.find_phrase(["quick", "brown", "fox"])
        urls = [r["url"] for r in results]
        assert "http://x.com/match" in urls
        assert "http://x.com/gap"   not in urls


# ---------------------------------------------------------------------------
# 2. Stemming
# ---------------------------------------------------------------------------

class TestStemming:
    """
    Unit tests for Indexer(stem=True).

    Each test isolates one aspect of stemming behaviour so a single failure
    points to a single cause.
    """

    # -- Inflected forms match ----------------------------------------------

    def test_stemmed_query_finds_inflected_form(self):
        """
        With stem=True, querying "running" must find a page that only contains
        "runs" because both stem to "run".
        """
        pages = {"http://x.com/p": "<html><body>she runs every morning</body></html>"}
        idx = Indexer(stem=True)
        idx.build(pages)
        results = idx.find(["running"])
        assert len(results) == 1
        assert results[0]["url"] == "http://x.com/p"

    def test_stemmed_query_finds_gerund_form(self):
        """
        'runs', 'running', and 'run' all stem to 'run' under Porter.
        A query for 'running' must find a page containing 'run'.
        Note: 'runner' stems to 'runner' and 'ran' stems to 'ran' —
        Porter does not normalise irregular past tense or agent nouns.
        """
        pages = {
            "http://x.com/run":     "<html><body>i run every day</body></html>",
            "http://x.com/imagine": "<html><body>imagine and imagining new worlds</body></html>",
        }
        idx = Indexer(stem=True)
        idx.build(pages)
        # "running" → "run"; page /run has "run" → "run" → should match
        results = idx.find(["running"])
        urls = {r["url"] for r in results}
        assert "http://x.com/run" in urls
        # "imagine"/"imagining" stem to "imagin", not "run" — must not appear
        assert "http://x.com/imagine" not in urls

    def test_stem_true_running_finds_runs(self):
        """Explicit: 'running' query finds page containing only 'runs'."""
        pages = {
            "http://x.com/p1": "<html><body>she runs daily</body></html>",
            "http://x.com/p2": "<html><body>completely unrelated text</body></html>",
        }
        idx = Indexer(stem=True)
        idx.build(pages)
        results = idx.find(["running"])
        urls = {r["url"] for r in results}
        assert "http://x.com/p1" in urls
        assert "http://x.com/p2" not in urls

    def test_stem_true_friends_finds_friend(self):
        """'friends' and 'friend' both stem to 'friend'."""
        pages = {
            "http://x.com/p": "<html><body>a true friend is rare</body></html>",
        }
        idx = Indexer(stem=True)
        idx.build(pages)
        # "friends" → "friend"; page has "friend" → "friend" → should match
        results = idx.find(["friends"])
        assert len(results) == 1

    def test_stem_true_unrelated_word_not_found(self):
        """Stemming must not conjure matches for unrelated words."""
        pages = {"http://x.com/p": "<html><body>she runs every day</body></html>"}
        idx = Indexer(stem=True)
        idx.build(pages)
        assert idx.find(["happiness"]) == []

    # -- Exact-match when stem=False ----------------------------------------

    def test_stem_false_running_does_not_find_runs(self):
        """
        Default stem=False must give exact-match only.  A query for 'running'
        must not match a page that only contains 'runs'.
        """
        pages = {"http://x.com/p": "<html><body>she runs every morning</body></html>"}
        idx = Indexer(stem=False)
        idx.build(pages)
        assert idx.find(["running"]) == []

    def test_stem_false_exact_match_still_works(self):
        """stem=False must still find exact matches."""
        pages = {"http://x.com/p": "<html><body>she runs every morning</body></html>"}
        idx = Indexer(stem=False)
        idx.build(pages)
        results = idx.find(["runs"])
        assert len(results) == 1

    # -- Numeric and mixed tokens are not stemmed ---------------------------

    def test_numeric_token_not_stemmed(self):
        """
        A token consisting of digits (e.g. '1984') must be indexed verbatim,
        not passed through Porter.  Porter is only applied to purely alphabetic
        tokens.
        """
        pages = {"http://x.com/p": "<html><body>published in 1984 by orwell</body></html>"}
        idx = Indexer(stem=True)
        idx.build(pages)
        # "1984" is not alpha → not stemmed → stored as "1984"
        assert idx.find(["1984"]) != []
        # There must be a posting for the exact string "1984"
        assert "1984" in idx.index

    def test_mixed_alphanumeric_token_not_stemmed(self):
        """A token like 'py3' (letters + digits) must not be stemmed."""
        pages = {"http://x.com/p": "<html><body>install using py3 today</body></html>"}
        idx = Indexer(stem=True)
        idx.build(pages)
        # "py3" contains a digit → not alpha → stored verbatim as "py3"
        assert "py3" in idx.index
        results = idx.find(["py3"])
        assert len(results) == 1

    def test_purely_numeric_query_not_stemmed(self):
        """Querying a number with stem=True still finds the exact number."""
        pages = {"http://x.com/p": "<html><body>chapter 42 begins here</body></html>"}
        idx = Indexer(stem=True)
        idx.build(pages)
        results = idx.find(["42"])
        assert len(results) == 1
        assert results[0]["url"] == "http://x.com/p"

    # -- Case insensitivity with stemming -----------------------------------

    def test_uppercase_stemmed_query_matches(self):
        """'Running' (capitalised) must match a page with 'runs' when stem=True."""
        pages = {"http://x.com/p": "<html><body>she runs fast</body></html>"}
        idx = Indexer(stem=True)
        idx.build(pages)
        results = idx.find(["Running"])
        assert len(results) == 1

    # -- Stem flag is instance-level ----------------------------------------

    def test_stem_flag_stored_on_instance(self):
        assert Indexer(stem=True).stem  is True
        assert Indexer(stem=False).stem is False
        assert Indexer().stem           is False  # default

    # -- Save / load with stemming ------------------------------------------

    def test_stemmed_index_usable_after_save_load(self):
        """
        build() with stem=True, save(), then load() into a fresh Indexer(stem=True)
        must produce identical find() results.

        Note: stem is an instance flag, not stored inside the JSON.  The caller
        is responsible for constructing Indexer(stem=True) when loading a stemmed
        index — the same contract as any other stateful configuration.
        """
        pages = {
            "http://x.com/p1": "<html><body>she runs every morning</body></html>",
            "http://x.com/p2": "<html><body>unrelated text here</body></html>",
        }
        original = Indexer(stem=True)
        original.build(pages)
        original_urls = {r["url"] for r in original.find(["running"])}

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            original.save(path)
            restored = Indexer(stem=True)   # stem=True must be passed explicitly
            restored.load(path)
            restored_urls = {r["url"] for r in restored.find(["running"])}
            assert original_urls == restored_urls
        finally:
            import os as _os
            _os.unlink(path)


# ---------------------------------------------------------------------------
# 3. Robots.txt — direct can_fetch mock
# ---------------------------------------------------------------------------

class TestRobotsCanFetch:
    """
    Tests that mock RobotFileParser.can_fetch directly (as opposed to the
    existing TestRobots, which configures the parser with parse()).

    Mocking can_fetch lets us:
      - Assert the exact arguments the crawler passes to can_fetch
      - Test disallow/allow without depending on the parser's internal logic
      - Verify _fetch makes exactly zero HTTP calls when disallowed
    """

    def _make_crawler(self) -> Crawler:
        c = Crawler("https://quotes.toscrape.com", politeness_window=6.0)
        c._robots.read = Mock()   # prevent real network call in crawl()
        return c

    @patch("crawler.time.sleep")
    def test_can_fetch_false_returns_none(self, _sleep: Mock) -> None:
        """
        When can_fetch returns False, _fetch must return None immediately
        without issuing any HTTP request.
        """
        c = self._make_crawler()
        c._robots.can_fetch = Mock(return_value=False)
        c._session.get = Mock()

        result = c._fetch("https://quotes.toscrape.com/secret")

        assert result is None
        c._session.get.assert_not_called()

    @patch("crawler.time.sleep")
    def test_can_fetch_called_with_user_agent_and_url(self, _sleep: Mock) -> None:
        """
        _fetch must pass the crawler's declared User-Agent string and the
        exact URL to can_fetch — not a wildcard or a different string.
        """
        c = self._make_crawler()
        c._robots.can_fetch = Mock(return_value=False)
        c._session.get = Mock()

        target_url = "https://quotes.toscrape.com/page/2"
        c._fetch(target_url)

        c._robots.can_fetch.assert_called_once()
        call_args = c._robots.can_fetch.call_args
        # First positional arg is the user-agent; second is the URL
        ua_arg, url_arg = call_args[0]
        assert "COMP3011" in ua_arg, (
            f"Expected User-Agent containing 'COMP3011', got: {ua_arg!r}"
        )
        assert url_arg == target_url

    @patch("crawler.time.sleep")
    def test_can_fetch_true_proceeds_with_request(self, _sleep: Mock) -> None:
        """
        When can_fetch returns True, _fetch must proceed normally and return
        the response body.
        """
        c = self._make_crawler()
        c._robots.can_fetch = Mock(return_value=True)
        c._session.get = Mock(return_value=_mock_response("<html>OK</html>"))

        result = c._fetch("https://quotes.toscrape.com/page/1")

        assert result == "<html>OK</html>"
        c._session.get.assert_called_once()

    @patch("crawler.time.sleep")
    def test_can_fetch_called_once_per_fetch(self, _sleep: Mock) -> None:
        """can_fetch must be checked exactly once per _fetch call."""
        c = self._make_crawler()
        c._robots.can_fetch = Mock(return_value=True)
        c._session.get = Mock(return_value=_mock_response("<html>OK</html>"))

        c._fetch("https://quotes.toscrape.com/page/1")

        assert c._robots.can_fetch.call_count == 1

    @patch("crawler.time.sleep")
    def test_disallowed_url_not_added_to_pages_in_full_crawl(self, _sleep: Mock) -> None:
        """
        When can_fetch returns False for all URLs, crawl() must return an
        empty pages dict — no pages are fetched.
        """
        html = '<html><body><a href="/page2">P2</a></body></html>'

        c = self._make_crawler()
        c._robots.can_fetch = Mock(return_value=False)
        c._session.get = Mock(return_value=_mock_response(html))

        pages = c.crawl()
        assert pages == {}


# ---------------------------------------------------------------------------
# 4. max_pages enforcement
# ---------------------------------------------------------------------------

class TestMaxPages:
    """
    Unit tests verifying that Crawler.max_pages stops the crawl at exactly
    the configured limit, regardless of how many pages are available.
    """

    BASE = "https://quotes.toscrape.com"

    def _linked_site(self, n: int) -> dict[str, str]:
        """
        Build a fake site of n pages.  The root links to /p1, /p1 links to
        /p2, …, /p(n-1) links to /pn.  This gives a chain where BFS will
        discover pages one by one.
        """
        pages: dict[str, str] = {}
        for i in range(n):
            url = self.BASE if i == 0 else f"{self.BASE}/p{i}"
            next_url = f"{self.BASE}/p{i + 1}" if i < n - 1 else ""
            link = f'<a href="/p{i + 1}">next</a>' if next_url else ""
            pages[url] = f"<html><body><p>page {i}</p>{link}</body></html>"
        return pages

    def _fake_get(self, site: dict[str, str]):
        def _get(url: str, **kw: object) -> Mock:
            return _mock_response(site.get(url.rstrip("/"), ""), 200 if url.rstrip("/") in site else 404)
        return _get

    @patch("crawler.time.sleep")
    def test_max_pages_one_stops_after_root(self, _sleep: Mock) -> None:
        """max_pages=1 must return exactly 1 page — the root."""
        site = self._linked_site(5)
        c = Crawler(self.BASE, politeness_window=6.0, max_pages=1)
        c._robots.parse(["User-agent: *", "Allow: /"])
        c._robots.read = Mock()
        c._session.get = Mock(side_effect=self._fake_get(site))

        pages = c.crawl()
        assert len(pages) == 1
        assert self.BASE in pages

    @patch("crawler.time.sleep")
    def test_max_pages_two_stops_after_two(self, _sleep: Mock) -> None:
        """max_pages=2 must return exactly 2 pages even with 5 available."""
        site = self._linked_site(5)
        c = Crawler(self.BASE, politeness_window=6.0, max_pages=2)
        c._robots.parse(["User-agent: *", "Allow: /"])
        c._robots.read = Mock()
        c._session.get = Mock(side_effect=self._fake_get(site))

        pages = c.crawl()
        assert len(pages) == 2

    @patch("crawler.time.sleep")
    def test_max_pages_larger_than_site_crawls_all(self, _sleep: Mock) -> None:
        """max_pages=100 on a 4-page site must return all 4 pages."""
        site = self._linked_site(4)
        c = Crawler(self.BASE, politeness_window=6.0, max_pages=100)
        c._robots.parse(["User-agent: *", "Allow: /"])
        c._robots.read = Mock()
        c._session.get = Mock(side_effect=self._fake_get(site))

        pages = c.crawl()
        assert len(pages) == 4

    @patch("crawler.time.sleep")
    def test_max_pages_none_crawls_all(self, _sleep: Mock) -> None:
        """max_pages=None (default) must impose no limit."""
        site = self._linked_site(4)
        c = Crawler(self.BASE, politeness_window=6.0, max_pages=None)
        c._robots.parse(["User-agent: *", "Allow: /"])
        c._robots.read = Mock()
        c._session.get = Mock(side_effect=self._fake_get(site))

        pages = c.crawl()
        assert len(pages) == 4

    @patch("crawler.time.sleep")
    def test_max_pages_never_exceeded(self, _sleep: Mock) -> None:
        """
        The number of fetched pages must never exceed max_pages, even when
        many links are discovered before the limit is hit.
        """
        # Flat site: root links to /p1 … /p9 all at once (wide, not deep)
        links = "".join(f'<a href="/p{i}">p{i}</a>' for i in range(1, 10))
        root_html = f"<html><body>{links}</body></html>"
        child_html = "<html><body>leaf</body></html>"

        def _get(url: str, **kw: object) -> Mock:
            if url.rstrip("/") == self.BASE:
                return _mock_response(root_html)
            return _mock_response(child_html)

        c = Crawler(self.BASE, politeness_window=6.0, max_pages=3)
        c._robots.parse(["User-agent: *", "Allow: /"])
        c._robots.read = Mock()
        c._session.get = Mock(side_effect=_get)

        pages = c.crawl()
        assert len(pages) <= 3

    @patch("crawler.time.sleep")
    def test_max_pages_stored_on_instance(self, _sleep: Mock) -> None:
        """max_pages value must be accessible as an attribute after init."""
        c = Crawler(self.BASE, politeness_window=6.0, max_pages=7)
        assert c.max_pages == 7

    def test_max_pages_default_is_none(self) -> None:
        c = Crawler(self.BASE, politeness_window=6.0)
        assert c.max_pages is None