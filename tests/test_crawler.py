"""
tests/test_crawler.py
---------------------
Unit tests for the Crawler class.

All HTTP requests are mocked using unittest.mock so these tests run
offline without ever touching the real website.  This is important both
for speed and for repeatability — we don't want tests to fail because
quotes.toscrape.com is temporarily unavailable.

Run with:
    pytest tests/test_crawler.py -v
    pytest tests/ -v --cov=src --cov-report=term-missing
"""

import sys
import os
from unittest.mock import Mock, call, patch

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from crawler import Crawler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIMPLE_HTML = """
<html><body>
  <p>Hello world</p>
  <a href="/page2">Page 2</a>
  <a href="/page3">Page 3</a>
  <a href="https://external.com/other">External</a>
</body></html>
"""
PAGE2_HTML = "<html><body><p>Page two content</p></body></html>"
PAGE3_HTML = "<html><body><p>Page three content</p></body></html>"


def _mock_response(text: str, status_code: int = 200) -> Mock:
    """Return a mock requests.Response."""
    resp = Mock()
    resp.text = text
    resp.status_code = status_code
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"{status_code} Error"
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestCrawlerInit:

    def test_valid_politeness_window(self):
        c = Crawler("https://example.com", politeness_window=6.0)
        assert c.politeness_window == 6.0

    def test_politeness_window_too_small_raises(self):
        with pytest.raises(ValueError, match="politeness_window must be >= 6"):
            Crawler("https://example.com", politeness_window=3.0)

    def test_politeness_window_exactly_six_is_valid(self):
        c = Crawler("https://example.com", politeness_window=6.0)
        assert c.politeness_window == 6.0

    def test_base_url_trailing_slash_stripped(self):
        c = Crawler("https://example.com/")
        assert c.base_url == "https://example.com"

    def test_allowed_netloc_extracted(self):
        c = Crawler("https://quotes.toscrape.com/")
        assert c._allowed_netloc == "quotes.toscrape.com"

    def test_pages_initially_empty(self):
        c = Crawler("https://example.com")
        assert c.pages == {}


# ---------------------------------------------------------------------------
# URL normalisation & domain filtering
# ---------------------------------------------------------------------------

class TestUrlHandling:

    def setup_method(self):
        self.crawler = Crawler("https://quotes.toscrape.com")

    def test_normalise_strips_fragment(self):
        url = self.crawler._normalise_url("https://example.com/page#section")
        assert url == "https://example.com/page"

    def test_normalise_strips_trailing_slash(self):
        url = self.crawler._normalise_url("https://example.com/page/")
        assert url == "https://example.com/page"

    def test_is_allowed_same_domain(self):
        assert self.crawler._is_allowed("https://quotes.toscrape.com/page/2")

    def test_is_allowed_rejects_external(self):
        assert not self.crawler._is_allowed("https://external.com/page")

    def test_is_allowed_rejects_different_scheme(self):
        assert not self.crawler._is_allowed("http://quotes.toscrape.com/page")

    def test_extract_links_filters_fragments(self):
        html = '<a href="#top">Top</a><a href="/real-page">Real</a>'
        links = self.crawler._extract_links(html, "https://quotes.toscrape.com")
        assert all("#" not in link for link in links)

    def test_extract_links_filters_javascript(self):
        html = '<a href="javascript:void(0)">JS</a><a href="/real">Real</a>'
        links = self.crawler._extract_links(html, "https://quotes.toscrape.com")
        assert all("javascript:" not in link for link in links)

    def test_extract_links_resolves_relative_urls(self):
        html = '<a href="/page2">Page 2</a>'
        links = self.crawler._extract_links(html, "https://quotes.toscrape.com")
        assert "https://quotes.toscrape.com/page2" in links

    def test_extract_links_deduplicates(self):
        html = '<a href="/page">P</a><a href="/page">P duplicate</a>'
        links = self.crawler._extract_links(html, "https://quotes.toscrape.com")
        assert links.count("https://quotes.toscrape.com/page") == 1

    def test_extract_links_ignores_external_domains(self):
        html = '<a href="https://other.com/page">External</a>'
        links = self.crawler._extract_links(html, "https://quotes.toscrape.com")
        assert links == []

    def test_extract_links_handles_empty_href(self):
        html = '<a href="">Empty</a><a href="/valid">Valid</a>'
        links = self.crawler._extract_links(html, "https://quotes.toscrape.com")
        assert "https://quotes.toscrape.com/valid" in links


# ---------------------------------------------------------------------------
# Fetch & retry logic
# ---------------------------------------------------------------------------

class TestFetch:

    def setup_method(self):
        self.crawler = Crawler("https://quotes.toscrape.com")
        # Prime the robots parser so _fetch doesn't block in unit tests
        self.crawler._robots.parse(["User-agent: *", "Allow: /"])

    @patch("crawler.time.sleep")
    def test_successful_fetch_returns_html(self, _sleep):
        self.crawler._session.get = Mock(
            return_value=_mock_response("<html>Hello</html>")
        )
        result = self.crawler._fetch("https://quotes.toscrape.com")
        assert result == "<html>Hello</html>"

    @patch("crawler.time.sleep")
    def test_http_404_retries_then_returns_none(self, _sleep):
        self.crawler._session.get = Mock(
            return_value=_mock_response("Not Found", status_code=404)
        )
        self.crawler.max_retries = 3
        result = self.crawler._fetch("https://quotes.toscrape.com/missing")
        assert result is None
        assert self.crawler._session.get.call_count == 3

    @patch("crawler.time.sleep")
    def test_connection_error_retries_then_returns_none(self, _sleep):
        self.crawler._session.get = Mock(
            side_effect=requests.exceptions.ConnectionError("refused")
        )
        self.crawler.max_retries = 2
        result = self.crawler._fetch("https://quotes.toscrape.com")
        assert result is None
        assert self.crawler._session.get.call_count == 2

    @patch("crawler.time.sleep")
    def test_timeout_retries_then_returns_none(self, _sleep):
        self.crawler._session.get = Mock(
            side_effect=requests.exceptions.Timeout()
        )
        self.crawler.max_retries = 2
        result = self.crawler._fetch("https://quotes.toscrape.com")
        assert result is None

    @patch("crawler.time.sleep")
    def test_unrecoverable_error_no_retry(self, _sleep):
        """Generic RequestException should bail immediately, no retry."""
        self.crawler._session.get = Mock(
            side_effect=requests.exceptions.RequestException("fatal")
        )
        self.crawler.max_retries = 3
        result = self.crawler._fetch("https://quotes.toscrape.com")
        assert result is None
        assert self.crawler._session.get.call_count == 1

    @patch("crawler.time.sleep")
    def test_successful_after_one_failure(self, _sleep):
        """Crawler should succeed on second attempt if first fails."""
        fail = Mock(side_effect=requests.exceptions.ConnectionError())
        success = _mock_response("<html>OK</html>")
        self.crawler._session.get = Mock(side_effect=[fail.side_effect, success])
        result = self.crawler._fetch("https://quotes.toscrape.com")
        assert result == "<html>OK</html>"


# ---------------------------------------------------------------------------
# Full crawl (mocked)
# ---------------------------------------------------------------------------

class TestCrawl:

    def _make_crawler(self):
        c = Crawler("https://quotes.toscrape.com", politeness_window=6.0)
        # Prime the robots parser and stub read() so crawl() never tries
        # to fetch robots.txt over the network during unit tests.
        c._robots.parse(["User-agent: *", "Allow: /"])
        c._robots.read = Mock()
        return c

    @patch("crawler.time.sleep")
    def test_crawl_visits_linked_pages(self, _sleep):
        responses = {
            "https://quotes.toscrape.com": _mock_response(SIMPLE_HTML),
            "https://quotes.toscrape.com/page2": _mock_response(PAGE2_HTML),
            "https://quotes.toscrape.com/page3": _mock_response(PAGE3_HTML),
        }

        def fake_get(url, **kwargs):
            return responses.get(url.rstrip("/"), _mock_response("", 404))

        c = self._make_crawler()
        c._session.get = Mock(side_effect=fake_get)
        pages = c.crawl()

        assert "https://quotes.toscrape.com" in pages
        assert "https://quotes.toscrape.com/page2" in pages
        assert "https://quotes.toscrape.com/page3" in pages

    @patch("crawler.time.sleep")
    def test_crawl_does_not_revisit_pages(self, _sleep):
        html = '<html><body><a href="/">Home</a><a href="/page2">P2</a></body></html>'

        def fake_get(url, **kwargs):
            if url.rstrip("/") == "https://quotes.toscrape.com":
                return _mock_response(html)
            return _mock_response("<html><body>P2</body></html>")

        c = self._make_crawler()
        c._session.get = Mock(side_effect=fake_get)
        c.crawl()

        fetched = [ca.args[0] for ca in c._session.get.call_args_list]
        assert len(fetched) == len(set(fetched)), "Some URLs were fetched more than once"

    @patch("crawler.time.sleep")
    def test_crawl_ignores_external_links(self, _sleep):
        html = '<html><body><a href="https://evil.com/page">External</a></body></html>'
        c = self._make_crawler()
        c._session.get = Mock(return_value=_mock_response(html))
        c.crawl()

        fetched = [ca.args[0] for ca in c._session.get.call_args_list]
        assert not any("evil.com" in u for u in fetched)

    @patch("crawler.time.sleep")
    def test_crawl_respects_politeness_window(self, mock_sleep):
        html = '<html><body><a href="/p2">P2</a></body></html>'

        def fake_get(url, **kwargs):
            return _mock_response(html if "p2" not in url else PAGE2_HTML)

        c = self._make_crawler()
        c._session.get = Mock(side_effect=fake_get)
        c.crawl()

        # sleep must be called with exactly 6.0 seconds
        for c_call in mock_sleep.call_args_list:
            assert c_call == call(6.0)

    @patch("crawler.time.sleep")
    def test_crawl_skips_failed_pages(self, _sleep):
        html = '<html><body><a href="/good">Good</a><a href="/bad">Bad</a></body></html>'

        def fake_get(url, **kwargs):
            url_s = url.rstrip("/")
            if url_s == "https://quotes.toscrape.com":
                return _mock_response(html)
            if "good" in url_s:
                return _mock_response("<html><body>Good</body></html>")
            return _mock_response("", 500)

        c = self._make_crawler()
        c._session.get = Mock(side_effect=fake_get)
        pages = c.crawl()

        assert "https://quotes.toscrape.com/good" in pages
        assert "https://quotes.toscrape.com/bad" not in pages

    @patch("crawler.time.sleep")
    def test_crawl_no_links_returns_root_only(self, _sleep):
        c = self._make_crawler()
        c._session.get = Mock(
            return_value=_mock_response("<html><body>No links.</body></html>")
        )
        pages = c.crawl()
        assert len(pages) == 1
        assert "https://quotes.toscrape.com" in pages

    @patch("crawler.time.sleep")
    def test_crawl_stores_html_content(self, _sleep):
        content = "<html><body><p>Unique content xyz</p></body></html>"
        c = self._make_crawler()
        c._session.get = Mock(return_value=_mock_response(content))
        pages = c.crawl()
        assert pages["https://quotes.toscrape.com"] == content


# ---------------------------------------------------------------------------
# robots.txt compliance
# ---------------------------------------------------------------------------

class TestRobots:
    """Tests for robots.txt fetching and URL gating."""

    def _make_crawler(self):
        return Crawler("https://quotes.toscrape.com", politeness_window=6.0)

    @patch("crawler.time.sleep")
    def test_disallowed_url_not_fetched(self, _sleep):
        """
        If robots.txt disallows our User-Agent on a URL, _fetch must return
        None without making an HTTP request for that URL.
        """
        c = self._make_crawler()
        c._robots.parse(["User-agent: *", "Disallow: /"])
        c._session.get = Mock()
        result = c._fetch("https://quotes.toscrape.com/private")
        assert result is None
        c._session.get.assert_not_called()

    @patch("crawler.time.sleep")
    def test_allowed_url_is_fetched(self, _sleep):
        """If robots.txt allows the URL, _fetch proceeds normally."""
        c = self._make_crawler()
        c._robots.parse(["User-agent: *", "Allow: /"])
        c._session.get = Mock(return_value=_mock_response("<html>OK</html>"))
        result = c._fetch("https://quotes.toscrape.com/page")
        assert result == "<html>OK</html>"

    @patch("crawler.time.sleep")
    def test_robots_txt_read_at_crawl_start(self, _sleep):
        """
        crawl() must call _robots.read() before making any page requests,
        so robots.txt is always consulted first.
        """
        c = self._make_crawler()
        c._session.get = Mock(return_value=_mock_response("<html><body></body></html>"))

        read_called_before_get = {"value": False}
        original_read = c._robots.read

        def tracking_read():
            # Mark that read() ran; allow everything so crawl completes
            read_called_before_get["value"] = True
            c._robots.parse(["User-agent: *", "Allow: /"])

        c._robots.read = tracking_read
        c.crawl()

        assert read_called_before_get["value"], "robots.txt was never read"

    @patch("crawler.time.sleep")
    def test_missing_robots_txt_does_not_crash_crawl(self, _sleep):
        """
        If robots.txt is unreachable the crawl must continue and treat
        all URLs as allowed (fail-open behaviour per RFC 9309).
        """
        c = self._make_crawler()
        c._session.get = Mock(return_value=_mock_response("<html><body></body></html>"))
        c._robots.read = Mock(side_effect=Exception("unreachable"))
        pages = c.crawl()
        assert len(pages) >= 1