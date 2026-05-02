"""
crawler.py
----------
Web crawler for the COMP3011 Search Engine Tool.

Crawls all pages of a target website, respecting a configurable politeness
window between successive HTTP requests. Returns raw page content (HTML) keyed
by URL for downstream indexing.

Typical usage
-------------
    from crawler import Crawler

    crawler = Crawler("https://quotes.toscrape.com/")
    pages = crawler.crawl()   # {url: html_string, ...}
"""

import logging
import time
import urllib.robotparser
from typing import cast
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Module-level logger — callers can configure the root logger as they wish
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


class Crawler:
    """
    A polite, breadth-first web crawler restricted to a single domain.

    Parameters
    ----------
    base_url : str
        The root URL from which crawling begins.  All discovered links that
        share the same scheme + netloc are followed; external links are
        silently ignored.
    politeness_window : float
        Minimum number of seconds to wait between successive HTTP GET
        requests.  Must be >= 6 to comply with the assignment brief.
        Defaults to 6.0.
    timeout : float
        Seconds to wait before treating an HTTP request as failed.
        Defaults to 10.0.
    max_retries : int
        Number of times to retry a failed request before giving up.
        Defaults to 3.
    max_pages : int | None
        Maximum number of pages to fetch.  ``None`` (default) means no
        limit.  The crawl stops as soon as this many pages are stored,
        even if the queue is non-empty.

    Attributes
    ----------
    pages : dict[str, str]
        Populated by :meth:`crawl`.  Maps each successfully fetched URL to
        its raw HTML content.

    Examples
    --------
    >>> c = Crawler("https://quotes.toscrape.com/")
    >>> pages = c.crawl()
    >>> len(pages)
    11  # exact count depends on site content at crawl time
    """

    DEFAULT_POLITENESS_WINDOW: float = 6.0
    DEFAULT_TIMEOUT: float = 10.0
    DEFAULT_MAX_RETRIES: int = 3
    DEFAULT_MAX_PAGES: int | None = None  # None = unlimited

    def __init__(
        self,
        base_url: str,
        politeness_window: float = DEFAULT_POLITENESS_WINDOW,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_pages: int | None = None,
    ) -> None:
        if politeness_window < 6.0:
            raise ValueError(
                f"politeness_window must be >= 6 seconds (got {politeness_window})."
            )

        self.base_url: str = base_url.rstrip("/")
        self.politeness_window: float = politeness_window
        self.timeout: float = timeout
        self.max_retries: int = max_retries
        self.max_pages: int | None = max_pages

        # Restrict crawling to this domain
        parsed = urlparse(base_url)
        self._allowed_scheme: str = parsed.scheme
        self._allowed_netloc: str = parsed.netloc

        # Stateful collections populated during crawl()
        self.pages: dict[str, str] = {}
        self._visited: set[str] = set()
        self._queue: list[str] = []

        # Reuse a single Session for connection pooling
        self._session = requests.Session()
        self._session.headers.update(
            {"User-Agent": "COMP3011-SearchBot/1.0 (educational project)"}
        )

        # robots.txt parser — populated at the start of crawl()
        self._robots: urllib.robotparser.RobotFileParser = (
            urllib.robotparser.RobotFileParser()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def crawl(self) -> dict[str, str]:
        """
        Crawl all reachable pages on the target domain starting from
        ``self.base_url``.

        The crawler performs a breadth-first traversal.  Each HTTP GET
        request is preceded by a sleep of ``self.politeness_window`` seconds
        (except the very first request) to avoid overloading the server.

        Failed requests (network errors, non-200 status codes) are logged
        and skipped; they do not halt the crawl.

        Returns
        -------
        dict[str, str]
            Mapping of ``{url: html_content}`` for every page that was
            successfully fetched.  The dict is also stored as
            ``self.pages``.
        """
        logger.info("Starting crawl from %s", self.base_url)

        self._visited.clear()
        self._queue.clear()
        self.pages.clear()

        # Fetch and parse robots.txt once before any other request.
        # If the file is absent or unreachable we treat all URLs as allowed,
        # which matches the behaviour of major crawlers (RFC 9309 s2.3).
        robots_url = f"{self._allowed_scheme}://{self._allowed_netloc}/robots.txt"
        self._robots.set_url(robots_url)
        try:
            self._robots.read()
            logger.info("robots.txt loaded from %s", robots_url)
        except Exception as exc:
            logger.warning("Could not load robots.txt (%s) — proceeding without it.", exc)
            # Fail-open: treat all URLs as allowed (RFC 9309 §2.3)
            self._robots.parse(["User-agent: *", "Allow: /"])

        self._enqueue(self.base_url)
        first_request = True

        while self._queue:
            url = self._queue.pop(0)

            if url in self._visited:
                continue

            # Politeness: sleep before every request except the very first
            if not first_request:
                logger.debug("Sleeping %.1fs (politeness window)", self.politeness_window)
                time.sleep(self.politeness_window)
            first_request = False

            # Honour max_pages limit if set
            if self.max_pages is not None and len(self.pages) >= self.max_pages:
                logger.info("max_pages=%d reached — stopping crawl.", self.max_pages)
                break

            html = self._fetch(url)
            if html is None:
                continue  # error already logged inside _fetch

            self._visited.add(url)
            self.pages[url] = html
            logger.info("Fetched %s  (total pages: %d)", url, len(self.pages))

            # Discover and enqueue new links
            for link in self._extract_links(html, url):
                if link not in self._visited:
                    self._enqueue(link)

        logger.info(
            "Crawl complete. %d pages fetched, %d URLs skipped.",
            len(self.pages),
            len(self._visited) - len(self.pages),
        )
        return self.pages

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch(self, url: str) -> str | None:
        """
        Fetch a single URL with retry logic.

        Respects robots.txt: if the parser has been loaded and the
        configured User-Agent is disallowed from *url*, the method logs a
        warning and returns ``None`` immediately without making a network
        request.

        Parameters
        ----------
        url : str
            The URL to fetch.

        Returns
        -------
        str | None
            The response body as a string, or ``None`` on failure or
            robots.txt disallowance.
        """
        raw_ua = self._session.headers.get("User-Agent", "*")
        user_agent: str = raw_ua if isinstance(raw_ua, str) else raw_ua.decode()
        if not self._robots.can_fetch(user_agent, url):
            logger.warning("robots.txt disallows %s — skipping.", url)
            return None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._session.get(url, timeout=self.timeout)
                response.raise_for_status()
                return response.text
            except requests.exceptions.HTTPError as exc:
                logger.warning(
                    "HTTP error on %s (attempt %d/%d): %s",
                    url, attempt, self.max_retries, exc,
                )
            except requests.exceptions.ConnectionError as exc:
                logger.warning(
                    "Connection error on %s (attempt %d/%d): %s",
                    url, attempt, self.max_retries, exc,
                )
            except requests.exceptions.Timeout:
                logger.warning(
                    "Timeout on %s (attempt %d/%d)", url, attempt, self.max_retries
                )
            except requests.exceptions.RequestException as exc:
                logger.error("Unrecoverable error fetching %s: %s", url, exc)
                return None  # No point retrying truly unrecoverable errors

            # Back-off before retry (but don't sleep after the last attempt)
            if attempt < self.max_retries:
                backoff = self.politeness_window * attempt
                logger.debug("Backing off %.1fs before retry", backoff)
                time.sleep(backoff)

        logger.error("Giving up on %s after %d attempts.", url, self.max_retries)
        return None

    def _extract_links(self, html: str, current_url: str) -> list[str]:
        """
        Parse all ``<a href=...>`` links from *html* and return those that
        belong to the allowed domain, normalised to absolute URLs.

        Parameters
        ----------
        html : str
            Raw HTML content of the current page.
        current_url : str
            The URL from which *html* was fetched (used to resolve relative
            hrefs).

        Returns
        -------
        list[str]
            Deduplicated list of absolute, same-domain URLs found on the page.
        """
        soup = BeautifulSoup(html, "html.parser")
        links: list[str] = []

        for anchor in soup.find_all("a", href=True):
            href: str = cast(str, anchor["href"]).strip()

            # Skip fragment-only and javascript: links
            if not href or href.startswith("#") or href.startswith("javascript:"):
                continue

            absolute = urljoin(current_url, href)
            absolute = self._normalise_url(absolute)

            if self._is_allowed(absolute) and absolute not in links:
                links.append(absolute)

        return links

    def _enqueue(self, url: str) -> None:
        """Add *url* to the crawl queue if it hasn't been visited or queued."""
        normalised = self._normalise_url(url)
        if normalised not in self._visited and normalised not in self._queue:
            self._queue.append(normalised)

    def _is_allowed(self, url: str) -> bool:
        """
        Return ``True`` if *url* belongs to the same scheme and domain as
        ``self.base_url``.
        """
        parsed = urlparse(url)
        return (
            parsed.scheme == self._allowed_scheme
            and parsed.netloc == self._allowed_netloc
        )

    @staticmethod
    def _normalise_url(url: str) -> str:
        """
        Strip URL fragments and trailing slashes for consistent deduplication.

        Parameters
        ----------
        url : str
            Raw URL string.

        Returns
        -------
        str
            Normalised URL without fragment or trailing slash.
        """
        parsed = urlparse(url)
        # Drop fragment; keep everything else
        normalised = parsed._replace(fragment="").geturl()
        return normalised.rstrip("/")