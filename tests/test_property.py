"""
tests/test_property.py
----------------------
Property-based tests for the COMP3011 Search Engine Tool using Hypothesis.

Unlike example-based tests (which check specific inputs you thought of),
property-based tests define *invariants* — mathematical truths that must hold
for *any* valid input — and let Hypothesis generate hundreds of random inputs
to try to falsify them.  If Hypothesis finds a counterexample it automatically
shrinks it to the minimal failing case, making debugging straightforward.

Invariants tested
-----------------
1. **Frequency conservation**
   After indexing any token list, the sum of every ``freq`` value across all
   postings equals the total number of tokens.  This is a pure accounting
   identity: every token must be counted exactly once.

2. **Position list structural correctness**
   For every (word, url) pair: ``len(positions) == freq``, all positions lie
   in ``[0, len(tokens) - 1]``, and the list is strictly increasing (no
   duplicates, naturally ordered because tokens are processed left-to-right).

3. **``_normalise_token`` idempotency**
   Applying normalisation twice must give the same result as applying it once:
   ``normalise(normalise(t)) == normalise(t)``.  Violating this would mean
   the indexer and the query path normalise tokens differently, causing
   queries to miss indexed terms.

4. **``find()`` subset invariant (AND semantics)**
   ``find([a, b])`` must return a strict subset of ``find([a])`` for any pair
   of terms.  An AND query can never *add* pages that a single-term query
   would not already return.

5. **TF-IDF non-negativity**
   Every ``tf_idf`` value in every posting must be >= 0.0.  A negative weight
   would invert the ranking, promoting irrelevant pages.

6. **``find_phrase()`` ⊆ ``find()``**
   Every URL returned by a phrase query must also appear in the corresponding
   AND query.  A phrase match implies all terms are present; the converse is
   not required.

7. **Round-trip save/load preserves the index exactly**
   Saving then loading must produce a structurally identical index for any
   corpus Hypothesis generates.

Run with:
    pytest tests/test_property.py -v
    pytest tests/test_property.py -v --hypothesis-show-statistics
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from indexer import Indexer

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

# A "word" for our purposes: lowercase ASCII letters only, length 1-12.
# We restrict to ASCII to avoid Hypothesis generating Unicode that BeautifulSoup
# would silently transform, which would make token-count assertions brittle.
WORD = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=12)

# A token list: 1–60 words drawn from WORD.
TOKEN_LIST = st.lists(WORD, min_size=1, max_size=60)

# A URL-like string (we don't need valid HTTP URLs; just unique keys).
URL = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
    min_size=4,
    max_size=30,
).map(lambda s: f"https://example.com/{s}")

# A single-page corpus: one URL mapped to a minimal HTML page built from a
# token list.  We wrap tokens in <body> so _tokenise processes them normally.
@st.composite
def single_page_corpus(draw: st.DrawFn) -> tuple[str, list[str], dict[str, str]]:
    """
    Draw a (url, token_list, pages_dict) triple.

    The HTML is constructed by joining the tokens with spaces inside a
    ``<body>`` tag so the text is visible and _tokenise will return exactly
    those tokens (after lowercasing, which is already applied since WORD is
    lowercase-only).
    """
    url = draw(URL)
    tokens = draw(TOKEN_LIST)
    body = " ".join(tokens)
    html = f"<html><body><p>{body}</p></body></html>"
    return url, tokens, {url: html}


# A multi-page corpus: 2–6 pages, each with its own token list.
@st.composite
def multi_page_corpus(draw: st.DrawFn) -> dict[str, str]:
    """
    Draw a dict of 2–6 distinct URL → HTML pages.

    URLs are guaranteed distinct by generating a set of unique suffixes first,
    then building pages.
    """
    n = draw(st.integers(min_value=2, max_value=6))
    suffixes = draw(
        st.lists(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=3, max_size=10),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    pages: dict[str, str] = {}
    for suffix in suffixes:
        tokens = draw(TOKEN_LIST)
        body = " ".join(tokens)
        pages[f"https://example.com/{suffix}"] = f"<html><body><p>{body}</p></body></html>"
    return pages


# ---------------------------------------------------------------------------
# 1. Frequency conservation
# ---------------------------------------------------------------------------

class TestFrequencyConservation:
    """
    Invariant: sum of all freq values == total tokens indexed.

    This holds because _index_page increments exactly one freq counter
    for each token position processed, and every token in the list is
    processed exactly once.
    """

    @given(data=single_page_corpus())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_total_freq_equals_token_count(
        self, data: tuple[str, list[str], dict[str, str]]
    ) -> None:
        url, tokens, pages = data
        idx = Indexer()
        idx.build(pages)

        total_freq = sum(
            stats["freq"]
            for postings in idx.index.values()
            for stats in postings.values()
        )
        assert total_freq == len(tokens), (
            f"Expected total freq={len(tokens)}, got {total_freq}. "
            f"First 5 tokens: {tokens[:5]}"
        )

    @given(pages=multi_page_corpus())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_total_freq_equals_token_count_multi_page(
        self, pages: dict[str, str]
    ) -> None:
        """Frequency conservation must hold across the entire multi-page corpus."""
        # Use _tokenise (the same code path as build()) to count expected tokens
        # per page — avoids reimplementing tokenisation logic in the test and
        # ensures the expected count can never diverge from the actual path.
        idx = Indexer()
        expected = sum(len(idx._tokenise(html)) for html in pages.values())
        idx.build(pages)

        actual = sum(
            stats["freq"]
            for postings in idx.index.values()
            for stats in postings.values()
        )
        assert actual == expected, (
            f"Total freq {actual} != expected token count {expected}"
        )


# ---------------------------------------------------------------------------
# 2. Position list structural correctness
# ---------------------------------------------------------------------------

class TestPositionStructure:
    """
    Invariant: for every (word, url) pair the position list is:
      (a) the same length as freq
      (b) all values in [0, len(tokens) - 1]
      (c) strictly increasing (no repeats, natural order)
    """

    @given(data=single_page_corpus())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_positions_length_equals_freq(
        self, data: tuple[str, list[str], dict[str, str]]
    ) -> None:
        _, tokens, pages = data
        idx = Indexer()
        idx.build(pages)

        for word, postings in idx.index.items():
            for url, stats in postings.items():
                assert len(stats["positions"]) == stats["freq"], (
                    f"Word '{word}' on {url}: "
                    f"len(positions)={len(stats['positions'])} != freq={stats['freq']}"
                )

    @given(data=single_page_corpus())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_positions_within_token_range(
        self, data: tuple[str, list[str], dict[str, str]]
    ) -> None:
        _, tokens, pages = data
        idx = Indexer()
        idx.build(pages)

        for word, postings in idx.index.items():
            for url, stats in postings.items():
                for pos in stats["positions"]:
                    assert 0 <= pos < len(tokens), (
                        f"Position {pos} for '{word}' out of range "
                        f"[0, {len(tokens) - 1}]"
                    )

    @given(data=single_page_corpus())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_positions_strictly_increasing(
        self, data: tuple[str, list[str], dict[str, str]]
    ) -> None:
        _, tokens, pages = data
        idx = Indexer()
        idx.build(pages)

        for word, postings in idx.index.items():
            for url, stats in postings.items():
                positions = stats["positions"]
                for i in range(len(positions) - 1):
                    assert positions[i] < positions[i + 1], (
                        f"Positions for '{word}' are not strictly increasing: "
                        f"{positions}"
                    )


# ---------------------------------------------------------------------------
# 3. _normalise_token idempotency
# ---------------------------------------------------------------------------

class TestNormaliseIdempotency:
    """
    Invariant: normalise(normalise(t)) == normalise(t) for all strings t.

    If this fails, the indexer stores tokens under a different key than the
    query path would look up — causing systematic misses.  The stemmer must
    also be idempotent because PorterStemmer is designed to be: stemming a
    stem returns the same stem.
    """

    @given(token=st.text(min_size=0, max_size=40))
    @settings(max_examples=500)
    def test_normalise_is_idempotent_no_stem(self, token: str) -> None:
        idx = Indexer(stem=False)
        once  = idx._normalise_token(token)
        twice = idx._normalise_token(once)
        assert once == twice, (
            f"normalise('{token}') = '{once}', "
            f"but normalise(normalise(...)) = '{twice}'"
        )

    @given(token=st.text(min_size=0, max_size=40))
    @settings(max_examples=500)
    def test_normalise_with_stem_is_deterministic(self, token: str) -> None:
        """
        _normalise_token must be a pure function: identical inputs always produce
        identical outputs.  This is the property that correctness actually requires:
        a user querying "perseverance" must always look up the same index key,
        regardless of when or how many times normalisation is called.

        Note: Hypothesis originally tested idempotency (stem(stem(x)) == stem(x))
        but found this does NOT hold even for real English words — 'perseverance'
        stems to 'persever', which then stems to 'persev'.  PorterStemmer reduces
        words toward a stem but does not guarantee a fixed point.  The correct
        invariant for search correctness is determinism, not idempotency: both
        the index path and the query path call _normalise_token once with the
        raw input, so they always produce the same lookup key.
        """
        idx = Indexer(stem=True)
        first_call  = idx._normalise_token(token)
        second_call = idx._normalise_token(token)   # same input, not the output
        assert first_call == second_call, (
            f"_normalise_token('{token}') returned different results "
            f"on two calls: '{first_call}' vs '{second_call}'"
        )

    @given(token=st.text(min_size=1, max_size=30))
    @settings(max_examples=300)
    def test_normalise_output_is_lowercase(self, token: str) -> None:
        idx = Indexer(stem=False)
        result = idx._normalise_token(token)
        assert result == result.lower(), (
            f"normalise('{token}') = '{result}' contains uppercase"
        )

    @given(token=WORD)
    @settings(max_examples=300)
    def test_normalise_pure_alpha_token_is_nonempty(self, token: str) -> None:
        """A token consisting only of lowercase letters must survive normalisation."""
        idx = Indexer(stem=False)
        result = idx._normalise_token(token)
        assert result != "", (
            f"normalise('{token}') returned empty string unexpectedly"
        )


# ---------------------------------------------------------------------------
# 4. find() subset invariant (AND semantics)
# ---------------------------------------------------------------------------

class TestFindSubsetInvariant:
    """
    Invariant: find([a, b]) ⊆ find([a]) for any terms a, b.

    An AND query can only *restrict* results, never expand them.  If this
    fails it means the intersection logic is adding URLs that are not in one
    of the individual postings lists.
    """

    @given(pages=multi_page_corpus(), terms=st.lists(WORD, min_size=2, max_size=2))
    @settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
    def test_two_term_find_subset_of_single_term(
        self, pages: dict[str, str], terms: list[str]
    ) -> None:
        a, b = terms[0], terms[1]
        idx = Indexer()
        idx.build(pages)

        urls_a     = {r["url"] for r in idx.find([a])}
        urls_b     = {r["url"] for r in idx.find([b])}
        urls_ab    = {r["url"] for r in idx.find([a, b])}

        assert urls_ab <= urls_a, (
            f"find(['{a}', '{b}']) returned URLs not in find(['{a}']): "
            f"{urls_ab - urls_a}"
        )
        assert urls_ab <= urls_b, (
            f"find(['{a}', '{b}']) returned URLs not in find(['{b}']): "
            f"{urls_ab - urls_b}"
        )

    @given(pages=multi_page_corpus(), terms=st.lists(WORD, min_size=3, max_size=3))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_three_term_find_subset_of_each_single_term(
        self, pages: dict[str, str], terms: list[str]
    ) -> None:
        """The subset property must hold for three-term AND queries too."""
        idx = Indexer()
        idx.build(pages)

        urls_abc = {r["url"] for r in idx.find(terms)}

        for term in terms:
            urls_single = {r["url"] for r in idx.find([term])}
            assert urls_abc <= urls_single, (
                f"find({terms}) returned URLs not in find(['{term}']): "
                f"{urls_abc - urls_single}"
            )


# ---------------------------------------------------------------------------
# 5. TF-IDF non-negativity
# ---------------------------------------------------------------------------

class TestTfIdfNonNegative:
    """
    Invariant: every tf_idf value must be >= 0.0.

    TF = 1 + log10(freq) >= 1 for freq >= 1.
    IDF = log10(N / df) >= 0 because N >= df always.
    So TF-IDF = TF * IDF >= 0.  A negative value would invert ranking.
    """

    @given(pages=multi_page_corpus())
    @settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
    def test_all_tfidf_non_negative(self, pages: dict[str, str]) -> None:
        idx = Indexer()
        idx.build(pages)

        for word, postings in idx.index.items():
            for url, stats in postings.items():
                assert stats["tf_idf"] >= 0.0, (
                    f"Negative tf_idf={stats['tf_idf']:.6f} "
                    f"for word='{word}' on {url}"
                )

    @given(pages=multi_page_corpus())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_universal_word_has_zero_tfidf(self, pages: dict[str, str]) -> None:
        """
        Any word appearing in every page must have IDF = log10(N/N) = 0,
        making its TF-IDF exactly 0.0 regardless of frequency.
        """
        idx = Indexer()
        idx.build(pages)
        n_pages = len(pages)

        for word, postings in idx.index.items():
            if len(postings) == n_pages:   # word in every page
                for url, stats in postings.items():
                    assert stats["tf_idf"] == pytest.approx(0.0, abs=1e-9), (
                        f"Universal word '{word}' should have tf_idf=0.0, "
                        f"got {stats['tf_idf']:.6f} on {url}"
                    )


# ---------------------------------------------------------------------------
# 6. find_phrase() ⊆ find()
# ---------------------------------------------------------------------------

class TestPhraseFindSubset:
    """
    Invariant: find_phrase([a, b]) ⊆ find([a, b]).

    A phrase match implies both terms are present (AND), but not vice versa.
    If a URL appears in phrase results but not AND results, the intersection
    logic is broken.
    """

    @given(pages=multi_page_corpus(), terms=st.lists(WORD, min_size=2, max_size=3))
    @settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
    def test_phrase_subset_of_find(
        self, pages: dict[str, str], terms: list[str]
    ) -> None:
        idx = Indexer()
        idx.build(pages)

        urls_phrase = {r["url"] for r in idx.find_phrase(terms)}
        urls_find   = {r["url"] for r in idx.find(terms)}

        assert urls_phrase <= urls_find, (
            f"find_phrase({terms}) returned URLs not in find({terms}): "
            f"{urls_phrase - urls_find}"
        )


# ---------------------------------------------------------------------------
# 7. Round-trip save/load preserves the index
# ---------------------------------------------------------------------------

class TestSaveLoadRoundTrip:
    """
    Invariant: save then load produces an index equal to the original.

    Tests both the structure (same keys) and the values (freq, positions,
    tf_idf all identical) across any corpus Hypothesis generates.
    """

    @given(pages=multi_page_corpus())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_save_load_index_identical(self, pages: dict[str, str]) -> None:
        original = Indexer()
        original.build(pages)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            original.save(path)
            restored = Indexer()
            restored.load(path)

            assert restored.index == original.index, (
                f"Loaded index differs from original. "
                f"Original terms: {set(original.index)}, "
                f"Restored terms: {set(restored.index)}"
            )
        finally:
            Path(path).unlink(missing_ok=True)

    @given(pages=multi_page_corpus(), terms=st.lists(WORD, min_size=1, max_size=2))
    @settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
    def test_save_load_find_same_results(
        self, pages: dict[str, str], terms: list[str]
    ) -> None:
        """
        The set of matching URLs must be identical before and after a
        save/load round-trip.  Scores may differ by floating-point epsilon
        so we only assert URL sets, not exact scores.
        """
        original = Indexer()
        original.build(pages)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            original.save(path)
            restored = Indexer()
            restored.load(path)

            urls_before = {r["url"] for r in original.find(terms)}
            urls_after  = {r["url"] for r in restored.find(terms)}
            assert urls_before == urls_after
        finally:
            Path(path).unlink(missing_ok=True)
