# COMP3011 Search Engine Tool

[![Tests](https://github.com/CodingOz/web-scraper/actions/workflows/test.yml/badge.svg)](https://github.com/CodingOz/web-scraper/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-2a6db2)](https://mypy.readthedocs.io/)

A command-line search engine that crawls [quotes.toscrape.com](https://quotes.toscrape.com), builds a TF-IDF weighted inverted index, and lets you query it interactively — all in Python.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Design Rationale](#design-rationale)
- [Installation](#installation)
- [Usage](#usage)
- [Commands](#commands)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [CI / GitHub Actions](#ci--github-actions)
- [Dependencies](#dependencies)

---

## Overview

This tool implements the three core components of a search engine from scratch:

1. **Crawler** — breadth-first web crawler with a 6-second politeness window, retry logic, and domain restriction
2. **Indexer** — builds an inverted index storing per-word, per-page frequency, token positions, and TF-IDF weight
3. **Search** — processes single and multi-word AND queries, returning results ranked by relevance score

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        main.py                          │
│              Interactive CLI shell (REPL)               │
└────────────┬────────────────────────────┬───────────────┘
             │                            │
             ▼                            ▼
┌────────────────────┐        ┌───────────────────────┐
│    crawler.py      │        │      search.py         │
│                    │        │                        │
│  Crawler           │        │  SearchEngine          │
│  ─ crawl()         │        │  ─ find()              │
│  ─ _fetch()        │        │  ─ print_word()        │
│  ─ _extract_links()│        └──────────┬────────────-┘
└────────┬───────────┘                   │
         │ pages: {url: html}            │ wraps
         ▼                               ▼
┌─────────────────────────────────────────────────────────┐
│                      indexer.py                         │
│                                                         │
│  Indexer                                                │
│  ─ build(pages)      ← tokenise, index, TF-IDF          │
│  ─ save(path)        ← serialise to JSON                │
│  ─ load(path)        ← deserialise from JSON            │
│  ─ find(terms)       ← AND query + TF-IDF ranking       │
│  ─ get_postings(word)← single-word lookup               │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│   data/index.json   │
│   Persisted index   │
└─────────────────────┘
```

Data flows in one direction: the crawler feeds raw HTML to the indexer, which produces a persistent index that the search engine reads from. Each layer has a single responsibility and can be tested independently.

---

## Design Rationale

### Inverted Index Data Structure

The index uses a nested dictionary:

```python
{
  "word": {
    "https://example.com/page1": {
      "freq":      3,
      "positions": [4, 17, 42],
      "tf_idf":    0.4812
    }
  }
}
```

**Why a nested dict over alternatives:**

| Structure | Word lookup | URL lookup within word | Memory |
|---|---|---|---|
| `dict[word → dict[url → stats]]` ✓ | O(1) | O(1) | Moderate |
| `dict[word → list[(url, stats)]]` | O(1) | O(n pages) | Lower |
| Flat list of `(word, url, stats)` | O(n terms) | O(n terms) | Low |
| SQLite table | O(log n) | O(log n) | Low |

A dict-of-dicts gives constant-time lookup at both levels. For the `find` command, this means computing the intersection of N postings sets in O(total matching docs) rather than O(total index entries). SQLite would add serialisation overhead and a dependency with no meaningful benefit for a single-site index of this scale.

**Positions list** — storing every token offset enables potential phrase/proximity search in future iterations and gives the marker visible evidence that the index is richer than a simple word count.

### TF-IDF Scoring

Results returned by `find` are ranked by TF-IDF rather than raw frequency:

```
TF  = 1 + log10(raw_frequency)    # log-normalised
IDF = log10(total_docs / doc_freq) # standard smooth IDF
TF-IDF = TF × IDF
```

**Why log-normalised TF:** A page mentioning "love" 30 times is not 30× more relevant than one mentioning it once. Log scaling compresses the range so longer pages do not automatically dominate.

**Why smooth IDF (not +1 variant):** Every word in the index appears in at least one document by construction, so the denominator is never zero — the +1 smoothing typically used to prevent division-by-zero is unnecessary here.

**IDF = 0 for universal terms:** Words appearing on every crawled page (e.g. "the", site navigation text) receive IDF = log10(N/N) = 0, effectively removing them from ranking. This is a lightweight alternative to maintaining an explicit stopword list.

### Crawler Design

**Breadth-first traversal** — a FIFO queue ensures shallow pages (the paginated quote listings) are crawled before any deeper links, producing a natural crawl order and making partial crawls useful.

**Single `requests.Session`** — reuses the underlying TCP connection pool across all requests, reducing connection overhead from O(pages) TCP handshakes to approximately O(1).

**URL normalisation** — fragments (`#section`) are stripped before deduplication because `page.html#s1` and `page.html#s2` return identical HTML. Without this, the same content would be indexed multiple times under different keys.

**Politeness window enforcement** — the `Crawler.__init__` method raises `ValueError` if `politeness_window < 6`, making it structurally impossible to misconfigure. The sleep happens at the start of each iteration (before the request), not after, so the window is guaranteed even if processing takes non-trivial time.

**Exponential back-off on retry** — wait time scales as `politeness_window × attempt_number`. This avoids hammering a temporarily overloaded server while still retrying transient failures.

### Separation of Concerns

`search.py` exists as a distinct layer even though it is thin. The reason: `Indexer.find()` returns raw data structures; `SearchEngine.find()` is responsible for formatting, ranking display, and writing to stdout. Keeping I/O out of the indexer means the indexer can be used in non-CLI contexts (e.g. a web API, a test) without capturing or suppressing output.

---

## Installation

**Requirements:** Python 3.10 or later.

```bash
# 1. Clone the repository
git clone https://github.com/CodingOz/web-scraper.git
cd web-scraper

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

Start the interactive shell from the `src/` directory:

```bash
cd src
python main.py
```

Optional flags:

```bash
python main.py --index ../data/index.json   # custom index path (default: data/index.json)
python main.py --url https://quotes.toscrape.com  # target URL (default: quotes.toscrape.com)
python main.py --max-pages 5               # stop crawl after 5 pages (default: unlimited)
python main.py --stem                      # enable Porter stemming for indexing and queries
python main.py --debug                     # enable verbose logging
```

**`--stem` demo:** run `find running` without `--stem` (no results if the page only has "runs"), then restart with `--stem` and run the same query — pages containing "runs", "run", or "running" all match because Porter reduces them to the same stem.

---

## Commands

### `build`

Crawls the target website, builds the inverted index, and saves it to disk.

```
> build
```

The crawl respects a **6-second politeness window** between requests. Crawling the full quotes.toscrape.com site (~11 pages) takes approximately 70 seconds.

---

### `load`

Loads a previously saved index from disk. Use this to skip re-crawling on subsequent sessions.

```
> load
```

Returns an error message (not a crash) if the index file does not exist yet.

---

### `print <word>`

Displays the full inverted index entry for a word: every page it appears on, its frequency, token positions, and TF-IDF weight. Results are sorted by TF-IDF descending.

```
> print nonsense
```

Example output:

```
  Inverted index entry for 'nonsense':
  ────────────────────────────────────────────────────────
  URL                                        freq  tf-idf
  ────────────────────────────────────────────────────────
  https://quotes.toscrape.com/page/3            2    0.3521
    positions: [142, 209]
  https://quotes.toscrape.com/                  1    0.1761
    positions: [87]
  ────────────────────────────────────────────────────────
  Total pages containing 'nonsense': 2
```

---

### `find <term> [<term> …]`

Returns all pages containing **every** query term (AND semantics), ranked by combined TF-IDF score. Works for single and multi-word queries.

```
> find indifference
> find good friends
```

Example output:

```
  Searching for pages containing: 'good' + 'friends'
  ────────────────────────────────────────────────────────────────
  Rank   Score  URL
  ────────────────────────────────────────────────────────────────
  1      0.6140  https://quotes.toscrape.com/page/4
          'good': freq=2, tf-idf=0.3521, positions=[14, 88]
          'friends': freq=1, tf-idf=0.2619, positions=[102]
  ────────────────────────────────────────────────────────────────
  1 page(s) found.
```

**Edge cases handled:**

| Input | Behaviour |
|---|---|
| Word not in index | Prints "not found", returns empty list |
| Empty query | Prints usage hint |
| No pages match all terms | Prints "No pages found" |
| Query terms with mixed case | Normalised to lowercase automatically |

---

### `phrase <term> [<term> …]`

Returns pages where the terms appear **consecutively and in order** — stricter than `find`, which only requires all terms to be present somewhere on the page.

```
> phrase good friends
```

Example: `find good friends` returns every page containing both words anywhere.  
`phrase good friends` returns only pages where "good" is immediately followed by "friends".

```
  Searching for phrase: 'good' 'friends'
  ────────────────────────────────────────────────────────────────
  Rank   Score  URL
  ────────────────────────────────────────────────────────────────
  1      0.6140  https://quotes.toscrape.com/page/4
          'good': freq=2, positions=[14, 88]
          'friends': freq=1, positions=[89]
  ────────────────────────────────────────────────────────────────
  1 page(s) found.
```

---

## Project Structure

```
.
├── .github/
│   └── workflows/
│       └── test.yml         # CI: mypy --strict + pytest on Python 3.11 & 3.12
├── src/
│   ├── crawler.py           # Crawler — BFS crawl, politeness window, robots.txt
│   ├── indexer.py           # Indexer — inverted index, TF-IDF, stemming, phrase search
│   ├── search.py            # SearchEngine — query processing and output formatting
│   └── main.py              # CLI shell (build / load / print / find / phrase)
├── scripts/
│   └── benchmark.py         # Timing table: build, save, load, find, find_phrase
├── tests/
│   ├── test_crawler.py      # 34 tests — init, URL handling, fetch/retry, robots, crawl
│   ├── test_indexer.py      # 62 tests — build, TF-IDF, save/load, find, phrase search
│   ├── test_search.py       # 27 tests — guards, print_word, find, phrase output
│   ├── test_integration.py  # 26 tests — full Crawler → Indexer → SearchEngine pipeline
│   ├── test_property.py     # 16 tests — Hypothesis invariants (freq, positions, TF-IDF)
│   └── test_new_features.py # 35 tests — stemming, phrase adjacency, robots, max-pages
├── data/
│   └── index.json           # Generated by `build` (not committed to git)
├── requirements.txt
└── README.md
```

---

## Testing

Run the full test suite from the repository root:

```bash
pytest tests/ -v
```

Run with coverage report:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

**200 tests across 6 files:**

| File | Tests | Focus |
|---|---|---|
| `test_crawler.py` | 34 | Init, URL handling, fetch/retry, robots.txt, crawl |
| `test_indexer.py` | 62 | Build, TF-IDF, save/load, `find`, phrase search |
| `test_search.py` | 27 | Guards, `print_word`, `find`, `find_phrase` output |
| `test_integration.py` | 26 | Full Crawler → Indexer → SearchEngine pipeline |
| `test_property.py` | 16 | Hypothesis invariants: freq conservation, position structure, TF-IDF non-negativity, `find` subset |
| `test_new_features.py` | 35 | Stemming behaviour, phrase adjacency/ordering, robots `can_fetch` mock, `--max-pages` enforcement |

**Coverage achieved:**

| Module       | Coverage |
|--------------|----------|
| `crawler.py` | 99%      |
| `indexer.py` | 99%      |
| `search.py`  | 100%     |
| `main.py`    | excluded (CLI glue, exercised via integration tests) |

200 tests across 6 files; overall coverage of tested modules: 99.1%.

**Testing strategy:**

- **No real HTTP requests** — all crawler tests mock `requests.Session.get` and `time.sleep` using `unittest.mock`. Tests run in milliseconds and pass offline.
- **Unit, integration, and property-based tests** — unit tests isolate each layer; integration tests wire Crawler → Indexer → SearchEngine end-to-end; Hypothesis property tests define mathematical invariants (e.g. sum of all `freq` values must equal total token count) and generate hundreds of random inputs to falsify them.
- **Direct `can_fetch` mocking** — robots tests mock `RobotFileParser.can_fetch` directly to assert the exact User-Agent and URL arguments passed, not just the outcome.
- **`MagicMock(spec=Indexer)`** — the `spec=` parameter means mocks raise `AttributeError` if code calls a method that doesn't exist on the real class, catching interface mismatches that plain `Mock()` would miss silently.
- **`capsys` for output testing** — pytest's built-in stdout capture verifies the human-readable output of `print_word` and `find` without coupling tests to exact string formatting.

---

## CI / GitHub Actions

Every push triggers the workflow defined in `.github/workflows/test.yml`:

| Step | What it does |
|---|---|
| **mypy --strict** | Type-checks all four source files; fails the build on any error |
| **pytest --cov** | Runs all 200 tests across Python 3.11 and 3.12 |
| **Upload artifact** | Saves `coverage.xml` so coverage history is visible per run |

The matrix runs both Python versions in parallel with `fail-fast: false` so you see all failures at once rather than stopping at the first broken version.

**Tagging a release**:
```bash
git tag -a v1.0.0 -m "Release v1.0.0 — full feature set"
git push origin v1.0.0
```

---

## Dependencies

```
requests>=2.31.0
beautifulsoup4>=4.12.0
pytest>=7.4.0         # testing only
pytest-cov>=4.1.0     # coverage reporting only
```

Install all dependencies:

```bash
pip install -r requirements.txt
```