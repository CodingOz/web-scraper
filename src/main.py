"""
main.py
-------
Command-line interface (REPL shell) for the COMP3011 Search Engine Tool.

Supported commands
------------------
    > build             Crawl the target website and build the inverted index.
    > load              Load a previously built index from disk.
    > print <word>      Print the inverted index entry for <word>.
    > find <terms…>     Find all pages containing every term (AND semantics),
                        ranked by TF-IDF score.
    > help              Display available commands.
    > quit | exit       Exit the shell.

Usage
-----
    python main.py

Optional flags
--------------
    --index   Path to the index file (default: data/index.json)
    --url     Target URL to crawl (default: https://quotes.toscrape.com)
    --debug   Enable DEBUG-level logging
"""

import argparse
import logging
import sys

from crawler import Crawler
from indexer import Indexer
from search import SearchEngine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_INDEX_PATH: str = "data/index.json"
DEFAULT_TARGET_URL: str = "https://quotes.toscrape.com"
PROMPT: str = "> "

HELP_TEXT: str = """
  Available commands:
  ───────────────────────────────────────────────────────
  build               Crawl the website and build the index
  load                Load a previously saved index from disk
  print <word>        Show inverted index entry for <word>
  find <term> …       Find pages containing ALL given terms
  help                Show this help message
  quit / exit         Exit the search tool
  ───────────────────────────────────────────────────────
"""


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

def run_shell(index_path: str, target_url: str, max_pages: int | None = None) -> None:
    """
    Start the interactive command-line shell.

    The shell loops indefinitely, reading one command per iteration from
    stdin, dispatching to the appropriate handler, and printing results.
    It exits cleanly on ``quit``, ``exit``, or EOF (Ctrl-D / Ctrl-Z).

    Parameters
    ----------
    index_path : str
        File system path where the index is saved / loaded.
    target_url : str
        Root URL to crawl when the ``build`` command is issued.
    """
    indexer = Indexer()
    engine = SearchEngine(indexer)

    print("  COMP3011 Search Engine Tool")
    limit = str(max_pages) if max_pages is not None else "unlimited"
    print(f"  Target: {target_url}  |  Index: {index_path}  |  Max pages: {limit}")
    print("  Type 'help' for available commands.\n")

    while True:
        try:
            raw = input(PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye.")
            sys.exit(0)

        if not raw:
            continue

        parts = raw.split()
        command = parts[0].lower()
        args = parts[1:]

        if command == "build":
            _cmd_build(indexer, target_url, index_path, max_pages)

        elif command == "load":
            _cmd_load(indexer, index_path)

        elif command == "print":
            _cmd_print(engine, args)

        elif command == "find":
            _cmd_find(engine, args)

        elif command == "help":
            print(HELP_TEXT)

        elif command in ("quit", "exit"):
            print("  Goodbye.")
            sys.exit(0)

        else:
            print(f"  Unknown command: '{command}'.  Type 'help' for options.")


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _cmd_build(indexer: Indexer, target_url: str, index_path: str, max_pages: int | None = None) -> None:
    """
    Handle the ``build`` command.

    Crawls *target_url*, builds the inverted index, and saves it to
    *index_path*.  Prints progress and a summary on completion.

    Parameters
    ----------
    indexer : Indexer
        The shared Indexer instance to populate.
    target_url : str
        Root URL to begin crawling from.
    index_path : str
        Destination path for the saved index file.
    """
    limit_str = str(max_pages) if max_pages is not None else "unlimited"
    print(f"\n  Starting crawl of {target_url} (max pages: {limit_str}) ...")
    print("  (Politeness window: 6 s between requests - this may take a few minutes)\n")

    try:
        crawler = Crawler(target_url, max_pages=max_pages)
        pages = crawler.crawl()
    except Exception as exc:
        print(f"  [!] Crawl failed: {exc}\n")
        return

    if not pages:
        print("  [!] No pages were fetched. Check the URL and your connection.\n")
        return

    print(f"  Crawl complete - {len(pages)} page(s) fetched.")
    print("  Building inverted index ...")

    try:
        indexer.build(pages)
    except Exception as exc:
        print(f"  [!] Index build failed: {exc}\n")
        return

    print(f"  Index built - {len(indexer.index)} unique terms.")

    try:
        indexer.save(index_path)
        print(f"  Index saved to '{index_path}'.\n")
    except Exception as exc:
        print(f"  [!] Failed to save index: {exc}\n")


def _cmd_load(indexer: Indexer, index_path: str) -> None:
    """
    Handle the ``load`` command.

    Parameters
    ----------
    indexer : Indexer
        The shared Indexer instance to populate.
    index_path : str
        Path to a previously saved index JSON file.
    """
    print(f"\n  Loading index from '{index_path}' ...")

    try:
        indexer.load(index_path)
        print(f"  Index loaded - {len(indexer.index)} unique terms.\n")
    except FileNotFoundError:
        print(
            f"  [!] Index file not found: '{index_path}'.\n"
            f"      Run 'build' first to create it.\n"
        )
    except Exception as exc:
        print(f"  [!] Failed to load index: {exc}\n")


def _cmd_print(engine: SearchEngine, args: list[str]) -> None:
    """
    Handle the ``print <word>`` command.

    Parameters
    ----------
    engine : SearchEngine
        The active SearchEngine instance.
    args : list[str]
        Tokenised arguments following the ``print`` keyword.
    """
    if not args:
        print("  Usage: print <word>\n")
        return

    word = args[0]

    try:
        engine.print_word(word)
    except RuntimeError as exc:
        print(f"  [!] {exc}\n")


def _cmd_find(engine: SearchEngine, args: list[str]) -> None:
    """
    Handle the ``find <term> [<term> ...]`` command.

    Parameters
    ----------
    engine : SearchEngine
        The active SearchEngine instance.
    args : list[str]
        Tokenised query terms following the ``find`` keyword.
    """
    if not args:
        print("  Usage: find <term> [<term> ...]\n")
        return

    try:
        engine.find(args)
    except RuntimeError as exc:
        print(f"  [!] {exc}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="COMP3011 Search Engine Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--index",
        default=DEFAULT_INDEX_PATH,
        help="Path to the index file",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_TARGET_URL,
        help="Target URL to crawl with the 'build' command",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        dest="max_pages",
        help="Maximum number of pages to crawl (default: unlimited)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_shell(index_path=args.index, target_url=args.url, max_pages=args.max_pages)