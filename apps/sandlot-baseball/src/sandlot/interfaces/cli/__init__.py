"""Command line entry point.

``sandlot analyze <video>`` and ``sandlot compare <a> <b>``. Arguments are
translated into use-case calls here and nowhere else.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sandlot import __version__

__all__ = ["DEFAULT_DATA_DIR", "main"]

# Sessions land in the user's home rather than the repo, so analysing a video
# does not leave files in a checkout. Every entry point takes --data-dir, and
# tests pass tmp_path, so no test can write here by accident.
DEFAULT_DATA_DIR = Path.home() / ".sandlot" / "sessions"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sandlot", description=__doc__)
    parser.add_argument("--version", action="version", version=f"sandlot {__version__}")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"where sessions are stored (default: {DEFAULT_DATA_DIR})",
    )

    commands = parser.add_subparsers(dest="command")

    analyse = commands.add_parser("analyze", help="measure one video")
    analyse.add_argument("video", type=Path)
    analyse.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="run the analysis N times and report whether the numbers agree",
    )

    compare = commands.add_parser("compare", help="difference between two sessions")
    compare.add_argument("session_a")
    compare.add_argument("session_b")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    # Task 5 wires these to the use cases. Failing loudly beats a command
    # that appears to succeed while measuring nothing.
    print(f"'{args.command}' is not implemented yet — see docs/plans/M3-sandlot-skeleton.md")
    return 2


if __name__ == "__main__":
    sys.exit(main())
