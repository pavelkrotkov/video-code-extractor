"""Command-line entry point.

Stub: the full ``vce extract <video>`` pipeline is wired up in the "CLI pipeline" issue.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vce", description=__doc__)
    parser.add_argument("--version", action="store_true", help="print version and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.version:
        from vce import __version__

        print(__version__)
        return 0
    print("vce: pipeline not yet wired up — see docs/architecture.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
