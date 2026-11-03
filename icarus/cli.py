"""Command-line entry point: `python -m icarus <command>`.

    ingest     scrape every source into data/raw/
    validate   data-quality gate (known closes, ranges, freshness)
    warehouse  build the DuckDB feature warehouse from sql/*.sql
    models     run every model and write data/processed/results.json
    report     render figures, README results block and dashboard data
    all        warehouse -> models -> report (on existing raw data)
    linkedin   render the 4-slide LinkedIn carousel, PDF, workbook and post text
"""

from __future__ import annotations

import argparse
import logging
import sys
import time


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="icarus", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["ingest", "validate", "warehouse", "models", "report", "all", "linkedin"])
    parser.add_argument("--sources", nargs="*", help="subset of sources for `ingest`")
    parser.add_argument("--fast", action="store_true", help="fewer Monte Carlo reps and windows (for CI)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s", datefmt="%H:%M:%S")
    t0 = time.time()

    if args.command == "ingest":
        from .ingest import run as ingest_run
        ingest_run.run(args.sources)
    elif args.command == "validate":
        from . import validate
        validate.run(strict=True)
    elif args.command in ("warehouse", "all"):
        from .warehouse import build
        build.run()
    if args.command in ("models", "all"):
        from .models import pipeline
        pipeline.run(fast=args.fast)
    if args.command in ("report", "all"):
        from .report import build as report_build
        report_build.run()
    if args.command == "linkedin":
        from .report import linkedin
        linkedin.run()

    logging.getLogger("icarus").info("done in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
