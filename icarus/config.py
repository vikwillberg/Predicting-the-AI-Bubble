"""Paths and configuration loading."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
WAREHOUSE = DATA_PROCESSED / "icarus.duckdb"
SQL_DIR = ROOT / "sql"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
DOCS = ROOT / "docs"
ENGINE_BIN = ROOT / "bin" / "lppls"


@lru_cache(maxsize=None)
def load(name: str) -> dict:
    """Load config/<name>.yaml."""
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def universe() -> dict:
    return load("universe")


def settings() -> dict:
    return load("settings")


def ai20_tickers() -> list[str]:
    return [row["ticker"] for row in universe()["ai20"]]


def panel_tickers() -> list[str]:
    groups = universe()["panel"]
    out: list[str] = []
    for tickers in groups.values():
        out.extend(tickers)
    return list(dict.fromkeys(out))


def all_price_tickers() -> list[str]:
    u = universe()
    tickers = ai20_tickers() + u["benchmarks"] + u["sectors"] + panel_tickers()
    return list(dict.fromkeys(tickers))


def ensure_dirs() -> None:
    for d in (DATA_RAW, DATA_PROCESSED, FIGURES, DOCS / "data", DOCS / "assets"):
        d.mkdir(parents=True, exist_ok=True)
