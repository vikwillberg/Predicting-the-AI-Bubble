"""Run every scraper, write Parquet snapshots to data/raw/, and record a manifest."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone

import pandas as pd

from .. import config
from . import attention, edgar, fred, prices

log = logging.getLogger(__name__)

PRICE_COLS = ["date", "ticker", "close", "adj_close", "volume", "split_factor", "source"]


def _write(df: pd.DataFrame, name: str) -> dict:
    path = config.DATA_RAW / f"{name}.parquet"
    df.to_parquet(path, index=False, compression="zstd")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    date_col = "date" if "date" in df.columns else ("end" if "end" in df.columns else None)
    meta = {"rows": int(len(df)), "sha256_16": digest}
    if date_col and len(df):
        meta["min_date"] = str(pd.to_datetime(df[date_col]).min().date())
        meta["max_date"] = str(pd.to_datetime(df[date_col]).max().date())
    return meta


def run(sources: list[str] | None = None) -> dict:
    config.ensure_dirs()
    u = config.universe()
    sources = sources or ["prices", "valuations", "fred", "edgar", "wikipedia", "gdelt", "multpl"]
    manifest_path = config.DATA_RAW / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["run_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest.setdefault("sources", {})

    def record(name, df, failures, **extra):
        meta = _write(df, name) if df is not None and len(df) else {"rows": 0}
        meta["failures"] = failures
        meta.update(extra)
        manifest["sources"][name] = meta

    if "prices" in sources:
        df, fails = prices.fetch_many(config.all_price_tickers())
        record("prices", df[PRICE_COLS] if len(df) else df, fails,
               tickers=int(df["ticker"].nunique()) if len(df) else 0)

    if "valuations" in sources:
        snap = prices.fetch_valuation_snapshot(config.ai20_tickers())
        path = config.DATA_RAW / "valuation_snapshots.parquet"
        if path.exists() and len(snap):
            snap = pd.concat([pd.read_parquet(path), snap], ignore_index=True)
            snap = snap.drop_duplicates(subset=["date", "ticker"], keep="last")
        record("valuation_snapshots", snap, {})

    if "fred" in sources:
        df, fails = fred.fetch_many(u["fred"])
        record("fred", df, fails)

    if "edgar" in sources:
        df, fails = edgar.fetch_many(u["edgar"]["tickers"], u["edgar"]["concepts"])
        record("edgar_facts", df, fails)
        # Yahoo statements: always collected (they cover the newest quarter fastest)
        # and the only fundamentals source when SEC rejects the request.
        df, fails = prices.fetch_yahoo_fundamentals(u["edgar"]["tickers"])
        record("yahoo_fundamentals", df, fails)

    if "wikipedia" in sources:
        df, fails = attention.fetch_pageviews(u["wikipedia"])
        record("wiki_pageviews", df, fails)

    if "gdelt" in sources:
        df, fails = attention.fetch_gdelt(u["gdelt"])
        record("gdelt", df, fails)

    if "multpl" in sources:
        df, fails = attention.fetch_multpl()
        record("multpl", df, fails)

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest
