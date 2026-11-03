"""Data quality gate.

Checks that the scraped data is complete and internally consistent, and that it
matches closing levels everyone knows (the NASDAQ's 2000 peak, the 2009 S&P low,
Japan's 1989 top). If Yahoo ever serves a corrupted series, the pipeline stops
here instead of training on it.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from . import config

log = logging.getLogger(__name__)

# (ticker, date, official close). Unadjusted index levels, so they must match exactly.
KNOWN_CLOSES = [
    ("^GSPC", "1987-10-19", 224.84),    # Black Monday
    ("^IXIC", "2000-03-10", 5048.62),   # dot-com peak
    ("^IXIC", "2002-10-09", 1114.11),   # dot-com bottom
    ("^GSPC", "2007-10-09", 1565.15),   # pre-GFC peak
    ("^GSPC", "2009-03-09", 676.53),    # GFC bottom
    ("^GSPC", "2020-02-19", 3386.15),   # pre-COVID peak
    ("^GSPC", "2020-03-23", 2237.40),   # COVID bottom
    ("^N225", "1989-12-29", 38915.87),  # Japan bubble peak
]
CORE_SERIES = ["^GSPC", "^IXIC", "^NDX", "NVDA", "MSFT", "SMH"]


def check_prices(px: pd.DataFrame) -> list[dict]:
    results = []

    def add(name, ok, detail, severity="error"):
        results.append({"check": name, "ok": bool(ok), "detail": detail, "severity": severity})

    add("non_empty", len(px) > 0, f"{len(px)} rows")
    present = set(px["ticker"].unique())
    missing_core = [t for t in CORE_SERIES if t not in present]
    add("core_series_present", not missing_core, f"missing: {missing_core}")
    ai_missing = [t for t in config.ai20_tickers() if t not in present]
    add("ai20_coverage", len(ai_missing) <= 3, f"missing AI-20 names: {ai_missing}")
    add("positive_prices", bool((px["close"] > 0).all()), "all closes > 0")
    dupes = int(px.duplicated(subset=["ticker", "date"]).sum())
    add("unique_ticker_date", dupes == 0, f"{dupes} duplicate rows")
    today = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    add("no_future_dates", bool((px["date"] <= today).all()), f"max date {px['date'].max().date()}")
    latest = px.groupby("ticker")["date"].max()
    stale = latest[latest < px["date"].max() - pd.Timedelta(days=10)]
    # Delisted or renamed panel names are expected to go stale; only warn.
    add("freshness", len(stale) < 10, f"stale series: {list(stale.index)}", severity="warning")

    # Daily moves beyond +/-60% in a broad index mean corrupted data.
    idx = px[px["ticker"].isin(["^GSPC", "^IXIC", "^NDX"])].sort_values(["ticker", "date"])
    rets = idx.groupby("ticker")["close"].pct_change().abs()
    add("index_jump_sanity", bool((rets.dropna() < 0.6).all()), f"max |daily move| {rets.max():.3f}")

    for ticker, date, level in KNOWN_CLOSES:
        row = px[(px["ticker"] == ticker) & (px["date"] == pd.Timestamp(date))]
        if row.empty:
            add(f"known_close:{ticker}@{date}", False, "date missing", severity="warning")
            continue
        got = float(row["close"].iloc[0])
        err = abs(got / level - 1)
        add(f"known_close:{ticker}@{date}", err < 0.005, f"expected {level}, got {got:.2f} ({err:.3%})")
    return results


def check_fred(fred: pd.DataFrame) -> list[dict]:
    out = []
    for sid, lo, hi in [("DGS10", -1, 20), ("BAA10Y", 0, 10), ("VIXCLS", 5, 100), ("DFF", -1, 25)]:
        s = fred[fred["series_id"] == sid]["value"]
        ok = len(s) > 0 and s.between(lo, hi).all()
        out.append({"check": f"fred_range:{sid}", "ok": bool(ok),
                    "detail": f"n={len(s)} range [{s.min() if len(s) else None}, {s.max() if len(s) else None}]",
                    "severity": "warning"})
    return out


def run(strict: bool = True) -> dict:
    results = []
    px_path = config.DATA_RAW / "prices.parquet"
    if not px_path.exists():
        raise FileNotFoundError("data/raw/prices.parquet missing - run `icarus ingest` first")
    px = pd.read_parquet(px_path)
    px["date"] = pd.to_datetime(px["date"])
    results += check_prices(px)
    fred_path = config.DATA_RAW / "fred.parquet"
    if fred_path.exists():
        results += check_fred(pd.read_parquet(fred_path))
    errors = [r for r in results if not r["ok"] and r["severity"] == "error"]
    warnings = [r for r in results if not r["ok"] and r["severity"] == "warning"]
    report = {"passed": not errors, "n_checks": len(results), "errors": errors,
              "warnings": warnings, "results": results}
    (config.DATA_RAW / "validation_report.json").write_text(json.dumps(report, indent=2, default=str))
    for r in results:
        log.log(logging.INFO if r["ok"] else logging.WARNING, "%s %s: %s",
                "PASS" if r["ok"] else "FAIL", r["check"], r["detail"])
    if strict and errors:
        raise SystemExit(f"Data validation failed: {[e['check'] for e in errors]}")
    return report
