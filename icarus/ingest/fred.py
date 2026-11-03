"""Macro series from FRED's public CSV endpoint (no API key required)."""

from __future__ import annotations

import io
import logging
import time

import pandas as pd

from .http import session

log = logging.getLogger(__name__)
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


def parse_fred_csv(text: str, series_id: str) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text))
    date_col = "observation_date" if "observation_date" in raw.columns else raw.columns[0]
    val_col = series_id if series_id in raw.columns else raw.columns[-1]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col]),
            "series_id": series_id,
            # FRED marks missing observations with "."
            "value": pd.to_numeric(raw[val_col], errors="coerce"),
        }
    )
    return df.dropna(subset=["value"])


def fetch_series(series_id: str) -> pd.DataFrame:
    resp = session().get(URL.format(sid=series_id), timeout=60)
    resp.raise_for_status()
    return parse_fred_csv(resp.text, series_id)


def fetch_many(series: dict[str, str]) -> tuple[pd.DataFrame, dict[str, str]]:
    frames, failures = [], {}
    for sid in series:
        try:
            df = fetch_series(sid)
            frames.append(df)
            log.info("FRED %s: %d obs (%s to %s)", sid, len(df), df.date.min().date(), df.date.max().date())
        except Exception as exc:
            failures[sid] = str(exc)
            log.error("FRED %s failed: %s", sid, exc)
        time.sleep(0.3)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "series_id", "value"])
    return out, failures
