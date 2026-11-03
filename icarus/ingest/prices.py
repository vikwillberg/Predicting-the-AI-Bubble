"""Daily OHLCV prices from Yahoo Finance (via yfinance), with a Stooq fallback.

Output schema (long format, one row per ticker per day):
    date, ticker, open, high, low, close, adj_close, volume, split_factor, source

`close` is split-adjusted; `adj_close` is split- and dividend-adjusted (total
return). `split_factor` is the split ratio effective that day (1.0 if none).
"""

from __future__ import annotations

import io
import logging
import time

import pandas as pd

from .http import session

log = logging.getLogger(__name__)

COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "split_factor", "source"]


def _normalise_yahoo(hist: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if hist is None or hist.empty:
        return pd.DataFrame(columns=COLUMNS)
    idx = hist.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(idx.normalize()),
            "ticker": ticker,
            "open": hist.get("Open"),
            "high": hist.get("High"),
            "low": hist.get("Low"),
            "close": hist.get("Close"),
            "adj_close": hist.get("Adj Close", hist.get("Close")),
            "volume": hist.get("Volume"),
            "split_factor": hist.get("Stock Splits", 0.0),
            "source": "yahoo",
        }
    ).reset_index(drop=True)
    df["split_factor"] = df["split_factor"].fillna(0.0).replace(0.0, 1.0)
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    # Yahoo occasionally emits two rows for the same session (e.g. a partial
    # intraday row). Keep the last one.
    return df.drop_duplicates(subset=["date"], keep="last")[COLUMNS]


def fetch_yahoo(ticker: str, retries: int = 4) -> pd.DataFrame:
    import yfinance as yf

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            hist = yf.Ticker(ticker).history(period="max", auto_adjust=False, actions=True)
            df = _normalise_yahoo(hist, ticker)
            if not df.empty:
                return df
            last_err = RuntimeError("empty response")
        except Exception as exc:  # yfinance raises a zoo of exception types
            last_err = exc
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Yahoo download failed for {ticker}: {last_err}")


def _stooq_symbol(ticker: str) -> str | None:
    if ticker.startswith("^") or "=" in ticker or "-" in ticker or "." in ticker:
        return None
    return f"{ticker.lower()}.us"


def fetch_stooq(ticker: str) -> pd.DataFrame:
    """Fallback for US equities. Stooq prices are split- and dividend-adjusted."""
    sym = _stooq_symbol(ticker)
    if sym is None:
        raise RuntimeError(f"No Stooq mapping for {ticker}")
    resp = session().get(f"https://stooq.com/q/d/l/?s={sym}&i=d", timeout=60)
    resp.raise_for_status()
    raw = pd.read_csv(io.StringIO(resp.text))
    if raw.empty or "Close" not in raw.columns:
        raise RuntimeError(f"Stooq returned no data for {ticker}")
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["Date"]),
            "ticker": ticker,
            "open": raw["Open"],
            "high": raw["High"],
            "low": raw["Low"],
            "close": raw["Close"],
            "adj_close": raw["Close"],
            "volume": raw.get("Volume"),
            "split_factor": 1.0,
            "source": "stooq",
        }
    )
    return df[df["close"] > 0][COLUMNS]


def fetch_many(tickers: list[str], pause: float = 0.6) -> tuple[pd.DataFrame, dict[str, str]]:
    frames, failures = [], {}
    for i, t in enumerate(tickers, 1):
        try:
            df = fetch_yahoo(t)
        except Exception as exc:
            log.warning("yahoo failed for %s (%s); trying stooq", t, exc)
            try:
                df = fetch_stooq(t)
            except Exception as exc2:
                failures[t] = f"yahoo: {exc}; stooq: {exc2}"
                log.error("all sources failed for %s", t)
                continue
        frames.append(df)
        log.info("[%d/%d] %s: %d rows (%s to %s)", i, len(tickers), t, len(df),
                 df["date"].min().date(), df["date"].max().date())
        time.sleep(pause)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    return out, failures


VALUATION_FIELDS = [
    "marketCap", "enterpriseValue", "trailingPE", "forwardPE", "priceToSalesTrailing12Months",
    "enterpriseToRevenue", "enterpriseToEbitda", "totalRevenue", "revenueGrowth", "grossMargins",
    "operatingMargins", "freeCashflow", "sharesOutstanding", "beta",
]


def fetch_valuation_snapshot(tickers: list[str], pause: float = 0.6) -> pd.DataFrame:
    """Point-in-time valuation multiples. Appended on every run so that a history
    of the AI complex's multiples accumulates over time."""
    import yfinance as yf

    rows = []
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    for t in tickers:
        try:
            info = yf.Ticker(t).info or {}
        except Exception as exc:
            log.warning("valuation snapshot failed for %s: %s", t, exc)
            continue
        row = {"date": today, "ticker": t}
        for f in VALUATION_FIELDS:
            v = info.get(f)
            row[f] = float(v) if isinstance(v, (int, float)) else None
        rows.append(row)
        time.sleep(pause)
    return pd.DataFrame(rows)
