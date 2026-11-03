"""The training panel: ~100 assets x up to 99 years of month-ends, with features
and forward-looking labels for "bear" (-20%) and "burst" (-40%) events.

Labels are built for horizons k = 1..24 months. A label is *known* only if the
event happened within k months, or the series has at least k months of
history after the sample date. Everything else is censored and dropped, which
is the discrete-time survival setup, so nothing peeks into the future.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .. import config
from ..warehouse.build import query
from . import psy

log = logging.getLogger(__name__)

FEATURES = [
    "r_1m", "r_3m", "r_6m", "r_12m", "r_24m", "runup_24m", "runup_z", "accel_12m", "accel_z",
    "dd", "episode_worst_dd", "dd_12m_high", "dist_ma200", "vol_3m", "vol_12m", "vol_5y", "vol_ratio",
    "skew_12m", "up_share_12m", "turnover_trend", "bsadf_excess",
]
FEATURE_LABELS = {
    "r_1m": "1-month return", "r_3m": "3-month return", "r_6m": "6-month return",
    "r_12m": "12-month return", "r_24m": "24-month return", "runup_24m": "Run-up from 2-year low",
    "runup_z": "2-year run-up in volatility units", "accel_12m": "Acceleration (this year vs last)",
    "accel_z": "Acceleration in volatility units", "dd": "Drawdown from all-time high",
    "episode_worst_dd": "Worst drawdown since the last peak",
    "dd_12m_high": "Drawdown from 12-month high", "dist_ma200": "Distance from 200-day average",
    "vol_3m": "3-month volatility", "vol_12m": "12-month volatility", "vol_5y": "5-year volatility",
    "vol_ratio": "Volatility ratio (3m / 12m)", "skew_12m": "Return skewness (12m)",
    "up_share_12m": "Share of up days (12m)", "turnover_trend": "Dollar-volume trend",
    "bsadf_excess": "PSY explosiveness (BSADF - 95% cv)", "horizon": "Forecast horizon (months)",
}
HORIZONS = [1, 2, 3, 6, 9, 12, 18, 24]


def rolling_bsadf_excess(monthly: pd.DataFrame, window: int, lags: int, reps: int, seed: int) -> pd.Series:
    """BSADF minus its 95% critical value, using at most `window` months of
    history, for every (ticker, month). One set of Monte Carlo critical values
    serves every series, since the rolling window makes the null comparable."""
    w0 = psy.min_window(window)
    cvs = psy.critical_values(window, lags=lags, w0=w0, wmax=window, reps=reps, seed=seed)
    cv95 = cvs[0.95]
    out = []
    for ticker, g in monthly.groupby("ticker", sort=False):
        y = np.log(g["price"].to_numpy(dtype=float))
        stat = psy.bsadf(y, lags=lags, w0=w0, wmax=window)
        idx = np.arange(len(y))
        cv = np.where(idx < len(cv95), cv95[np.minimum(idx, len(cv95) - 1)], cv95[-1])
        out.append(pd.Series(stat - cv, index=g.index))
    return pd.concat(out)


def load_monthly(fast: bool = False) -> pd.DataFrame:
    s = config.settings()
    df = query("SELECT * FROM features_monthly ORDER BY ticker, date")
    df["date"] = pd.to_datetime(df["date"])
    for c in ("next_bear_date", "next_burst_date", "last_date", "episode_peak_date"):
        df[c] = pd.to_datetime(df[c])
    # Volatility-normalised run-up and acceleration make a 100% run-up in bitcoin
    # comparable with a 40% run-up in the S&P 500.
    df["runup_z"] = np.log1p(df["runup_24m"]) / (df["vol_5y"] * np.sqrt(2))
    df["accel_z"] = df["accel_12m"] / df["vol_5y"]
    p = s["psy"]
    reps = 200 if fast else p["mc_reps"]
    df["bsadf_excess"] = rolling_bsadf_excess(df, p["rolling_window_months"], p["lags"], reps, p["seed"])
    return df


def build_labels(monthly: pd.DataFrame, event: str, horizons: list[int] = HORIZONS) -> pd.DataFrame:
    """Stack (sample, horizon) rows with a known 0/1 label.

    event: 'bear' or 'burst'. Rows whose current drawdown episode has already
    crossed the threshold are dropped: an asset that fell 45% from its peak and
    bounced to -35% has already burst, even though it is now above -40%.
    """
    ev = config.settings()["events"]
    thr = ev["bear_drawdown"] if event == "bear" else ev["burst_drawdown"]
    col = "next_bear_date" if event == "bear" else "next_burst_date"
    # Keep only samples whose current drawdown episode has NOT yet crossed the
    # threshold: the event must still be ahead of us.
    base = monthly[(monthly["episode_worst_dd"] > -thr) & (monthly["history_years"] >= 2.5)].copy()
    base = base.dropna(subset=["r_24m", "vol_5y", "vol_12m"])
    frames = []
    for k in horizons:
        horizon_end = base["date"] + pd.DateOffset(months=k)
        hit = base[col].notna() & (base[col] <= horizon_end)
        known = hit | (horizon_end <= base["last_date"])
        f = base[known].copy()
        f["horizon"] = k
        f["y"] = hit[known].astype(int)
        f["label_end"] = horizon_end[known]
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    log.info("%s labels: %d stacked rows, %d series, positive rate %.3f",
             event, len(out), out["ticker"].nunique(), out["y"].mean())
    return out


def training_universe() -> list[str]:
    """Panel assets; the AI-20 index itself is excluded from training (it's the
    target, and its long history is a hindsight-picked basket)."""
    return config.panel_tickers()
