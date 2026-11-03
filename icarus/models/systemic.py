"""Systemic fragility: how tightly coupled is the market?

* Absorption ratio (Kritzman, Li, Page & Rigobon 2011): the share of total
  variance explained by the top principal components of asset returns. When it
  is high, risks are concentrated and shocks spread fast. A sharp rise in it
  (the standardised shift) has historically come before large drawdowns.
* AI-20 average pairwise correlation: a crowded trade moves as one block.
* Breadth (from the SQL warehouse): how many AI-20 members are already in bear
  markets while the index sits near its highs.
* Concentration: equal-weight vs cap-weight S&P 500.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..warehouse.build import query


def _returns_wide(tickers: list[str], start: str) -> pd.DataFrame:
    tl = ",".join(f"'{t}'" for t in tickers)
    d = query(f"SELECT date, ticker, r FROM returns WHERE ticker IN ({tl}) AND date >= '{start}'")
    d["date"] = pd.to_datetime(d["date"])
    return d.pivot(index="date", columns="ticker", values="r").sort_index()


def absorption_ratio(R: pd.DataFrame, window: int = 500, frac: float = 0.2, step: int = 5,
                     min_assets: int = 5) -> pd.DataFrame:
    """Rolling absorption ratio on exponentially weighted covariance (half-life
    of 250 days, as in Kritzman et al.). Computed every `step` days."""
    out = []
    lam = 0.5 ** (1 / 250)
    for end in range(window, len(R) + 1, step):
        win = R.iloc[end - window:end]
        win = win.loc[:, win.notna().mean() > 0.95].dropna()
        if win.shape[1] < min_assets or len(win) < window * 0.8:
            continue
        w = lam ** np.arange(len(win))[::-1]
        w /= w.sum()
        X = win.to_numpy() - (w[:, None] * win.to_numpy()).sum(0)
        cov = (X * w[:, None]).T @ X
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
        n = max(1, int(round(frac * len(ev))))
        out.append({"date": win.index[-1], "ar": float(ev[:n].sum() / ev.sum()), "n_assets": win.shape[1]})
    df = pd.DataFrame(out).set_index("date")
    # Standardised shift: 15-day vs 1-year average, in units of the 1-year stdev.
    per_year = max(1, 252 // step)
    short = df["ar"].rolling(max(1, 15 // step)).mean()
    long = df["ar"].rolling(per_year).mean()
    sd = df["ar"].rolling(per_year).std()
    df["delta_ar"] = (short - long) / sd
    return df


def avg_pairwise_corr(R: pd.DataFrame, window: int = 63, step: int = 5) -> pd.Series:
    vals = {}
    for end in range(window, len(R) + 1, step):
        win = R.iloc[end - window:end]
        win = win.loc[:, win.notna().mean() > 0.9]
        if win.shape[1] < 5:
            continue
        c = win.corr().to_numpy()
        iu = np.triu_indices_from(c, 1)
        vals[win.index[-1]] = float(np.nanmean(c[iu]))
    return pd.Series(vals, name="avg_corr")


def analyse() -> dict:
    sectors = config.universe()["sectors"]
    R_sec = _returns_wide(sectors, "1998-12-01")
    ar = absorption_ratio(R_sec)
    R_ai = _returns_wide(config.ai20_tickers(), "2014-01-01")
    corr = avg_pairwise_corr(R_ai)
    breadth = query("SELECT * FROM ai20_breadth ORDER BY date")
    breadth["date"] = pd.to_datetime(breadth["date"])
    conc = query("SELECT * FROM concentration ORDER BY date")
    conc["date"] = pd.to_datetime(conc["date"])

    def pct(series: pd.Series, value: float) -> float:
        s = series.dropna()
        return float((s <= value).mean()) if len(s) else float("nan")

    last_b = breadth.iloc[-1]
    return {
        "absorption": ar, "ai20_corr": corr, "breadth": breadth, "concentration": conc,
        "summary": {
            "absorption_ratio": float(ar["ar"].iloc[-1]),
            "absorption_ratio_pctile": pct(ar["ar"], ar["ar"].iloc[-1]),
            "delta_ar": float(ar["delta_ar"].iloc[-1]),
            "ai20_avg_corr": float(corr.iloc[-1]),
            "ai20_avg_corr_pctile": pct(corr, corr.iloc[-1]),
            "breadth_date": str(last_b["date"].date()),
            "ai20_share_in_bear": float(last_b["share_in_bear"]),
            "ai20_n_members": int(last_b["n_members"]),
            "ai20_n_in_bear": int(round(last_b["share_in_bear"] * last_b["n_members"])),
            "ai20_share_burst": float(last_b["share_burst"]),
            "ai20_share_above_ma200": float(last_b["share_above_ma200"]),
            "ai20_median_dd": float(last_b["median_dd_from_ath"]),
            "rsp_spy": float(conc["rsp_spy"].iloc[-1]),
            # A low equal/cap ratio means a narrow, top-heavy market: score the
            # inverse percentile so that "more concentrated" = higher risk.
            "concentration_pctile": 1 - pct(conc["rsp_spy"], conc["rsp_spy"].iloc[-1]),
        },
    }
