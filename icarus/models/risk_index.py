"""ICARUS Bubble Risk Index: seven perspectives, one 0-100 gauge.

Each pillar is scored 0-100, where higher means more bubble-like or more
fragile. Wherever a long history exists the score is a percentile against that
history, so "90" means "higher than 90% of all past readings". Pillars:

  price_dynamics    run-up in volatility units, PSY explosiveness, LPPLS confidence
  valuation         Shiller CAPE and S&P 500 P/E percentiles (1871+)
  fundamentals      hyperscaler capex vs operating cash flow
  market_structure  AI-20 breadth, S&P concentration, AI-20 correlation, absorption ratio
  credit_macro      3-month change in credit spreads and financial conditions; real yields
  sentiment         public attention to AI and to bubbles (Wikipedia, GDELT)
  issuance          mega-IPO dollars raised in the last 12 months

Because pillar weights are a judgment call, the composite is also reported
under 5,000 random Dirichlet weightings; the conclusion should not depend on
any single choice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..warehouse.build import query


def pct_rank(history: pd.Series, value: float) -> float:
    h = pd.Series(history).dropna()
    if not len(h) or value is None or not np.isfinite(value):
        return float("nan")
    return float(100 * (h <= value).mean())


def _macro() -> pd.DataFrame:
    m = query("SELECT * FROM macro_monthly ORDER BY month")
    m["month"] = pd.to_datetime(m["month"])
    return m.set_index("month")


def price_dynamics(panel_monthly: pd.DataFrame, lppls_conf: float | None) -> dict:
    ai = panel_monthly[panel_monthly["ticker"] == "AI20"].sort_values("date")
    ref = panel_monthly[panel_monthly["ticker"] != "AI20"]
    last = ai.iloc[-1]
    recent = ai[ai["date"] > last["date"] - pd.DateOffset(months=12)]
    comps = {
        "runup_z_pctile": pct_rank(ref["runup_z"], last["runup_z"]),
        "max_bsadf_excess_12m_pctile": pct_rank(ref["bsadf_excess"], recent["bsadf_excess"].max()),
        "lppls_confidence_x100": 100 * lppls_conf if lppls_conf is not None else float("nan"),
    }
    return {"score": float(np.nanmean(list(comps.values()))), "components": comps}


def valuation() -> dict:
    c = query("SELECT * FROM cape_monthly ORDER BY month")
    cape, pe = c["cape"].dropna(), c["pe"].dropna()
    comps = {"cape": float(cape.iloc[-1]), "cape_pctile": pct_rank(cape, cape.iloc[-1]),
             "pe": float(pe.iloc[-1]), "pe_pctile": pct_rank(pe, pe.iloc[-1]),
             "cape_dotcom_peak": float(cape.max())}
    return {"score": float(np.nanmean([comps["cape_pctile"], comps["pe_pctile"]])), "components": comps}


def fundamentals(fund: dict) -> dict:
    cap = fund["capex"]
    base = cap.get("baseline_capex_to_ocf") or 0.4
    ratio = cap["latest"]["capex_to_ocf"]
    # 0 at the pre-boom baseline, 100 when capex consumes all operating cash flow.
    strain = float(np.clip((ratio - base) / (1.0 - base), 0, 1) * 100)
    share = 100 * cap["latest"]["companies_spending_more_than_ocf"] / cap["latest"]["companies"]
    nv = fund.get("nvidia", {})
    gap = float(np.clip(1 - nv.get("coverage_ratio", np.nan), 0, 1) * 100) if nv else float("nan")
    comps = {"capex_to_ocf": ratio, "capex_strain": strain, "share_capex_above_ocf": share,
             "ai_revenue_gap": gap}
    return {"score": float(np.nanmean([strain, share, gap])), "components": comps}


def market_structure(sys: dict) -> dict:
    s = sys["summary"]
    comps = {"ai20_share_in_bear_x100": 100 * s["ai20_share_in_bear"],
             "concentration_pctile": 100 * s["concentration_pctile"],
             "ai20_corr_pctile": 100 * s["ai20_avg_corr_pctile"],
             "absorption_ratio_pctile": 100 * s["absorption_ratio_pctile"]}
    return {"score": float(np.nanmean(list(comps.values()))), "components": comps}


def credit_macro() -> dict:
    m = _macro()
    comps = {}
    for col, label in (("BAA10Y", "baa_spread_3m_change_pctile"), ("BAMLH0A0HYM2", "hy_oas_3m_change_pctile"),
                       ("NFCI", "nfci_3m_change_pctile")):
        if col in m:
            d = m[col].dropna().diff(3)
            comps[label] = pct_rank(d, d.iloc[-1]) if len(d) else float("nan")
    if "DFII10" in m:
        r = m["DFII10"].dropna()
        comps["real_yield_pctile"] = pct_rank(r, r.iloc[-1])
    latest = {c: float(m[c].dropna().iloc[-1]) for c in ("BAA10Y", "BAMLH0A0HYM2", "NFCI", "DFII10", "T10Y3M", "DFF")
              if c in m and m[c].notna().any()}
    return {"score": float(np.nanmean(list(comps.values()))), "components": comps, "latest_levels": latest}


def sentiment() -> dict:
    """AI hype, scored; bubble fear, reported as context only.

    Hype (attention to AI, NVIDIA, AI spending) is bubble fuel, so a high
    percentile means a high score. Fear (searches for "stock market crash", news
    about the dot-com bubble) cuts both ways: low fear can mean complacency, and
    loud bubble talk often comes long before a top. It is reported but not scored.
    """
    w = query("SELECT * FROM wiki_monthly ORDER BY month")
    w["month"] = pd.to_datetime(w["month"])
    w = w[w["days"] >= 25]                                   # complete months only
    g = query("SELECT * FROM gdelt_monthly ORDER BY month")

    def wiki(art):
        s = w[w["article"] == art].set_index("month")["views"].rolling(3).mean().dropna()
        return pct_rank(s, s.iloc[-1]) if len(s) > 24 else float("nan")

    def news(key):
        s = g[g["query_key"] == key].set_index("month")["per_million_articles"].rolling(3).mean().dropna()
        return pct_rank(s, s.iloc[-1]) if len(s) > 24 else float("nan")

    hype = {"wiki_Artificial_intelligence_pctile": wiki("Artificial_intelligence"),
            "wiki_Nvidia_pctile": wiki("Nvidia"),
            "gdelt_ai_capex_pctile": news("ai_capex")}
    fear = {"wiki_Stock_market_crash_pctile": wiki("Stock_market_crash"),
            "wiki_Dot-com_bubble_pctile": wiki("Dot-com_bubble"),
            "gdelt_dotcom_bubble_pctile": news("dotcom_bubble"),
            "gdelt_ai_bubble_pctile": news("ai_bubble")}
    ab = w[w["article"] == "AI_bubble"]
    vals = [v for v in hype.values() if np.isfinite(v)]
    return {"score": float(np.mean(vals)) if vals else float("nan"), "components": hype,
            "fear_context": fear,
            "extra": {"wiki_ai_bubble_last_month_views": int(ab["views"].iloc[-1]) if len(ab) else None}}


def issuance(as_of: pd.Timestamp) -> dict:
    ipos = pd.DataFrame(config.load("private_marks")["mega_ipos"])
    ipos["date"] = pd.to_datetime(ipos["date"])
    def ttm(d):
        return float(ipos[(ipos["date"] > d - pd.DateOffset(years=1)) & (ipos["date"] <= d)]["raised_b"].sum())
    hist = [ttm(d) for d in ipos["date"]]
    prior = [h for d, h in zip(ipos["date"], hist) if d < as_of - pd.DateOffset(years=1)]
    prior_max = max(prior) if prior else max(hist, default=1.0) or 1.0
    now = ttm(as_of)
    return {"score": float(min(100.0, 100 * now / (2 * prior_max))),
            "components": {"mega_ipo_raised_ttm_b": now, "prior_record_ttm_b": prior_max,
                           "note": "100 = twice the previous 12-month record; Anthropic (pending) not included"}}


def composite(pillars: dict, weights: dict, n_draws: int = 5000, seed: int = 3) -> dict:
    names = [k for k in weights if k in pillars and np.isfinite(pillars[k]["score"])]
    s = np.array([pillars[k]["score"] for k in names])
    w = np.array([weights[k] for k in names], dtype=float)
    base = float((s * w).sum() / w.sum())
    rng = np.random.default_rng(seed)
    draws = rng.dirichlet(np.ones(len(names)), size=n_draws) @ s
    return {"score": base, "pillars_used": names,
            "dirichlet_p10": float(np.quantile(draws, 0.1)), "dirichlet_p50": float(np.quantile(draws, 0.5)),
            "dirichlet_p90": float(np.quantile(draws, 0.9))}


def historical_ndx_index(panel_monthly: pd.DataFrame) -> pd.DataFrame:
    """A point-in-time version of three pillars for the Nasdaq-100 since 1990,
    using expanding-window percentiles only (no look-ahead), to check that the
    gauge would have been high in 1999-2000 and 2021."""
    ndx = panel_monthly[panel_monthly["ticker"] == "^NDX"].set_index("date").sort_index()
    ref = panel_monthly[panel_monthly["ticker"] != "AI20"][["date", "runup_z"]].dropna().sort_values("date")
    c = query("SELECT month, cape FROM cape_monthly ORDER BY month")
    c["month"] = pd.to_datetime(c["month"])
    cape = c.set_index("month")["cape"].dropna()
    m = _macro()
    baa = m["BAA10Y"].dropna().diff(3) if "BAA10Y" in m else pd.Series(dtype=float)
    ref_dates = ref["date"].to_numpy()
    ref_vals = ref["runup_z"].to_numpy()
    rows = []
    for d, r in ndx[ndx.index >= "1990-01-01"].iterrows():
        mo = d + pd.offsets.MonthEnd(0)
        n = np.searchsorted(ref_dates, np.datetime64(d), side="right")   # ref is sorted by date
        p1 = float(100 * (ref_vals[:n] <= r["runup_z"]).mean()) if n and np.isfinite(r["runup_z"]) else float("nan")
        p2 = pct_rank(cape[cape.index <= mo], cape[cape.index <= mo].iloc[-1]) if (cape.index <= mo).any() else np.nan
        b = baa[baa.index <= mo]
        p3 = pct_rank(b, b.iloc[-1]) if len(b) > 24 else np.nan
        rows.append({"date": d, "price_dynamics": p1, "valuation": p2, "credit": p3,
                     "index": float(np.nanmean([p1, p2, p3]))})
    return pd.DataFrame(rows)
