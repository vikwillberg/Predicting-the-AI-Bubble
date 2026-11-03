"""Historical analogs: "if this is a bubble, which month of the bubble is it?"

For each famous bubble, the AI-20's last two years of weekly log prices are
slid across that bubble's history (from four years before its peak to two years
after) at several time-stretch factors, because bubbles run at different
speeds. The best-fitting alignment gives:

  * the analog's date that corresponds to "today",
  * how far that was from the analog's peak (in our time scale), and
  * how far the analog fell afterwards.

Weighting each analog by its fit quality gives a distribution of "months to
peak" and "drawdown ahead". The method is conditional by construction: it says
where we are *if* this is a bubble, so it is reported next to the models that
estimate *whether* it is.

The module also runs the classic "dot-com clock": AI-20 since the ChatGPT launch
against the Nasdaq-100 since the Netscape IPO, on calendar-aligned axes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..warehouse.build import query

EPISODES = [
    ("^NDX", "2000-03-27", "Nasdaq-100 2000"),
    ("^SOX", "2000-03-10", "Semiconductors 2000"),
    ("CSCO", "2000-03-27", "Cisco 2000"),
    ("^N225", "1989-12-29", "Nikkei 1989"),
    ("000001.SS", "2007-10-16", "Shanghai 2007"),
    ("000001.SS", "2015-06-12", "Shanghai 2015"),
    ("^HSI", "2007-10-30", "Hang Seng 2007"),
    ("CL=F", "2008-07-03", "Crude oil 2008"),
    ("BTC-USD", "2017-12-16", "Bitcoin 2017"),
    ("BTC-USD", "2021-11-08", "Bitcoin 2021"),
    ("ARKK", "2021-02-12", "ARK Innovation 2021"),
    ("NVDA", "2021-11-29", "NVIDIA 2021"),
]
STRETCH = [0.6, 0.75, 0.9, 1.0, 1.15, 1.35, 1.6, 2.0]
WINDOW_WEEKS = 104


def _weekly(ticker: str) -> pd.Series:
    d = query(f"SELECT date, price FROM series WHERE ticker = '{ticker}' ORDER BY date")
    d["date"] = pd.to_datetime(d["date"])
    return np.log(d.set_index("date")["price"]).resample("W-FRI").last().dropna()


def _znorm(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 0 else 1.0)


def best_alignment(current: np.ndarray, hist: pd.Series, peak: pd.Timestamp) -> dict | None:
    """Slide the current path over the historical one at several stretch factors."""
    cur = _znorm(current)
    L = len(current)
    hvals = hist.to_numpy()
    hidx = hist.index
    lo = hidx.searchsorted(peak - pd.DateOffset(years=4))
    hi = hidx.searchsorted(peak + pd.DateOffset(years=2))
    best = None
    for s in STRETCH:
        span = int(round(L * s))
        if span < 20:
            continue
        for end in range(max(lo, span), min(hi, len(hvals) - 1) + 1):
            seg = hvals[end - span:end + 1]
            # resample the stretched historical segment to L points
            seg_l = np.interp(np.linspace(0, len(seg) - 1, L), np.arange(len(seg)), seg)
            d = float(np.sqrt(np.mean((cur - _znorm(seg_l)) ** 2)))
            if best is None or d < best["rmse"]:
                best = {"rmse": d, "stretch": s, "end": end, "amp_ratio": float(np.ptp(current) / max(np.ptp(seg_l), 1e-9))}
    if best is None:
        return None
    end = best["end"]
    today_equiv = hidx[end]
    weeks_to_peak_hist = (peak - today_equiv).days / 7
    fwd = hvals[end:min(len(hvals), end + 104)]
    return {
        **best,
        "today_equivalent": str(today_equiv.date()),
        "months_to_peak": float(weeks_to_peak_hist / best["stretch"] / 4.345),
        "drawdown_ahead": float(np.exp(fwd.min() - hvals[end]) - 1) if len(fwd) else None,
        "return_12m_ahead": float(np.exp(hvals[min(len(hvals) - 1, end + 52)] - hvals[end]) - 1),
    }


def analyse() -> dict:
    ai = _weekly("AI20")
    current = ai.iloc[-WINDOW_WEEKS:].to_numpy()
    rows = []
    for ticker, peak, label in EPISODES:
        try:
            h = _weekly(ticker)
        except Exception:
            continue
        if h.empty or h.index[0] > pd.Timestamp(peak) - pd.DateOffset(years=3):
            continue
        res = best_alignment(current, h, pd.Timestamp(peak))
        if res:
            rows.append({"episode": label, "ticker": ticker, "peak": peak, **res})
    df = pd.DataFrame(rows).sort_values("rmse")
    w = 1 / df["rmse"] ** 2
    w = w / w.sum()
    df["weight"] = w

    def wq(values, weights, q):
        o = np.argsort(values)
        cw = np.cumsum(np.asarray(weights)[o])
        return float(np.asarray(values)[o][np.searchsorted(cw, q * cw[-1])])

    summary = {
        "n_analogs": int(len(df)),
        "best_match": df.iloc[0][["episode", "today_equivalent", "rmse", "months_to_peak", "drawdown_ahead"]].to_dict(),
        "months_to_peak_weighted_median": wq(df["months_to_peak"].to_numpy(), w.to_numpy(), 0.5),
        "months_to_peak_q25": wq(df["months_to_peak"].to_numpy(), w.to_numpy(), 0.25),
        "months_to_peak_q75": wq(df["months_to_peak"].to_numpy(), w.to_numpy(), 0.75),
        "share_past_peak": float(w[df["months_to_peak"] < 0].sum()),
        "drawdown_ahead_weighted_median": wq(df["drawdown_ahead"].to_numpy(), w.to_numpy(), 0.5),
    }
    return {"table": df, "summary": summary, "clock": dotcom_clock()}


def dotcom_clock() -> dict:
    """Calendar-aligned paths: Nasdaq-100 from the Netscape IPO, AI-20 from ChatGPT."""
    anchors = config.load("events")["analog_anchors"]
    a_dot, a_ai = pd.Timestamp(anchors["dotcom"]), pd.Timestamp(anchors["ai"])
    ndx_all = np.exp(_weekly("^NDX"))
    ai = np.exp(_weekly("AI20"))
    ndx = ndx_all[(ndx_all.index >= a_dot) & (ndx_all.index <= a_dot + pd.DateOffset(years=8))]
    ndx_now = ndx_all[ndx_all.index >= a_ai]          # apples-to-apples: same index, new boom
    ai = ai[ai.index >= a_ai]

    def months(idx, a):
        return (idx - a).days / 30.4375

    peak_dot = pd.Timestamp("2000-03-27")
    months_to_peak = (peak_dot - a_dot).days / 30.4375
    return {
        "ndx": pd.DataFrame({"months": months(ndx.index, a_dot), "level": 100 * ndx / ndx.iloc[0]}),
        "ai20": pd.DataFrame({"months": months(ai.index, a_ai), "level": 100 * ai / ai.iloc[0]}),
        "ndx_now": pd.DataFrame({"months": months(ndx_now.index, a_ai), "level": 100 * ndx_now / ndx_now.iloc[0]}),
        "dotcom_months_anchor_to_peak": float(months_to_peak),
        "ai_months_since_anchor": float(months(ai.index, a_ai)[-1]),
        "calendar_equivalent_peak": str((a_ai + pd.DateOffset(days=int(months_to_peak * 30.4375))).date()),
        "ndx_multiple_at_dotcom_peak": float(ndx.loc[:peak_dot].iloc[-1] / ndx.iloc[0]),
        "ndx_multiple_since_chatgpt": float(ndx_now.iloc[-1] / ndx_now.iloc[0]),
        "ndx_multiple_since_chatgpt_peak": float(ndx_now.max() / ndx_now.iloc[0]),
        # The AI-20 is picked with hindsight (today's winners), which inflates its run-up.
        "ai20_multiple_now": float(ai.iloc[-1] / ai.iloc[0]),
        "ai20_multiple_peak": float(ai.max() / ai.iloc[0]),
    }
