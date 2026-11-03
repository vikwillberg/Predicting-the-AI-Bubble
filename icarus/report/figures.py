"""README / dashboard figures (1600x1000). Every headline states the finding,
not the topic, and is filled in from reports/results.json, so the charts
re-word themselves when the data changes."""

from __future__ import annotations

import json

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from .. import config
from ..warehouse.build import query
from .style import (AXIS, BAD, BAD_TINT, BLUE, GOOD, GRID, INK, INK2, MUTED, NEUTRAL, NEUTRAL_DK, SURFACE,
                    Frame, hbar, legend_row, pct, vbar)

ART = config.DATA_PROCESSED / "artifacts"
OUT = config.FIGURES
SRC_PX = "Data: Yahoo Finance, FRED, multpl, Wikimedia, GDELT, company filings via Yahoo. Model: ICARUS."


def _r() -> dict:
    return json.loads((config.REPORTS / "results.json").read_text())


def _art(name: str) -> pd.DataFrame:
    df = pd.read_parquet(ART / f"{name}.parquet")
    for c in df.columns:
        if "date" in c.lower() or c in ("month", "cal_quarter", "period_end"):
            try:
                df[c] = pd.to_datetime(df[c])
            except (ValueError, TypeError):
                pass
    return df


def _date_axis(a, fmt="%Y", minticks=4, maxticks=8):
    loc = mdates.AutoDateLocator(minticks=minticks, maxticks=maxticks)
    a.xaxis.set_major_locator(loc)
    a.xaxis.set_major_formatter(mdates.DateFormatter(fmt))


# --------------------------------------------------------------------------- 1
def dotcom_clock(r: dict) -> None:
    c = r["analogs"]["clock"]
    ndx, ai, now = _art("clock_ndx"), _art("clock_ai20"), _art("clock_ndx_now")
    f = Frame()
    f.header("The dot-com clock", "Same index, new boom: the Nasdaq-100 is far below its 2000 run",
             f"Index level, 100 = the catalyst: the Netscape IPO (Aug 1995) and the ChatGPT launch (Nov 2022). "
             f"Log scale.", title_size=44)
    a = f.plot_axes(110, 230, 1270, 570)
    a.plot(ndx["months"], ndx["level"], color=NEUTRAL_DK, lw=2.5, label="Nasdaq-100 after Netscape IPO")
    a.plot(now["months"], now["level"], color=BLUE, lw=3, label="Nasdaq-100 after ChatGPT")
    a.plot(ai["months"], ai["level"], color=INK, lw=2.5, label="AI-20 after ChatGPT")
    a.set_yscale("log")
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    a.set_yticks([100, 200, 400, 800])
    a.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    a.set_xlim(0, 96)
    a.set_xticks(range(0, 97, 12))
    a.set_xticklabels([f"{m // 12} yr" if m else "Catalyst" for m in range(0, 97, 12)])
    pk = c["dotcom_months_anchor_to_peak"]
    a.scatter([pk], [100 * c["ndx_multiple_at_dotcom_peak"]], s=90, color=NEUTRAL_DK, edgecolor=SURFACE, lw=2, zorder=5)
    a.annotate(f"Mar 2000 peak: {c['ndx_multiple_at_dotcom_peak']:.1f}x", (pk, 100 * c["ndx_multiple_at_dotcom_peak"]),
               xytext=(10, 10), textcoords="offset points", fontsize=f.pt(20), color=INK2, fontweight=600)
    t = c["ai_months_since_anchor"]
    a.scatter([t], [100 * c["ndx_multiple_since_chatgpt"]], s=90, color=BLUE, edgecolor=SURFACE, lw=2, zorder=5)
    a.annotate(f"Today: {c['ndx_multiple_since_chatgpt']:.1f}x", (t, 100 * c["ndx_multiple_since_chatgpt"]),
               xytext=(10, -26), textcoords="offset points", fontsize=f.pt(20), color=INK2, fontweight=600)
    a.scatter([t], [100 * c["ai20_multiple_now"]], s=90, color=INK, edgecolor=SURFACE, lw=2, zorder=5)
    a.set_ylim(top=1400)
    a.annotate(f"AI-20 today: {c['ai20_multiple_now']:.1f}x", (t, 100 * c["ai20_multiple_now"]),
               xytext=(30, 1150), textcoords="data", ha="right", va="center", fontsize=f.pt(20), color=INK2,
               fontweight=600, arrowprops=dict(arrowstyle="-", color=NEUTRAL_DK, lw=1.2))
    legend_row(f, 892, [(NEUTRAL_DK, "Nasdaq-100 after Netscape IPO", "line"), (BLUE, "Nasdaq-100 after ChatGPT", "line"),
                        (INK, "AI-20 after ChatGPT (picked with hindsight)", "line")])
    f.footer("AI-20 = equal-weight basket of 20 AI leaders chosen in 2026, so its run-up is flattered by hindsight. " + "Data: Yahoo Finance.")
    f.save(OUT / "01_dotcom_clock.png")


# --------------------------------------------------------------------------- 2
def burst_odds(r: dict) -> None:
    tg = r["survival"]["burst"]["targets"]
    ai = tg["AI20"]
    f = Frame()
    f.header("When does it burst?",
             f"{pct(ai['p12'])} odds the AI-20 crashes 40%+ within a year, {pct(ai['p24'])} within two",
             f"Probability of a fall of 40% or more from the {pd.Timestamp(ai['peak_date']):%b %-d, %Y} peak, by months from "
             f"{pd.Timestamp(ai['as_of']):%b %-d, %Y}. Band: 80% year-block bootstrap.", title_size=44)
    a = f.plot_axes(110, 230, 1250, 590)
    k = np.arange(1, 25)
    a.fill_between(k, ai["cdf_p10"], ai["cdf_p90"], color=BAD, alpha=0.12, lw=0)
    a.plot(k, ai["cdf"], color=BAD, lw=3.2)
    for tk, col in (("^NDX", NEUTRAL_DK), ("^GSPC", NEUTRAL)):
        t = tg.get(tk)
        if t and t.get("cdf"):
            a.plot(k, t["cdf"], color=col, lw=2.5)
            a.text(24.4, t["cdf"][-1], f"{t['name']} {pct(t['cdf'][-1])}", fontsize=f.pt(19), color=INK2, va="center")
    a.text(24.4, ai["cdf"][-1], f"AI-20 {pct(ai['cdf'][-1])}", fontsize=f.pt(21), color=INK, va="center", fontweight=700)
    for m in (12,):
        a.scatter([m], [ai["cdf"][m - 1]], s=110, color=BAD, edgecolor=SURFACE, lw=2.5, zorder=6)
        a.annotate(f"{pct(ai['cdf'][m - 1])} by {pd.Timestamp(ai['as_of']) + pd.DateOffset(months=m):%b %Y}",
                   (m, ai["cdf"][m - 1]), xytext=(-20, 18), textcoords="offset points", ha="right",
                   fontsize=f.pt(21), fontweight=700, color=INK)
    a.set_xlim(1, 24)
    a.set_ylim(0, 1)
    a.set_xticks([1, 3, 6, 9, 12, 15, 18, 21, 24])
    a.set_xlabel("Months ahead", fontsize=f.pt(19), color=MUTED)
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    sk = r["survival"]["burst"]["metrics"]["h12"]["p_ens"]
    f.footer(f"Monotone survival ensemble (GBM + logit), trained on {r['survival']['burst']['n_series']} assets since 1927. "
             f"Out-of-sample AUC {sk['auc']:.2f} at 12 months.")
    f.save(OUT / "02_burst_odds.png")


# --------------------------------------------------------------------------- 3
def backtest(r: dict) -> None:
    fam = r["survival"]["burst"]["famous"]
    ai = r["survival"]["burst"]["targets"]["AI20"]
    rows = [(e["episode"], e["p12_at_peak"], e["climatology"], False) for e in fam]
    rows.append(("AI-20 today", ai["p12"], np.mean([e["climatology"] for e in fam]), True))
    rows.sort(key=lambda x: x[1])
    f = Frame()
    hits = sum(e["lift"] >= 1.5 for e in fam)
    f.header("Backtest", f"ICARUS flagged {hits} of {len(fam)} past bubble peaks at 1.5x+ the normal crash risk",
             "Out-of-sample probability of a 40%+ crash within 12 months, at the month of each peak, vs. the base rate.",
             title_size=44)
    x0, x1, top, row = 420, 1430, 250, 58
    scale = (x1 - x0)
    for i, (lab, p, clim, me) in enumerate(rows):
        y = top + i * row
        f.text(x0 - 24, y, lab, 22, 700 if me else 400, INK if me else INK2, ha="right", va="center")
        hbar(f.ax, x0, y, p * scale, 30, BAD if me else NEUTRAL_DK)
        f.ax.plot([x0 + clim * scale] * 2, [y - 22, y + 22], color=INK, lw=2.2, zorder=5)
        f.text(x0 + p * scale + 12, y, pct(p), 22, 700, INK, va="center")
    yb = top + len(rows) * row - 18
    f.ax.plot([x0, x1], [yb, yb], color=AXIS, lw=1)
    for v in (0, .25, .5, .75, 1):
        f.text(x0 + v * scale, yb + 10, f"{v:.0%}", 18, 400, MUTED, ha="center")
    legend_row(f, yb + 70, [(NEUTRAL_DK, "Past bubble at its peak", "box"), (BAD, "AI-20 now", "box"),
                            (INK, "Base rate (tick)", "line")], x=x0)
    f.footer("Walk-forward: each year predicted by a model trained only on labels resolved before that year.")
    f.save(OUT / "03_backtest.png")


# --------------------------------------------------------------------------- 4
def lppls_figure(r: dict) -> None:
    ex = r["verdict"].get("lppls_ex_ante")
    s = query("SELECT date, price FROM series WHERE ticker='AI20' AND date >= '2025-06-01' ORDER BY date")
    s["date"] = pd.to_datetime(s["date"])
    fits = _art("lppls_fits_AI20")
    ci = _art("lppls_ci_AI20")
    f = Frame()
    title = (f"LPPLS saw the October top coming: it forecast {pd.Timestamp(ex['tc_median']):%b %-d}, the peak came {pd.Timestamp(ex['peak']):%b %-d}"
             if ex else "LPPLS bubble clock for the AI-20")
    f.header("Sornette's bubble clock (LPPLS)", title,
             "Top: AI-20 with the critical times implied by qualified fits. Bottom: share of 26 windows (6 months to 3 years) "
             "that fit a bubble.", title_size=40)
    a = f.plot_axes(110, 220, 1380, 400)
    a.plot(s["date"], s["price"] / s["price"].iloc[0] * 100, color=INK, lw=2.5)
    pre = fits[fits["t2_date"] < pd.Timestamp(ex["peak"])] if ex else fits
    if len(pre):
        lo, hi = a.get_ylim()
        for tc in pd.to_datetime(pre["tc_date"]):
            a.plot([tc, tc], [lo, lo + (hi - lo) * 0.10], color=BAD, alpha=0.35, lw=2)
        a.set_ylim(lo, hi)
    if ex:
        pk = pd.Timestamp(ex["peak"])
        yv = float(s.loc[s["date"] == pk, "price"].iloc[0] / s["price"].iloc[0] * 100)
        a.scatter([pk], [yv], s=110, color=BAD, edgecolor=SURFACE, lw=2.5, zorder=6)
        a.annotate(f"Peak {pk:%b %-d}\nForecast on {pd.Timestamp(ex['forecast_made']):%b %-d}: {pd.Timestamp(ex['tc_median']):%b %-d}",
                   (pk, yv), xytext=(-260, -10), textcoords="offset points", fontsize=f.pt(20), fontweight=600, color=INK,
                   arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8))
    _date_axis(a, "%b %Y")
    b = f.plot_axes(110, 680, 1380, 190)
    b.bar(ci["date"], ci["pos_conf"], width=8, color=BAD, alpha=0.85)
    b.set_ylim(0, 1)
    b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    b.set_xlim(a.get_xlim())
    _date_axis(b, "%b %Y")
    bias = r["verdict"].get("lppls_historical_bias_days") or {}
    f.footer(f"Red ticks = critical times from qualified fits made before the peak. In 7 past bubbles LPPLS ran early by "
             f"{abs(bias.get('median', 0)):.0f} days (median).")
    f.save(OUT / "04_lppls.png")


# --------------------------------------------------------------------------- 5
def psy_figure(r: dict) -> None:
    p = _art("psy_NDX")
    info = r["psy"]["^NDX"]
    f = Frame()
    f.header("Phillips-Shi-Yu explosive-root test",
             ("The Nasdaq-100 is explosive again, as in 1997-2000 and 2020-22"
              if r["psy"]["^NDX"]["explosive_now"] else "The Nasdaq-100's explosive (bubble) regimes since 1985"),
             "Monthly. Shaded: BSADF above its 95% Monte Carlo critical value (two-step procedure: global GSADF test, then dating).",
             title_size=42)
    px = query("SELECT date, price FROM series WHERE ticker='^NDX' ORDER BY date")
    px["date"] = pd.to_datetime(px["date"])
    a = f.plot_axes(110, 220, 1380, 320)
    a.plot(px["date"], px["price"], color=INK, lw=2)
    a.set_yscale("log")
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    a.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    b = f.plot_axes(110, 590, 1380, 240)
    b.plot(p["date"], p["bsadf"], color=INK, lw=2, label="BSADF")
    b.plot(p["date"], p["cv95"], color=NEUTRAL_DK, lw=2, label="95% critical value")
    for e in info["episodes"]:
        for ax_ in (a, b):
            ax_.axvspan(pd.Timestamp(e["start"]), pd.Timestamp(e["end"]), color=BAD_TINT, lw=0, zorder=0)
    for ax_ in (a, b):
        _date_axis(ax_, "%Y")
        ax_.set_xlim(p["date"].min(), p["date"].max())
    legend_row(f, 897, [(INK, "BSADF statistic", "line"), (NEUTRAL_DK, "95% critical value", "line"),
                        (BAD_TINT, "Explosive (bubble) regime", "box")])
    f.footer(f"GSADF {info['gsadf']:.2f} vs 95% critical value {info['gsadf_cv95']:.2f}: the random-walk null is rejected.")
    f.save(OUT / "05_psy.png")


# --------------------------------------------------------------------------- 6
def pillars(r: dict) -> None:
    ri = r["risk_index"]
    names = {"price_dynamics": "Price dynamics", "valuation": "Valuation", "fundamentals": "Fundamentals (capex strain)",
             "market_structure": "Market structure", "credit_macro": "Credit & macro", "sentiment": "AI hype",
             "issuance": "Mega-IPO issuance"}
    rows = sorted([(names[k], v["score"]) for k, v in ri["pillars"].items() if v["score"] is not None], key=lambda x: -x[1])
    comp = ri["composite"]
    f = Frame()
    words = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "all seven"}
    n_hot = sum(sc >= 70 for _, sc in rows)
    f.header("ICARUS Bubble Risk Index",
             f"{comp['score']:.0f} / 100: {words[n_hot]}{' of seven' if n_hot < 7 else ''} perspectives flash warning",
             f"Each pillar is a 0-100 percentile vs its own history (100 = most bubble-like). "
             f"Random-weight range: {comp['dirichlet_p10']:.0f}-{comp['dirichlet_p90']:.0f}.", title_size=44)
    x0, x1, top, row = 480, 1400, 260, 72
    for i, (lab, sc) in enumerate(rows):
        y = top + i * row
        f.text(x0 - 24, y, lab, 23, 400, INK2, ha="right", va="center")
        hbar(f.ax, x0, y, sc / 100 * (x1 - x0), 34, BAD if sc >= 70 else NEUTRAL_DK)
        f.text(x0 + sc / 100 * (x1 - x0) + 12, y, f"{sc:.0f}", 24, 700, INK, va="center")
    yb = top + len(rows) * row - 22
    f.ax.plot([x0 + 0.7 * (x1 - x0)] * 2, [top - 34, yb], color=INK, lw=2)
    f.text(x0 + 0.7 * (x1 - x0) + 8, top - 48, "Warning zone (70+)", 19, 600, INK2)
    legend_row(f, yb + 60, [(BAD, "70 or higher", "box"), (NEUTRAL_DK, "Below 70", "box")], x=x0)
    f.footer("Point-in-time check: the same gauge (3 pillars) read 98 at the 2000 Nasdaq peak and 84 in 2021.")
    f.save(OUT / "06_risk_index.png")


# --------------------------------------------------------------------------- 7
def capex(r: dict) -> None:
    q = _art("capex_quarterly")
    full_n = q["n"].max()
    q = q[q["cal_quarter"] >= q.loc[q["n"] == full_n, "cal_quarter"].min()]
    # One slot per calendar quarter; a quarter with missing filings is shown as a gap, not dropped.
    quarters = pd.date_range(q["cal_quarter"].min(), q["cal_quarter"].max(), freq="QS")
    fc = r["fundamentals"]["capex"]
    f = Frame()
    f.header("The capex test", "Big Tech's capex now exceeds the cash its businesses generate",
             "Microsoft, Alphabet, Amazon, Meta, Oracle: quarterly capital spending vs operating cash flow, $B.", title_size=44)
    a = f.plot_axes(130, 220, 1340, 560)
    for i, qd in enumerate(quarters):
        row = q[(q["cal_quarter"] == qd) & (q["n"] == full_n)]
        if row.empty:
            a.text(i, 8, "Data gap:\nAmazon Q2\nmissing\nat source", ha="center", va="bottom",
                   fontsize=f.pt(17), color=MUTED)
            continue
        c, o = float(row["capex"].iloc[0]) / 1e9, float(row["ocf"].iloc[0]) / 1e9
        a.bar(i - 0.18, c, width=0.32, color=BAD if c > o else NEUTRAL_DK)
        a.bar(i + 0.18, o, width=0.32, color=NEUTRAL)
        a.text(i - 0.18, c + 4, f"{c:.0f}", ha="center", fontsize=f.pt(19), color=INK, fontweight=700)
        a.text(i + 0.18, o + 4, f"{o:.0f}", ha="center", fontsize=f.pt(19), color=INK2)
    a.set_xticks(range(len(quarters)))
    a.set_xticklabels([f"Q{d.quarter} {d.year}" for d in quarters])
    a.set_xlim(-0.6, len(quarters) - 0.4)
    legend_row(f, 862, [(BAD, "Capex above cash flow", "box"), (NEUTRAL_DK, "Capex below cash flow", "box"),
                        (NEUTRAL, "Operating cash flow", "box")])
    lt = fc["latest"]
    f.footer(f"Latest quarter capex/OCF {lt['capex_to_ocf']:.2f}x, {lt['companies_spending_more_than_ocf']} of "
             f"{lt['companies']} spent more than they generated (2023: {fc.get('baseline_capex_to_ocf', 0):.2f}x). "
             "Company statements via Yahoo Finance.")
    f.save(OUT / "07_capex.png")


# --------------------------------------------------------------------------- 8
def anthropic(r: dict) -> None:
    pv = r["fundamentals"]["private"]
    rounds = pd.DataFrame(pv["rounds"])
    rounds = rounds[(rounds["lab"] == "anthropic") & rounds["multiple"].notna()]
    rdcf = r["fundamentals"]["reverse_dcf"]
    rd = rdcf["scenarios"]
    bm = r["fundamentals"]["benchmarks"]
    f = Frame()
    ipo = rd["IPO ($1.5T)"]["required_revenue_b"]
    f.header("The Anthropic IPO",
             f"At $1.5T, Anthropic needs ~${ipo['p50']:.0f}B of revenue by {rdcf['target_year']}",
             "Left: price as a multiple of revenue run-rate. Right: revenue needed in 10 years to justify each price.",
             title_size=44)
    a = f.plot_axes(110, 250, 520, 560)
    labels = list(rounds["event"]) + ["IPO\n$1.5T"]
    vals = list(rounds["multiple"]) + [pv["implied_ipo_multiples"]["IPO low ($1.5T)"]["x_run_rate"]]
    cols = [NEUTRAL_DK] * len(rounds) + [BAD]
    a.bar(range(len(vals)), vals, width=0.55, color=cols)
    for i, v in enumerate(vals):
        a.text(i, v + 0.8, f"{v:.0f}x", ha="center", fontsize=f.pt(21), fontweight=700, color=INK)
    a.set_xticks(range(len(vals)))
    a.set_xticklabels([lb.replace("Series ", "Series\n") for lb in labels], fontsize=f.pt(18))
    a.set_yticks([])
    a.grid(False)
    # Right: one row per price (median and 10-90% range), then today's giants as reference rows.
    names = {"MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon"}
    refs = sorted([(names[k], v["revenue_b"]) for k, v in bm.items() if k in names], key=lambda x: -x[1])
    x0, x1, top, row = 980, 1490, 280, 64
    vmax = 1600.0
    X = lambda v: x0 + v / vmax * (x1 - x0)
    f.text(x0 - 260, top - 44, "Revenue needed by " + str(rdcf["target_year"]) + " ($B)", 20, 600, MUTED)
    for i, (k, v) in enumerate(rd.items()):
        y = top + i * row
        q = v["required_revenue_b"]
        col = BAD if "IPO" in k else NEUTRAL_DK
        f.text(x0 - 20, y, k.replace(" (", "  ("), 20, 600 if "IPO" in k else 400, INK if "IPO" in k else INK2,
               ha="right", va="center")
        f.ax.plot([X(q["p10"]), X(q["p90"])], [y, y], color=col, lw=9, alpha=0.3, solid_capstyle="round", zorder=3)
        f.ax.scatter([X(q["p50"])], [y], s=120, color=col, edgecolor=SURFACE, linewidth=2.5, zorder=5)
        f.text(min(X(q["p90"]) + 12, 1500), y, f"${q['p50']:.0f}B", 20, 700, INK, va="center")
    yr = top + len(rd) * row + 30
    f.ax.plot([x0 - 250, x1], [yr - 34, yr - 34], color=GRID, lw=1)
    for j, (n, v) in enumerate(refs):
        y = yr + j * 50
        f.text(x0 - 20, y, f"{n} revenue today", 19, 400, MUTED, ha="right", va="center")
        hbar(f.ax, x0, y, X(v) - x0, 16, NEUTRAL)
        f.text(X(v) + 10, y, f"${v:.0f}B", 19, 400, INK2, va="center")
    s1 = pv["s1"]
    f.footer(f"S-1 (Oct 26, 2026): H1-2026 revenue ${s1['revenue_h1_2026_b']:.1f}B, {pv['yoy_h1_growth']:.0f}x a year earlier. "
             f"Reverse DCF: 20,000 draws of discount rate, margin, exit multiple, dilution.")
    f.save(OUT / "08_anthropic_ipo.png")


# --------------------------------------------------------------------------- 9
def breadth(r: dict) -> None:
    d = query("""SELECT f.ticker, f.dd, m.name FROM features_daily f JOIN ai20_members m USING (ticker)
                 WHERE f.date = (SELECT max(date) FROM features_daily WHERE ticker='AI20') ORDER BY f.dd""")
    s = r["systemic"]
    n_bear = int((d["dd"] <= -0.2).sum())
    f = Frame(1600, 1100)
    f.header("Breadth", f"{n_bear} of the 20 AI leaders are already in bear markets",
             "Each company's fall from its all-time high. The generals hold while the troops fall: the 2000 pattern.",
             title_size=46)
    x0, x1, top = 360, 1380, 236
    row = 36
    for i, rw in enumerate(d.itertuples()):
        y = top + i * row
        f.text(x0 - 20, y, rw.name, 19, 400, INK2, ha="right", va="center")
        L = -rw.dd * (x1 - x0) / 0.8
        hbar(f.ax, x0, y, L, 22, BAD if rw.dd <= -0.2 else NEUTRAL_DK)
        xr = x0 + 0.2 * (x1 - x0) / 0.8
        lx = x0 + L + 10
        if lx < xr + 10 and lx + 56 > xr - 4:          # never let a label sit on the reference line
            lx = xr + 10
        f.text(lx, y, f"{rw.dd:.0%}", 19, 700, INK, va="center")
    yb = top + 20 * row - 10
    xr = x0 + 0.2 * (x1 - x0) / 0.8
    f.ax.plot([xr, xr], [top - 26, yb], color=INK, lw=2, zorder=6)
    f.text(xr + 8, top - 40, "Bear market (-20%)", 18, 600, INK2)
    legend_row(f, yb + 40, [(BAD, "In a bear market", "box"), (NEUTRAL_DK, "Down less than 20%", "box")], x=x0)
    f.footer(f"Median member {s['ai20_median_dd']:.0%} from its high; AI-20 index itself "
             f"{r['verdict']['ai20_drawdown_from_peak']:.0%}. As of {s['breadth_date']}.")
    f.save(OUT / "09_breadth.png")


# -------------------------------------------------------------------------- 10
def regimes(r: dict) -> None:
    h = _art("hmm_ndx")
    px = query("SELECT date, price FROM series WHERE ticker='^NDX' ORDER BY date")
    px["date"] = pd.to_datetime(px["date"])
    reg = r["regimes"]
    ai = _art("hmm_ai20")
    ai = ai.set_index(pd.to_datetime(ai[ai.columns[0]]))
    is_stress = ai["regime"] == "stress"
    since = None
    if is_stress.iloc[-1]:
        run_start = is_stress[::-1].ne(True).idxmax()          # last non-stress week
        since = ai.index[ai.index > run_start][0] if (ai.index > run_start).any() else ai.index[0]
    title = (f"The AI-20 has been in the stress regime since {since:%b %-d} ({pct(reg['current']['probabilities']['stress'])})"
             if since is not None else f"The AI-20 is in the {reg['current']['regime']} regime")
    f = Frame()
    f.header("Hidden Markov regimes", title,
             "HMM trained on 40 years of weekly Nasdaq-100 returns and volatility. Shaded: Nasdaq-100 stress weeks.",
             title_size=42)
    a = f.plot_axes(110, 220, 1380, 620)
    a.plot(px["date"], px["price"], color=INK, lw=1.8)
    a.set_yscale("log")
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    st = h.set_index(h.columns[0])["regime"] == "stress"
    blocks, start = [], None
    for d, v in st.items():
        if v and start is None:
            start = d
        if not v and start is not None:
            blocks.append((start, d)); start = None
    if start is not None:
        blocks.append((start, st.index[-1]))
    for s0, s1 in blocks:
        a.axvspan(s0, s1, color=BAD_TINT, lw=0, zorder=0)
    _date_axis(a, "%Y")
    v = reg["validation_share_stress"]
    f.footer(f"Sanity check: stress share of weeks in the 2000-02 bust {v['2000-02 dot-com bust']:.0%}, GFC {v['2008 GFC']:.0%}, "
             f"2013-14 calm bull {v['2013-14 calm bull']:.0%}. Expected stress spell: {reg['expected_duration_weeks']['stress']:.0f} weeks.")
    f.save(OUT / "10_regimes.png")


# -------------------------------------------------------------------------- 11
def calibration(r: dict) -> None:
    rel = _art("reliability_burst_h12")
    m = r["survival"]["burst"]["metrics"]
    f = Frame()
    f.header("Is the model honest?", "When ICARUS says 30%, crashes happen about 30% of the time",
             "Out-of-sample reliability, 12-month burst forecasts, 1995-2024 (deciles of predicted probability).", title_size=44)
    a = f.plot_axes(140, 220, 640, 640)
    a.plot([0, 1], [0, 1], color=NEUTRAL, lw=2)
    a.plot(rel["pred"], rel["obs"], color=INK, lw=2.5)
    a.scatter(rel["pred"], rel["obs"], s=110, color=INK, edgecolor=SURFACE, lw=2.5, zorder=5)
    lim = max(rel["pred"].max(), rel["obs"].max()) * 1.1
    a.set_xlim(0, lim)
    a.set_ylim(0, lim)
    a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    a.set_xlabel("Predicted probability", fontsize=f.pt(19), color=MUTED)
    a.set_ylabel("Observed frequency", fontsize=f.pt(19), color=MUTED)
    x0, y = 880, 260
    f.text(x0, y, "Out-of-sample skill (walk-forward)", 24, 700, INK)
    hdr = ["Horizon", "Model", "AUC", "Brier skill"]
    cols = [x0, x0 + 150, x0 + 420, x0 + 540]
    for c, t in zip(cols, hdr):
        f.text(c, y + 56, t, 19, 600, MUTED)
    yy = y + 100
    names = {"p_ens": "ICARUS ensemble", "p_logit": "Logistic only", "p_gbm": "Gradient boosting only"}
    for h in ("h3", "h12", "h24"):
        for k in ("p_ens", "p_logit", "p_gbm"):
            v = m[h][k]
            bold = 700 if k == "p_ens" else 400
            f.text(cols[0], yy, h[1:] + " months" if k == "p_ens" else "", 20, bold, INK)
            f.text(cols[1], yy, names[k], 20, bold, INK if k == "p_ens" else INK2)
            f.text(cols[2], yy, f"{v['auc']:.2f}", 20, bold, INK)
            f.text(cols[3], yy, f"{v['brier_skill_vs_climatology']:+.0%}", 20, bold, INK)
            yy += 38
        yy += 16
    f.footer("Brier skill = improvement over always forecasting the historical base rate. AUC 0.5 = coin flip, 1.0 = perfect.")
    f.save(OUT / "11_calibration.png")


# -------------------------------------------------------------------------- 12
def analogs(r: dict) -> None:
    t = pd.DataFrame(r["analogs"]["table"]).sort_values("months_to_peak")
    t = t[t["months_to_peak"].between(-24, 36)]
    sm = r["analogs"]["summary"]
    f = Frame()
    f.header("Historical analogs", f"{pct(sm['share_past_peak'])} of the analog weight says the AI peak is already behind us",
             "The AI-20's last two years matched to 12 famous bubbles (shape, time-stretched). Where is 'today'?",
             title_size=42)
    x0, x1, top, row = 470, 1480, 260, 50
    lo, hi = -12, 30
    X = lambda m: x0 + (m - lo) / (hi - lo) * (x1 - x0)
    for i, rw in enumerate(t.itertuples()):
        y = top + i * row
        f.text(x0 - 22, y, f"{rw.episode}", 21, 400, INK2, ha="right", va="center")
        f.ax.plot([x0, x1], [y, y], color=GRID, lw=1, zorder=1)
        col = BAD if rw.months_to_peak < 0 else NEUTRAL_DK
        f.ax.scatter([X(np.clip(rw.months_to_peak, lo, hi))], [y], s=120 + 1600 * rw.weight, color=col,
                     edgecolor=SURFACE, lw=2, zorder=4)
    yb = top + len(t) * row - 20
    f.ax.plot([X(0), X(0)], [top - 30, yb], color=INK, lw=2)
    f.text(X(0), top - 44, "Bubble peak", 19, 700, INK, ha="center")
    for m in range(lo, hi + 1, 6):
        f.text(X(m), yb + 10, f"{m:+d} mo" if m else "0", 18, 400, MUTED, ha="center")
    f.text(X(-6), yb + 46, "Past the peak", 20, 600, INK2, ha="center")
    f.text(X(15), yb + 46, "Peak still ahead", 20, 600, INK2, ha="center")
    f.footer(f"Dot size = fit quality. Weighted median drawdown that followed the matched point: "
             f"{sm['drawdown_ahead_weighted_median']:.0%}. Conditional on this being a bubble.")
    f.save(OUT / "12_analogs.png")


def run() -> list[str]:
    OUT.mkdir(parents=True, exist_ok=True)
    r = _r()
    done = []
    for fn in (dotcom_clock, burst_odds, backtest, lppls_figure, psy_figure, pillars, capex, anthropic, breadth,
               regimes, calibration, analogs):
        fn(r)
        done.append(fn.__name__)
    return done
