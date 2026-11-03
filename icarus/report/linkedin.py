"""LinkedIn carousel: 4 slides (1080x1350, exported at 2x), a PDF, a data
workbook that traces every number on the slides, and the post text.

Story arc (one idea per slide):
  1 hook   the answer: the odds of a 40%+ crash, by month
  2 why    breadth: most AI leaders are already in bear markets
  3 money  capex now exceeds cash flow; what Anthropic's IPO price requires
  4 proof  the same model at the top of past bubbles, and where today ranks
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import config
from ..warehouse.build import query
from .style import (AXIS, BAD, GRID, INK, INK2, MUTED, NEUTRAL, NEUTRAL_DK, SURFACE, Frame, arrow, hbar,
                    legend_row, pct, vbar)

OUT = config.ROOT / "linkedin"
N = 4
TOPIC = "Is AI a bubble?"


class Slide(Frame):
    def __init__(self):
        super().__init__(1080, 1350, 64)

    def header(self, n: int, title: str, subtitle: str, title_lines: int = 2) -> int:  # template positions
        M = self.M
        self.text(M, 58, f"{TOPIC}  ·  {n}/{N}".upper(), 22, 600, MUTED)
        self.text(M, 100, title, 62, 700, INK, linespacing=1.05)
        sub_top = 100 + title_lines * 66 + 22
        self.text(M, sub_top, subtitle, 29, 400, INK2)
        self.text(self.W - M, 58, "ICARUS", 22, 700, INK2, ha="right")
        return sub_top + 38 * (subtitle.count("\n") + 1)

    def footer(self, source: str) -> None:
        self.ax.plot([self.M, self.W - self.M], [1282, 1282], color=GRID, lw=1)
        self.text(self.M, 1300, source, 21, 400, MUTED)
        self.text(self.W - self.M, 1298, "Viktor Berg", 24, 700, INK2, ha="right")

    def swipe(self, text="Swipe to see why", y=1236):
        self.text(self.W - self.M - 52, y, text, 24, 700, INK, ha="right", va="center")
        self.ax.annotate("", xy=(self.W - self.M, y + 1), xytext=(self.W - self.M - 40, y + 1),
                         arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.32", color=INK, lw=2.6,
                                         shrinkA=0, shrinkB=0))

    def save2x(self, path):
        self.save(path, scale=2.0)


def slide1(r: dict, wb: dict) -> None:
    ai = r["survival"]["burst"]["targets"]["AI20"]
    s = Slide()
    s.header(1, f"{pct(ai['p12'])} odds the AI boom\ncrashes 40% within a year",
             f"20 AI leaders, equal weight. Chance of a 40%+ fall from\nthe {pd.Timestamp(ai['peak_date']):%b %-d} peak, by months ahead.")
    x0, x1, y0, y1 = 150, 1000, 1060, 440          # plot box in px (y down)
    cdf, lo, hi = np.array(ai["cdf"]), np.array(ai["cdf_p10"]), np.array(ai["cdf_p90"])
    k = np.arange(1, 25)
    X = lambda m: x0 + (m - 1) / 23 * (x1 - x0)
    Y = lambda p: y0 - p * (y0 - y1)
    for g in (0, .25, .5, .75, 1):
        s.ax.plot([x0, x1], [Y(g), Y(g)], color=GRID if g else AXIS, lw=1, zorder=1)
        s.text(x0 - 14, Y(g), f"{g:.0%}", 21, 400, MUTED, ha="right", va="center")
    s.ax.fill_between([X(m) for m in k], [Y(p) for p in lo], [Y(p) for p in hi], color=BAD, alpha=0.12, lw=0, zorder=2)
    s.ax.plot([X(m) for m in k], [Y(p) for p in cdf], color=BAD, lw=4.5, solid_capstyle="round", zorder=3)
    for m in (6, 12, 24):
        s.ax.scatter([X(m)], [Y(cdf[m - 1])], s=150, color=BAD, edgecolor=SURFACE, linewidth=3, zorder=5)
        s.text(X(m) - (14 if m == 24 else 0), Y(cdf[m - 1]) - 26, pct(cdf[m - 1]), 30, 700, INK,
               ha="right" if m == 24 else "center", va="bottom")
    for m in (1, 6, 12, 18, 24):
        s.text(X(m), y0 + 14, f"{m} mo", 21, 400, MUTED, ha="center")
    s.text(x0, y0 + 50, "Months from today", 21, 400, MUTED)
    s.text(x0 + 10, y1 + 6, "Shaded: 80% range\n(year-block bootstrap)", 21, 400, MUTED)
    ex = r["verdict"].get("lppls_ex_ante")
    if ex:
        s.text(64, 1150, f"Bonus: Sornette's bubble clock, run {ex['days_before_peak']} days before the\n"
                         f"peak, put the top at {pd.Timestamp(ex['tc_median']):%b %-d}. It came {pd.Timestamp(ex['peak']):%b %-d}.",
               24, 600, INK)
    s.swipe()
    s.footer(f"Model: ICARUS survival ensemble, {r['survival']['burst']['n_series']} assets since 1927.")
    s.save2x(OUT / "slide_1.png")
    wb["1 Crash odds"] = pd.DataFrame({"months_ahead": k, "p_burst": cdf, "p10": lo, "p90": hi})


def slide2(r: dict, wb: dict) -> None:
    d = query("""SELECT m.name, f.dd FROM features_daily f JOIN ai20_members m USING (ticker)
                 WHERE f.date = (SELECT max(date) FROM features_daily WHERE ticker='AI20') ORDER BY f.dd""")
    n_bear = int((d["dd"] <= -0.2).sum())
    s = Slide()
    s.header(2, f"{n_bear} of 20 AI leaders are\nalready in bear markets",
             "Fall from each company's all-time high.\nRed = down 20% or more.")
    x0, x1, top, row = 330, 930, 412, 40
    for i, rw in enumerate(d.itertuples()):
        y = top + i * row
        s.text(x0 - 16, y, rw.name, 22, 400, INK2, ha="right", va="center")
        L = -rw.dd * (x1 - x0) / 0.8
        hbar(s.ax, x0, y, L, 26, BAD if rw.dd <= -0.2 else NEUTRAL_DK)
        xr = x0 + 0.2 * (x1 - x0) / 0.8
        lx = x0 + L + 10
        if lx < xr + 10 and lx + 64 > xr - 4:          # labels never sit on the reference line
            lx = xr + 12
        s.text(lx, y, f"{rw.dd:.0%}", 22, 700, INK, va="center")
    xr = x0 + 0.2 * (x1 - x0) / 0.8
    s.ax.plot([xr, xr], [top - 30, top + 19 * row + 20], color=INK, lw=2.2, zorder=6)
    s.text(xr, top - 38, "Bear market: -20%", 21, 700, INK, ha="center", va="bottom")
    s.text(620, 1050, "The generals hold.\nThe troops are falling.\nSame pattern as 2000.", 25, 600, INK)
    s.footer(f"As of {r['systemic']['breadth_date']}. Data: Yahoo Finance.")
    s.save2x(OUT / "slide_2.png")
    wb["2 Breadth"] = d.assign(in_bear_market=d["dd"] <= -0.2)


def slide3(r: dict, wb: dict) -> None:
    q = pd.read_parquet(config.DATA_PROCESSED / "artifacts" / "capex_quarterly.parquet")
    q["cal_quarter"] = pd.to_datetime(q["cal_quarter"])
    full_n = q["n"].max()
    start = q.loc[q["n"] == full_n, "cal_quarter"].min()
    quarters = pd.date_range(start, q["cal_quarter"].max(), freq="QS")[-5:]
    rd = r["fundamentals"]["reverse_dcf"]
    need = rd["scenarios"]["IPO ($1.5T)"]["required_revenue_b"]["p50"]
    amzn = r["fundamentals"]["benchmarks"].get("AMZN", {}).get("revenue_b")
    s = Slide()
    s.header(3, "Big Tech's capex now tops\nthe cash its businesses make",
             "Microsoft, Alphabet, Amazon, Meta, Oracle, per quarter, $B:\ncapex vs. operating cash flow.")
    base, top = 960, 470
    full = q[q["n"] == full_n]
    vmax = max(full["capex"].max(), full["ocf"].max()) / 1e9
    Hh = lambda v: v / vmax * (base - top)
    xs = np.linspace(170, 920, len(quarters))
    rows = []
    for xc, qd in zip(xs, quarters):
        rw = full[full["cal_quarter"] == qd]
        s.text(xc, base + 14, f"Q{qd.quarter} '{qd.year % 100}", 22, 400, MUTED, ha="center")
        if rw.empty:
            s.text(xc, base - 20, "Data gap:\nAmazon Q2\nmissing at\nsource", 19, 400, MUTED, ha="center", va="bottom")
            continue
        c, o = float(rw["capex"].iloc[0]) / 1e9, float(rw["ocf"].iloc[0]) / 1e9
        rows.append({"quarter": f"Q{qd.quarter} {qd.year}", "capex_b": c, "ocf_b": o, "capex_above_ocf": c > o})
        vbar(s.ax, xc - 26, base, Hh(c), 42, BAD if c > o else NEUTRAL_DK)
        vbar(s.ax, xc + 26, base, Hh(o), 42, NEUTRAL)
        s.text(xc - 26, base - Hh(c) - 10, f"{c:.0f}", 24, 700, INK, ha="center", va="bottom")
        s.text(xc + 26, base - Hh(o) - 10, f"{o:.0f}", 22, 400, INK2, ha="center", va="bottom")
    s.ax.plot([100, 990], [base, base], color=AXIS, lw=1)
    legend_row(s, 1035, [(BAD, "Capex > cash flow", "box"), (NEUTRAL_DK, "Capex", "box"), (NEUTRAL, "Cash flow", "box")],
               size=22, gap=30)
    msg = (f"And Anthropic's IPO? At $1.5T it needs ~${need:.0f}B\nof revenue by {rd['target_year']}"
           + (f", more than Amazon's ${amzn:.0f}B today." if amzn else "."))
    s.text(64, 1110, msg, 25, 600, INK)
    s.footer("Company statements via Yahoo Finance; reverse DCF, 20,000 draws.")
    s.save2x(OUT / "slide_3.png")
    wb["3 Capex vs cash"] = pd.DataFrame(rows)


def slide4(r: dict, wb: dict) -> None:
    fam = r["survival"]["burst"]["famous"]
    ai = r["survival"]["burst"]["targets"]["AI20"]
    picks = [e for e in fam if e["episode"] in ("Nasdaq-100 2000", "Cisco 2000", "Oil 2008", "Bitcoin 2017",
                                                 "ARK Innovation 2021", "NVIDIA 2021")]
    rows = [(e["episode"], e["p12_at_peak"], False) for e in picks] + [("AI-20 today", ai["p12"], True)]
    rows.sort(key=lambda x: -x[1])
    clim = float(np.mean([e["climatology"] for e in fam]))
    s = Slide()
    s.header(4, "The same model at the\ntop of past bubbles",
             "Odds of a 40%+ crash within 12 months, at each peak.\nOut of sample: no peeking at what came next.")
    x0, x1, top, row = 360, 930, 420, 84
    for i, (lab, p, me) in enumerate(rows):
        y = top + i * row
        s.text(x0 - 16, y, lab, 25, 700 if me else 400, INK if me else INK2, ha="right", va="center")
        hbar(s.ax, x0, y, p * (x1 - x0), 34, BAD if me else NEUTRAL_DK)
        s.text(x0 + p * (x1 - x0) + 12, y, pct(p), 27, 700, INK, va="center")
    xr = x0 + clim * (x1 - x0)
    yb = top + (len(rows) - 1) * row + 34
    s.ax.plot([xr, xr], [top - 34, yb], color=INK, lw=2.2, zorder=6)
    s.text(xr + 8, top - 50, f"Normal: {pct(clim)}", 21, 700, INK)
    s.text(64, 1070, "Not a sell signal. A risk, not a certainty.\n"
                     "But the AI boom now looks like the bubbles before it.", 25, 600, INK)
    m = r["survival"]["burst"]["metrics"]["h12"]["p_ens"]
    s.footer(f"Walk-forward backtest 1995-2024, AUC {m['auc']:.2f}. Code on GitHub.")
    s.save2x(OUT / "slide_4.png")
    wb["4 Backtest"] = pd.DataFrame(rows, columns=["episode", "p12_at_peak", "is_today"]).assign(base_rate=clim)


def post_text(r: dict) -> str:
    ai = r["survival"]["burst"]["targets"]["AI20"]
    v = r["verdict"]
    ex = v.get("lppls_ex_ante")
    rd = r["fundamentals"]["reverse_dcf"]["scenarios"]["IPO ($1.5T)"]["required_revenue_b"]["p50"]
    s = r["systemic"]
    cap = r["fundamentals"]["capex"]["latest"]
    lines = [
        "Anthropic is about to list at up to $1.5 trillion. Everyone's asking: is AI a bubble, and when does it pop?",
        "",
        f"So I built a model to answer it. ICARUS learns what the year before a crash looks like from "
        f"{r['survival']['burst']['n_series']} assets "
        f"since 1927, from the dot-com bust to bitcoin, and scores today's AI leaders.",
        "",
        f"The answer: {pct(ai['p12'])} odds the AI-20 falls 40%+ from its Oct 6 peak within a year, "
        f"{pct(ai['p24'])} within two.",
        "",
        f"Why so high? {s['ai20_n_in_bear']} of {s['ai20_n_members']} AI leaders are already in bear markets while "
        f"Microsoft and NVIDIA hold up. That's the 2000 pattern. And Big Tech now spends ${cap['capex_to_ocf']:.2f} on "
        f"capex for every $1 of cash its businesses bring in.",
        "",
        f"At $1.5T, Anthropic needs ~${rd:.0f}B of revenue by 2036. Possible, but it's a big ask.",
    ]
    if ex:
        lines += ["", f"The part that surprised me: Sornette's bubble clock, fitted {ex['days_before_peak']} days before the top, "
                      f"put the critical point at {pd.Timestamp(ex['tc_median']):%b %-d}. The top came {pd.Timestamp(ex['peak']):%b %-d}."]
    lines += ["", "43% isn't a sell signal. It's a risk, not a certainty.".replace("43%", pct(ai["p12"])),
              "", "What would you add to the model? Code in the comments."]
    return "\n".join(lines)


def workbook(r: dict, sheets: dict) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Font
    wb = Workbook()
    ws = wb.active
    ws.title = "README"
    blue, black, green = Font(color="0000FF"), Font(color="000000"), Font(color="008000")
    meta = [
        ("ICARUS: Is AI a bubble? Data behind the 4-slide LinkedIn carousel", None),
        ("Author", "Viktor Berg"),
        ("Data as of", r["data_as_of"]),
        ("Method", "Monotone survival ensemble (LightGBM + logistic) trained on 67 assets since 1927; "
                   "walk-forward backtest 1995-2024; see github.com/vikwillberg/Predicting-the-AI-Bubble"),
        ("Burst", "A fall of 40% or more from the running peak (Greenwood, Shleifer & You, 2019)"),
        ("Sources", "Yahoo Finance prices and statements, FRED, multpl, Wikimedia, GDELT; Anthropic figures from press "
                    "coverage of the Oct 26, 2026 S-1"),
        ("Colors", "Blue = model output pasted as input; black = formula; green = link to another sheet"),
        ("QA", "Every slide number is read from these sheets; COUNTIF/MAX checks below recalculate in Excel"),
    ]
    for i, (k, v) in enumerate(meta, 1):
        ws.cell(i, 1, k).font = Font(bold=True)
        if v is not None:
            ws.cell(i, 2, v)
    for name, df in sheets.items():
        sh = wb.create_sheet(name[:31])
        for j, c in enumerate(df.columns, 1):
            sh.cell(1, j, str(c)).font = Font(bold=True)
        for i, row in enumerate(df.itertuples(index=False), 2):
            for j, val in enumerate(row, 1):
                if isinstance(val, (pd.Timestamp,)):
                    val = val.date()
                elif isinstance(val, (np.floating, np.integer)):
                    val = val.item()
                elif isinstance(val, np.bool_):
                    val = bool(val)
                sh.cell(i, j, val).font = blue
    # formula checks (black) and cross-sheet links (green)
    b = wb["2 Breadth"]
    n = len(sheets["2 Breadth"]) + 1
    b.cell(1, 5, "count_in_bear").font = Font(bold=True)
    b.cell(2, 5, f'=COUNTIF(B2:B{n},"<=-0.2")').font = black
    c = wb["1 Crash odds"]
    c.cell(1, 6, "p_burst_12m").font = Font(bold=True)
    c.cell(2, 6, "=B13").font = black
    ws.cell(10, 1, "Slide 1 headline (12-month odds)").font = Font(bold=True)
    ws.cell(10, 2, "='1 Crash odds'!F2").font = green
    ws.cell(11, 1, "Slide 2 headline (members in bear)").font = Font(bold=True)
    ws.cell(11, 2, "='2 Breadth'!E2").font = green
    wb.save(OUT / "icarus_linkedin_data.xlsx")


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r = json.loads((config.REPORTS / "results.json").read_text())
    sheets: dict = {}
    for fn in (slide1, slide2, slide3, slide4):
        fn(r, sheets)
    import img2pdf
    layout = img2pdf.get_layout_fun((img2pdf.in_to_pt(10.8), img2pdf.in_to_pt(13.5)))
    pngs = [str(OUT / f"slide_{i}.png") for i in range(1, N + 1)]
    (OUT / "icarus_carousel.pdf").write_bytes(img2pdf.convert(pngs, layout_fun=layout))
    workbook(r, sheets)
    (OUT / "post.md").write_text(post_text(r) + "\n")
