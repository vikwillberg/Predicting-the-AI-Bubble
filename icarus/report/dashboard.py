"""Export a compact JSON for the interactive dashboard (docs/index.html)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..warehouse.build import query

ART = config.DATA_PROCESSED / "artifacts"


def _series(df: pd.DataFrame, x: str, ys: list[str], every: int = 1, digits: int = 4) -> dict:
    d = df.iloc[::every]
    out = {"x": [str(pd.Timestamp(v).date()) for v in d[x]]}
    for y in ys:
        out[y] = [None if pd.isna(v) else round(float(v), digits) for v in d[y]]
    return out


def build() -> dict:
    r = json.loads((config.REPORTS / "results.json").read_text())
    ai = query("SELECT date, price FROM series WHERE ticker='AI20' AND date >= '2022-11-30' ORDER BY date")
    ai["price"] = 100 * ai["price"] / ai["price"].iloc[0]
    ndx = query("SELECT date, price FROM series WHERE ticker='^NDX' AND date >= '2022-11-30' ORDER BY date")
    ndx["price"] = 100 * ndx["price"] / ndx["price"].iloc[0]
    breadth = query("""SELECT m.name, m.layer, f.dd FROM features_daily f JOIN ai20_members m USING (ticker)
                       WHERE f.date = (SELECT max(date) FROM features_daily WHERE ticker='AI20') ORDER BY f.dd""")
    capq = pd.read_parquet(ART / "capex_quarterly.parquet")
    ci = pd.read_parquet(ART / "lppls_ci_AI20.parquet")
    hist = pd.read_parquet(ART / "risk_index_ndx_history.parquet")
    dot = pd.read_parquet(ART / "clock_ndx.parquet")
    burst, bear = r["survival"]["burst"], r["survival"]["bear"]
    data = {
        "generated_utc": r["generated_utc"], "data_as_of": r["data_as_of"], "verdict": r["verdict"],
        "targets": {"burst": burst["targets"], "bear": bear["targets"]},
        "skill": {"burst": burst["metrics"], "bear": bear["metrics"], "n_series": burst["n_series"],
                  "n_rows": burst["n_rows"]},
        "famous": burst["famous"],
        "pillars": {k: {"score": v["score"], "components": v["components"]} for k, v in r["risk_index"]["pillars"].items()},
        "composite": r["risk_index"]["composite"],
        "ai20": _series(ai, "date", ["price"], every=2, digits=2),
        "ndx": _series(ndx, "date", ["price"], every=2, digits=2),
        "lppls_ci": _series(ci, "date", ["pos_conf"]),
        "lppls": r["lppls"]["current"].get("AI20", {}),
        "lppls_episodes": r["lppls"]["episodes"],
        "breadth": breadth.round(4).to_dict("records"),
        "capex": [{"quarter": f"Q{pd.Timestamp(q).quarter} {pd.Timestamp(q).year}", "capex_b": round(c / 1e9, 1),
                   "ocf_b": round(o / 1e9, 1), "complete": bool(n == capq["n"].max())}
                  for q, c, o, n in zip(capq["cal_quarter"], capq["capex"], capq["ocf"], capq["n"])],
        "anthropic": {"rounds": [x for x in r["fundamentals"]["private"]["rounds"] if x["lab"] == "anthropic"],
                      "reverse_dcf": r["fundamentals"]["reverse_dcf"], "benchmarks": r["fundamentals"]["benchmarks"],
                      "s1": r["fundamentals"]["private"]["s1"],
                      "implied": r["fundamentals"]["private"]["implied_ipo_multiples"],
                      "run_rate": r["fundamentals"]["private"]["latest_run_rate_b"]},
        "nvidia": r["fundamentals"]["nvidia"], "valuation": {k: v for k, v in r["fundamentals"]["valuation"].items() if k != "members"},
        "analogs": r["analogs"]["summary"], "analog_table": [
            {k: x[k] for k in ("episode", "today_equivalent", "months_to_peak", "drawdown_ahead", "weight")}
            for x in r["analogs"]["table"]],
        "clock": r["analogs"]["clock"], "regime": r["regimes"]["current"],
        "dotcom": {"months": [round(float(m), 2) for m in dot["months"]], "level": [round(float(v), 1) for v in dot["level"]]},
        "systemic": r["systemic"], "psy": {k: {kk: v[kk] for kk in ("label", "global_reject", "episodes", "explosive_now")}
                                           for k, v in r["psy"].items()},
        "ndx_index_history": _series(hist, "date", ["index"], every=3, digits=1),
    }
    return json.loads(json.dumps(data, default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else str(o)))


TEMPLATE = Path(__file__).parent / "templates" / "dashboard.html"


def render_html(data: dict) -> str:
    """The dashboard as an HTML fragment with the data inlined (no fetch, so the
    page is complete the moment it loads)."""
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE.read_text().replace("/*__ICARUS_DATA__*/null", payload)


def run() -> None:
    data = build()
    out = config.DOCS / "data" / "dashboard.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":")))
    # GitHub Pages serves docs/index.html: a full document (doctype avoids quirks mode).
    head = ('<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}'
            '[hidden]{display:none!important}img{max-width:100%}</style>\n')
    (config.DOCS / "index.html").write_text(head + render_html(data))
