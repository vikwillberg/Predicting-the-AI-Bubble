"""Fundamentals: is the money behind the AI boom real?

1. Capex strain: hyperscaler capital spending vs the operating cash flow that
   has to pay for it. Spending above cash flow means the build-out runs on
   debt and equity issuance, which is how capex booms end (telecom, 2001).
2. "AI's $600B question" (David Cahn, Sequoia, 2024), refreshed with live data:
   NVIDIA's data-center run-rate x2 (GPUs are about half of data-center cost)
   x2 (the end customer needs a ~50% gross margin) = the annual end-user AI
   revenue needed to justify the build-out.
3. NVIDIA cash conversion: operating cash flow / net income.
4. Frontier-lab valuations: Anthropic and OpenAI marks vs revenue run-rates.
5. Anthropic IPO reverse DCF: what must be true for $1.5T to be fair value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..warehouse.build import query

HYPERSCALERS = ["MSFT", "GOOGL", "AMZN", "META", "ORCL"]
NVDA_DATACENTER_SHARE = 0.89   # Data center share of NVIDIA revenue (FY2026 10-K, ~89%); documented assumption


def _yahoo(freq: str) -> pd.DataFrame:
    path = config.DATA_RAW / "yahoo_fundamentals.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "period_end", "freq", "item", "value"])
    y = pd.read_parquet(path)
    y["period_end"] = pd.to_datetime(y["period_end"])
    return y[y["freq"] == freq]


def capex_cycle() -> dict:
    q = query("SELECT * FROM fundamentals_quarterly")
    q["cal_quarter"] = pd.to_datetime(q["cal_quarter"])
    hs = q[q["ticker"].isin(HYPERSCALERS) & q["concept_group"].isin(["capex", "ocf"])]
    wide = hs.pivot_table(index=["cal_quarter", "ticker"], columns="concept_group", values="value").reset_index()
    wide = wide.dropna(subset=["capex", "ocf"])
    by_q = wide.groupby("cal_quarter").agg(capex=("capex", "sum"), ocf=("ocf", "sum"),
                                          n=("ticker", "nunique"),
                                          n_capex_gt_ocf=("capex", lambda s: int((s > wide.loc[s.index, "ocf"]).sum())))
    by_q["capex_to_ocf"] = by_q["capex"] / by_q["ocf"]
    full = by_q[by_q["n"] == by_q["n"].max()]
    latest = full.iloc[-1] if len(full) else by_q.iloc[-1]

    a = _yahoo("yearly")
    a = a[a["ticker"].isin(HYPERSCALERS) & a["item"].isin(["CapitalExpenditure", "OperatingCashFlow",
                                                            "DepreciationAndAmortization", "TotalRevenue"])].copy()
    a["year"] = a["period_end"].dt.year
    annual = a.pivot_table(index=["year", "ticker"], columns="item", values="value").reset_index()
    annual["capex"] = -annual["CapitalExpenditure"]
    ann = annual.groupby("year").agg(capex=("capex", "sum"), ocf=("OperatingCashFlow", "sum"),
                                     depreciation=("DepreciationAndAmortization", "sum"),
                                     revenue=("TotalRevenue", "sum"), n=("ticker", "nunique"))
    ann = ann[ann["n"] == len(HYPERSCALERS)]
    ann["capex_to_ocf"] = ann["capex"] / ann["ocf"]
    guide = config.load("private_marks")["capex_guidance_2026"]
    per_company = wide[wide["cal_quarter"] == latest.name][["ticker", "capex", "ocf"]] if len(full) else wide.iloc[0:0]
    return {
        "quarterly": by_q.reset_index(),
        "annual": ann.reset_index(),
        "latest_quarter": str(latest.name.date()) if hasattr(latest.name, "date") else str(latest.name),
        "latest": {"capex_b": float(latest["capex"] / 1e9), "ocf_b": float(latest["ocf"] / 1e9),
                   "capex_to_ocf": float(latest["capex_to_ocf"]),
                   "companies_spending_more_than_ocf": int(latest["n_capex_gt_ocf"]),
                   "companies": int(latest["n"])},
        "per_company_latest": per_company.assign(capex_b=lambda d: d["capex"] / 1e9, ocf_b=lambda d: d["ocf"] / 1e9)
                                         [["ticker", "capex_b", "ocf_b"]].to_dict("records"),
        "baseline_capex_to_ocf": float(ann.loc[ann.index.min(), "capex_to_ocf"]) if len(ann) else None,
        "baseline_year": int(ann.index.min()) if len(ann) else None,
        "guidance_2026_total_b": guide["total_midpoint"], "actual_2025_total_b": guide["total_2025"],
    }


def nvidia() -> dict:
    q = _yahoo("quarterly")
    n = q[q["ticker"] == "NVDA"].pivot_table(index="period_end", columns="item", values="value").sort_index()
    if n.empty:
        return {}
    n["cash_conversion"] = n["OperatingCashFlow"] / n["NetIncome"]
    last = n.iloc[-1]
    run_rate = float(last["TotalRevenue"] * 4)
    required = run_rate * NVDA_DATACENTER_SHARE * 2 * 2
    marks = config.load("private_marks")
    labs = marks["anthropic"]["run_rate_history"][-1]["run_rate_b"] + marks["openai"]["run_rate_history"][-1]["run_rate_b"]
    return {
        "quarterly": n.reset_index(),
        "latest_quarter_end": str(n.index[-1].date()),
        "latest_revenue_b": float(last["TotalRevenue"] / 1e9),
        "revenue_run_rate_b": run_rate / 1e9,
        "cash_conversion_latest": float(last["cash_conversion"]),
        "cash_conversion_prior_avg": float(n["cash_conversion"].iloc[:-1].mean()),
        "cahn_required_end_user_revenue_b": required / 1e9,
        "frontier_lab_run_rate_b": float(labs),
        "coverage_ratio": float(labs / (required / 1e9)),
        "datacenter_share_assumption": NVDA_DATACENTER_SHARE,
    }


def private_marks() -> dict:
    m = config.load("private_marks")
    rows = []
    for lab in ("anthropic", "openai"):
        rr = pd.DataFrame(m[lab].get("run_rate_history", []))
        if len(rr):
            rr["date"] = pd.to_datetime(rr["date"])
        for r in m[lab]["rounds"]:
            d = pd.Timestamp(r["date"])
            run = r.get("run_rate_b")
            if run is None and len(rr):
                prior = rr[rr["date"] <= d + pd.Timedelta(days=45)]
                run = float(prior["run_rate_b"].iloc[-1]) if len(prior) else None
            rows.append({"lab": lab, "date": d, "event": r["event"], "post_money_b": r["post_money_b"],
                         "run_rate_b": run, "multiple": r["post_money_b"] / run if run else None,
                         "confidence": r["confidence"]})
    rounds = pd.DataFrame(rows).sort_values(["lab", "date"])
    a = m["anthropic"]
    latest_rr = a["run_rate_history"][-1]["run_rate_b"]
    s1 = a["s1"]
    h1_annualised = s1["revenue_h1_2026_b"] * 2
    targets = {"IPO low ($1.5T)": 1500.0, "IPO high ($1.6T)": 1600.0, "Earlier talk ($2T)": 2000.0}
    implied = {k: {"x_run_rate": v / latest_rr, "x_h1_2026_annualised": v / h1_annualised} for k, v in targets.items()}
    return {
        "rounds": rounds,
        "anthropic_run_rate_history": pd.DataFrame(a["run_rate_history"]),
        "latest_run_rate_b": latest_rr,
        "s1": s1,
        "h1_2026_annualised_b": h1_annualised,
        "yoy_h1_growth": s1["revenue_h1_2026_b"] / s1["revenue_h1_2025_b"],
        "implied_ipo_multiples": implied,
        "ipo_targets": a["ipo_targets"], "bear_case": a["bear_case"],
    }


def reverse_dcf(valuations_b: dict[str, float], n_sims: int = 20000, years: int = 10, seed: int = 1) -> dict:
    """What revenue must Anthropic reach in `years` years for each valuation to be fair?

    Value today = terminal value discounted, after share dilution:
        V = R_N * margin * multiple * (1 - dilution)^N / (1 + r)^N
    Every assumption is drawn from a range so the answer is a distribution:
        starting run-rate R0  U(50, 70)  $B  (reported ~$65B; gross-vs-net revenue debate)
        discount rate r       U(9%, 13%)
        mature FCF margin     U(20%, 35%)    (Alphabet ~25%, Microsoft ~30%)
        exit EV/FCF multiple  U(18, 30)
        annual dilution       U(1%, 3%)     (stock-based compensation)
    """
    rng = np.random.default_rng(seed)
    R0 = rng.uniform(50, 70, n_sims)
    r = rng.uniform(0.09, 0.13, n_sims)
    margin = rng.uniform(0.20, 0.35, n_sims)
    mult = rng.uniform(18, 30, n_sims)
    dil = rng.uniform(0.01, 0.03, n_sims)
    out = {}
    for name, V in valuations_b.items():
        RN = V * (1 + r) ** years / ((1 - dil) ** years * margin * mult)
        cagr = (RN / R0) ** (1 / years) - 1
        out[name] = {
            "valuation_b": V,
            "required_revenue_b": {q: float(np.quantile(RN, p)) for q, p in (("p10", .1), ("p50", .5), ("p90", .9))},
            "required_cagr": {q: float(np.quantile(cagr, p)) for q, p in (("p10", .1), ("p50", .5), ("p90", .9))},
        }
    return {"years": years, "target_year": pd.Timestamp.now().year + years, "scenarios": out}


def megacap_revenue_benchmarks() -> dict:
    a = _yahoo("yearly")
    a = a[a["item"] == "TotalRevenue"].sort_values("period_end").groupby("ticker").tail(1)
    return {row.ticker: {"revenue_b": float(row.value / 1e9), "fiscal_year_end": str(row.period_end.date())}
            for row in a.itertuples() if row.ticker in ("MSFT", "GOOGL", "AMZN", "META", "NVDA", "ORCL")}


def valuation_snapshot() -> dict:
    path = config.DATA_RAW / "valuation_snapshots.parquet"
    if not path.exists():
        return {}
    v = pd.read_parquet(path)
    v["date"] = pd.to_datetime(v["date"])
    v = v[v["date"] == v["date"].max()]
    cap = v["marketCap"].sum()
    pos = v[v["trailingPE"] > 0]
    fpos = v[v["forwardPE"] > 0]
    ps = v[(v["priceToSalesTrailing12Months"] > 0) & (v["ticker"] != "TSM")]  # TSM: TWD revenue vs USD cap
    return {
        "date": str(v["date"].max().date()),
        "ai20_market_cap_t": float(cap / 1e12),
        "cap_weighted_trailing_pe": float(pos["marketCap"].sum() / (pos["marketCap"] / pos["trailingPE"]).sum()),
        "cap_weighted_forward_pe": float(fpos["marketCap"].sum() / (fpos["marketCap"] / fpos["forwardPE"]).sum()),
        "cap_weighted_price_to_sales": float(ps["marketCap"].sum() / (ps["marketCap"] / ps["priceToSalesTrailing12Months"]).sum()),
        "members": v[["ticker", "marketCap", "trailingPE", "forwardPE", "priceToSalesTrailing12Months"]].to_dict("records"),
    }


def analyse() -> dict:
    pm = private_marks()
    vals = {"Series G ($380B)": 380.0, "Series H ($965B)": 965.0, "IPO ($1.5T)": 1500.0, "IPO ($2T)": 2000.0}
    return {
        "capex": capex_cycle(),
        "nvidia": nvidia(),
        "private": pm,
        "reverse_dcf": reverse_dcf(vals),
        "benchmarks": megacap_revenue_benchmarks(),
        "valuation": valuation_snapshot(),
    }
