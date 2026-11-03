"""Run every model and synthesise the verdict.

Outputs
  reports/results.json             every headline number, metric and caveat
  data/processed/artifacts/*.parquet  series behind each chart
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .. import config
from ..warehouse.build import query
from . import analogs, fundamentals, lppls, panel, psy, regimes, risk_index, survival, systemic

log = logging.getLogger(__name__)
ART = config.DATA_PROCESSED / "artifacts"
TARGETS = {"AI20": "ICARUS AI-20", "^NDX": "Nasdaq-100", "SMH": "Semiconductors (SMH)",
           "NVDA": "NVIDIA", "^GSPC": "S&P 500"}
FAMOUS = [("^NDX", "2000-03-27", "Nasdaq-100 2000"), ("^SOX", "2000-03-10", "Semiconductors 2000"),
          ("CSCO", "2000-03-27", "Cisco 2000"), ("000001.SS", "2007-10-16", "Shanghai 2007"),
          ("CL=F", "2008-07-03", "Oil 2008"), ("000001.SS", "2015-06-12", "Shanghai 2015"),
          ("BTC-USD", "2017-12-16", "Bitcoin 2017"), ("ARKK", "2021-02-12", "ARK Innovation 2021"),
          ("NVDA", "2021-11-29", "NVIDIA 2021"), ("BTC-USD", "2021-11-08", "Bitcoin 2021")]
LPPLS_EPISODES = [("^NDX", "2000-03-27", "Nasdaq-100 2000"), ("^N225", "1989-12-29", "Nikkei 1989"),
                  ("000001.SS", "2007-10-16", "Shanghai 2007"), ("000001.SS", "2015-06-12", "Shanghai 2015"),
                  ("BTC-USD", "2017-12-16", "Bitcoin 2017"), ("ARKK", "2021-02-12", "ARK Innovation 2021"),
                  ("NVDA", "2021-11-29", "NVIDIA 2021"), ("CL=F", "2008-07-03", "Oil 2008")]


def _save(df: pd.DataFrame, name: str) -> None:
    ART.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ART / f"{name}.parquet", index=False)


def _daily(ticker: str) -> pd.Series:
    d = query(f"SELECT date, price FROM series WHERE ticker = '{ticker}' ORDER BY date")
    d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date")["price"]


def _clean(o):
    """Make results JSON-safe."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return str(o.date()) if hasattr(o, "date") else str(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ----------------------------------------------------------------- survival block
def survival_block(monthly: pd.DataFrame, fast: bool) -> dict:
    s = config.settings()
    first_year = s["validation"]["first_test_year"]
    train_monthly = monthly[monthly["ticker"].isin(panel.training_universe())]
    out = {}
    for event in ("burst", "bear"):
        t0 = time.time()
        thr = s["events"]["burst_drawdown" if event == "burst" else "bear_drawdown"]
        stacked = panel.build_labels(train_monthly, event)
        oos = survival.walk_forward(stacked, first_year, fast=fast)
        metrics = survival.evaluate(oos)
        rel = survival.reliability(oos[oos["horizon"] == 12], "p_ens", bins=10)
        _save(oos[oos["horizon"].isin([3, 12, 24])], f"oos_{event}")
        _save(rel, f"reliability_{event}_h12")

        # Famous bubbles, out of sample: P(event within 12m) around each peak.
        famous = []
        for tk, peak, label in FAMOUS:
            sub = oos[(oos["ticker"] == tk) & (oos["horizon"] == 12)].sort_values("date")
            pk = pd.Timestamp(peak)
            win = sub[(sub["date"] >= pk - pd.DateOffset(months=24)) & (sub["date"] <= pk + pd.DateOffset(months=1))]
            if len(win):
                at_peak = win[win["date"] <= pk + pd.DateOffset(days=31)].iloc[-1]
                famous.append({"episode": label, "ticker": tk, "peak": peak,
                               "p12_at_peak": float(at_peak["p_ens"]), "climatology": float(at_peak["p_clim"]),
                               "lift": float(at_peak["p_ens"] / max(at_peak["p_clim"], 1e-9)),
                               "p12_24m_before": float(win["p_ens"].iloc[0])})
                _save(win.assign(episode=label), f"famous_{event}_{label.replace(' ', '_').replace('/', '')}")

        w_final = survival.best_weight(oos["p_gbm"].to_numpy(), oos["p_logit"].to_numpy(), oos["y"].to_numpy())
        ens = survival.EnsembleSurvival(fast=fast, w_gbm=w_final).fit(stacked)

        targets = {}
        for tk, name in TARGETS.items():
            rows = monthly[monthly["ticker"] == tk].sort_values("date")
            if rows.empty:
                continue
            row = rows.iloc[[-1]].copy()
            dd = float(row["dd"].iloc[0])
            worst = float(row["episode_worst_dd"].iloc[0])
            entry = {"name": name, "as_of": str(row["date"].iloc[0].date()), "drawdown_from_peak": dd,
                     "worst_drawdown_since_peak": worst,
                     "peak_date": str(pd.Timestamp(row["episode_peak_date"].iloc[0]).date())}
            if worst <= -thr:
                entry["already_happened"] = True
            else:
                cdf = ens.cdf(row)
                entry.update({"already_happened": False, "cdf": cdf.tolist(),
                              "p3": float(cdf[2]), "p6": float(cdf[5]), "p12": float(cdf[11]), "p24": float(cdf[23]),
                              "median_months": int(np.argmax(cdf >= 0.5) + 1) if cdf[-1] >= 0.5 else None})
            targets[tk] = entry

        ai_row = monthly[monthly["ticker"] == "AI20"].sort_values("date").iloc[[-1]]
        if not targets.get("AI20", {}).get("already_happened", True):
            B = 6 if fast else 30
            boot = survival.bootstrap_cdfs(stacked, ai_row, B=B, fast=fast, w_gbm=w_final)
            targets["AI20"]["cdf_p10"] = np.quantile(boot, 0.1, axis=0).tolist()
            targets["AI20"]["cdf_p90"] = np.quantile(boot, 0.9, axis=0).tolist()
            targets["AI20"]["p12_interval"] = [float(np.quantile(boot[:, 11], 0.1)), float(np.quantile(boot[:, 11], 0.9))]
            targets["AI20"]["p24_interval"] = [float(np.quantile(boot[:, 23], 0.1)), float(np.quantile(boot[:, 23], 0.9))]
            contrib = ens.gbm.contributions(ai_row, horizon=12)
            targets["AI20"]["drivers_h12"] = [{"feature": f, "label": panel.FEATURE_LABELS.get(f, f),
                                               "log_odds": float(v), "value": float(ai_row[f].iloc[0]) if f in ai_row else None}
                                              for f, v in contrib.head(8).items()]
        out[event] = {"threshold": thr, "metrics": metrics, "ensemble_weight_gbm": w_final, "famous": famous, "targets": targets,
                      "n_rows": int(len(stacked)), "n_series": int(stacked["ticker"].nunique()),
                      "n_events": int(stacked[stacked["horizon"] == 1]["y"].sum()),
                      "seconds": round(time.time() - t0, 1)}
        log.info("survival %s done in %.0fs", event, time.time() - t0)
    return out


# -------------------------------------------------------------------- LPPLS block
def lppls_block(fast: bool) -> dict:
    cfg = lppls.LPPLSConfig.from_settings()
    step = 10 if fast else 5
    out = {"engine": "go" if lppls._go_available() else "python", "current": {}, "episodes": []}
    for tk in ("AI20", "^NDX", "NVDA", "SMH"):
        s = _daily(tk)
        s = s[s.index >= "2018-01-01"]
        n = len(s)
        t2s = list(range(n - 1 - 126, n, step))
        if t2s[-1] != n - 1:
            t2s.append(n - 1)
        fits = lppls.scan(s, t2s, cfg)
        ci = lppls.confidence_indicator(fits)
        _save(ci, f"lppls_ci_{tk.replace('^', '')}")
        _save(fits[fits["qualified"] & (fits["bubble"] == 1)][["t2_date", "window", "tc_date", "m", "omega"]],
              f"lppls_fits_{tk.replace('^', '')}")
        # Ex-ante test: if the series has already turned down from a recent peak,
        # what did fits made strictly BEFORE that peak say about the critical time?
        peak = s.idxmax()
        ex_ante = None
        if s.iloc[-1] / s.max() - 1 <= -0.08 and peak >= s.index[-130]:
            pre = fits[(fits["t2_date"] < peak) & (fits["t2_date"] >= peak - pd.DateOffset(months=3))
                       & fits["qualified"] & (fits["bubble"] == 1)]
            if len(pre):
                last_t2 = pre["t2_date"].max()
                tcs = pd.to_datetime(pre[pre["t2_date"] == last_t2]["tc_date"])
                ex_ante = {"peak": str(peak.date()), "forecast_made": str(pd.Timestamp(last_t2).date()),
                           "days_before_peak": int((peak - pd.Timestamp(last_t2)).days),
                           "tc_median": str(tcs.median().date()), "tc_min": str(tcs.min().date()),
                           "tc_max": str(tcs.max().date()), "n_fits": int(len(tcs)),
                           "error_days": int((tcs.median() - peak).days)}
        out["current"][tk] = {"ex_ante": ex_ante, "confidence_latest": float(ci["pos_conf"].iloc[-1]),
                              "confidence_max_6m": float(ci["pos_conf"].max()),
                              "confidence_max_date": str(pd.Timestamp(ci.loc[ci["pos_conf"].idxmax(), "date"]).date()),
                              "negative_confidence_latest": float(ci["neg_conf"].iloc[-1]),
                              "tc": lppls.tc_distribution(fits, last_n_dates=6)}
    for tk, peak, label in LPPLS_EPISODES:
        s = _daily(tk)
        pk = pd.Timestamp(peak)
        idx = s.index
        start = idx.searchsorted(pk - pd.DateOffset(months=18))
        stop = idx.searchsorted(pk + pd.DateOffset(months=2))
        if start < cfg.window_max:
            continue
        t2s = list(range(start, min(stop, len(s) - 1), 15 if fast else 10))
        fits = lppls.scan(s, t2s, cfg)
        ci = lppls.confidence_indicator(fits)
        _save(ci.assign(episode=label), f"lppls_episode_{label.replace(' ', '_')}")
        pre = ci[ci["date"] <= pk]
        q = fits[fits["qualified"] & (fits["bubble"] == 1) & (fits["t2_date"] <= pk) &
                 (fits["t2_date"] >= pk - pd.DateOffset(months=3))]
        out["episodes"].append({
            "episode": label, "peak": peak,
            "max_conf_before_peak": float(pre["pos_conf"].max()) if len(pre) else None,
            "max_conf_date": str(pd.Timestamp(pre.loc[pre["pos_conf"].idxmax(), "date"]).date()) if len(pre) and pre["pos_conf"].max() > 0 else None,
            "median_tc_last_3m": str(pd.to_datetime(q["tc_date"]).median().date()) if len(q) else None,
            "tc_error_days": float((pd.to_datetime(q["tc_date"]).median() - pk).days) if len(q) else None,
        })
    errs = [e["tc_error_days"] for e in out["episodes"] if e["tc_error_days"] is not None]
    out["historical_tc_bias_days"] = {"median": float(np.median(errs)), "min": float(min(errs)),
                                      "max": float(max(errs)), "n": len(errs)} if errs else None
    return out


# ---------------------------------------------------------------------- PSY block
def psy_block(fast: bool) -> dict:
    reps = 199 if fast else 999
    out = {}
    specs = [("AI20", "2015-01-01", "W-FRI", "ICARUS AI-20 (weekly, 2015+)"),
             ("NVDA", "2015-01-01", "W-FRI", "NVIDIA (weekly, 2015+)"),
             ("SMH", "2015-01-01", "W-FRI", "Semiconductors (weekly, 2015+)"),
             ("^NDX", "1985-10-01", "ME", "Nasdaq-100 (monthly, 1985+)"),
             ("^SOX", "1994-05-01", "ME", "PHLX Semiconductor (monthly, 1994+)")]
    for tk, start, freq, label in specs:
        s = _daily(tk)
        s = s[s.index >= start].resample(freq).last().dropna()
        res = psy.date_stamp(s, lags=1, reps=reps, seed=config.settings()["psy"]["seed"])
        _save(res["series"].reset_index().rename(columns={"index": "date"}), f"psy_{tk.replace('^', '')}")
        last = res["series"].iloc[-1]
        out[tk] = {"label": label, "gsadf": res["gsadf"], "gsadf_cv95": res["gsadf_cv95"],
                   "global_reject": res["global_reject"], "episodes": res["episodes"],
                   "explosive_now": bool(last["explosive"]) and res["global_reject"],
                   "bsadf_latest": float(last["bsadf"]), "cv95_latest": float(last["cv95"])}
    return out


# --------------------------------------------------------------------- synthesis
def synthesise(r: dict) -> dict:
    burst = r["survival"]["burst"]["targets"].get("AI20", {})
    bear = r["survival"]["bear"]["targets"].get("AI20", {})
    psy_ai = r["psy"].get("AI20", {})
    ai_eps = psy_ai.get("episodes", [])
    lp = r["lppls"]["current"].get("AI20", {})
    reg = r["regimes"]["current"]
    ana = r["analogs"]["summary"]
    idx = r["risk_index"]["composite"]["score"]

    # A bubble call needs two independent kinds of evidence: statistical
    # explosiveness in the AI era (PSY on any AI-linked series since 2023, or
    # qualified LPPLS fits before the peak) AND an extreme composite risk index.
    ai_era = [tk for tk, v in r["psy"].items()
              if any(e["end"] >= "2023-01-01" for e in v.get("episodes", []))]
    lppls_ex_ante = lp.get("ex_ante")
    is_bubble = bool(idx >= 70 and (ai_era or lppls_ex_ante))
    peak_date = burst.get("peak_date")
    dd = burst.get("drawdown_from_peak")
    if burst.get("already_happened"):
        stage = "burst"
    elif reg["regime"] == "stress" and dd is not None and dd <= -0.15:
        stage = "deflating"
    elif dd is not None and dd > -0.10:
        stage = "inflating"
    else:
        stage = "wobbling"
    p12, p24 = burst.get("p12"), burst.get("p24")
    cdf = np.array(burst.get("cdf", [np.nan] * 24))
    # Month with the largest increase in burst probability = most likely burst month.
    inc = np.diff(np.concatenate([[0.0], cdf]))
    peak_month = int(np.nanargmax(inc) + 1) if np.isfinite(inc).any() else None
    as_of = pd.Timestamp(burst.get("as_of", pd.Timestamp.now()))
    cond_median = None
    if p24 and p24 > 0:
        cond = cdf / p24
        cond_median = int(np.argmax(cond >= 0.5) + 1)
    return {
        "is_bubble": is_bubble,
        "stage": stage,
        "ai20_peak_date": peak_date,
        "ai20_drawdown_from_peak": dd,
        "bubble_risk_index": idx,
        "p_burst_12m": p12, "p_burst_24m": p24,
        "p_burst_12m_interval": burst.get("p12_interval"), "p_burst_24m_interval": burst.get("p24_interval"),
        "p_bear_3m": bear.get("p3") if not bear.get("already_happened") else 1.0,
        "bear_already": bear.get("already_happened", False),
        "most_likely_burst_month": str((as_of + pd.DateOffset(months=peak_month)).strftime("%Y-%m")) if peak_month else None,
        "conditional_median_burst_month": str((as_of + pd.DateOffset(months=cond_median)).strftime("%Y-%m")) if cond_median else None,
        "burst_level_ai20": None,
        "psy_bubble_episodes_ai20": ai_eps,
        "psy_explosive_series_since_2023": ai_era,
        "lppls_ex_ante": lppls_ex_ante,
        "lppls_historical_bias_days": r["lppls"].get("historical_tc_bias_days"),
        "lppls_confidence_max_6m": lp.get("confidence_max_6m"),
        "lppls_tc": lp.get("tc"),
        "regime": reg["regime"], "regime_probabilities": reg["probabilities"],
        "analog_share_past_peak": ana["share_past_peak"],
        "analog_months_to_peak_median": ana["months_to_peak_weighted_median"],
        "analog_drawdown_ahead_median": ana["drawdown_ahead_weighted_median"],
        "dotcom_clock_equivalent_peak": r["analogs"]["clock"]["calendar_equivalent_peak"],
    }


def run(fast: bool = False) -> dict:
    t_start = time.time()
    config.ensure_dirs()
    s = config.settings()
    px_max = query("SELECT max(date) AS d FROM series WHERE ticker = 'AI20'")["d"].iloc[0]
    results: dict = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "data_as_of": str(pd.Timestamp(px_max).date()), "fast_mode": fast,
                     "settings": s}

    log.info("panel: loading monthly features + rolling PSY")
    monthly = panel.load_monthly(fast=fast)
    _save(monthly[monthly["ticker"].isin(list(TARGETS))], "monthly_targets")

    log.info("survival models")
    results["survival"] = survival_block(monthly, fast)

    log.info("PSY explosive-root tests")
    results["psy"] = psy_block(fast)

    log.info("LPPLS scans")
    results["lppls"] = lppls_block(fast)

    log.info("HMM regimes")
    reg = regimes.analyse(**{"n_states": s["hmm"]["n_states"], "restarts": 4 if fast else s["hmm"]["n_restarts"],
                             "seed": s["hmm"]["seed"]})
    _save(reg["ndx_states"].rename_axis("date").reset_index(), "hmm_ndx")
    _save(reg["ai20_states"].rename_axis("date").reset_index(), "hmm_ai20")
    results["regimes"] = {k: v for k, v in reg.items() if k not in ("ndx_states", "ai20_states", "transition")}
    results["regimes"]["transition"] = reg["transition"].round(4).to_dict()

    log.info("systemic fragility")
    sysm = systemic.analyse()
    _save(sysm["absorption"].reset_index(), "absorption")
    _save(sysm["ai20_corr"].rename_axis("date").reset_index(), "ai20_corr")
    _save(sysm["breadth"], "breadth")
    _save(sysm["concentration"], "concentration")
    results["systemic"] = sysm["summary"]

    log.info("analogs")
    an = analogs.analyse()
    _save(an["table"], "analogs")
    for k in ("ndx", "ai20", "ndx_now"):
        _save(an["clock"][k], f"clock_{k}")
    results["analogs"] = {"summary": an["summary"], "table": an["table"].to_dict("records"),
                          "clock": {k: v for k, v in an["clock"].items() if k not in ("ndx", "ai20", "ndx_now")}}

    log.info("fundamentals")
    fund = fundamentals.analyse()
    _save(fund["capex"]["quarterly"], "capex_quarterly")
    _save(fund["capex"]["annual"], "capex_annual")
    if fund["nvidia"]:
        _save(fund["nvidia"]["quarterly"], "nvidia_quarterly")
    _save(fund["private"]["rounds"], "private_rounds")
    results["fundamentals"] = {
        "capex": {k: v for k, v in fund["capex"].items() if k not in ("quarterly", "annual")},
        "capex_annual": fund["capex"]["annual"].to_dict("records"),
        "nvidia": {k: v for k, v in fund["nvidia"].items() if k != "quarterly"},
        "private": {k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else v) for k, v in fund["private"].items()},
        "reverse_dcf": fund["reverse_dcf"], "benchmarks": fund["benchmarks"],
        "valuation": fund["valuation"],
    }

    log.info("risk index")
    pillars = {
        "price_dynamics": risk_index.price_dynamics(monthly, results["lppls"]["current"].get("AI20", {}).get("confidence_max_6m")),
        "valuation": risk_index.valuation(),
        "fundamentals": risk_index.fundamentals(fund),
        "market_structure": risk_index.market_structure(sysm),
        "credit_macro": risk_index.credit_macro(),
        "sentiment": risk_index.sentiment(),
        "issuance": risk_index.issuance(pd.Timestamp(px_max)),
    }
    comp = risk_index.composite(pillars, s["risk_index"]["weights"])
    hist = risk_index.historical_ndx_index(monthly)
    _save(hist, "risk_index_ndx_history")
    results["risk_index"] = {"pillars": pillars, "composite": comp,
                             "ndx_history_check": {
                                 "peak_1999_2000": float(hist[(hist["date"] >= "1999-01-01") & (hist["date"] <= "2000-06-30")]["index"].max()),
                                 "peak_2021": float(hist[(hist["date"] >= "2021-01-01") & (hist["date"] <= "2021-12-31")]["index"].max()),
                                 "median_all": float(hist["index"].median()),
                                 "latest": float(hist["index"].iloc[-1])}}

    results["verdict"] = synthesise(results)
    results["runtime_seconds"] = round(time.time() - t_start, 1)
    out = config.REPORTS / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_clean(results), indent=2))
    log.info("wrote %s (%.0fs)", out, time.time() - t_start)
    return results
