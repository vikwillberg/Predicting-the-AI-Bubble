"""Survival model, ensemble weighting, analogs and risk-index arithmetic."""

import numpy as np
import pandas as pd

from icarus.models import analogs, risk_index, survival
from icarus.models.panel import FEATURES


def _toy_panel(n_dates=180, n_assets=12, seed=0):
    """Monthly panel where one feature (vol_12m) drives crash risk."""
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(n_assets):
        for d in pd.date_range("1990-01-31", periods=n_dates, freq="ME"):
            x = {f: rng.normal() for f in FEATURES}
            risk = 1 / (1 + np.exp(-(1.5 * x["vol_12m"] - 2.5)))
            for k in (1, 3, 6, 12):
                y = rng.random() < 1 - (1 - risk / 6) ** k
                rows.append({"date": d, "ticker": f"A{a}", "horizon": k, "y": int(y),
                             "label_end": d + pd.DateOffset(months=k), **x})
    return pd.DataFrame(rows)


def test_cdf_is_monotone_and_learns_the_signal():
    df = _toy_panel()
    ens = survival.EnsembleSurvival(fast=True).fit(df)
    calm = df.iloc[[0]].copy()
    calm[FEATURES] = 0.0
    hot = calm.copy()
    hot["vol_12m"] = 2.5
    c_calm, c_hot = ens.cdf(calm), ens.cdf(hot)
    assert np.all(np.diff(c_calm) >= -1e-12) and np.all(np.diff(c_hot) >= -1e-12)
    assert c_hot[11] > c_calm[11]


def test_walk_forward_never_trains_on_the_test_period():
    df = _toy_panel(n_dates=96)
    oos = survival.walk_forward(df, first_year=1994, fast=True)
    assert set(oos["fold"]) <= set(range(1994, 1998))
    # every prediction belongs to its own fold's calendar year
    assert (oos["date"].dt.year == oos["fold"]).all()
    assert set(oos["w_gbm"]).issubset(set(survival.WEIGHT_GRID))


def test_best_weight_prefers_the_better_model():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 5000)
    good = np.clip(y * 0.7 + 0.15, 0, 1)
    noise = rng.random(5000)
    assert survival.best_weight(good, noise, y) == 1.0
    assert survival.best_weight(noise, good, y) == 0.0


def test_analog_alignment_finds_a_planted_match():
    idx = pd.date_range("1995-01-06", periods=600, freq="W-FRI")
    rng = np.random.default_rng(3)
    path = np.cumsum(rng.normal(0.004, 0.03, 600))
    hist = pd.Series(path, index=idx)
    peak = idx[400]
    current = path[250:354]                            # planted: "today" = week 353
    res = analogs.best_alignment(current, hist, peak)
    assert res["stretch"] == 1.0
    assert pd.Timestamp(res["today_equivalent"]) == idx[353]
    assert res["months_to_peak"] > 0


def test_percentile_and_composite():
    assert risk_index.pct_rank(pd.Series([1, 2, 3, 4]), 3) == 75.0
    pillars = {"a": {"score": 90.0}, "b": {"score": 30.0}}
    c = risk_index.composite(pillars, {"a": 1.0, "b": 1.0})
    assert c["score"] == 60.0
    assert 30 <= c["dirichlet_p10"] <= c["dirichlet_p50"] <= c["dirichlet_p90"] <= 90
