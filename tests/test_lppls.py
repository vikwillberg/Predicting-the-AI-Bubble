"""LPPLS: recovery of known critical times, and Go/Python engine parity."""

import numpy as np
import pandas as pd
import pytest

from icarus import config
from icarus.models import lppls


def synthetic_bubble(n=400, tc=420.0, m=0.45, w=9.0, A=5.0, B=-0.06, C1=0.005, C2=-0.004, noise=0.0, seed=0):
    t = np.arange(n, dtype=float)
    dt = tc - t
    f = dt ** m
    y = A + B * f + C1 * f * np.cos(w * np.log(dt)) + C2 * f * np.sin(w * np.log(dt))
    if noise:
        y = y + np.random.default_rng(seed).normal(0, noise, n)
    return t, y


def test_linear_fit_is_exact_at_true_nonlinear_params():
    t, y = synthetic_bubble()
    beta, sse = lppls.linear_fit(t, y, 420.0, 0.45, 9.0)
    assert sse < 1e-18
    np.testing.assert_allclose(beta, [5.0, -0.06, 0.005, -0.004], rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("tc_true", [410.0, 430.0])
def test_python_fit_recovers_tc(tc_true):
    t, y = synthetic_bubble(tc=tc_true, noise=0.002)
    fit = lppls.fit_window(t, y, lppls.LPPLSConfig())
    assert abs(fit["tc"] - tc_true) < 8
    assert fit["bubble"] == 1


def test_random_walk_is_rarely_a_qualified_bubble():
    rng = np.random.default_rng(42)
    hits = 0
    for _ in range(10):
        y = np.cumsum(rng.normal(0, 0.01, 300))
        fit = lppls.fit_window(np.arange(300.0), y, lppls.LPPLSConfig())
        hits += fit["qualified"] and fit["bubble"] == 1
    assert hits <= 3


@pytest.mark.skipif(not config.ENGINE_BIN.exists(), reason="Go engine not built (make engine)")
def test_go_and_python_engines_agree():
    _, y = synthetic_bubble(n=420, tc=440.0, noise=0.003, seed=3)
    t = np.arange(len(y), dtype=float)
    jobs = [(0, 299), (50, 349), (119, 419)]
    cfg = lppls.LPPLSConfig()
    go = lppls.fit_jobs(t, y, jobs, cfg, engine="go")
    py = lppls.fit_jobs(t, y, jobs, cfg, engine="python")
    for g, p in zip(go, py):
        assert g["qualified"] == p["qualified"]
        assert abs(g["tc"] - p["tc"]) < 0.5
        assert g["sse"] == pytest.approx(p["sse"], rel=1e-5, abs=1e-12)


def test_scan_maps_tc_to_calendar_dates():
    _, y = synthetic_bubble(n=300, tc=320.0)
    idx = pd.bdate_range("2025-01-01", periods=300)
    s = pd.Series(np.exp(y), index=idx)
    cfg = lppls.LPPLSConfig(window_min=150, window_max=250, window_step=50)
    fits = lppls.scan(s, [299], cfg, engine="python")
    assert len(fits) == 3
    assert (fits["tc_date"] > idx[-1]).all()            # critical time is in the future
    ci = lppls.confidence_indicator(fits)
    assert 0 <= ci["pos_conf"].iloc[0] <= 1
