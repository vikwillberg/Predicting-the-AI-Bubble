"""PSY bubble test: correct size under the null, power against explosive bubbles,
and agreement of the prefix-sum shortcut with a plain OLS regression."""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from icarus.models import psy


def test_prefix_sum_adf_matches_statsmodels_ols():
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(size=120))
    lags = 1
    X, z = psy._design(y, lags)
    px = psy._prefix(X, z)
    s, e = 10, 99
    t_fast = psy._adf_tstats(px, e - lags - 1, np.array([s]), X.shape[1])[0]
    # Plain OLS over the same window of y.
    yy = y[s:e + 1]
    dy = np.diff(yy)
    reg = np.column_stack([np.ones(len(yy) - 1 - lags), yy[lags:-1], dy[:-1]])
    res = sm.OLS(dy[lags:], reg).fit()
    assert abs(t_fast - res.tvalues[1]) < 1e-8


def test_size_under_random_walk_null():
    """GSADF should reject a true random walk about 5% of the time at the 95% level."""
    n, reps = 100, 300
    cvs = psy.critical_values(n, lags=0, reps=reps, seed=1)
    rng = np.random.default_rng(99)
    rejections = 0
    trials = 150
    for _ in range(trials):
        y = np.cumsum(rng.standard_normal(n) + 1.0 / n)
        rejections += np.nanmax(psy.bsadf(y, lags=0)) > cvs["gsadf"][0.95]
    assert 0.0 <= rejections / trials <= 0.12


def _bubble_path(seed, n=200, start=120, end=150, rho=1.06):
    """Log-price random walk, then an upward explosive phase (the deviation from the
    pre-bubble level grows by rho each period), then a collapse back to the base."""
    rng = np.random.default_rng(seed)
    y = np.empty(n)
    y[0] = 4.0
    for t in range(1, start):
        y[t] = y[t - 1] + rng.normal(0, 0.05)
    base = y[start - 1]
    dev = 0.1
    for t in range(start, end):
        dev *= rho
        y[t] = base + dev + rng.normal(0, 0.01)
    for t in range(end, n):
        y[t] = base + 0.6 * (y[t - 1] - base) + rng.normal(0, 0.05)
    return pd.Series(np.exp(y), index=pd.date_range("2000-01-31", periods=n, freq="ME"))


def test_detects_and_dates_an_explosive_episode():
    prices = _bubble_path(seed=5)
    out = psy.date_stamp(prices, lags=0, reps=300, seed=3)
    assert out["global_reject"], "GSADF failed to reject on a clear bubble"
    assert out["episodes"], "no explosive episode detected"
    # The strongest episode is dated inside the explosive phase.
    main = max(out["episodes"], key=lambda e: e["peak_bsadf"])
    assert prices.index[118] <= pd.Timestamp(main["start"]) <= prices.index[150]


def test_two_step_procedure_keeps_false_episodes_rare():
    n = 150
    rng = np.random.default_rng(2024)
    false = 0
    trials = 40
    for i in range(trials):
        y = np.cumsum(rng.normal(0, 0.05, n))
        prices = pd.Series(np.exp(4 + y), index=pd.date_range("2000-01-31", periods=n, freq="ME"))
        false += bool(psy.date_stamp(prices, lags=0, reps=100, seed=i)["episodes"])
    assert false / trials <= 0.15
