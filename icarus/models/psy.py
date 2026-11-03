"""Phillips-Shi-Yu (2015) real-time bubble detection: the GSADF / BSADF test.

Under normal conditions log prices behave like a random walk (a unit root). A
bubble shows up as an *explosive* root: prices grow faster than any random walk
would. The backward sup-ADF statistic BSADF(t) takes, at each date, the largest
right-tailed ADF statistic over all windows ending at that date. When it
crosses its Monte Carlo critical value, the market is in an explosive regime.
The test only uses data up to t, so it works as a real-time detector.

Performance trick: every ADF regression needs only the window's cross-product
sums, so prefix sums turn each window into O(1) work and a full BSADF sequence
costs O(T^2) small batched solves rather than O(T^3).

Reference: Phillips, Shi & Yu (2015), "Testing for Multiple Bubbles",
International Economic Review 56(4).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd


def _design(y: np.ndarray, lags: int):
    """Regression arrays for dy_t = a + b*y_{t-1} + sum_j phi_j dy_{t-j} + e_t."""
    dy = np.diff(y)
    T = len(y)
    rows = T - 1 - lags                       # usable regression observations
    if rows <= 0:
        return None, None
    X = np.empty((rows, 2 + lags))
    X[:, 0] = 1.0
    X[:, 1] = y[lags:T - 1]                   # y_{t-1}
    for j in range(1, lags + 1):
        X[:, 1 + j] = dy[lags - j:T - 1 - j]  # dy_{t-j}
    target = dy[lags:]
    return X, target


def _prefix(X: np.ndarray, z: np.ndarray):
    p = X.shape[1]
    xx = np.zeros((len(X) + 1, p, p))
    xx[1:] = np.cumsum(X[:, :, None] * X[:, None, :], axis=0)
    xz = np.zeros((len(X) + 1, p))
    xz[1:] = np.cumsum(X * z[:, None], axis=0)
    zz = np.zeros(len(X) + 1)
    zz[1:] = np.cumsum(z * z)
    return xx, xz, zz


def _adf_tstats(px, end_row: int, start_rows: np.ndarray, p: int) -> np.ndarray:
    """t-statistics of b for regressions over rows [start, end_row] (inclusive)."""
    xx, xz, zz = px
    XtX = xx[end_row + 1] - xx[start_rows]
    Xtz = xz[end_row + 1] - xz[start_rows]
    ztz = zz[end_row + 1] - zz[start_rows]
    n = end_row + 1 - start_rows
    try:
        inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(XtX)
    beta = np.einsum("nij,nj->ni", inv, Xtz)
    rss = ztz - np.einsum("ni,ni->n", beta, Xtz)
    dof = np.maximum(n - p, 1)
    sigma2 = np.maximum(rss, 1e-300) / dof
    se = np.sqrt(np.maximum(sigma2 * inv[:, 1, 1], 1e-300))
    return beta[:, 1] / se


def min_window(n: int) -> int:
    """PSY's recommended minimum window: r0 = 0.01 + 1.8 / sqrt(T)."""
    return max(int(np.floor(n * (0.01 + 1.8 / np.sqrt(n)))), 8)


def bsadf(y: np.ndarray, lags: int = 1, w0: int | None = None, wmax: int | None = None) -> np.ndarray:
    """BSADF sequence for log-price array y. Entry i corresponds to y[i]; NaN where
    fewer than w0 observations are available. wmax caps the look-back (rolling)."""
    y = np.asarray(y, dtype=float)
    T = len(y)
    w0 = w0 or min_window(T if wmax is None else min(T, wmax))
    out = np.full(T, np.nan)
    X, z = _design(y, lags)
    if X is None:
        return out
    p = X.shape[1]
    px = _prefix(X, z)
    # Regression row r corresponds to observation index i = r + lags + 1 in y.
    # A window of y over [s, e] uses regression rows [s, e - lags - 1].
    for e in range(w0 - 1, T):
        s_max = e - w0 + 1
        s_min = 0 if wmax is None else max(0, e - wmax + 1)
        if s_max < s_min:
            continue
        starts = np.arange(s_min, s_max + 1)
        end_row = e - lags - 1
        start_rows = starts
        valid = (end_row - start_rows + 1) > p + 1
        if not valid.any():
            continue
        stats = _adf_tstats(px, end_row, start_rows[valid], p)
        out[e] = np.nanmax(stats)
    return out


@lru_cache(maxsize=16)   # pure function of its arguments; series of equal length share one simulation
def critical_values(n: int, lags: int = 1, w0: int | None = None, wmax: int | None = None,
                    reps: int = 1000, seed: int = 7, quantiles=(0.90, 0.95, 0.99)) -> dict[float, np.ndarray]:
    """Monte Carlo critical values for the BSADF sequence under the null of a
    random walk with asymptotically negligible drift (y_t = T^-1 + y_{t-1} + e_t)."""
    rng = np.random.default_rng(seed)
    w0 = w0 or min_window(n if wmax is None else min(n, wmax))
    sims = np.empty((reps, n))
    for r in range(reps):
        e = rng.standard_normal(n)
        y = np.cumsum(e + 1.0 / n)
        sims[r] = bsadf(y, lags, w0, wmax)
    out: dict = {q: np.nanquantile(sims, q, axis=0) for q in quantiles}
    for arr in out.values():
        arr.setflags(write=False)          # results are cached and shared: make them read-only
    # GSADF = sup of the BSADF sequence; its critical value is the quantile of
    # that sup across simulations (not the sup of pointwise quantiles).
    out["gsadf"] = {q: float(np.nanquantile(np.nanmax(sims, axis=1), q)) for q in quantiles}
    return out


def date_stamp(series: pd.Series, lags: int = 1, wmax: int | None = None, reps: int = 1000,
               seed: int = 7, min_duration: int | None = None) -> dict:
    """Run BSADF on a price series and date-stamp explosive episodes.

    Follows PSY's two-step procedure, which keeps false episodes rare:
      1. global test: does GSADF (the sup of BSADF) exceed its 95% critical value?
      2. only if so, an episode runs while BSADF exceeds its pointwise 95%
         critical value, and it must last at least log(T) periods.
    Testing every date on its own would flag a short "episode" in roughly 4 of 10
    pure random walks; the two-step rule brings that down to about 5%.
    (tests/test_psy.py measures both rates.)
    """
    s = series.dropna()
    s = s[s > 0]
    y = np.log(s.to_numpy(dtype=float))
    n = len(y)
    w0 = min_window(n if wmax is None else min(n, wmax))
    stat = bsadf(y, lags, w0, wmax)
    sim_len = n if wmax is None else min(n, wmax)
    cvs = critical_values(sim_len, lags, w0, wmax, reps=reps, seed=seed, quantiles=(0.90, 0.95, 0.99))
    cv95 = cvs[0.95]
    if wmax is not None and n > sim_len:
        # With a rolling cap the null distribution is stationary once a full
        # window is available, so the last simulated value applies thereafter.
        cv95 = np.concatenate([cv95, np.full(n - sim_len, cv95[-1])])
        cv90 = np.concatenate([cvs[0.90], np.full(n - sim_len, cvs[0.90][-1])])
        cv99 = np.concatenate([cvs[0.99], np.full(n - sim_len, cvs[0.99][-1])])
    else:
        cv90, cv99 = cvs[0.90], cvs[0.99]
    df = pd.DataFrame({"bsadf": stat, "cv90": cv90, "cv95": cv95, "cv99": cv99}, index=s.index)
    df["explosive"] = df["bsadf"] > df["cv95"]
    min_duration = min_duration or int(np.ceil(np.log(n)))
    gsadf = float(np.nanmax(stat)) if np.isfinite(stat).any() else float("nan")
    gsadf_cv95 = cvs["gsadf"][0.95] if wmax is None else None
    global_reject = True if gsadf_cv95 is None else bool(gsadf > gsadf_cv95)

    episodes, start = [], None
    flags = df["explosive"].to_numpy()
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        if (not f or i == n - 1) and start is not None:
            end = i if f else i - 1
            if end - start + 1 >= min_duration:
                seg = df.iloc[start:end + 1]
                episodes.append({"start": str(seg.index[0].date()), "end": str(seg.index[-1].date()),
                                 "periods": int(end - start + 1), "ongoing": bool(f and i == n - 1),
                                 "peak_bsadf": float(seg["bsadf"].max())})
            start = None
    if not global_reject:
        episodes = []
    return {"series": df, "episodes": episodes, "gsadf": gsadf, "gsadf_cv95": gsadf_cv95,
            "global_reject": global_reject, "min_duration": min_duration, "w0": int(w0)}
