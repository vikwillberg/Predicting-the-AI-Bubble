"""Log-Periodic Power Law Singularity (LPPLS) model: Sornette's bubble clock.

A bubble, in this view, is a period of *faster-than-exponential* growth driven by
herding and positive feedback, decorated with accelerating log-periodic
oscillations. The fit yields a critical time ``tc``: the most probable end of
the bubble regime (a crash or a regime change, not a guaranteed crash).

Two engines share one algorithm (coarse grid search, then Nelder-Mead from the
best grid points, with the linear parameters solved exactly):

* ``engine/`` (Go) runs every window in parallel and is used in production.
* this module is the readable reference and the fallback when Go isn't built.

tests/test_lppls.py checks that both engines agree, and that the fit recovers
the true ``tc`` of synthetic bubbles.

References: Johansen, Ledoit & Sornette (2000); Filimonov & Sornette (2013);
Sornette, Demos, Zhang et al. (2015); Demos & Sornette (2017).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config

log = logging.getLogger(__name__)

GRID_TC, GRID_M, GRID_W, STARTS, MAX_ITER = 12, 8, 12, 3, 400


@dataclass(frozen=True)
class LPPLSConfig:
    window_min: int = 125
    window_max: int = 750
    window_step: int = 25
    m_range: tuple[float, float] = (0.01, 0.99)
    omega_range: tuple[float, float] = (2.0, 25.0)
    tc_fit_frac: float = 0.2
    tc_qual_frac: tuple[float, float] = (-0.05, 0.10)
    min_oscillations: float = 2.5
    min_damping: float = 0.5

    @classmethod
    def from_settings(cls) -> "LPPLSConfig":
        s = config.settings()["lppls"]
        return cls(s["window_min"], s["window_max"], s["window_step"], tuple(s["m_range"]),
                   tuple(s["omega_range"]), s["tc_fit_frac"], tuple(s["tc_qual_frac"]),
                   s["min_oscillations"], s["min_damping"])

    @property
    def windows(self) -> list[int]:
        return list(range(self.window_min, self.window_max + 1, self.window_step))


# ----------------------------------------------------------------- core algorithm
def _solve4(a: np.ndarray, b: np.ndarray):
    """Gaussian elimination with partial pivoting (mirrors engine/lppls.solve4)."""
    a = a.copy()
    b = b.copy()
    scale = float(np.max(np.abs(np.diag(a))))
    if scale == 0:
        return None
    for col in range(4):
        piv = col + int(np.argmax(np.abs(a[col:, col])))
        if abs(a[piv, col]) < 1e-13 * scale:
            return None
        if piv != col:
            a[[col, piv]] = a[[piv, col]]
            b[[col, piv]] = b[[piv, col]]
        for r in range(col + 1, 4):
            f = a[r, col] / a[col, col]
            a[r, col:] -= f * a[col, col:]
            b[r] -= f * b[col]
    x = np.zeros(4)
    for r in range(3, -1, -1):
        x[r] = (b[r] - a[r, r + 1:] @ x[r + 1:]) / a[r, r]
    return x


def linear_fit(t: np.ndarray, y: np.ndarray, tc: float, m: float, w: float):
    """Solve (A, B, C1, C2) exactly for fixed (tc, m, omega). Returns (beta, sse)."""
    dt = np.maximum(np.abs(tc - t), 1e-8)
    lg = np.log(dt)
    f = np.exp(m * lg)
    X = np.column_stack([np.ones_like(t), f, f * np.cos(w * lg), f * np.sin(w * lg)])
    beta = _solve4(X.T @ X, X.T @ y)
    if beta is None:
        return None, np.inf
    r = y - X @ beta
    sse = float(r @ r)
    return (beta, sse) if np.isfinite(sse) else (None, np.inf)


def _nelder_mead(f, x0: np.ndarray, max_iter: int):
    """Bounded Nelder-Mead on the unit cube (mirrors engine/lppls.nelderMead)."""
    nd = 3
    x0 = np.clip(x0, 0, 1)
    simplex = [x0.copy()]
    for k in range(nd):
        p = x0.copy()
        step = 0.1 if p[k] + 0.1 <= 1 else -0.1
        p[k] += step
        simplex.append(p)
    simplex = np.array(simplex)
    fv = np.array([f(p) for p in simplex])
    for _ in range(max_iter):
        order = np.argsort(fv, kind="stable")
        simplex, fv = simplex[order], fv[order]
        spread = abs(fv[nd] - fv[0])
        size = float(np.max(np.abs(simplex[1:] - simplex[0])))
        if spread <= 1e-12 * (1 + abs(fv[0])) and size < 1e-7:
            break
        c = simplex[:nd].mean(axis=0)
        xr = np.clip(c + (c - simplex[nd]), 0, 1)
        fr = f(xr)
        if fr < fv[0]:
            xe = np.clip(c + 2 * (xr - c), 0, 1)
            fe = f(xe)
            simplex[nd], fv[nd] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fv[nd - 1]:
            simplex[nd], fv[nd] = xr, fr
        else:
            xc = c + 0.5 * (xr - c) if fr < fv[nd] else c + 0.5 * (simplex[nd] - c)
            xc = np.clip(xc, 0, 1)
            fc = f(xc)
            if fc < min(fr, fv[nd]):
                simplex[nd], fv[nd] = xc, fc
            else:
                for i in range(1, nd + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fv[i] = f(simplex[i])
    best = int(np.argmin(fv))
    return simplex[best], float(fv[best])


def fit_window(t: np.ndarray, y: np.ndarray, cfg: LPPLSConfig) -> dict:
    t1, t2 = float(t[0]), float(t[-1])
    dt = t2 - t1
    lo = np.array([t2 - cfg.tc_fit_frac * dt, cfg.m_range[0], cfg.omega_range[0]])
    hi = np.array([t2 + cfg.tc_fit_frac * dt, cfg.m_range[1], cfg.omega_range[1]])

    def params(u):
        return lo + np.clip(u, 0, 1) * (hi - lo)

    def obj(u):
        tc, m, w = params(u)
        return linear_fit(t, y, tc, m, w)[1]

    grid = [np.array([a, b, c]) for a in np.linspace(0, 1, GRID_TC)
            for b in np.linspace(0, 1, GRID_M) for c in np.linspace(0, 1, GRID_W)]
    sse = np.array([obj(u) for u in grid])
    order = np.argsort(sse, kind="stable")
    best_u, best_f = grid[order[0]], sse[order[0]]
    for k in order[:STARTS]:
        u, fval = _nelder_mead(obj, grid[k], MAX_ITER)
        if fval < best_f:
            best_u, best_f = u, fval
    tc, m, w = params(best_u)
    beta, sse_final = linear_fit(t, y, tc, m, w)
    out = {"t1": t1, "t2": t2, "tc": tc, "m": m, "omega": w, "sse": sse_final, "ok": beta is not None}
    if beta is None:
        return out
    out.update(dict(zip(["A", "B", "C1", "C2"], map(float, beta))))
    return qualify(out, cfg)


def qualify(fit: dict, cfg: LPPLSConfig) -> dict:
    dt = fit["t2"] - fit["t1"]
    c = float(np.hypot(fit["C1"], fit["C2"]))
    num = max(abs(fit["tc"] - fit["t1"]), 1e-8)
    den = max(abs(fit["tc"] - fit["t2"]), 1e-8)
    fit["oscillations"] = (fit["omega"] / np.pi) * np.log(num / den)
    fit["damping"] = (fit["m"] * abs(fit["B"])) / (fit["omega"] * c) if c > 0 else np.inf
    fit["bubble"] = 1 if fit["B"] < 0 else -1
    fit["qualified"] = bool(
        cfg.m_range[0] <= fit["m"] <= cfg.m_range[1]
        and cfg.omega_range[0] <= fit["omega"] <= cfg.omega_range[1]
        and fit["t2"] + cfg.tc_qual_frac[0] * dt <= fit["tc"] <= fit["t2"] + cfg.tc_qual_frac[1] * dt
        and fit["oscillations"] >= cfg.min_oscillations
        and fit["damping"] >= cfg.min_damping
    )
    return fit


# ------------------------------------------------------------------------ engines
def _go_available() -> bool:
    return config.ENGINE_BIN.exists() or shutil.which("lppls") is not None


def _run_go(t: np.ndarray, y: np.ndarray, jobs: list[tuple[int, int]], cfg: LPPLSConfig) -> list[dict]:
    exe = str(config.ENGINE_BIN) if config.ENGINE_BIN.exists() else shutil.which("lppls")
    req = {
        "t": t.tolist(), "y": y.tolist(),
        "jobs": [{"start": a, "end": b} for a, b in jobs],
        "tc_fit_frac": cfg.tc_fit_frac, "m_range": list(cfg.m_range), "omega_range": list(cfg.omega_range),
        "qual": {"tc_lo_frac": cfg.tc_qual_frac[0], "tc_hi_frac": cfg.tc_qual_frac[1],
                 "min_oscillations": cfg.min_oscillations, "min_damping": cfg.min_damping},
        "search": {"GridTc": GRID_TC, "GridM": GRID_M, "GridW": GRID_W, "Starts": STARTS, "MaxIter": MAX_ITER},
    }
    proc = subprocess.run([exe], input=json.dumps(req), capture_output=True, text=True, check=True)
    rows = json.loads(proc.stdout)
    out = []
    for r in rows:
        r.pop("job", None)
        out.append(r)
    return out


def fit_jobs(t: np.ndarray, y: np.ndarray, jobs: list[tuple[int, int]], cfg: LPPLSConfig,
             engine: str = "auto") -> list[dict]:
    """Fit every (start, end) index window. engine: 'auto' | 'go' | 'python'."""
    use_go = engine == "go" or (engine == "auto" and _go_available())
    if use_go:
        return _run_go(t, y, jobs, cfg)
    return [fit_window(t[a:b + 1], y[a:b + 1], cfg) for a, b in jobs]


# ---------------------------------------------------------------------- analysis
def trading_index_to_date(dates: pd.DatetimeIndex, x: float) -> pd.Timestamp:
    """Map a fractional trading-day index (possibly beyond the data) to a date."""
    last = len(dates) - 1
    if x <= last:
        return dates[int(np.clip(round(x), 0, last))]
    return dates[-1] + pd.offsets.BDay(int(round(x - last)))


def scan(series: pd.Series, t2_positions: list[int], cfg: LPPLSConfig | None = None,
         engine: str = "auto") -> pd.DataFrame:
    """Multi-scale LPPLS scan: for every end date t2, fit every window length.

    series: price series indexed by trading date (prices, not logs).
    Returns one row per (t2, window) fit, with tc mapped to a calendar date.
    """
    cfg = cfg or LPPLSConfig.from_settings()
    s = series.dropna()
    s = s[s > 0]
    dates = pd.DatetimeIndex(s.index)
    y = np.log(s.to_numpy(dtype=float))
    t = np.arange(len(y), dtype=float)
    jobs, meta = [], []
    for t2 in t2_positions:
        for w in cfg.windows:
            if t2 - w >= 0:
                jobs.append((t2 - w, t2))
                meta.append((t2, w))
    if not jobs:
        return pd.DataFrame()
    fits = fit_jobs(t, y, jobs, cfg, engine=engine)
    df = pd.DataFrame(fits)
    df["t2_idx"] = [m[0] for m in meta]
    df["window"] = [m[1] for m in meta]
    df["t2_date"] = [dates[m[0]] for m in meta]
    df["tc_date"] = [trading_index_to_date(dates, x) for x in df["tc"]]
    df["tc_days_ahead"] = df["tc"] - df["t2"]
    return df


def confidence_indicator(fits: pd.DataFrame) -> pd.DataFrame:
    """Share of windows whose fit is a qualified positive / negative bubble."""
    if fits.empty:
        return pd.DataFrame(columns=["date", "pos_conf", "neg_conf", "n_windows"])
    g = fits.groupby("t2_date")
    return pd.DataFrame({
        "pos_conf": g.apply(lambda d: float(((d["qualified"]) & (d["bubble"] == 1)).mean()), include_groups=False),
        "neg_conf": g.apply(lambda d: float(((d["qualified"]) & (d["bubble"] == -1)).mean()), include_groups=False),
        "n_windows": g.size(),
    }).reset_index().rename(columns={"t2_date": "date"})


def tc_distribution(fits: pd.DataFrame, last_n_dates: int = 5) -> dict:
    """Distribution of critical times from qualified positive-bubble fits over the
    most recent scan dates. Returns quantiles as dates (or None if no fit qualifies)."""
    if fits.empty:
        return {"n_qualified": 0}
    recent_dates = sorted(fits["t2_date"].unique())[-last_n_dates:]
    q = fits[fits["t2_date"].isin(recent_dates) & fits["qualified"] & (fits["bubble"] == 1)]
    out = {"n_qualified": int(len(q)), "n_fits": int(fits["t2_date"].isin(recent_dates).sum())}
    if len(q):
        tcs = pd.to_datetime(q["tc_date"]).sort_values()
        for p in (0.1, 0.25, 0.5, 0.75, 0.9):
            out[f"q{int(p * 100)}"] = str(tcs.quantile(p).date())
    return out
