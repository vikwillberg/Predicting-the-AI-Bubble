"""Hidden Markov regimes: which "weather" is the market in right now?

A 3-state Gaussian HMM is fitted to weekly Nasdaq-100 log returns and log
realised volatility since 1985 (40 years, spanning the dot-com bubble and its
bust, the GFC, COVID and 2022). States are labelled after fitting by their
average volatility: calm, choppy, stress. The fitted transition matrix gives
the expected duration of each regime and the probability of moving into stress
within N weeks. The trained model then decodes the AI-20's recent path.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from ..warehouse.build import query

log = logging.getLogger(__name__)
logging.getLogger("hmmlearn").setLevel(logging.ERROR)  # benign "not converging" chatter
NAMES = ["calm", "choppy", "stress"]


def weekly_obs(ticker: str, start: str | None = None) -> pd.DataFrame:
    d = query(f"SELECT date, price, r FROM series WHERE ticker = '{ticker}' ORDER BY date")
    d["date"] = pd.to_datetime(d["date"])
    d = d.set_index("date")
    if start:
        d = d[d.index >= start]
    w = pd.DataFrame({
        "ret": np.log(d["price"]).resample("W-FRI").last().diff(),
        "rv": d["r"].pow(2).resample("W-FRI").sum().pow(0.5) * np.sqrt(52),
    }).dropna()
    w = w[w["rv"] > 0]
    w["log_rv"] = np.log(w["rv"])
    # The current week is still open: label it with the last real trading day,
    # not with a Friday that hasn't happened yet.
    if len(w) and w.index[-1] > d.index[-1]:
        w.index = w.index[:-1].append(pd.DatetimeIndex([d.index[-1]]))
    return w


def fit(train: pd.DataFrame, n_states: int = 3, restarts: int = 12, seed: int = 11) -> tuple[GaussianHMM, np.ndarray]:
    X = train[["ret", "log_rv"]].to_numpy()
    best, best_ll = None, -np.inf
    for k in range(restarts):
        m = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=500, tol=1e-5,
                        random_state=seed + k)
        try:
            m.fit(X)
            ll = m.score(X)
        except Exception:
            continue
        if ll > best_ll:
            best, best_ll = m, ll
    if best is None:
        raise RuntimeError("HMM failed to converge")
    order = np.argsort(best.means_[:, 1])        # sort states by mean log-volatility
    return best, order


def analyse(n_states: int = 3, restarts: int = 12, seed: int = 11) -> dict:
    ndx = weekly_obs("^NDX")
    model, order = fit(ndx, n_states, restarts, seed)
    rank = {s: i for i, s in enumerate(order)}           # raw state -> 0 calm .. 2 stress
    names = {s: NAMES[rank[s]] for s in range(n_states)}

    def decode(df: pd.DataFrame) -> pd.DataFrame:
        post = model.predict_proba(df[["ret", "log_rv"]].to_numpy())
        out = pd.DataFrame({names[s]: post[:, s] for s in range(n_states)}, index=df.index)
        out["regime"] = out[NAMES].idxmax(axis=1)
        return out

    ndx_states = decode(ndx)
    ai20 = weekly_obs("AI20", start="2015-01-01")
    ai_states = decode(ai20)

    A = model.transmat_
    trans = pd.DataFrame(A, index=[names[s] for s in range(n_states)],
                         columns=[names[s] for s in range(n_states)]).loc[NAMES, NAMES]
    durations = {names[s]: float(1 / max(1 - A[s, s], 1e-9)) for s in range(n_states)}
    stats = {names[s]: {"weekly_return_mean": float(model.means_[s, 0]),
                        "annualised_vol": float(np.exp(model.means_[s, 1]))} for s in range(n_states)}

    # Forecast the regime h weeks ahead from today's filtered state probabilities.
    p0 = ai_states[[names[s] for s in range(n_states)]].iloc[-1].to_numpy()
    stress_raw = order[-1]

    def p_stress_at(h: int) -> float:
        return float((p0 @ np.linalg.matrix_power(A, h))[stress_raw])

    # Stationary distribution: the long-run share of time spent in each regime.
    evals, evecs = np.linalg.eig(A.T)
    stat = np.real(evecs[:, np.argmin(np.abs(evals - 1))])
    stat = stat / stat.sum()

    # Validation: the share of weeks in the stress regime during known crashes.
    checks = {}
    for label, a, b in [("2000-02 dot-com bust", "2000-04-01", "2002-10-31"),
                        ("2008 GFC", "2008-09-01", "2009-03-31"),
                        ("2020 COVID", "2020-02-20", "2020-04-30"),
                        ("2013-14 calm bull", "2013-01-01", "2014-12-31")]:
        seg = ndx_states.loc[a:b]
        checks[label] = float((seg["regime"] == "stress").mean()) if len(seg) else None

    return {
        "ndx_states": ndx_states, "ai20_states": ai_states, "transition": trans,
        "expected_duration_weeks": durations, "state_stats": stats,
        "current": {"date": str(ai_states.index[-1].date()),
                    "regime": ai_states["regime"].iloc[-1],
                    "probabilities": {k: float(v) for k, v in ai_states[NAMES].iloc[-1].items()},
                    "p_stress_in_4w": p_stress_at(4),
                    "p_stress_in_13w": p_stress_at(13),
                    "p_stress_in_26w": p_stress_at(26),
                    "long_run_share_stress": float(stat[stress_raw])},
        "validation_share_stress": checks,
    }
