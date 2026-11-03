"""Monotone multi-horizon survival model: P(event within k months | features).

A gradient-boosted model is trained on the stacked (sample, horizon) panel with
a monotone-increasing constraint on the horizon, so every predicted
time-to-event curve is a valid CDF (it never decreases), and it is blended with
a regularised logistic regression. That turns a classifier into a "when" model:
the full distribution of time until a bear market or a burst.

Evaluation is strict walk-forward. For test year Y, the model only sees samples
whose labels were fully observed before 1 January of Y (purged), so nothing from
the test period leaks into training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .panel import FEATURES

log = logging.getLogger(__name__)

LGB_PARAMS = dict(
    objective="binary", learning_rate=0.03, n_estimators=450, num_leaves=15, min_child_samples=300,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0, verbose=-1,
)


def _xy(df: pd.DataFrame, features: list[str]):
    X = df[features + ["horizon"]].astype(float)
    return X, df["y"].to_numpy()


@dataclass
class SurvivalModel:
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    params: dict = field(default_factory=lambda: dict(LGB_PARAMS))
    seed: int = 0
    model: lgb.LGBMClassifier | None = None

    def fit(self, df: pd.DataFrame) -> "SurvivalModel":
        X, y = _xy(df, self.features)
        mono = [0] * len(self.features) + [1]          # increasing in horizon
        self.model = lgb.LGBMClassifier(**self.params, monotone_constraints=mono,
                                        monotone_constraints_method="advanced", random_state=self.seed)
        self.model.fit(X, y)
        return self

    def raw_predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features + ["horizon"]].astype(float)
        return self.model.predict_proba(X)[:, 1]


    def contributions(self, row: pd.DataFrame, horizon: int = 12) -> pd.Series:
        """Exact TreeSHAP contributions (log-odds) for one sample at one horizon."""
        r = row.iloc[[0]].copy()
        r["horizon"] = horizon
        X = r[self.features + ["horizon"]].astype(float)
        contrib = self.model.booster_.predict(X, pred_contrib=True)[0]
        return pd.Series(contrib[:-1], index=self.features + ["horizon"]).sort_values(key=np.abs, ascending=False)


def baseline_logit(features: list[str]):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=2000))


WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def best_weight(p_gbm: np.ndarray, p_logit: np.ndarray, y: np.ndarray) -> float:
    """Ensemble weight on the GBM that minimises log loss on the given predictions."""
    losses = [log_loss(y, np.clip(w * p_gbm + (1 - w) * p_logit, 1e-6, 1 - 1e-6)) for w in WEIGHT_GRID]
    return float(WEIGHT_GRID[int(np.argmin(losses))])


@dataclass
class EnsembleSurvival:
    """Weighted blend of the monotone GBM and a regularised logistic regression.
    Trees capture interactions, the logit extrapolates smoothly, and they make
    different mistakes. The blend weight is learned walk-forward from
    out-of-sample log loss (see walk_forward), never from the test year itself."""
    fast: bool = False
    seed: int = 0
    w_gbm: float = 0.5
    gbm: SurvivalModel | None = None
    logit: object | None = None

    def fit(self, df: pd.DataFrame) -> "EnsembleSurvival":
        params = dict(LGB_PARAMS)
        if self.fast:
            params.update(n_estimators=200, learning_rate=0.06)
        self.gbm = SurvivalModel(params=params, seed=self.seed).fit(df)
        self.logit = baseline_logit(FEATURES)
        self.logit.fit(_logit_design(df), df["y"])
        return self

    def components(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self.gbm.raw_predict(df), self.logit.predict_proba(_logit_design(df))[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        g, lg = self.components(df)
        return self.w_gbm * g + (1 - self.w_gbm) * lg

    def cdf(self, row: pd.DataFrame, horizons=range(1, 25)) -> np.ndarray:
        """Time-to-event CDF for one sample at monthly horizons. The GBM is monotone
        in the horizon and the logit's horizon coefficient is positive, so the
        running maximum only guards against floating-point ties."""
        grid = pd.concat([row.iloc[[0]]] * len(horizons), ignore_index=True)
        grid["horizon"] = list(horizons)
        return np.maximum.accumulate(self.predict(grid))


def _logit_design(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES + ["horizon"]].astype(float).assign(horizon=np.log(df["horizon"].astype(float)))


def walk_forward(stacked: pd.DataFrame, first_year: int, fast: bool = False, seed: int = 0) -> pd.DataFrame:
    """Out-of-sample predictions for every test year, with purging. The ensemble
    weight is nested too: year Y uses a weight fitted on earlier years only."""
    years = sorted(y for y in stacked["date"].dt.year.unique() if y >= first_year)
    preds = []
    for Y in years:
        cutoff = pd.Timestamp(f"{Y}-01-01")
        train = stacked[stacked["label_end"] < cutoff]
        test = stacked[stacked["date"].dt.year == Y]
        if test.empty or train["y"].sum() < 30:
            continue
        ens = EnsembleSurvival(fast=fast, seed=seed).fit(train)
        clim = train.groupby("horizon")["y"].mean()
        out = test[["date", "ticker", "horizon", "y"]].copy()
        out["p_gbm"], out["p_logit"] = ens.components(test)
        out["p_raw"] = (out["p_gbm"] + out["p_logit"]) / 2
        out["p_clim"] = test["horizon"].map(clim).fillna(train["y"].mean()).to_numpy()
        out["fold"] = Y
        preds.append(out)
    oos = pd.concat(preds, ignore_index=True)
    # Nested model averaging: the GBM/logit weight used in year Y is the one with
    # the lowest log loss on the out-of-sample predictions of years before Y.
    oos["p_ens"] = oos["p_raw"]
    oos["w_gbm"] = 0.5
    for Y in sorted(oos["fold"].unique()):
        prior = oos[oos["fold"] < Y]
        if prior["y"].sum() >= 50:
            w = best_weight(prior["p_gbm"].to_numpy(), prior["p_logit"].to_numpy(), prior["y"].to_numpy())
            m = oos["fold"] == Y
            oos.loc[m, "p_ens"] = w * oos.loc[m, "p_gbm"] + (1 - w) * oos.loc[m, "p_logit"]
            oos.loc[m, "w_gbm"] = w
    return oos


def score(oos: pd.DataFrame, pcol: str) -> dict:
    y, p = oos["y"].to_numpy(), np.clip(oos[pcol].to_numpy(), 1e-6, 1 - 1e-6)
    brier = brier_score_loss(y, p)
    brier_clim = brier_score_loss(y, np.clip(oos["p_clim"], 1e-6, 1 - 1e-6))
    return {
        "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
        "brier": float(brier),
        "brier_skill_vs_climatology": float(1 - brier / brier_clim),
        "log_loss": float(log_loss(y, p)),
        "n": int(len(y)),
        "positives": int(y.sum()),
    }


def evaluate(oos: pd.DataFrame) -> dict:
    """Metrics overall and at the headline horizons, for GBM, logit and climatology."""
    res = {"overall": {c: score(oos, c) for c in ("p_gbm", "p_logit", "p_raw", "p_ens") if c in oos}}
    for k in (3, 12, 24):
        sub = oos[oos["horizon"] == k]
        if len(sub) and 0 < sub["y"].sum() < len(sub):
            res[f"h{k}"] = {c: score(sub, c) for c in ("p_gbm", "p_logit", "p_raw", "p_ens") if c in sub}
    return res


def reliability(oos: pd.DataFrame, pcol: str, bins: int = 10) -> pd.DataFrame:
    q = pd.qcut(oos[pcol].rank(method="first"), bins, labels=False)
    return oos.groupby(q).agg(pred=(pcol, "mean"), obs=("y", "mean"), n=("y", "size")).reset_index(drop=True)


def bootstrap_cdfs(stacked: pd.DataFrame, row: pd.DataFrame, B: int, fast: bool, seed: int = 0,
                   w_gbm: float = 0.5) -> np.ndarray:
    """Year-block bootstrap: resample calendar years with replacement, refit the
    ensemble and predict the target's CDF. Crashes cluster in time, so
    resampling whole years gives an honest uncertainty band."""
    rng = np.random.default_rng(seed)
    years = stacked["date"].dt.year.unique()
    by_year = {y: g for y, g in stacked.groupby(stacked["date"].dt.year)}
    out = []
    for b in range(B):
        pick = rng.choice(years, size=len(years), replace=True)
        sample = pd.concat([by_year[y] for y in pick], ignore_index=True)
        ens = EnsembleSurvival(fast=fast, seed=b, w_gbm=w_gbm).fit(sample)
        out.append(ens.cdf(row))
    return np.array(out)
