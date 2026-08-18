"""Shared evaluation metrics and CV splitting for Part 2
(`Part_2_Classical_ML_Pipeline.ipynb`) and Part 3 (`Part_3_Transformer.ipynb`).

Both notebooks import this rather than each defining their own -- the same
reasoning as `feature_engineering.py`: if Part 3 computed QWK or built its
train/test split even slightly differently from Part 2, "compare it against
Part 2's model" (the brief's own words for Part 3) would silently stop being
an apples-to-apples comparison. An import makes that impossible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold

N_CLASSES = 5  # human_score in 0..4

# Ceilings established in EDA_PART_01.ipynb -- Section 3 (human-human
# agreement) and Section 4 (group-mean oracle). Repeated here as constants
# (not recomputed) because both are properties of the *labels*, not of any
# model, so they don't change when Part 2/3's models change.
HUMAN_QWK_CEIL = 0.795
HUMAN_MAE_CEIL = 0.484
ORACLE_MAE_CEIL = 0.743


def action_of(score) -> np.ndarray:
    """App decision the product would take: 0-1 -> re-prompt, 2 -> correct, 3-4 -> advance."""
    s = np.asarray(score)
    return np.where(s <= 1, 0, np.where(s == 2, 1, 2))


def evaluate(y_true, y_pred) -> dict:
    """All headline metrics for one (y_true, y_pred) pair, including
    `false_praise` -- the brief's named worst failure mode (advancing a
    learner whose true score was <=1)."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    err = np.abs(yt - yp)
    advance = action_of(yp) == 2
    bad = yt <= 1
    return {
        "qwk": round(float(cohen_kappa_score(yt, yp, weights="quadratic", labels=list(range(N_CLASSES)))), 4),
        "mae": round(float(err.mean()), 4),
        "exact": round(float((err == 0).mean()), 4),
        "within1": round(float((err <= 1).mean()), 4),
        "macro_f1": round(float(f1_score(yt, yp, average="macro", labels=list(range(N_CLASSES)), zero_division=0)), 4),
        "severe_err": round(float((err >= 2).mean()), 4),
        "false_praise": round(float((advance & bad).sum() / max(1, bad.sum())), 4),
        "n": int(len(yt)),
    }


def bootstrap_ci(y_true, y_pred, metric: str = "qwk", n_boot: int = 500, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI -- essential when n=2000 and some slices (it, C1) are much smaller."""
    rng = np.random.default_rng(seed)
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(yt), len(yt))
        if len(np.unique(yt[idx])) < 2:
            continue
        vals.append(evaluate(yt[idx], yp[idx])[metric])
    return (round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4))


class OptimizedRounder:
    """Learn thresholds that convert a continuous regression output into
    integer classes by maximizing QWK directly (scipy.optimize), rather
    than plain round() -- a regressor minimizing MSE rarely predicts the
    extremes (0 or 4) because that pulls it away from the mean."""

    def __init__(self, n_classes: int = N_CLASSES):
        self.n_classes = n_classes
        self.thresholds_ = None

    def fit(self, y_pred_cont, y_true):
        y_pred_cont = np.asarray(y_pred_cont)
        init = np.linspace(y_pred_cont.min(), y_pred_cont.max(), self.n_classes - 1)

        def neg_qwk(thr):
            thr = np.sort(thr)
            y_hat = np.digitize(y_pred_cont, thr).clip(0, self.n_classes - 1)
            return -cohen_kappa_score(y_true, y_hat, weights="quadratic", labels=list(range(self.n_classes)))

        res = minimize(neg_qwk, init, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-4})
        self.thresholds_ = np.sort(res.x)
        return self

    def predict(self, y_pred_cont):
        return np.digitize(np.asarray(y_pred_cont), self.thresholds_).clip(0, self.n_classes - 1)


def make_splits(df_in: pd.DataFrame, scheme: str = "transcript", n_splits: int = 5, seed: int = 0):
    """Returns a list of (train_idx, test_idx) index arrays.

    'random':     StratifiedKFold -- LEAKY; only used to quantify the gap (Part 2 Section 5)
    'transcript': StratifiedGroupKFold by asr_transcript -- primary, used everywhere else
    'prompt':     GroupKFold by prompt -- stress test (entirely new prompt)
    """
    y_ = df_in["human_score"].values
    if scheme == "random":
        return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(df_in, y_))
    if scheme == "transcript":
        groups = df_in["asr_transcript"].values
        return list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(df_in, y_, groups))
    if scheme == "prompt":
        groups = df_in["prompt"].values
        k = min(df_in["prompt"].nunique(), n_splits)
        return list(GroupKFold(n_splits=k).split(df_in, y_, groups))
    raise ValueError(f"unknown scheme {scheme!r}")


def get_fold0_split(df_in: pd.DataFrame, n_splits: int = 5, seed: int = 0):
    """The first fold of the group-safe CV, as a single (train_idx, test_idx)
    pair -- Part 3's fixed held-out set, so a Part 2 vs Part 3 comparison is
    on the exact same rows rather than a re-randomized split."""
    return make_splits(df_in, scheme="transcript", n_splits=n_splits, seed=seed)[0]
