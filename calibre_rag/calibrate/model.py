"""v1: calibrators that turn candidate features into P(relevant).

All in NumPy (no scikit-learn dependency):
  - LogisticModel: L2-regularised logistic regression fitted by Newton/IRLS
  - IsotonicModel: monotone fused-score -> probability map (pool-adjacent-violators)
  - Platt re-fit for cheap recalibration after a dataset or model change
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_EPS = 1e-12


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, max_iter: int = 50, tol: float = 1e-8) -> np.ndarray:
    """Newton-Raphson (IRLS) for L2-regularised logistic regression.
    X must already include a bias column (last column, not penalised). Returns weights."""
    n, d = X.shape
    w = np.zeros(d)
    reg = np.full(d, l2)
    reg[-1] = 0.0                      # don't penalise the bias
    for _ in range(max_iter):
        p = sigmoid(X @ w)
        grad = X.T @ (p - y) + reg * w
        W = p * (1 - p) + _EPS
        H = (X * W[:, None]).T @ X + np.diag(reg) + 1e-9 * np.eye(d)
        step = np.linalg.solve(H, grad)
        w -= step
        if np.max(np.abs(step)) < tol:
            break
    return w


@dataclass
class LogisticModel:
    l2: float = 1.0
    mean_: Optional[np.ndarray] = None
    scale_: Optional[np.ndarray] = None
    w_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticModel":
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        self.w_ = fit_logistic(self._design(X), y.astype(float), self.l2)
        return self

    def _design(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mean_) / self.scale_
        return np.hstack([Z, np.ones((len(Z), 1))])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self._design(X) @ self.w_)


def pav(x: np.ndarray, y: np.ndarray):
    """Pool-adjacent-violators: non-decreasing step fit of y on x.
    Returns (knots_x, values) for use with np.interp."""
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order].astype(float)
    # merge tied x first
    ux, inv = np.unique(xs, return_inverse=True)
    sums = np.bincount(inv, weights=ys)
    cnts = np.bincount(inv).astype(float)
    vals, wts, xs_blocks = [], [], []
    for xv, s, c in zip(ux, sums, cnts):
        vals.append(s / c); wts.append(c); xs_blocks.append([xv, xv])
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, b2 = vals.pop(), wts.pop(), xs_blocks.pop()
            v1, w1, b1 = vals.pop(), wts.pop(), xs_blocks.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2)); wts.append(w1 + w2); xs_blocks.append([b1[0], b2[1]])
    kx, kv = [], []
    for (lo, hi), v in zip(xs_blocks, vals):
        kx += [lo, hi]; kv += [v, v]
    return np.array(kx), np.array(kv)


@dataclass
class IsotonicModel:
    """Monotone map from one score column (default: fused score) to probability."""
    column: int = -1
    knots_: Optional[np.ndarray] = None
    values_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "IsotonicModel":
        self.knots_, self.values_ = pav(X[:, self.column], y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.interp(X[:, self.column], self.knots_, self.values_)


@dataclass
class Platt:
    """p' = sigmoid(a * logit(p) + b). Two parameters: cheap to refit on 25-50 new queries."""
    a: float = 1.0
    b: float = 0.0

    def fit(self, p: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> "Platt":
        X = np.column_stack([logit(p), np.ones(len(p))])
        w = fit_logistic(X, y.astype(float), l2=l2)
        self.a, self.b = float(w[0]), float(w[1])
        return self

    def __call__(self, p: np.ndarray) -> np.ndarray:
        if self.a == 1.0 and self.b == 0.0:
            return p
        return sigmoid(self.a * logit(p) + self.b)


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """ECE = sum_b (n_b / n) * |mean(y)_b - mean(p)_b| over equal-width bins."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) == 0:
        return 0.0
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)
