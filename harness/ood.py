"""Out-of-distribution (OOD) detector: the "warning light".

Fit on the features the ML model sees during training conditions. At run time, score how
far the current features are from that training cloud (Mahalanobis distance), averaged over
a window of slots so a single odd slot does not trigger an alarm. If the window score is
above the 99th percentile of training windows, raise a flag.

Two detectors:
  MahalanobisOOD       "do the inputs look like training data?"
  CalibrationMonitor   "does the model's own prediction still match reality?" It compares
                       the P(ACK) the model predicted for its chosen MCS with the ACK rate
                       actually observed in the window. Uses only ACK/NACK feedback, which
                       a real gNB / call box has.

Important: OOD is not the same as failure. A detector can flag a condition where the model
still does fine (false alarm), and miss one where it fails but the inputs look normal
(e.g. an impairment that only shows up in ACK/NACK patterns). E3 measures both.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class MahalanobisOOD:
    def __init__(self, window: int = 500, quantile: float = 0.99):
        self.window, self.quantile = window, quantile

    def fit(self, X: np.ndarray) -> "MahalanobisOOD":
        self.mu = X.mean(0)
        cov = np.cov(X, rowvar=False) + 1e-3 * np.eye(X.shape[1])   # ridge for stability
        self.prec = np.linalg.inv(cov)
        self.threshold = float(np.quantile(self.window_scores(X), self.quantile))
        return self

    def slot_scores(self, X: np.ndarray) -> np.ndarray:
        d = X - self.mu
        return np.einsum("ij,jk,ik->i", d, self.prec, d)

    def window_scores(self, X: np.ndarray) -> np.ndarray:
        s = self.slot_scores(X)
        n = len(s) // self.window
        return s[: n * self.window].reshape(n, self.window).mean(1)

    def save(self, path):
        np.savez(path, mu=self.mu, prec=self.prec, threshold=self.threshold,
                 window=self.window, quantile=self.quantile)

    @classmethod
    def load(cls, path) -> "MahalanobisOOD":
        z = np.load(path)
        o = cls(int(z["window"]), float(z["quantile"]))
        o.mu, o.prec, o.threshold = z["mu"], z["prec"], float(z["threshold"])
        return o


class CalibrationMonitor:
    """Window score = |observed ACK rate - mean predicted P(ACK)|."""

    def __init__(self, window: int = 500, quantile: float = 0.99):
        self.window, self.quantile = window, quantile

    def window_scores(self, preds: np.ndarray, acks: np.ndarray) -> np.ndarray:
        n = min(len(preds), len(acks)) // self.window
        p = np.asarray(preds[: n * self.window]).reshape(n, self.window).mean(1)
        o = np.asarray(acks[: n * self.window]).reshape(n, self.window).mean(1)
        return np.abs(o - p)

    def fit(self, scores: np.ndarray) -> "CalibrationMonitor":
        self.threshold = float(np.quantile(scores, self.quantile))
        return self
