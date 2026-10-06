"""What the scheduler (gNB / call box) can observe, turned into a feature vector.

Only information a real gNB has: the latest SNR report and its age, the trend between
reports, rough Doppler and delay-spread estimates, ACK/NACK feedback, and its own past
MCS choices. A background OLLA-style offset is included as a feature, because the ACK
loop is the only signal that reveals errors the SNR report hides (such as impairments).
"""
from __future__ import annotations

from collections import deque

import numpy as np

FEATURE_NAMES = ["snr_report", "report_trend", "report_age", "log_doppler", "delay_spread",
                 "ack_rate_8", "ack_rate_32", "olla_offset", "last_mcs", "nack_streak"]
N_FEATURES = len(FEATURE_NAMES)


class FeatureTracker:
    def __init__(self, olla_step_db: float = 0.5, bler_target: float = 0.1):
        self.down, self.up = olla_step_db, olla_step_db * bler_target / (1 - bler_target)
        self.reset()

    def reset(self):
        self.report = self.prev_report = 0.0
        self.t_measured = 0
        self.doppler = self.ds_ns = 0.0
        self.acks = deque([1.0] * 32, maxlen=32)
        self.offset, self.last_mcs, self.nack_streak = 0.0, 10, 0

    def on_report(self, report_db, t_measured, doppler_est, ds_est_ns):
        self.prev_report, self.report = self.report, report_db
        self.t_measured, self.doppler, self.ds_ns = t_measured, doppler_est, ds_est_ns

    def on_decision(self, mcs):
        self.last_mcs = mcs

    def on_feedback(self, ack: bool):
        self.acks.append(float(ack))
        self.offset = float(np.clip(self.offset + (self.up if ack else -self.down), -15, 5))
        self.nack_streak = 0 if ack else self.nack_streak + 1

    def vector(self, t_now: int) -> np.ndarray:
        a = np.fromiter(self.acks, float)
        return np.array([
            self.report / 30, (self.report - self.prev_report) / 5,
            (t_now - self.t_measured) / 20, np.log10(1 + self.doppler) / 3, self.ds_ns / 300,
            a[-8:].mean(), a.mean(), self.offset / 10, self.last_mcs / 28,
            min(self.nack_streak, 5) / 5], dtype=np.float32)
