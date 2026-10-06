"""Outer-loop link adaptation (OLLA), the standard baseline the ML model must beat.

Two loops:
  inner loop  map the reported SNR (+ offset) to the highest MCS whose AWGN reference
              curve reaches the BLER target at that SNR
  outer loop  offset += step_up on ACK, offset -= step_down on NACK

With step_up / step_down = t / (1 - t), the offset settles where the long-run BLER = t,
because 0.9 * step_up = 0.1 * step_down for t = 10%.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OLLA:
    thresholds_db: dict             # {mcs: SNR needed for target BLER on the reference curve}
    bler_target: float = 0.1
    step_down_db: float = 0.5       # tune this; too small reacts slowly, too large jitters
    offset_db: float = 0.0
    offset_limits: tuple = (-15.0, 5.0)
    name: str = field(default="OLLA", init=False)

    def __post_init__(self):
        t = self.bler_target
        self.step_up_db = self.step_down_db * t / (1 - t)
        self._mcs = np.array(sorted(self.thresholds_db))
        self._thr = np.array([self.thresholds_db[m] for m in self._mcs])

    @classmethod
    def from_table(cls, table, reference: str = "AWGN__clean", **kw) -> "OLLA":
        thr = {m: table.snr_at_bler(reference, m, kw.get("bler_target", 0.1))
               for m in table.mcs_list(reference)}
        return cls(thresholds_db=thr, **kw)

    def reset(self):
        self.offset_db = 0.0

    def select(self, obs: dict) -> int:
        eff = obs["snr_report_db"] + self.offset_db
        ok = self._thr <= eff
        return int(self._mcs[ok][-1]) if ok.any() else int(self._mcs[0])

    def update(self, ack: bool, obs: dict | None = None):
        self.offset_db += self.step_up_db if ack else -self.step_down_db
        self.offset_db = float(np.clip(self.offset_db, *self.offset_limits))
