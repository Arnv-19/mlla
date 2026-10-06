"""The "airbag": ML with a self-check, a fallback to OLLA, and diagnosis.

SafeML runs the ML model and keeps OLLA warm in the background (fed with the same ACKs).
It compares, over the last `window` resolved slots, the ML model's predicted P(ACK) for
its chosen MCS with the ACK rate actually observed. If the model is over-confident by
more than `margin`, it raises an alarm and hands control to OLLA.

What happens next depends on the diagnosis (if a diagnoser is given):
  diagnosis = an impairment  -> stay on OLLA (hardware faults do not go away)
  diagnosis = clean / uncertain / none -> retry ML after `hold` slots (may be transient)

margin = 0.16 (not 0.08): in slow fading, a 100 ms window can dip ~0.05-0.1 below the
prediction by chance, while real failures show gaps of 0.3-0.8. 0.08 caused false alarms
that cost ML its gains in good conditions.

Limitation by design: the self-check only sees over-confidence. If the model is too
cautious (as on AWGN), its own choices keep succeeding and nothing looks wrong.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class SafeML:
    def __init__(self, ml, olla, window: int = 200, margin: float = 0.16, hold: int = 2000,
                 diagnosis: str | None = None):
        self.ml, self.olla = ml, olla
        self.window, self.margin, self.hold = window, margin, hold
        self.diagnosis = diagnosis
        self.name = "SafeML+diag" if diagnosis is not None else "SafeML"
        self.reset()

    def reset(self):
        self.ml.reset(); self.olla.reset()
        self.pending, self.hist = deque(), deque(maxlen=self.window)
        self.mode, self.t, self.until, self.persistent = "ml", 0, 0, False
        self.alarms, self.ml_slots = 0, 0

    def select(self, obs):
        self.t += 1
        if self.mode == "fallback" and not self.persistent and self.t >= self.until:
            self.mode = "ml"; self.hist.clear()
        if self.mode == "ml":
            m = self.ml.select(obs)
            self.pending.append(self.ml.preds[-1]); self.ml_slots += 1
        else:
            m = self.olla.select(obs)
            self.pending.append(None)
        return m

    def update(self, ack):
        p = self.pending.popleft()
        self.olla.update(ack)                         # keep OLLA's offset warm
        if p is None:
            return
        self.hist.append((p, float(ack)))
        if self.mode == "ml" and len(self.hist) == self.window:
            pred, obs = np.mean(self.hist, axis=0)
            if pred - obs > self.margin:              # model over-confident -> alarm
                self.alarms += 1
                self.mode, self.until = "fallback", self.t + self.hold
                self.persistent = self.diagnosis not in (None, "clean", "uncertain")
