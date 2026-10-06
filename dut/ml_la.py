"""ML link adaptation: the device under test.

The network predicts, for every MCS at once, the probability that a TB sent with that MCS
in the current slot will be ACKed, given only gNB-side features (dut/features.py).
Choosing an MCS is then explicit:
  mode "tput"    argmax over MCS of TBS * P(ACK)        (maximise expected throughput)
  mode "bler10"  highest TBS with P(ACK) >= 0.9         (same 10% BLER target as OLLA)

Why predict P(ACK) per MCS instead of classifying "the best MCS": the BLER tables give a
label for every MCS in every slot, so there is no need to explore, the outputs are
interpretable (calibration can be checked), and the BLER target can change without
retraining.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from sim.phy_chain import MCS_RANGE, LinkConfig, mcs_info
from .features import N_FEATURES

MCS_LIST = np.array(list(MCS_RANGE))


class AckNet(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_FEATURES, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, len(MCS_LIST)))

    def forward(self, x):            # logits; sigmoid gives P(ACK) per MCS
        return self.net(x)


class MLPolicy:
    """Runs the trained network in numpy (much faster than torch for one sample at a time)."""

    def __init__(self, model: AckNet, mode: str = "tput", epsilon: float = 0.0,
                 link_cfg: LinkConfig | None = None, seed: int = 0):
        cfg = link_cfg or LinkConfig()
        self.W = [(l.weight.detach().numpy().T, l.bias.detach().numpy())
                  for l in model.net if isinstance(l, nn.Linear)]
        self.tbs = np.array([mcs_info(cfg, m).tbs for m in MCS_LIST])
        self.mode, self.epsilon = mode, epsilon
        self.rng = np.random.default_rng(seed)
        self.name = f"ML-{mode}"
        self.preds = []

    def p_ack(self, f):
        h = f
        for i, (w, b) in enumerate(self.W):
            h = h @ w + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0)
        return 1 / (1 + np.exp(-h))

    def reset(self):
        self.preds = []          # predicted P(ACK) of each chosen MCS, for the monitor

    def update(self, ack): ...

    def select(self, obs):
        p = self.p_ack(obs["features"])
        if self.mode == "tput":
            i = int(np.argmax(self.tbs * p))
        else:
            ok = np.nonzero(p >= 0.9)[0]
            i = int(ok[-1]) if ok.size else int(np.argmax(p))
        if self.epsilon and self.rng.random() < self.epsilon:      # exploration for data
            i = int(np.clip(i + self.rng.integers(-4, 5), 0, len(MCS_LIST) - 1))
        self.preds.append(float(p[i]))
        return int(MCS_LIST[i])


def save(model: AckNet, path):
    torch.save(model.state_dict(), path)


def load(path) -> AckNet:
    m = AckNet()
    m.load_state_dict(torch.load(path, weights_only=True))
    return m.eval()
