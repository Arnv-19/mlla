"""Train the ML link adaptation model.

Data: run the closed loop with a behaviour policy, record the gNB-side features in every
slot, and label each slot with P(ACK) for every MCS from the BLER table at the TRUE slot
SNR. Two rounds:
  round 0  behaviour = OLLA (random step size) with random MCS exploration
  round 1  behaviour = the round-0 model itself (DAgger), so the model also sees the ACK
           patterns its own decisions produce, not only OLLA's

Training distribution is deliberately limited (default: TDLA30-10 and TDLB100-400, clean).
Everything else is out of distribution for the later stress tests.

  python -m dut.train                       # -> results/models/acknet.pt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from baseline.olla import OLLA
from sim.bler_tables import BLERTable
from sim.system_sim import LoopConfig, make_gains, run_loop
from .ml_la import AckNet, MLPolicy, save

TRAIN_SCENARIOS = ["TDLA30-10__clean", "TDLB100-400__clean"]


class ExploringOLLA(OLLA):
    def __init__(self, *a, epsilon=0.15, seed=0, **kw):
        super().__init__(*a, **kw)
        self.epsilon, self.rng = epsilon, np.random.default_rng(seed)

    def select(self, obs):
        m = super().select(obs)
        if self.rng.random() < self.epsilon:
            m = int(np.clip(m + self.rng.integers(-4, 5), self._mcs[0], self._mcs[-1]))
        return m


def collect(table, scenarios, make_policy, episodes, n_slots, seed0):
    X, Y = [], []
    rng = np.random.default_rng(seed0)
    for scn in scenarios:
        for e in range(episodes):
            seed = seed0 + 1000 * scenarios.index(scn) + e      # never overlaps test seeds 0-99
            cfg = LoopConfig(avg_snr_db=float(rng.uniform(0, 28)), n_slots=n_slots, seed=seed)
            r = run_loop(make_policy(seed), table, scn, cfg,
                         gains=make_gains(scn, n_slots, seed), record_features=True)
            X.append(r["features"])
            Y.append(1 - table.bler_all(scn, r["log"]["true_snr"]))
    return np.concatenate(X), np.concatenate(Y).astype(np.float32)


def fit(X, Y, epochs=10, seed=0, log=print):
    torch.manual_seed(seed)
    n = len(X); idx = np.random.default_rng(seed).permutation(n)
    tr, va = idx[: int(0.9 * n)], idx[int(0.9 * n):]
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    model = AckNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train()
        for b in np.array_split(np.random.default_rng(ep).permutation(tr), max(1, len(tr) // 1024)):
            opt.zero_grad()
            loss = loss_fn(model(Xt[b]), Yt[b])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xt[va]), Yt[va]).item()
        log(f"  epoch {ep + 1:2d}  val loss {v:.4f}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="results/tables/bler_fits.json")
    ap.add_argument("--episodes", type=int, default=40, help="per scenario per round")
    ap.add_argument("--slots", type=int, default=4000)
    ap.add_argument("--out", default="results/models/acknet.pt")
    a = ap.parse_args()
    table = BLERTable.load(a.tables)
    t0 = time.time()

    print("round 0: data from exploring OLLA")
    X0, Y0 = collect(table, TRAIN_SCENARIOS,
                     lambda s: ExploringOLLA.from_table(table, step_down_db=float(
                         np.random.default_rng(s).uniform(0.2, 1.0)), seed=s),
                     a.episodes, a.slots, seed0=10_000)
    print(f"  {len(X0)} samples ({time.time() - t0:.0f} s)")
    model = fit(X0, Y0)

    print("round 1: data from the model itself (DAgger)")
    X1, Y1 = collect(table, TRAIN_SCENARIOS,
                     lambda s: MLPolicy(model, "tput", epsilon=0.1, seed=s),
                     a.episodes, a.slots, seed0=20_000)
    model = fit(np.concatenate([X0, X1]), np.concatenate([Y0, Y1]))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    save(model, a.out)
    print(f"saved {a.out} ({time.time() - t0:.0f} s total)")


if __name__ == "__main__":
    main()
