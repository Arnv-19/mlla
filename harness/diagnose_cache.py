"""Run the diagnosis CNN on a few slots of a scenario (majority vote), with caching.

In a real test this would be a short measurement of the received constellation; here we
simulate it with the same PHY chain (no decoding needed).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from diagnose.cnn import DiagCNN
from diagnose.dataset import CLASSES, views
from sim.bler_tables import Scenario
from sim.phy_chain import PDSCHLink, simulate_symbols


class Diagnoser:
    def __init__(self, model_path="results/models/diag_cnn.pt",
                 cache_path="results/diag_cache.json", n_slots: int = 8, qm: int = 4):
        self.model = DiagCNN(4, True)
        self.model.load_state_dict(torch.load(model_path, weights_only=True)); self.model.eval()
        self.cache_path, self.n_slots, self.qm = Path(cache_path), n_slots, qm
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self.link = PDSCHLink(device="cpu")

    MIN_SNR_DB = 15.0   # below this, E4 shows the CNN is unreliable -> report "uncertain"

    @torch.no_grad()
    def __call__(self, scenario: str, snr_db: float, seed: int = 0) -> str:
        if snr_db < self.MIN_SNR_DB:
            return "uncertain"
        key = f"{scenario}|{snr_db:g}|{seed}"
        if key not in self.cache:
            torch.manual_seed(10_000 + seed)
            scn = Scenario.parse(scenario)
            r = simulate_symbols(self.link, self.qm, snr_db, scn.spec, scn.imp_dict, self.n_slots)
            X = torch.from_numpy(np.stack([views(r["x_hat"][i], self.qm) for i in range(self.n_slots)]))
            pred = self.model(X, torch.full((self.n_slots,), float(snr_db))).argmax(1).numpy()
            self.cache[key] = CLASSES[int(np.bincount(pred, minlength=len(CLASSES)).argmax())]
            self.cache_path.write_text(json.dumps(self.cache))
        return self.cache[key]
