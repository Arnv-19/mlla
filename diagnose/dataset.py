"""Diagnosis dataset: four "instrument views" of the equalised symbols in one slot.

A signal analyser (VSA) does not only show the constellation; it also shows error views
that separate impairments. We give the CNN four 32x32 images, each a 2D histogram:

  0 constellation       received symbols in the IQ plane
  1 AM/AM error         radial error vs reference amplitude. PA compression pulls the
                        outer points inward, so the error grows with amplitude.
  2 image leakage       error on subcarrier k multiplied by the reference symbol on the
                        mirror subcarrier -k. IQ imbalance leaks -k into k, so this forms
                        an off-centre cluster; other impairments stay centred at zero.
  3 phase vs time       phase error per OFDM symbol across the slot. CFO drifts
                        steadily, phase noise wanders randomly.

The reference symbols are hard decisions (nearest constellation point), which any
receiver can compute, so no genie knowledge is used. Views 1-3 exist because the
constellation alone cannot separate PA compression from IQ imbalance (see the smoke test).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from sim.channels import ChannelSpec
from sim.impairments import SEVERITY
from sim.phy_chain import PDSCHLink, simulate_symbols

CLASSES = ["clean", "pa", "iq", "pn", "cfo"]
VIEW_NAMES = ["constellation", "AM/AM error", "image leakage", "phase vs time"]
CHANNELS = [ChannelSpec("AWGN"), ChannelSpec("A30", 10), ChannelSpec("B100", 400),
            ChannelSpec("C300", 100)]
QMS = [2, 4, 6]
BINS = 32


def hard_decision(z: torch.Tensor, qm: int) -> torch.Tensor:
    m = 2 ** (qm // 2)                                    # levels per axis
    scale = math.sqrt(2 * (4 ** (qm // 2) - 1) / 3)       # unit average power
    def axis(v):
        lv = torch.clamp(2 * torch.round((v * scale + (m - 1)) / 2) - (m - 1), -(m - 1), m - 1)
        return lv / scale
    return torch.complex(axis(z.real), axis(z.imag))


def _hist(xv, yv, xr, yr):
    h, _, _ = np.histogram2d(xv, yv, bins=BINS, range=[xr, yr])
    h = np.log1p(h)
    return h / max(h.max(), 1e-9)


def views(x_hat: torch.Tensor, qm: int) -> np.ndarray:
    """x_hat [n_sym, n_sc] for one slot -> [4, 32, 32] float32."""
    ref = hard_decision(x_hat, qm)
    err = x_hat - ref
    n_sym, n_sc = x_hat.shape
    v0 = _hist(x_hat.real.flatten(), x_hat.imag.flatten(), [-1.6, 1.6], [-1.6, 1.6])
    v1 = _hist(ref.abs().flatten(), (x_hat.abs() - ref.abs()).flatten(), [0, 1.6], [-0.5, 0.5])
    j = torch.arange(1, n_sc)                              # mirror of j is n_sc - j
    leak = err[:, j] * ref[:, n_sc - j]
    v2 = _hist(leak.real.flatten(), leak.imag.flatten(), [-0.4, 0.4], [-0.4, 0.4])
    ph = torch.angle(x_hat * ref.conj())
    t = (torch.arange(n_sym).float()[:, None] + 0.5).expand(n_sym, n_sc) / n_sym * 1.6
    v3 = _hist(t.flatten(), ph.flatten(), [0, 1.6], [-0.6, 0.6])
    return np.stack([v0, v1, v2, v3]).astype(np.float32)


def generate(n_batches_per_class: int, batch: int = 16, seed: int = 0,
             snr_range=(5, 35), link: PDSCHLink | None = None, log=print) -> dict:
    link = link or PDSCHLink(device="cpu")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    X, y, meta = [], [], []
    for ci, cls in enumerate(CLASSES):
        for b in range(n_batches_per_class):
            sev = None if cls == "clean" else str(rng.choice(["mild", "moderate", "severe"]))
            ch = CHANNELS[rng.integers(len(CHANNELS))]
            qm = int(rng.choice(QMS))
            snr = float(rng.uniform(*snr_range))
            r = simulate_symbols(link, qm, snr, ch, None if sev is None else {cls: sev}, batch)
            for i in range(batch):
                X.append(views(r["x_hat"][i], qm)); y.append(ci)
                meta.append((snr, sev or "none", ch.name, qm))
        log(f"  class {cls}: {n_batches_per_class * batch} samples")
    return dict(X=np.stack(X), y=np.array(y), snr=np.array([m[0] for m in meta]),
                sev=np.array([m[1] for m in meta]), channel=np.array([m[2] for m in meta]),
                qm=np.array([m[3] for m in meta]))
