"""Sweep each impairment's parameter and report the post-equalisation SINR ceiling.

Use this to (re)set the mild/moderate/severe presets in sim/impairments.py whenever the
numerology or receiver changes.  Usage:  python -m scripts.calibrate_impairments
"""
import torch

from sim.channels import ChannelSpec
from sim.phy_chain import PDSCHLink

torch.manual_seed(0)
link, awgn = PDSCHLink(), ChannelSpec("AWGN")
sweeps = {
    "pa": [dict(ibo_db=v) for v in (10, 8, 6, 5, 4, 3)],
    "iq": [dict(gain_db=g, phase_deg=p)
           for g, p in ((0.2, 1), (0.3, 2), (0.5, 3), (0.8, 5), (1.0, 7), (1.5, 10))],
    "pn": [dict(linewidth_hz=v) for v in (5, 10, 20, 50, 100, 200)],
    "cfo": [dict(eps=v) for v in (0.005, 0.01, 0.015, 0.02, 0.03)],
}


def ceiling(imp):
    return link.simulate(3, 45.0, awgn, imp, batch=16)["sinr_post_db"].median().item()


print(f"clean ceiling: {ceiling(None):.1f} dB")
for kind, params in sweeps.items():
    for p in params:
        print(f"{kind:4s} {str(p):40s} -> {ceiling({kind: p}):5.1f} dB")
