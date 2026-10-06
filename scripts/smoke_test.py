"""Quick end-to-end check of the PHY chain (~3 min on one CPU core, seconds on a GPU).

1. AWGN BLER around the expected 10% points for a few MCS.
2. Constellation plots showing each impairment's signature (results/constellations.png).

Usage (from repo root):  python -m scripts.smoke_test [--skip-bler]
"""
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from sim.channels import ChannelSpec
from sim.phy_chain import PDSCHLink, bler_point, mcs_info

torch.manual_seed(0)
link = PDSCHLink()
awgn = ChannelSpec("AWGN")
print(f"device={link.device}  fs={link.cfg.fs/1e6:.2f} MHz  data REs/slot={link.cfg.n_data_re}")

if "--skip-bler" not in sys.argv:
    # (mcs, SNR below / near / above the 10% point). Expect ~1.0, in between, ~0.0.
    # Reference run (CPU): MCS3 ~0 dB, MCS10 ~4.8 dB, MCS20 ~13.2 dB, MCS27 ~20 dB.
    checks = [(3, -3, 0, 2), (10, 3, 5, 7), (20, 11, 13, 15), (27, 18, 20, 22)]
    t0 = time.time()
    for mcs, *snrs in checks:
        info = mcs_info(link.cfg, mcs)
        row = [bler_point(link, mcs, s, awgn, max_tb=64, min_errors=64)["bler"] for s in snrs]
        print(f"MCS {mcs:2d} (Qm={info.qm}, R={info.rate:.3f}, TBS={info.tbs:5d})  "
              + "  ".join(f"{s:>3} dB: {b:.2f}" for s, b in zip(snrs, row)))
    print(f"BLER checks took {time.time() - t0:.0f} s")

cases = [("clean", None), ("PA compression", {"pa": "severe"}),
         ("IQ imbalance", {"iq": "severe"}), ("phase noise", {"pn": "severe"}),
         ("residual CFO", {"cfo": "severe"})]
fig, axes = plt.subplots(1, len(cases), figsize=(3.2 * len(cases), 3.5))
for ax, (title, imp) in zip(axes, cases):
    r = link.simulate(12, 30.0, awgn, imp, batch=4, return_symbols=True)
    z = r["x_hat"].flatten().cpu()
    ax.hist2d(z.real.numpy(), z.imag.numpy(), bins=96, range=[[-1.6, 1.6], [-1.6, 1.6]],
              cmap="magma")
    ax.set_title(f"{title}\nSINR {r['sinr_post_db'].median():.1f} dB", fontsize=9)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("16QAM (MCS 12), AWGN, 30 dB SNR, severe level of each impairment", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig("results/constellations.png", dpi=130)
print("saved results/constellations.png")
