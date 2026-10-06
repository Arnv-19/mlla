"""Plot fitted BLER curves for every scenario in a fits file.

  python -m scripts.plot_tables results/tables/bler_fits.json
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim.bler_tables import BLERTable

path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/tables/bler_fits.json")
tab = BLERTable.load(path)
s = np.linspace(-6, 35, 400)
for scn in tab.scenarios:
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in tab.mcs_list(scn):
        ax.semilogy(s, np.clip(tab.bler(scn, m, s), 1e-4, 1), lw=1,
                    color=plt.cm.viridis((m - 3) / 25))
    ax.axhline(0.1, ls="--", c="grey", lw=0.8)
    ax.set(xlabel="slot SNR [dB]", ylabel="BLER", title=f"{scn}  (colour: MCS 3 to 28)",
           ylim=(1e-3, 1.05))
    ax.grid(alpha=0.3)
    out = path.parent / f"bler_{scn}.png"
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    print("saved", out)
