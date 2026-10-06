"""Compact figures for the report (reads results/e2_stress.csv from the stress test).

  python -m scripts.paper_figures   -> report/fig_e2_compact.png, report/fig_e5_summary.png
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rows = list(csv.DictReader(open("results/e2_stress.csv")))
snrs = sorted({float(r["snr"]) for r in rows})
imps = ["clean"] + [f"{k}-{l}" for k in ("pa", "iq", "pn", "cfo") for l in ("mild", "moderate", "severe")]

def grid(key):
    return np.array([[np.mean([float(r[key]) for r in rows
                               if r["scenario"].split("__")[1] == imp and float(r["snr"]) == s])
                      for s in snrs] for imp in imps])

# E2 compact: impairment setting x SNR, averaged over the 4 channels
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.9), sharey=True)
for ax, key, title in ((axes[0], "gain_pct", "ML alone"), (axes[1], "safe_diag_gain_pct", "SafeML + diagnosis")):
    M = grid(key)
    im = ax.imshow(M, cmap="RdBu", vmin=-60, vmax=60, aspect="auto")
    for i in range(len(imps)):
        for j in range(len(snrs)):
            ax.text(j, i, f"{M[i, j]:+.0f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(snrs)), [f"{s:.0f} dB" for s in snrs], fontsize=8)
    ax.set_title(title, fontsize=9)
axes[0].set_yticks(range(len(imps)), imps, fontsize=7)
cb = fig.colorbar(im, ax=axes, shrink=0.9); cb.set_label("throughput vs OLLA [%]", fontsize=8)
fig.savefig("report/fig_e2_compact.png", dpi=200, bbox_inches="tight")

# E5 summary: distribution of per-cell gain for each policy
fig, ax = plt.subplots(figsize=(3.5, 2.6))
data = [[float(r[k]) for r in rows] for k in ("gain_pct", "safe_gain_pct", "safe_diag_gain_pct")]
ax.boxplot(data, widths=0.5, showfliers=False)
rng = np.random.default_rng(0)
for i, d in enumerate(data, 1):
    ax.scatter(i + rng.uniform(-0.18, 0.18, len(d)), d, s=5, alpha=0.5)
ax.axhline(0, c="grey", lw=0.8)
ax.set_xticks([1, 2, 3], ["ML alone", "SafeML", "SafeML\n+diag"], fontsize=8)
ax.set_ylabel("throughput vs OLLA [%]", fontsize=8); ax.tick_params(labelsize=7)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig("report/fig_e5_summary.png", dpi=200)
print("saved report figures")
