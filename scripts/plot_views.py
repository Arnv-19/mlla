"""Show the four diagnosis views for each impairment (severe, AWGN, 25 dB, 16QAM).

  python -m scripts.plot_views     -> results/diagnosis_views.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from diagnose.dataset import CLASSES, VIEW_NAMES, views
from sim.channels import ChannelSpec
from sim.phy_chain import PDSCHLink, simulate_symbols

torch.manual_seed(0)
link = PDSCHLink(device="cpu")
fig, axes = plt.subplots(len(VIEW_NAMES), len(CLASSES), figsize=(2.3 * len(CLASSES), 2.3 * len(VIEW_NAMES)))
for j, cls in enumerate(CLASSES):
    r = simulate_symbols(link, 4, 25.0, ChannelSpec("AWGN"), None if cls == "clean" else {cls: "severe"}, 1)
    v = views(r["x_hat"][0], 4)
    for i in range(len(VIEW_NAMES)):
        ax = axes[i, j]
        ax.imshow(v[i].T, origin="lower", cmap="magma", aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0: ax.set_title(cls, fontsize=10)
        if j == 0: ax.set_ylabel(VIEW_NAMES[i], fontsize=9)
fig.suptitle("Diagnosis views per impairment (severe, AWGN, 25 dB, 16QAM)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig("results/diagnosis_views.png", dpi=120)
print("saved results/diagnosis_views.png")
