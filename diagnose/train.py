"""Experiment E4: train and evaluate the impairment classifier.

Trains two models to show why the extra views matter:
  constellation-only   view 0, no SNR input (what you would try first)
  full                 all 4 views + the known SNR

  python -m diagnose.train     -> results/models/diag_cnn.pt, results/e4_diagnosis.png
Data is cached in results/diag_{train,test}.npz (delete to regenerate).
"""
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from .cnn import DiagCNN
from .dataset import CLASSES, generate


def data(path, n, seed):
    if Path(path).exists():
        return dict(np.load(path))
    print(f"generating {path} ...")
    d = generate(n, seed=seed)
    np.savez_compressed(path, **d)
    return d


def train(model, d, views, epochs=15, seed=0):
    torch.manual_seed(seed)
    X = torch.from_numpy(d["X"][:, views]); y = torch.from_numpy(d["y"]).long()
    s = torch.from_numpy(d["snr"]).float()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for ep in range(epochs):
        model.train()
        for b in torch.randperm(len(y)).split(128):
            opt.zero_grad()
            nn.functional.cross_entropy(model(X[b], s[b]), y[b]).backward()
            opt.step()
    return model.eval()


@torch.no_grad()
def predict(model, d, views):
    return model(torch.from_numpy(d["X"][:, views]), torch.from_numpy(d["snr"]).float()).argmax(1).numpy()


def main():
    Path("results/models").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tr = data("results/diag_train.npz", 100, seed=1)
    te = data("results/diag_test.npz", 30, seed=2)          # different seed = unseen slots
    print(f"data ready: {len(tr['y'])} train / {len(te['y'])} test ({time.time() - t0:.0f} s)")

    variants = {"constellation-only": ([0], False), "full (4 views + SNR)": ([0, 1, 2, 3], True)}
    preds = {}
    for name, (views, use_snr) in variants.items():
        m = train(DiagCNN(len(views), use_snr), tr, views)
        preds[name] = predict(m, te, views)
        print(f"{name:22s} test accuracy {np.mean(preds[name] == te['y']):.1%}")
    torch.save(m.state_dict(), "results/models/diag_cnn.pt")       # the full model

    full = preds["full (4 views + SNR)"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (name, p) in zip(axes[:2], preds.items()):
        cm = np.zeros((len(CLASSES), len(CLASSES)))
        for t, q in zip(te["y"], p):
            cm[t, q] += 1
        cm /= cm.sum(1, keepdims=True)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(CLASSES)), CLASSES); ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{name}: {np.mean(p == te['y']):.0%}", fontsize=10)
    bins = np.arange(5, 40, 5)
    for name, p in preds.items():
        acc = [np.mean((p == te["y"])[(te["snr"] >= lo) & (te["snr"] < lo + 5)]) for lo in bins[:-1]]
        axes[2].plot(bins[:-1] + 2.5, acc, "-o", label=name)
    axes[2].set(xlabel="SNR [dB]", ylabel="accuracy", title="accuracy vs SNR", ylim=(0, 1.02))
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
    for sev in ("mild", "moderate", "severe"):
        k = te["sev"] == sev
        acc = [np.mean((full == te["y"])[k & (te["snr"] >= lo) & (te["snr"] < lo + 5)]) for lo in bins[:-1]]
        axes[3].plot(bins[:-1] + 2.5, acc, "-o", label=sev)
    axes[3].set(xlabel="SNR [dB]", ylabel="accuracy", title="full model, by severity", ylim=(0, 1.02))
    axes[3].legend(fontsize=8); axes[3].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("results/e4_diagnosis.png", dpi=120)
    print(f"saved results/e4_diagnosis.png ({time.time() - t0:.0f} s total)")


if __name__ == "__main__":
    main()
