"""Experiment E1: ML vs OLLA vs Oracle across channels and SNRs (5 seeds each).

In-distribution: TDLA30-10, TDLB100-400 (what the model was trained on).
Out-of-distribution: AWGN, TDLC300-100 (never seen in training).

  python -m scripts.compare          -> results/e1_compare.csv, results/e1_compare.png
"""
import argparse
import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baseline.olla import OLLA
from dut.ml_la import MLPolicy, load
from dut.train import TRAIN_SCENARIOS
from sim.bler_tables import BLERTable
from sim.system_sim import LoopConfig, OraclePolicy, make_gains, run_loop

ap = argparse.ArgumentParser()
ap.add_argument("--tables", default="results/tables/bler_fits.json")
ap.add_argument("--model", default="results/models/acknet.pt")
ap.add_argument("--snr", type=float, nargs="*", default=[5, 10, 15, 20, 25])
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--slots", type=int, default=10_000)
a = ap.parse_args()

table, model = BLERTable.load(a.tables), load(a.model)
policies = {
    "OLLA": lambda: OLLA.from_table(table, step_down_db=0.5),
    "ML-tput": lambda: MLPolicy(model, "tput"),
    "ML-bler10": lambda: MLPolicy(model, "bler10"),
    "Oracle": lambda: OraclePolicy(table),
}
rows, t0 = [], time.time()
SCNS = [s for s in table.scenarios if s.endswith("__clean")]     # E1 = clean channels only
for scn in SCNS:
    tag = "in-dist" if scn in TRAIN_SCENARIOS else "OOD"
    for snr in a.snr:
        res = {p: [] for p in policies}
        for seed in range(a.seeds):
            g = make_gains(scn, a.slots, seed)             # same fading for every policy
            for name, mk in policies.items():
                r = run_loop(mk(), table, scn, LoopConfig(avg_snr_db=snr, n_slots=a.slots,
                                                          seed=seed), gains=g)
                res[name].append((r["throughput_mbps"], r["bler"]))
        base = np.mean([x[0] for x in res["OLLA"]])
        for name, v in res.items():
            tp, bl = np.array(v).T
            rows.append(dict(scenario=scn, split=tag, snr=snr, policy=name,
                             tput=tp.mean(), tput_std=tp.std(), bler=bl.mean(),
                             gain_pct=100 * (tp.mean() / base - 1)))
    print(f"{scn} done ({time.time() - t0:.0f} s)")

with open("results/e1_compare.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)

print(f"\n{'scenario':22s} {'split':7s} {'SNR':>4s} | " +
      " | ".join(f"{p:>16s}" for p in policies))
for scn in SCNS:
    for snr in a.snr:
        rs = {r["policy"]: r for r in rows if r["scenario"] == scn and r["snr"] == snr}
        cells = [f"{rs[p]['gain_pct']:+5.1f}% b={rs[p]['bler']:.2f}" if p != "OLLA"
                 else f"{rs[p]['tput']:5.1f}Mb b={rs[p]['bler']:.2f}" for p in policies]
        print(f"{scn:22s} {rs['OLLA']['split']:7s} {snr:4.0f} | " +
              " | ".join(f"{c:>16s}" for c in cells))

fig, axes = plt.subplots(1, len(SCNS), figsize=(4.2 * len(SCNS), 3.6), sharey=True)
for ax, scn in zip(axes, SCNS):
    for p, st in zip(policies, ["-o", "-s", "--^", ":k"]):
        rs = [r for r in rows if r["scenario"] == scn and r["policy"] == p]
        ax.plot([r["snr"] for r in rs], [r["gain_pct"] for r in rs], st, label=p, ms=4)
    tag = "in-distribution" if scn in TRAIN_SCENARIOS else "OUT of distribution"
    ax.set_title(f"{scn.split('__')[0]}\n({tag})", fontsize=9)
    ax.axhline(0, c="grey", lw=0.8); ax.set_xlabel("average SNR [dB]"); ax.grid(alpha=0.3)
axes[0].set_ylabel("throughput gain vs OLLA [%]"); axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig("results/e1_compare.png", dpi=130)
print("\nsaved results/e1_compare.csv and results/e1_compare.png")
