"""Experiments E2 + E3 + E5: stress-test the ML model, check the warning lights, test the airbag.

E2  For every scenario in the tables (channels x impairments) and every SNR, compare the
    ML model (BLER-10% mode) against OLLA on identical fading and noise. Heatmap of the gain.
E3  Split every run into 250 ms windows. Label a window "ML fails" if ML delivers >5% less
    data than OLLA in it. Check whether the OOD score flags those windows (AUROC), and how
    often it raises false alarms in the training conditions.

E5  Same runs with SafeML (self-check + fallback to OLLA) and SafeML+diag (fallback that
    also asks the diagnosis CNN whether the problem is a lasting hardware fault).

  python -m scripts.stress_test        -> results/e2_heatmap.png, e2_stress.csv, e3_windows.csv
"""
import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from baseline.olla import OLLA
from dut.ml_la import MLPolicy, load
from dut.train import TRAIN_SCENARIOS, collect
from harness.diagnose_cache import Diagnoser
from harness.fallback import SafeML
from harness.ood import CalibrationMonitor, MahalanobisOOD
from sim.bler_tables import BLERTable, Scenario
from sim.system_sim import LoopConfig, make_gains, run_loop

ap = argparse.ArgumentParser()
ap.add_argument("--tables", default="results/tables/bler_fits.json")
ap.add_argument("--model", default="results/models/acknet.pt")
ap.add_argument("--ood", default="results/models/ood.npz")
ap.add_argument("--snr", type=float, nargs="*", default=[5, 10, 15, 20, 25])
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--slots", type=int, default=10_000)
ap.add_argument("--fail-margin", type=float, default=0.05)
a = ap.parse_args()

table, model = BLERTable.load(a.tables), load(a.model)
if Path(a.ood).exists():
    ood = MahalanobisOOD.load(a.ood)
else:
    print("fitting OOD detector on training-condition features ...")
    X, _ = collect(table, TRAIN_SCENARIOS, lambda s: MLPolicy(model, "bler10"),
                   episodes=20, n_slots=4000, seed0=30_000)
    ood = MahalanobisOOD().fit(X); ood.save(a.ood)
W = ood.window

# calibration monitor threshold: training conditions, seeds disjoint from the test seeds
cal_train = []
for i, scn in enumerate(TRAIN_SCENARIOS):
    for e in range(10):
        seed = 40_000 + 100 * i + e
        pol = MLPolicy(model, "bler10")
        r = run_loop(pol, table, scn, LoopConfig(avg_snr_db=float(np.random.default_rng(seed)
                     .uniform(0, 28)), n_slots=a.slots, seed=seed), gains=make_gains(scn, a.slots, seed))
        cal_train.append(CalibrationMonitor(W).window_scores(pol.preds, r["log"]["ack"]))
cal = CalibrationMonitor(W).fit(np.concatenate(cal_train))
diag = Diagnoser()
mk_olla = lambda: OLLA.from_table(table, step_down_db=0.5)

import json
cache_dir = Path("results/stress_cache"); cache_dir.mkdir(parents=True, exist_ok=True)
cfg_key = f"snr{'-'.join(f'{x:g}' for x in a.snr)}_s{a.seeds}_n{a.slots}"
cells, wins, t0 = [], [], time.time()
for scn in table.scenarios:
    cf = cache_dir / f"{scn}__{cfg_key}.json"
    if cf.exists():                                   # resumable: skip finished scenarios
        c = json.loads(cf.read_text()); cells += c["cells"]; wins += c["wins"]; continue
    n_cells, n_wins = len(cells), len(wins)
    for snr in a.snr:
        g_list, f_list, c_list = [], [], []
        d = diag(scn, snr)
        extra = {"SafeML": [], "SafeML+diag": []}
        for seed in range(a.seeds):
            g = make_gains(scn, a.slots, seed)
            cfg = LoopConfig(avg_snr_db=snr, n_slots=a.slots, seed=seed)
            ro = run_loop(mk_olla(), table, scn, cfg, gains=g)
            pol = MLPolicy(model, "bler10")
            rm = run_loop(pol, table, scn, cfg, gains=g, record_features=True)
            cs = cal.window_scores(pol.preds, rm["log"]["ack"])
            base = max(ro["throughput_mbps"], 1e-9)
            g_list.append(rm["throughput_mbps"] / base - 1)
            for name, dg in (("SafeML", None), ("SafeML+diag", d)):
                rs = run_loop(SafeML(MLPolicy(model, "bler10"), mk_olla(), diagnosis=dg),
                              table, scn, cfg, gains=g)
                extra[name].append(rs["throughput_mbps"] / base - 1)
            n = a.slots // W
            bo = ro["log"]["bits"][: n * W].reshape(n, W).sum(1)
            bm = rm["log"]["bits"][: n * W].reshape(n, W).sum(1)
            sc = ood.window_scores(rm["features"])
            for i in range(n):
                fail = bm[i] < (1 - a.fail_margin) * bo[i]
                wins.append(dict(scenario=scn, snr=snr, seed=seed, window=i,
                                 in_dist=scn in TRAIN_SCENARIOS, ml_fails=bool(fail),
                                 ood_score=float(sc[i]), flagged=bool(sc[i] > ood.threshold),
                                 cal_score=float(cs[i]), cal_flag=bool(cs[i] > cal.threshold)))
            f_list.append(np.mean(sc > ood.threshold)); c_list.append(np.mean(cs > cal.threshold))
        cells.append(dict(scenario=scn, snr=snr, gain_pct=100 * np.mean(g_list),
                          gain_std=100 * np.std(g_list), flag_rate=float(np.mean(f_list)),
                          cal_flag_rate=float(np.mean(c_list)), diagnosis=d,
                          safe_gain_pct=100 * np.mean(extra["SafeML"]),
                          safe_diag_gain_pct=100 * np.mean(extra["SafeML+diag"])))
    cf.write_text(json.dumps({"cells": cells[n_cells:], "wins": wins[n_wins:]}, default=float))
    print(f"{scn} done ({time.time() - t0:.0f} s)", flush=True)

Path("results").mkdir(exist_ok=True)
for name, rows in (("e2_stress.csv", cells), ("e3_windows.csv", wins)):
    with open(f"results/{name}", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)

# ---------------- E2 heatmap
scns = table.scenarios
G = np.array([[next(c["gain_pct"] for c in cells if c["scenario"] == s and c["snr"] == x)
               for x in a.snr] for s in scns])
F = np.array([[next(c["flag_rate"] for c in cells if c["scenario"] == s and c["snr"] == x)
               for x in a.snr] for s in scns])
C = np.array([[next(c["cal_flag_rate"] for c in cells if c["scenario"] == s and c["snr"] == x)
               for x in a.snr] for s in scns])
lim = max(10, np.nanmax(np.abs(G)))
fig, ax = plt.subplots(figsize=(1.3 * len(a.snr) + 4, 0.42 * len(scns) + 1.5))
im = ax.imshow(G, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
for i in range(len(scns)):
    for j in range(len(a.snr)):
        ax.text(j, i, f"{G[i, j]:+.0f}%" + (" M" if F[i, j] > 0.5 else "") + (" C" if C[i, j] > 0.5 else ""),
                ha="center", va="center", fontsize=7)
ax.set_xticks(range(len(a.snr)), [f"{x:.0f} dB" for x in a.snr])
ax.set_yticks(range(len(scns)),
              [s.replace("__", "  ") + ("  [train]" if s in TRAIN_SCENARIOS else "") for s in scns],
              fontsize=7)
ax.set_title("ML vs OLLA (blue = ML better, red = worse). Flags: M = input-OOD, C = calibration", fontsize=9)
fig.colorbar(im, ax=ax, label="gain vs OLLA [%]")
fig.tight_layout(); fig.savefig("results/e2_heatmap.png", dpi=130)

# ---------------- E3 summary
y = np.array([w["ml_fails"] for w in wins]).astype(bool)
ind = np.array([w["in_dist"] for w in wins]).astype(bool)
print(f"\nE2: heatmap saved to results/e2_heatmap.png")
print(f"E3: {len(wins)} windows of {W} slots; ML fails (>{a.fail_margin:.0%} below OLLA) in {y.mean():.1%}")
for label, sk, fk in (("input-OOD (Mahalanobis)", "ood_score", "flagged"),
                      ("calibration monitor", "cal_score", "cal_flag")):
    s_ = np.array([w[sk] for w in wins]); f_ = np.array([w[fk] for w in wins]).astype(bool)
    auc = roc_auc_score(y, s_) if 0 < y.sum() < len(y) else float("nan")
    print(f"  {label:25s} AUROC {auc:.3f} | catches {f_[y].mean():5.1%} of failures | "
          f"false alarms: {f_[ind & ~y].mean():5.1%} in training conditions, "
          f"{f_[~y].mean():5.1%} overall")

# ---------------- E5 summary
gm = np.array([c["gain_pct"] for c in cells]); gs = np.array([c["safe_gain_pct"] for c in cells])
gd = np.array([c["safe_diag_gain_pct"] for c in cells])
fail, win = gm < -5, gm > 2
print(f"\nE5 airbag ({fail.sum()} cells where ML loses >5%, {win.sum()} where it wins >2%):")
for name, g_ in (("ML alone", gm), ("SafeML", gs), ("SafeML+diag", gd)):
    print(f"  {name:12s} avg gain vs OLLA: in ML-failure cells {g_[fail].mean():+6.1f}% | "
          f"in ML-win cells {g_[win].mean():+6.1f}% | worst cell {g_.min():+6.1f}%")
truth = [Scenario.parse(c["scenario"]).impairment for c in cells]
true_lbl = np.array([t[0][0] if t else "clean" for t in truth]); pred_lbl = np.array([c["diagnosis"] for c in cells])
print(f"  diagnosis correct in {np.mean(true_lbl == pred_lbl):.0%} of cells, "
      f"{np.mean((true_lbl == pred_lbl)[fail]):.0%} of ML-failure cells")

GD = np.array([[next(c["safe_diag_gain_pct"] for c in cells if c["scenario"] == s and c["snr"] == x)
                for x in a.snr] for s in scns])
fig, axes = plt.subplots(1, 2, figsize=(2.6 * len(a.snr) + 6, 0.42 * len(scns) + 1.5), sharey=True)
for ax, M, title in ((axes[0], G, "ML alone vs OLLA"), (axes[1], GD, "SafeML+diag vs OLLA")):
    im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(scns)):
        for j in range(len(a.snr)):
            ax.text(j, i, f"{M[i, j]:+.0f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(a.snr)), [f"{x:.0f} dB" for x in a.snr]); ax.set_title(title, fontsize=10)
axes[0].set_yticks(range(len(scns)), [s.replace("__", "  ") for s in scns], fontsize=7)
fig.colorbar(im, ax=axes, label="throughput gain vs OLLA [%]")
fig.savefig("results/e5_airbag.png", dpi=130, bbox_inches="tight")
print("saved results/e5_airbag.png")
