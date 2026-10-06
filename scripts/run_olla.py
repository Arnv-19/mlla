"""Run the OLLA baseline in the closed loop and check it holds the 10% BLER target.

  python -m scripts.run_olla                       # uses results/tables/bler_fits.json
  python -m scripts.run_olla --synthetic           # fake tables, only to test the code

Also sweeps OLLA's step size, so the baseline you compare against is a well-tuned one.
"""
import argparse

from baseline.olla import OLLA
from sim.bler_tables import BLERTable
from sim.system_sim import LoopConfig, run_loop

ap = argparse.ArgumentParser()
ap.add_argument("--tables", default="results/tables/bler_fits.json")
ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--snr", type=float, nargs="*", default=[5, 15, 25])
a = ap.parse_args()

tab = BLERTable.synthetic() if a.synthetic else BLERTable.load(a.tables)
if a.synthetic:
    print("WARNING: synthetic tables. Numbers below test the code, they are not results.\n")
print(f"{'scenario':28s} {'SNR':>4s} {'step':>5s} {'BLER':>6s} {'Mbps':>6s} {'MCS':>5s}")
for scn in tab.scenarios:
    for snr in a.snr:
        for step in (0.1, 0.3, 0.5, 1.0, 2.0):
            r = run_loop(OLLA.from_table(tab, step_down_db=step), tab, scn,
                         LoopConfig(avg_snr_db=snr, n_slots=20_000))
            print(f"{scn:28s} {snr:4.0f} {step:5.1f} {r['bler']:6.3f} "
                  f"{r['throughput_mbps']:6.1f} {r['mean_mcs']:5.1f}")
        print()
