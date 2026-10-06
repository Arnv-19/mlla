"""Generate BLER tables. Run this on a Kaggle GPU; it is resumable if the session dies.

  python -m scripts.make_tables --preset clean            # 4 channels, no impairments
  python -m scripts.make_tables --preset full             # 52 scenarios (adds impairments)
  python -m scripts.make_tables --preset dev --n-tb 32    # tiny CPU check

Output: results/tables/<scenario>.npz (raw per-block records) and bler_fits.json (fits).
"""
import argparse

import torch

from sim.bler_tables import build_tables, preset
from sim.phy_chain import MCS_RANGE

ap = argparse.ArgumentParser()
ap.add_argument("--preset", default="clean", choices=["dev", "clean", "full"])
ap.add_argument("--n-tb", type=int, default=256, help="transport blocks per SNR point")
ap.add_argument("--batch", type=int, default=256, help="lower this if the GPU runs out of memory")
ap.add_argument("--out", default="results/tables")
ap.add_argument("--mcs", type=int, nargs="*", default=list(MCS_RANGE))
a = ap.parse_args()
torch.manual_seed(0)
build_tables(preset(a.preset), a.out, n_tb=a.n_tb, batch=a.batch, mcs_list=a.mcs)
