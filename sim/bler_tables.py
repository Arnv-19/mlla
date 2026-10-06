"""BLER tables: run the full PHY once, then look results up during fast closed-loop runs.

For each scenario (channel x impairment) and each MCS we simulate transport blocks over a
range of average SNRs and record, per block, the slot SNR (wideband SNR of that particular
fading realisation) and whether it decoded. We then fit

    BLER(s) = 1 - (1 - floor) * sigmoid(k * (s - s50))

against slot SNR s, by maximum likelihood. Three numbers per (scenario, MCS):
  s50    SNR where the waterfall is centred
  k      steepness (steep in AWGN, flatter under frequency-selective fading)
  floor  error floor; > 0 when an impairment caps SINR below what this MCS needs

The slot SNR deliberately excludes impairments: the scheduler cannot see them, so an
impaired scenario simply needs more SNR (or never works) for the same MCS. That hidden
gap is what the ML model and the OOD detector are later tested against.

Abstraction caveat (for the report): conditioning on wideband slot SNR treats the
frequency selectivity within a slot as randomness inside each curve, rather than mapping it
exactly (as EESM/MIESM would).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .channels import ChannelSpec
from .phy_chain import MCS_RANGE, PDSCHLink, mcs_info

# ---------------------------------------------------------------- scenarios
CONFORMANCE_CHANNELS = {
    "AWGN": ChannelSpec("AWGN"),
    "TDLA30-10": ChannelSpec("A30", 10),
    "TDLB100-400": ChannelSpec("B100", 400),
    "TDLC300-100": ChannelSpec("C300", 100),
}
IMPAIRMENTS = [None] + [{k: lvl} for k in ("pa", "iq", "pn", "cfo")
                        for lvl in ("mild", "moderate", "severe")]


@dataclass(frozen=True)
class Scenario:
    channel: str                 # key of CONFORMANCE_CHANNELS
    impairment: tuple = ()       # e.g. (("pn", "severe"),)

    @property
    def name(self) -> str:
        imp = "+".join(f"{k}-{v}" for k, v in self.impairment) or "clean"
        return f"{self.channel}__{imp}"

    @property
    def spec(self) -> ChannelSpec:
        return CONFORMANCE_CHANNELS[self.channel]

    @property
    def imp_dict(self) -> dict | None:
        return dict(self.impairment) or None

    @classmethod
    def parse(cls, name: str) -> "Scenario":
        ch, imp = name.split("__")
        pairs = () if imp == "clean" else tuple(tuple(p.split("-")) for p in imp.split("+"))
        return cls(ch, pairs)


def preset(name: str) -> list[Scenario]:
    if name == "dev":        # quick CPU check
        return [Scenario("AWGN"), Scenario("TDLA30-10")]
    if name == "clean":      # all channels, no impairments
        return [Scenario(c) for c in CONFORMANCE_CHANNELS]
    if name == "full":       # 4 channels x 13 impairment settings = 52 scenarios
        return [Scenario(c, tuple(i.items()) if i else ())
                for c in CONFORMANCE_CHANNELS for i in IMPAIRMENTS]
    raise ValueError(name)


# ---------------------------------------------------------------- generation
def shannon_center_db(se: float, eff: float = 0.75) -> float:
    """Rough 10% BLER point in AWGN from spectral efficiency (within ~2 dB of the real
    curves); only used to centre the SNR sweep, never as a result."""
    return 10 * math.log10(2 ** (se / eff) - 1)


def sweep_snrs(scn: Scenario, mcs: int, cfg) -> list[float]:
    c = shannon_center_db(mcs_info(cfg, mcs).spectral_efficiency)
    # AWGN curves are a near-vertical cliff, so they need a fine grid (0.5 dB) or the
    # fitted threshold is only known to within the grid spacing.
    offs = ([x / 2 for x in range(-6, 11)] if scn.channel == "AWGN"
            else [-4, 0, 4, 8, 12, 16])
    return [round(c + o, 1) for o in offs]


def generate_scenario(link: PDSCHLink, scn: Scenario, n_tb: int = 256, batch: int = 128,
                      mcs_list=MCS_RANGE, log=print) -> dict:
    """Simulate one scenario. Returns per-block records as numpy arrays."""
    rec = {"mcs": [], "avg_snr": [], "slot_snr": [], "ok": []}
    for mcs in mcs_list:
        zero_err_in_a_row = 0
        for snr in sweep_snrs(scn, mcs, link.cfg):
            oks, slots = [], []
            for _ in range(max(1, n_tb // batch)):
                r = link.simulate(mcs, snr, scn.spec, scn.imp_dict, min(batch, n_tb))
                oks.append(r["tb_ok"].cpu().numpy())
                slots.append(r["snr_slot_db"].cpu().numpy())
            ok, sl = np.concatenate(oks), np.concatenate(slots)
            rec["mcs"].append(np.full(ok.size, mcs)); rec["avg_snr"].append(np.full(ok.size, snr))
            rec["slot_snr"].append(sl); rec["ok"].append(ok)
            # stop climbing once the curve is clearly finished (saves ~30% of the budget)
            zero_err_in_a_row = zero_err_in_a_row + 1 if ok.mean() > 0.99 else 0
            if zero_err_in_a_row >= 2:
                break
        log(f"  {scn.name}  MCS {mcs:2d} done")
    return {k: np.concatenate(v) for k, v in rec.items()}


# ---------------------------------------------------------------- fitting
def _model(s, s50, k, floor):
    return 1 - (1 - floor) * expit(k * (s - s50))   # expit avoids exp overflow warnings


def fit_curve(slot_snr: np.ndarray, ok: np.ndarray) -> dict:
    """Maximum-likelihood fit of (s50, k, floor) to per-block pass/fail outcomes."""
    fail = (~ok.astype(bool)).astype(float)
    if fail.mean() in (0.0, 1.0):          # degenerate: always passes / always fails
        s50 = slot_snr.min() - 3 if fail.mean() == 0 else 60.0
        return dict(s50=float(s50), k=2.0, floor=0.0, n=int(ok.size))

    def nll(p):
        s50, logk, lf = p
        pf = np.clip(_model(slot_snr, s50, math.exp(logk), 1 / (1 + math.exp(-lf))),
                     1e-9, 1 - 1e-9)
        return -(fail * np.log(pf) + (1 - fail) * np.log(1 - pf)).sum()

    s0 = float(np.median(slot_snr[fail > 0]))
    res = minimize(nll, x0=[s0, math.log(1.5), -6.0], method="L-BFGS-B",
                   bounds=[(-20, 60), (math.log(0.1), math.log(30)), (-12, 12)])
    s50, logk, lf = res.x
    return dict(s50=float(s50), k=float(math.exp(logk)),
                floor=float(1 / (1 + math.exp(-lf))), n=int(ok.size))


def fit_records(rec: dict) -> dict:
    return {int(m): fit_curve(rec["slot_snr"][rec["mcs"] == m], rec["ok"][rec["mcs"] == m])
            for m in np.unique(rec["mcs"])}


# ---------------------------------------------------------------- lookup
class BLERTable:
    """fits: {scenario_name: {mcs: {s50, k, floor, n}}}"""

    def __init__(self, fits: dict):
        self.fits = {s: {int(m): v for m, v in d.items()} for s, d in fits.items()}

    @classmethod
    def load(cls, path: str | Path) -> "BLERTable":
        return cls(json.loads(Path(path).read_text()))

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.fits, indent=1))

    @property
    def scenarios(self) -> list[str]:
        return list(self.fits)

    def mcs_list(self, scenario: str) -> list[int]:
        return sorted(self.fits[scenario])

    def bler(self, scenario: str, mcs: int, slot_snr_db):
        p = self.fits[scenario][int(mcs)]
        return _model(np.asarray(slot_snr_db, dtype=float), p["s50"], p["k"], p["floor"])

    def bler_all(self, scenario: str, slot_snr_db):
        """BLER of every MCS at once: returns [..., n_mcs] in mcs_list order (vectorised)."""
        if not hasattr(self, "_arr"):
            self._arr = {}
        if scenario not in self._arr:
            ps = [self.fits[scenario][m] for m in self.mcs_list(scenario)]
            self._arr[scenario] = tuple(np.array([p[k] for p in ps]) for k in ("s50", "k", "floor"))
        s50, k, fl = self._arr[scenario]
        s = np.asarray(slot_snr_db, dtype=float)[..., None]
        return _model(s, s50, k, fl)

    def snr_at_bler(self, scenario: str, mcs: int, target: float = 0.1) -> float:
        """Slot SNR needed for BLER = target (inf if the error floor is above target)."""
        p = self.fits[scenario][int(mcs)]
        if p["floor"] >= target:
            return math.inf
        q = (1 - target) / (1 - p["floor"])          # required sigmoid value
        return p["s50"] + math.log(q / (1 - q)) / p["k"]

    @classmethod
    def synthetic(cls, cfg=None) -> "BLERTable":
        """FAKE table from a Shannon-gap formula. Only for testing code paths; never report it."""
        from .phy_chain import LinkConfig
        cfg = cfg or LinkConfig()
        fits = {}
        for name, k, shift in (("AWGN__clean", 3.0, 0.0), ("TDLA30-10__clean", 0.8, 1.0)):
            fits[name] = {m: dict(s50=shannon_center_db(mcs_info(cfg, m).spectral_efficiency)
                                  + shift - 0.7, k=k, floor=0.0, n=0) for m in MCS_RANGE}
        return cls(fits)


# ---------------------------------------------------------------- driver
def build_tables(scenarios: list[Scenario], out_dir: str | Path, n_tb: int = 256,
                 batch: int = 128, mcs_list=MCS_RANGE, device: str | None = None) -> BLERTable:
    """Resumable: each scenario is saved as soon as it finishes and skipped next time."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    link = PDSCHLink(device=device)
    print(f"device={link.device}, {len(scenarios)} scenarios, {n_tb} TBs per SNR point")
    for i, scn in enumerate(scenarios, 1):
        f = out / f"{scn.name}.npz"
        if f.exists():
            print(f"[{i}/{len(scenarios)}] {scn.name}: exists, skipping"); continue
        t0 = time.time()
        rec = generate_scenario(link, scn, n_tb, batch, mcs_list)
        np.savez_compressed(f, **rec)
        print(f"[{i}/{len(scenarios)}] {scn.name}: {rec['ok'].size} TBs in {time.time()-t0:.0f} s")
    fits = {f.stem: fit_records(dict(np.load(f))) for f in sorted(out.glob("*.npz"))}
    table = BLERTable(fits)
    table.save(out / "bler_fits.json")
    print(f"saved {out / 'bler_fits.json'} ({len(fits)} scenarios)")
    return table
