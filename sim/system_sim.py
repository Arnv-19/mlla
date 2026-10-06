"""Fast closed-loop link adaptation simulation: one loop iteration per slot, no decoding.

Per slot t:
  1. the channel has a true slot SNR (from a TDL fading time series)
  2. the scheduler sees only a delayed, noisy, quantised SNR report (like a CQI report),
     rough Doppler / delay-spread estimates, and ACK/NACK feedback from earlier slots
  3. the policy (OLLA, the ML model, or the oracle) picks an MCS
  4. ACK/NACK is drawn from the BLER table at the TRUE slot SNR for the TRUE scenario
     (including any impairment the scheduler cannot see)

Simplifications: single user, full buffer, no HARQ retransmissions (a NACK loses that
slot's TB), CSI report modelled as wideband SNR rather than a CQI index.

Fair comparisons: pass the same `gains` and seed to every policy, so they all face the
same fading and the same report errors (common random numbers).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from dut.features import N_FEATURES, FeatureTracker
from .bler_tables import BLERTable, Scenario
from .channels import slot_gain_series
from .phy_chain import LinkConfig, mcs_info

DELAY_SPREAD_NS = {"AWGN": 0.0, "A30": 30.0, "B100": 100.0, "C300": 300.0}


@dataclass
class LoopConfig:
    avg_snr_db: float = 15.0
    n_slots: int = 20_000           # 10 s of 0.5 ms slots
    csi_period: int = 10            # slots between SNR reports (5 ms)
    csi_delay: int = 4              # slots from measurement to use
    csi_noise_db: float = 1.0       # measurement error std
    csi_step_db: float = 1.0        # report quantisation
    ack_delay: int = 2              # slots before ACK/NACK reaches the scheduler
    est_error: float = 0.2          # relative error of Doppler / delay-spread estimates
    seed: int = 0


def make_gains(scenario: str, n_slots: int, seed: int, link_cfg: LinkConfig | None = None):
    import torch
    link_cfg = link_cfg or LinkConfig()
    torch.manual_seed(seed)
    spec = Scenario.parse(scenario).spec
    return slot_gain_series(spec, link_cfg.fc, link_cfg.slot_duration, n_slots,
                            link_cfg.num_rx_ant).numpy()


def run_loop(policy, table: BLERTable, scenario: str, cfg: LoopConfig,
             link_cfg: LinkConfig | None = None, gains: np.ndarray | None = None,
             record_features: bool = False) -> dict:
    link_cfg = link_cfg or LinkConfig()
    rng = np.random.default_rng(cfg.seed)
    spec = Scenario.parse(scenario).spec
    if gains is None:
        gains = make_gains(scenario, cfg.n_slots, cfg.seed, link_cfg)
    gains = gains[:cfg.n_slots]
    true_snr = cfg.avg_snr_db + 10 * np.log10(np.maximum(gains, 1e-12))
    tbs = {m: mcs_info(link_cfg, m).tbs for m in table.mcs_list(scenario)}
    dopp_true = 0.0 if spec.profile == "AWGN" else spec.doppler_hz
    ds_true = DELAY_SPREAD_NS.get(spec.profile, spec.delay_spread * 1e9)

    policy.reset()
    tracker = FeatureTracker()
    pending = deque()                       # (slot when feedback arrives, ack)
    report = 0.0
    log = {k: np.zeros(cfg.n_slots) for k in ("true_snr", "report", "mcs", "ack", "bits")}
    feats = np.zeros((cfg.n_slots, N_FEATURES), np.float32) if record_features else None
    for t in range(cfg.n_slots):
        if t % cfg.csi_period == 0:
            src = max(t - cfg.csi_delay, 0)
            noisy = true_snr[src] + rng.normal(0, cfg.csi_noise_db)
            report = cfg.csi_step_db * np.round(noisy / cfg.csi_step_db)
            e1, e2 = rng.lognormal(0, cfg.est_error, 2)
            tracker.on_report(report, src, dopp_true * e1, ds_true * e2)
        while pending and pending[0][0] <= t:
            _, a = pending.popleft()
            policy.update(a)
            tracker.on_feedback(a)
        f = tracker.vector(t)
        obs = dict(snr_report_db=report, features=f,
                   _true_snr=true_snr[t], _scenario=scenario)   # "_" = genie, oracle only
        mcs = policy.select(obs)
        tracker.on_decision(mcs)
        ack = rng.random() >= table.bler(scenario, mcs, true_snr[t])
        pending.append((t + cfg.ack_delay, bool(ack)))
        log["true_snr"][t], log["report"][t], log["mcs"][t] = true_snr[t], report, mcs
        log["ack"][t], log["bits"][t] = ack, tbs[mcs] * ack
        if record_features:
            feats[t] = f

    dur = cfg.n_slots * link_cfg.slot_duration
    out = dict(log=log, bler=1 - log["ack"].mean(),
               throughput_mbps=log["bits"].sum() / dur / 1e6,
               spectral_eff=log["bits"].sum() / dur / (link_cfg.n_sc * link_cfg.scs),
               mean_mcs=log["mcs"].mean())
    if record_features:
        out["features"] = feats
    return out


class OraclePolicy:
    """Upper bound: knows the true SNR of the current slot and the true scenario."""
    name = "Oracle"

    def __init__(self, table: BLERTable, link_cfg: LinkConfig | None = None):
        cfg = link_cfg or LinkConfig()
        self.table = table
        self.mcs = np.array(sorted(next(iter(table.fits.values()))))
        self.tbs = np.array([mcs_info(cfg, m).tbs for m in self.mcs])

    def reset(self): ...
    def update(self, ack): ...

    def select(self, obs):
        p_ack = 1 - self.table.bler_all(obs["_scenario"], obs["_true_snr"])
        return int(self.mcs[np.argmax(self.tbs * p_ack)])
