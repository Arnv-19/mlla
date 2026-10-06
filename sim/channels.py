"""Fading channels for the link-level chain, wrapping Sionna's 3GPP TDL models.

Profiles:
  "AWGN"                     no fading (unit gain), for sanity checks
  "A30", "B100", "C300"      fixed-delay TS 38.101-4 conformance profiles. Combined with a
                             Doppler they give the familiar test names, e.g.
                             TDLA30-10, TDLB100-400, TDLC300-100.
  "A".."E"                   TR 38.901 scalable profiles; set delay_spread.

Approximation (documented in the report): the channel is sampled once per OFDM symbol and
held constant within the symbol. This captures channel ageing between the DMRS and data
symbols, which is what hurts link adaptation at high Doppler, but ignores intra-symbol ICI
from Doppler (small at 30 kHz SCS: 400 Hz / 30 kHz ~ 1.3%). It is roughly 500x cheaper than
per-sample fading on a CPU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.channel import cir_to_time_channel

SPEED_OF_LIGHT = 299_792_458.0
FIXED_PROFILES = {"A30", "B100", "C300", "A10", "C60", "D10", "D30"}
# Approximate maximum normalised delay of the scalable TR 38.901 profiles.
MAX_NORM_DELAY = {"A": 9.66, "B": 4.79, "C": 8.66, "D": 12.53, "E": 20.66}


@dataclass(frozen=True)
class ChannelSpec:
    profile: str = "A30"
    doppler_hz: float = 10.0
    delay_spread: float = 100e-9     # only used by scalable profiles A..E

    @property
    def name(self) -> str:
        if self.profile == "AWGN":
            return "AWGN"
        if self.profile in FIXED_PROFILES:
            return f"TDL{self.profile}-{int(self.doppler_hz)}"
        return f"TDL{self.profile}-{int(self.delay_spread * 1e9)}ns-{int(self.doppler_hz)}Hz"


class TimeVaryingChannel:
    """Generates discrete-time taps h[b, rx_ant, symbol, lag] at the sample rate fs."""

    def __init__(self, spec: ChannelSpec, fc: float, fs: float, symbol_duration: float,
                 num_rx_ant: int = 1, device: str = "cpu"):
        self.spec, self.fs, self.num_rx_ant, self.device = spec, fs, num_rx_ant, device
        self.symbol_rate = 1.0 / symbol_duration
        if spec.profile == "AWGN":
            self.tdl, self.l_min, self.l_max = None, 0, 0
            return
        speed = spec.doppler_hz * SPEED_OF_LIGHT / fc
        kw = dict(carrier_frequency=fc, min_speed=speed, num_rx_ant=num_rx_ant, device=device)
        if spec.profile in FIXED_PROFILES:
            self.tdl = TDL(spec.profile, **kw)
            max_delay = 3e-6
        else:
            self.tdl = TDL(spec.profile, delay_spread=spec.delay_spread, **kw)
            max_delay = MAX_NORM_DELAY[spec.profile] * spec.delay_spread
        # sinc-interpolated taps need a few extra lags either side
        self.l_min = -6
        self.l_max = int(math.ceil(max_delay * fs)) + 6

    def taps(self, batch: int, num_symbols: int) -> torch.Tensor:
        """Returns complex taps [batch, num_rx_ant, num_symbols, L] (lag index starts at self.l_min)."""
        if self.tdl is None:
            h = torch.ones(batch, self.num_rx_ant, num_symbols, 1, dtype=torch.complex64,
                           device=self.device)
            return h
        a, tau = self.tdl(batch, num_symbols, self.symbol_rate)
        # Each call draws fresh sum-of-sinusoids phases, so batches are independent fades.
        # normalize=False on purpose: Sionna's normalisation rescales every slot to unit
        # energy, which would erase the slot-to-slot power fades that link adaptation
        # has to track. TDL path powers already average to one across realisations.
        h = cir_to_time_channel(self.fs, a, tau, self.l_min, self.l_max, normalize=False)
        # h: [b, rx=1, rx_ant, tx=1, tx_ant=1, time, L]
        return h[:, 0, :, 0, 0]


def apply_taps(x: torch.Tensor, h: torch.Tensor, l_min: int, samples_per_symbol: int) -> torch.Tensor:
    """y[n] = sum_l h_{sym(n)}[l] x[n - l - l_min] for each receive antenna.

    x: [b, N] complex, N = num_symbols * samples_per_symbol
    h: [b, rx_ant, num_symbols, L]
    returns y: [b, rx_ant, N]
    """
    b, n = x.shape
    L = h.shape[-1]
    h_n = h.repeat_interleave(samples_per_symbol, dim=2)          # [b, rx, N, L]
    # pad so that negative and positive lags are both available
    pad_left = max(L - 1 + l_min, 0)
    pad_right = max(-l_min, 0)
    xp = torch.nn.functional.pad(torch.view_as_real(x).permute(0, 2, 1), (pad_left, pad_right))
    xp = torch.view_as_complex(xp.permute(0, 2, 1).contiguous())  # [b, N + pads]
    y = torch.zeros(b, h.shape[1], n, dtype=x.dtype, device=x.device)
    for li in range(L):
        lag = li + l_min                                          # actual delay in samples
        start = pad_left - lag
        y += h_n[..., li] * xp[:, None, start:start + n]
    return y


def slot_gain_series(spec: ChannelSpec, fc: float, slot_duration: float, n_slots: int,
                     num_rx_ant: int = 1, device: str = "cpu") -> torch.Tensor:
    """Wideband channel power gain per slot (linear, mean ~1), for the fast closed-loop sim.

    Sampled once per slot from the same TDL model as the full PHY, so the fading speed
    matches the Doppler. Summed over RX antennas (MRC gain).
    """
    if spec.profile == "AWGN":
        return torch.full((n_slots,), float(num_rx_ant))
    ch = TimeVaryingChannel(spec, fc, 1.0 / slot_duration, slot_duration, num_rx_ant, device)
    a, _ = ch.tdl(1, n_slots, 1.0 / slot_duration)          # [1,1,rx,1,1,paths,T]
    return (a.abs() ** 2).sum(dim=5)[0, 0, :, 0, 0].sum(0).cpu()
