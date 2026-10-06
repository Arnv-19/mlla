"""RF impairment models applied to complex baseband time-domain samples.

All blocks take a complex tensor of shape [..., num_samples] and return the same shape.
Where impairments sit in the downlink chain (call box = gNB transmitter, UE = receiver):

    TX: bits -> QAM -> OFDM -> [PA nonlinearity] -> channel -> AWGN
    RX: -> [CFO] -> [phase noise] -> [IQ imbalance] -> OFDM demod -> ...

Thermal noise is added before the RX impairments, because the LO and IQ mixer act on
signal plus noise together.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import torch


class Impairment:
    name = "base"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def params(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class RappPA(Impairment):
    """Rapp solid-state PA model (AM/AM only, no AM/PM).

    out_amp = r / (1 + (r / A_sat)^(2p))^(1/(2p))

    ibo_db: input back-off, i.e. saturation power relative to the mean input power.
            Lower = more compression. Typical: 10 dB mild, 4 dB severe.
    p:      smoothness. p=2..3 is a common solid-state value; p -> inf is a hard clipper.
    """
    ibo_db: float = 10.0
    p: float = 2.0
    name: str = field(default="pa", init=False)

    def __call__(self, x):
        p_in = (x.abs() ** 2).mean(dim=-1, keepdim=True)
        a_sat = torch.sqrt(p_in * 10 ** (self.ibo_db / 10))
        r = x.abs()
        g = 1.0 / (1.0 + (r / a_sat) ** (2 * self.p)) ** (1.0 / (2 * self.p))
        return x * g


@dataclass
class IQImbalance(Impairment):
    """Receiver IQ imbalance.

    y = mu * x + nu * conj(x)
    mu = (1 + g e^{-j phi}) / 2,  nu = (1 - g e^{+j phi}) / 2

    In OFDM this leaks subcarrier k onto its mirror -k. Image rejection ratio
    IRR = |mu|^2 / |nu|^2, reported by .irr_db().
    """
    gain_db: float = 0.5
    phase_deg: float = 3.0
    name: str = field(default="iq", init=False)

    def _coeffs(self):
        g = 10 ** (self.gain_db / 20)
        phi = math.radians(self.phase_deg)
        mu = complex(1 + g * math.cos(-phi), g * math.sin(-phi)) / 2
        nu = complex(1 - g * math.cos(phi), -g * math.sin(phi)) / 2
        return mu, nu

    def irr_db(self) -> float:
        mu, nu = self._coeffs()
        return 10 * math.log10(abs(mu) ** 2 / max(abs(nu) ** 2, 1e-30))

    def __call__(self, x):
        mu, nu = self._coeffs()
        return mu * x + nu * torch.conj(x)


@dataclass
class PhaseNoise(Impairment):
    """Free-running oscillator phase noise as a Wiener process (Lorentzian spectrum).

    phi[n] = phi[n-1] + w[n],  w ~ N(0, 2*pi*linewidth/fs)

    linewidth_hz: 3-dB linewidth of the oscillator. Each slot starts at a random
                  phase; the DMRS-based channel estimate absorbs the common part,
                  the drift across the slot (CPE) and ICI remain.
    """
    linewidth_hz: float = 100.0
    fs: float = 15.36e6
    name: str = field(default="pn", init=False)

    def __call__(self, x):
        var = 2 * math.pi * self.linewidth_hz / self.fs
        steps = torch.randn(x.shape, device=x.device) * math.sqrt(var)
        phi0 = torch.rand(x.shape[:-1] + (1,), device=x.device) * 2 * math.pi
        phi = phi0 + torch.cumsum(steps, dim=-1)
        return x * torch.exp(1j * phi).to(x.dtype)


@dataclass
class CFO(Impairment):
    """Residual carrier frequency offset after the UE's frequency correction.

    eps: offset normalised to the subcarrier spacing (e.g. 0.02 = 2% of SCS = 600 Hz at 30 kHz).
    Causes phase rotation across symbols plus inter-carrier interference.
    """
    eps: float = 0.02
    fft_size: int = 512
    name: str = field(default="cfo", init=False)

    def __call__(self, x):
        n = torch.arange(x.shape[-1], device=x.device, dtype=torch.float32)
        rot = torch.exp(1j * 2 * math.pi * self.eps * n / self.fft_size)
        return x * rot.to(x.dtype)


# ---------------------------------------------------------------------------
# Severity presets. These are the knobs the harness sweeps and the CNN classifies.
# Values are illustrative, not calibrated to a specific radio.
# ---------------------------------------------------------------------------
SEVERITY = {
    "pa":  {"mild": dict(ibo_db=8.0),  "moderate": dict(ibo_db=5.0),  "severe": dict(ibo_db=3.0)},
    "iq":  {"mild": dict(gain_db=0.3, phase_deg=2.0),
            "moderate": dict(gain_db=1.0, phase_deg=7.0),
            "severe": dict(gain_db=1.5, phase_deg=10.0)},
    "pn":  {"mild": dict(linewidth_hz=10.0), "moderate": dict(linewidth_hz=50.0),
            "severe": dict(linewidth_hz=200.0)},
    "cfo": {"mild": dict(eps=0.01), "moderate": dict(eps=0.015), "severe": dict(eps=0.02)},
}
# Calibrated so each level caps post-equalisation SINR at roughly
# mild ~27-30 dB, moderate ~18-21 dB, severe ~12-16 dB (AWGN, 45 dB SNR, median over slots).
# Re-run scripts/calibrate_impairments.py if you change the numerology or receiver.
TX_SIDE = {"pa"}


@dataclass
class ImpairmentSet:
    """A named combination of impairments, split into TX-side and RX-side chains."""
    tx: list = field(default_factory=list)
    rx: list = field(default_factory=list)

    @classmethod
    def from_spec(cls, spec: dict | None, fs: float, fft_size: int) -> "ImpairmentSet":
        """spec examples: None, {"pn": "moderate"}, {"pa": "mild", "iq": "severe"}."""
        out = cls()
        for kind, level in (spec or {}).items():
            kw = dict(SEVERITY[kind][level]) if isinstance(level, str) else dict(level)
            if kind == "pa":
                out.tx.append(RappPA(**kw))
            elif kind == "iq":
                out.rx.append(IQImbalance(**kw))
            elif kind == "pn":
                out.rx.append(PhaseNoise(fs=fs, **kw))
            elif kind == "cfo":
                out.rx.append(CFO(fft_size=fft_size, **kw))
            else:
                raise ValueError(f"unknown impairment {kind}")
        # Fixed physical order at the receiver: CFO, then phase noise, then IQ imbalance.
        order = {"cfo": 0, "pn": 1, "iq": 2}
        out.rx.sort(key=lambda b: order[b.name])
        return out

    def apply_tx(self, x):
        for b in self.tx:
            x = b(x)
        return x

    def apply_rx(self, x):
        for b in self.rx:
            x = b(x)
        return x

    def describe(self) -> list[dict]:
        return [{"type": b.name, **b.params()} for b in self.tx + self.rx]
