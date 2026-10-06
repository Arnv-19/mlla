"""NR-like PDSCH link-level simulator (single layer, 1 TX antenna, 1..N RX antennas).

Chain per slot:
  TB bits -> TB CRC + segmentation + LDPC + rate matching + scrambling (Sionna TBEncoder)
  -> QAM -> resource grid with DMRS -> OFDM (IFFT + CP) -> [TX impairments]
  -> TDL fading -> AWGN -> [RX impairments] -> CP removal + FFT
  -> LS channel estimate on DMRS, linear interpolation in frequency and time
  -> MRC equalisation -> soft demapping -> LDPC decode -> TB CRC

Numerology: 30 kHz SCS, 24 PRBs (8.64 MHz occupied), FFT 512, fs = 15.36 MHz, normal CP,
14 symbols per slot (0.5 ms), DMRS type-1 comb on symbols 2 and 11 with no data on the
DMRS symbols (+3 dB DMRS power boost, as for 2 CDM groups without data). MCS from
TS 38.214 Table 5.1.3.1-1 (up to 64QAM), TBS from TS 38.214 Sec. 5.1.3.2.

Simplifications (state these in the report):
  * no PDCCH region, PTRS, CSI-RS or SSB in the slot
  * no HARQ soft combining (each slot is a first transmission)
  * receiver knows the true noise variance (no noise estimation)
  * channel is constant within an OFDM symbol (see sim/channels.py)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import torch
from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.nr import TBEncoder, TBDecoder
from sionna.phy.nr.utils import calculate_tb_size, decode_mcs_index

from .channels import ChannelSpec, TimeVaryingChannel, apply_taps
from .impairments import ImpairmentSet

NUM_MCS = 29  # Table 5.1.3.1-1, indices 0..28
# Sionna's LDPC encoder rejects code rates below 1/5, which rules out MCS 0-2 (R = 0.12-0.19).
# Real NR reaches them through repetition in rate matching. We restrict the action space to
# MCS 3..28; the lowest-SNR region (below about -2 dB) is therefore out of scope.
MCS_RANGE = range(3, 29)


@dataclass
class LinkConfig:
    scs: float = 30e3
    n_prb: int = 24
    fft_size: int = 512
    cp_len: int = 36
    n_sym: int = 14
    dmrs_symbols: tuple = (2, 11)
    fc: float = 3.5e9
    num_rx_ant: int = 1
    bp_iter: int = 20

    @property
    def n_sc(self) -> int:
        return 12 * self.n_prb

    @property
    def fs(self) -> float:
        return self.fft_size * self.scs

    @property
    def sym_len(self) -> int:
        return self.fft_size + self.cp_len

    @property
    def symbol_duration(self) -> float:
        return self.sym_len / self.fs

    @property
    def slot_duration(self) -> float:
        return 1e-3 / (self.scs / 15e3)

    @property
    def data_symbols(self) -> list[int]:
        return [s for s in range(self.n_sym) if s not in self.dmrs_symbols]

    @property
    def n_data_re(self) -> int:
        return self.n_sc * len(self.data_symbols)


@dataclass
class MCSInfo:
    index: int
    qm: int
    rate: float
    tbs: int
    n_coded_bits: int

    @property
    def spectral_efficiency(self) -> float:
        return self.qm * self.rate


def mcs_info(cfg: LinkConfig, mcs: int) -> MCSInfo:
    qm, r = decode_mcs_index(mcs, table_index=1, is_pusch=False)
    qm, r = int(qm), float(r)
    tbs = int(calculate_tb_size(qm, r, num_prbs=cfg.n_prb, num_ofdm_symbols=cfg.n_sym,
                                num_dmrs_per_prb=12 * len(cfg.dmrs_symbols))[0])
    return MCSInfo(mcs, qm, r, tbs, cfg.n_data_re * qm)


class PDSCHLink:
    def __init__(self, cfg: LinkConfig | None = None, device: str | None = None):
        self.cfg = cfg or LinkConfig()
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":          # Sionna only accepts explicit indices like "cuda:0"
            self.device = "cuda:0"
        c, dev = self.cfg, self.device
        # active subcarriers k = -n_sc/2 .. n_sc/2-1 mapped to FFT bins
        k = torch.arange(c.n_sc) - c.n_sc // 2
        self.bins = (k % c.fft_size).long().to(dev)
        # DMRS: type-1 comb (even subcarriers), fixed QPSK sequence, +3 dB boost
        g = torch.Generator().manual_seed(1234)
        bits = torch.randint(0, 2, (len(c.dmrs_symbols), c.n_sc // 2, 2), generator=g).float()
        qpsk = ((1 - 2 * bits[..., 0]) + 1j * (1 - 2 * bits[..., 1])) / math.sqrt(2)
        self.dmrs = (math.sqrt(2) * qpsk).to(torch.complex64).to(dev)  # [n_dmrs_sym, n_sc/2]
        self.pilot_sc = torch.arange(0, c.n_sc, 2, device=dev)
        self._codecs: dict[int, tuple] = {}
        self._mappers: dict[int, Mapper] = {}

    # ------------------------------------------------------------------ codec
    def _codec(self, mcs: int):
        if mcs not in MCS_RANGE:
            raise ValueError(f"MCS {mcs} outside supported range {MCS_RANGE.start}..{MCS_RANGE.stop - 1}")
        if mcs not in self._codecs:
            info = mcs_info(self.cfg, mcs)
            d = self.device
            enc = TBEncoder(target_tb_size=info.tbs, num_coded_bits=info.n_coded_bits,
                            target_coderate=info.rate, num_bits_per_symbol=info.qm,
                            channel_type="PDSCH", device=d)
            dec = TBDecoder(enc, num_bp_iter=self.cfg.bp_iter, device=d)
            self._codecs[mcs] = (info, enc, dec, Mapper("qam", info.qm, device=d),
                                 Demapper("app", "qam", info.qm, device=d))
        return self._codecs[mcs]

    # ------------------------------------------------------------------ TX
    def _build_grid(self, x_data: torch.Tensor) -> torch.Tensor:
        """x_data [b, n_data_re] -> frequency grid [b, n_sym, n_sc] (frequency-first mapping)."""
        c, b = self.cfg, x_data.shape[0]
        grid = torch.zeros(b, c.n_sym, c.n_sc, dtype=torch.complex64, device=self.device)
        grid[:, c.data_symbols, :] = x_data.reshape(b, len(c.data_symbols), c.n_sc)
        for i, s in enumerate(c.dmrs_symbols):
            grid[:, s, self.pilot_sc] = self.dmrs[i]
        return grid

    def _ofdm_mod(self, grid: torch.Tensor) -> torch.Tensor:
        c, b = self.cfg, grid.shape[0]
        full = torch.zeros(b, c.n_sym, c.fft_size, dtype=torch.complex64, device=self.device)
        full[..., self.bins] = grid
        t = torch.fft.ifft(full, norm="ortho")
        t = torch.cat([t[..., -c.cp_len:], t], dim=-1)               # add CP
        return t.reshape(b, -1)

    def _ofdm_demod(self, y: torch.Tensor) -> torch.Tensor:
        """y [b, rx, N] -> grid [b, rx, n_sym, n_sc]."""
        c = self.cfg
        y = y.reshape(*y.shape[:2], c.n_sym, c.sym_len)[..., c.cp_len:]
        return torch.fft.fft(y, norm="ortho")[..., self.bins]

    # ------------------------------------------------------------------ RX
    def _estimate_channel(self, Y: torch.Tensor) -> torch.Tensor:
        """LS on DMRS + linear interpolation. Y [b, rx, n_sym, n_sc] -> H [b, rx, n_sym, n_sc]."""
        c = self.cfg
        hp = Y[:, :, list(c.dmrs_symbols)][..., self.pilot_sc] / self.dmrs   # [b, rx, 2, n_sc/2]
        # frequency: pilots on even subcarriers; odd ones = mean of neighbours, last = nearest
        hf = torch.zeros(*hp.shape[:-1], c.n_sc, dtype=hp.dtype, device=self.device)
        hf[..., 0::2] = hp
        hf[..., 1:-1:2] = 0.5 * (hp[..., :-1] + hp[..., 1:])
        hf[..., -1] = hp[..., -1]
        # time: linear inter/extrapolation between the two DMRS symbols
        s0, s1 = c.dmrs_symbols
        w = (torch.arange(c.n_sym, dtype=torch.float32, device=self.device) - s0) / (s1 - s0)
        w = w.view(1, 1, -1, 1)
        return hf[:, :, :1] + (hf[:, :, 1:2] - hf[:, :, :1]) * w

    # ------------------------------------------------------------------ main
    @torch.no_grad()
    def simulate(self, mcs: int, snr_db: float, channel: ChannelSpec,
                 impairments: dict | None = None, batch: int = 32,
                 return_symbols: bool = False) -> dict:
        """Simulate `batch` independent slots. SNR is average Es/N0 per data RE."""
        c = self.cfg
        info, enc, dec, mapper, demapper = self._codec(mcs)
        imp = ImpairmentSet.from_spec(impairments, c.fs, c.fft_size)
        ch = TimeVaryingChannel(channel, c.fc, c.fs, c.symbol_duration, c.num_rx_ant, self.device)

        bits = torch.randint(0, 2, (batch, info.tbs), device=self.device).float()
        x = mapper(enc(bits))                                         # [b, n_data_re]
        tx = imp.apply_tx(self._ofdm_mod(self._build_grid(x)))         # [b, N]

        h = ch.taps(batch, c.n_sym)
        y = apply_taps(tx, h, ch.l_min, c.sym_len)                     # [b, rx, N]
        n0 = 10 ** (-snr_db / 10)
        noise = torch.randn(*y.shape, 2, device=self.device) * math.sqrt(n0 / 2)
        y = imp.apply_rx(y + torch.view_as_complex(noise))

        Y = self._ofdm_demod(y)
        H = self._estimate_channel(Y)
        ds = c.data_symbols
        Yd, Hd = Y[:, :, ds], H[:, :, ds]                              # [b, rx, n_ds, n_sc]
        hpow = (Hd.abs() ** 2).sum(dim=1).clamp_min(1e-9)              # MRC
        x_hat = ((Hd.conj() * Yd).sum(dim=1) / hpow).reshape(batch, -1)
        no_eff = (n0 / hpow).reshape(batch, -1)
        llr = demapper(x_hat, no_eff)
        _, tb_ok = dec(llr)

        # post-equalisation SINR per slot (includes estimation error and impairments)
        sinr = (x.abs() ** 2).mean(-1) / ((x_hat - x).abs() ** 2).mean(-1)
        # wideband slot SNR: what a perfect wideband measurement of this slot would report
        gain = (h.abs() ** 2).sum(-1).mean(-1).sum(1)                   # [b], MRC over RX
        snr_slot = snr_db + 10 * torch.log10(gain.clamp_min(1e-12))
        out = dict(tb_ok=tb_ok.bool(), tbs=info.tbs, snr_slot_db=snr_slot,
                   sinr_post_db=10 * torch.log10(sinr),
                   impairments=imp.describe(), channel=channel.name)
        if return_symbols:
            out["x_hat"], out["x"] = x_hat, x
        return out


@torch.no_grad()
def simulate_symbols(link: "PDSCHLink", qm: int, snr_db: float, channel: ChannelSpec,
                     impairments: dict | None = None, batch: int = 16) -> dict:
    """Same TX/channel/RX chain as simulate(), but random coded bits and no LDPC decoding.

    For the diagnosis dataset: we only need the equalised symbols, so skipping the encoder
    and decoder makes this ~20x faster and fine on a CPU.
    Returns x_hat and x as [batch, n_data_symbols, n_sc].
    """
    c = link.cfg
    if qm not in link._mappers:
        link._mappers[qm] = Mapper("qam", qm, device=link.device)
    imp = ImpairmentSet.from_spec(impairments, c.fs, c.fft_size)
    ch = TimeVaryingChannel(channel, c.fc, c.fs, c.symbol_duration, c.num_rx_ant, link.device)
    bits = torch.randint(0, 2, (batch, c.n_data_re * qm), device=link.device).float()
    x = link._mappers[qm](bits)
    tx = imp.apply_tx(link._ofdm_mod(link._build_grid(x)))
    h = ch.taps(batch, c.n_sym)
    y = apply_taps(tx, h, ch.l_min, c.sym_len)
    n0 = 10 ** (-snr_db / 10)
    y = imp.apply_rx(y + torch.view_as_complex(
        torch.randn(*y.shape, 2, device=link.device) * math.sqrt(n0 / 2)))
    Y = link._ofdm_demod(y)
    H = link._estimate_channel(Y)
    ds = c.data_symbols
    Yd, Hd = Y[:, :, ds], H[:, :, ds]
    x_hat = (Hd.conj() * Yd).sum(1) / (Hd.abs() ** 2).sum(1).clamp_min(1e-9)
    return dict(x_hat=x_hat.cpu(), x=x.reshape(batch, len(ds), c.n_sc).cpu())


def bler_point(link: PDSCHLink, mcs: int, snr_db: float, channel: ChannelSpec,
               impairments: dict | None = None, max_tb: int = 2000, min_errors: int = 100,
               batch: int = 32) -> dict:
    """Monte Carlo BLER with early stopping once `min_errors` block errors are seen."""
    n = err = 0
    while n < max_tb and err < min_errors:
        r = link.simulate(mcs, snr_db, channel, impairments, batch)
        n += batch
        err += int((~r["tb_ok"]).sum())
    bler = err / n
    tput = (1 - bler) * r["tbs"] / link.cfg.slot_duration
    return dict(mcs=mcs, snr_db=snr_db, channel=channel.name, bler=bler, n_tb=n,
                n_err=err, tbs=r["tbs"], throughput_mbps=tput / 1e6)
