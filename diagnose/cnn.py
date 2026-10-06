"""Small CNN that names the impairment from the diagnosis views (+ the known SNR).

The SNR input matters for PA compression: in OFDM its distortion looks like extra noise
(Bussgang), so it is only recognisable as "more noise than this SNR explains". A call box
knows the SNR because it sets the noise level itself.
"""
import torch
from torch import nn

from .dataset import CLASSES


class DiagCNN(nn.Module):
    def __init__(self, n_views: int = 4, use_snr: bool = True):
        super().__init__()
        self.n_views, self.use_snr = n_views, use_snr
        self.conv = nn.Sequential(
            nn.Conv2d(n_views, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16x16
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),        # 8x8
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))        # 4x4
        self.head = nn.Sequential(nn.Linear(64 * 16 + int(use_snr), 64), nn.ReLU(),
                                  nn.Linear(64, len(CLASSES)))

    def forward(self, x, snr_db):
        z = self.conv(x).flatten(1)
        if self.use_snr:
            z = torch.cat([z, (snr_db[:, None] - 20) / 10], dim=1)
        return self.head(z)
