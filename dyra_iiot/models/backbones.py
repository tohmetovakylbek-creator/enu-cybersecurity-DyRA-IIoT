"""
dyra_iiot/models/backbones.py
─────────────────────────────────────────────────────────────────────────────
Five backbone architectures evaluated in DyRA-IIoT (Section 3.1.3).

All models:
  • Accept input  (B, L, F) — batch × window_len × n_features.
  • Return raw logits  (B,)   — sigmoid applied OUTSIDE the model.
  • Use BCEWithLogitsLoss for numerically stable training.

Parameter counts at L=50, F=36 (as reported in Table 6):
  TiDE               627,201
  1D-CNN              20,097
  LSTM               217,217
  DLinear            230,657
  Vanilla-Transformer 203,137
"""

from __future__ import annotations
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Shared building block
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Dense residual block with LayerNorm (used in TiDE encoder)."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x + self.net(x))


# ─────────────────────────────────────────────────────────────────────────────
# TiDE — Time-series Dense Encoder
# ─────────────────────────────────────────────────────────────────────────────

class TiDE(nn.Module):
    """
    Deep residual MLP (TiDE) for window-level binary classification.

    Architecture (Section 3.1.3):
      Input  → Feature projection → N residual blocks → Sigmoid classifier
    """
    def __init__(
        self,
        window_len: int,
        n_features: int,
        hidden_dim: int  = 256,
        n_blocks:   int  = 2,
        dropout:    float = 0.1,
    ):
        super().__init__()
        in_dim = window_len * n_features
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.res  = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x.flatten(1))           # (B, hidden)
        h = self.res(h)
        return self.head(h).squeeze(-1)        # (B,) raw logit


# ─────────────────────────────────────────────────────────────────────────────
# 1D-CNN
# ─────────────────────────────────────────────────────────────────────────────

class CNN1D(nn.Module):
    """
    Two 1D-convolutional layers with max-pooling (Section 3.1.3).
    """
    def __init__(
        self,
        window_len: int,
        n_features: int,
        n_filters: int   = 64,
        kernel_size: int = 3,
        dropout: float   = 0.1,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, n_filters, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(n_filters, n_filters, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
        )
        # Compute flat dimension with a dummy forward pass
        dummy = torch.zeros(1, n_features, window_len)
        flat_dim = self.conv(dummy).shape[-1]
        self.head = nn.Linear(flat_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.permute(0, 2, 1))    # (B, F, L) → flatten
        return self.head(h).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# LSTM
# ─────────────────────────────────────────────────────────────────────────────

class LSTMBackbone(nn.Module):
    """
    Two stacked LSTM layers; final hidden state → linear classifier.
    """
    def __init__(
        self,
        n_features: int,
        hidden_dim: int  = 128,
        n_layers:   int  = 2,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)              # h: (n_layers, B, hidden)
        return self.head(h[-1]).squeeze(-1)   # last layer hidden state


# ─────────────────────────────────────────────────────────────────────────────
# DLinear — Linear decomposition model
# ─────────────────────────────────────────────────────────────────────────────

class DLinear(nn.Module):
    """
    Splits input into trend (moving average) and seasonal residual, then
    projects both through parallel linear layers and concatenates for
    classification (Section 3.1.3).
    """
    def __init__(
        self,
        window_len:  int,
        n_features:  int,
        out_dim:     int  = 64,
        kernel_size: int  = 25,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.kernel_size  = kernel_size
        self.trend_proj   = nn.Linear(window_len * n_features, out_dim)
        self.season_proj  = nn.Linear(window_len * n_features, out_dim)
        self.dropout      = nn.Dropout(dropout)
        self.head         = nn.Linear(2 * out_dim, 1)

    def _moving_avg(self, x: torch.Tensor) -> torch.Tensor:
        """Compute moving-average trend component."""
        B, L, F = x.shape
        xp = nn.functional.avg_pool1d(
            x.permute(0, 2, 1),                # (B, F, L)
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        )
        return xp[..., :L].permute(0, 2, 1)   # back to (B, L, F)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend  = self._moving_avg(x)
        season = x - trend
        t = self.dropout(self.trend_proj(trend.flatten(1)))
        s = self.dropout(self.season_proj(season.flatten(1)))
        return self.head(torch.cat([t, s], dim=-1)).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Vanilla-Transformer
# ─────────────────────────────────────────────────────────────────────────────

class VanillaTransformer(nn.Module):
    """
    Single Transformer encoder block with mean-pooling aggregation
    (Section 3.1.3).
    """
    def __init__(
        self,
        n_features: int,
        d_model:    int   = 128,
        n_heads:    int   = 4,
        ff_dim:     int   = 512,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.head    = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.proj(x))        # (B, L, d_model)
        h = h.mean(dim=1)                     # mean-pooling over time
        return self.head(h).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

BACKBONE_NAMES = ["TiDE", "1D-CNN", "LSTM", "DLinear", "Vanilla-Transformer"]


def build_backbone(
    name: str,
    window_len: int,
    n_features: int,
    cfg: dict | None = None,
) -> nn.Module:
    """
    Instantiate a backbone by name using hyperparameters from ``cfg``.
    If ``cfg`` is None, paper defaults are used.

    Parameters
    ----------
    name       : one of BACKBONE_NAMES.
    window_len : L (default 50).
    n_features : F (default 36 for Edge-IIoTset).
    cfg        : dict from dyra_iiot.config  (or equivalent).
    """
    from dyra_iiot import config as C   # late import to avoid circular deps

    if name == "TiDE":
        return TiDE(
            window_len, n_features,
            hidden_dim=getattr(cfg, "TIDE_HIDDEN",  C.TIDE_HIDDEN)   if cfg else C.TIDE_HIDDEN,
            n_blocks=  getattr(cfg, "TIDE_N_BLOCKS",C.TIDE_N_BLOCKS) if cfg else C.TIDE_N_BLOCKS,
            dropout=   getattr(cfg, "TIDE_DROPOUT", C.TIDE_DROPOUT)  if cfg else C.TIDE_DROPOUT,
        )
    elif name == "1D-CNN":
        return CNN1D(
            window_len, n_features,
            n_filters=  getattr(cfg, "CNN_FILTERS", C.CNN_FILTERS)  if cfg else C.CNN_FILTERS,
            kernel_size=getattr(cfg, "CNN_KERNEL",  C.CNN_KERNEL)   if cfg else C.CNN_KERNEL,
            dropout=    getattr(cfg, "CNN_DROPOUT", C.CNN_DROPOUT)  if cfg else C.CNN_DROPOUT,
        )
    elif name == "LSTM":
        return LSTMBackbone(
            n_features,
            hidden_dim=getattr(cfg, "LSTM_HIDDEN",  C.LSTM_HIDDEN)  if cfg else C.LSTM_HIDDEN,
            n_layers=  getattr(cfg, "LSTM_LAYERS",  C.LSTM_LAYERS)  if cfg else C.LSTM_LAYERS,
            dropout=   getattr(cfg, "LSTM_DROPOUT", C.LSTM_DROPOUT) if cfg else C.LSTM_DROPOUT,
        )
    elif name == "DLinear":
        return DLinear(
            window_len, n_features,
            out_dim=    getattr(cfg, "DLINEAR_DIM",    C.DLINEAR_DIM)    if cfg else C.DLINEAR_DIM,
            kernel_size=getattr(cfg, "DLINEAR_KERNEL", C.DLINEAR_KERNEL) if cfg else C.DLINEAR_KERNEL,
            dropout=    getattr(cfg, "DLINEAR_DROPOUT",C.DLINEAR_DROPOUT)if cfg else C.DLINEAR_DROPOUT,
        )
    elif name == "Vanilla-Transformer":
        return VanillaTransformer(
            n_features,
            d_model= getattr(cfg, "VT_DMODEL",  C.VT_DMODEL)  if cfg else C.VT_DMODEL,
            n_heads= getattr(cfg, "VT_HEADS",   C.VT_HEADS)   if cfg else C.VT_HEADS,
            ff_dim=  getattr(cfg, "VT_FF_DIM",  C.VT_FF_DIM)  if cfg else C.VT_FF_DIM,
            dropout= getattr(cfg, "VT_DROPOUT", C.VT_DROPOUT) if cfg else C.VT_DROPOUT,
        )
    else:
        raise ValueError(f"Unknown backbone: {name!r}. Choose from {BACKBONE_NAMES}.")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
