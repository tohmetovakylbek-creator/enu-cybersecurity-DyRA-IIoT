"""
dyra_iiot/training/trainer.py
─────────────────────────────────────────────────────────────────────────────
Training loop and evaluation for DyRA-IIoT backbones (Section 4.1.6).

Key design choices that match the paper:
  • BCEWithLogitsLoss + pos_weight for class imbalance
  • Adam  (lr=5e-4, Section 4.1.6)
  • ReduceLROnPlateau  (factor=0.5, patience=2)
  • Checkpoint selection on test-partition F1
  • Models return raw logits; sigmoid applied at eval time
"""

from __future__ import annotations
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

import dyra_iiot.config as C

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class WindowDataset(Dataset):
    """PyTorch dataset for normalised window tensors."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    ds = WindowDataset(X, y)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _compute_pos_weight(y: np.ndarray) -> float:
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    return n_neg / (n_pos + 1e-8)


def train_backbone(
    model: nn.Module,
    train_loader: DataLoader,
    device: str = "cpu",
    epochs: int = C.EPOCHS,
    lr: float = C.LR,
    lr_patience: int = C.LR_PATIENCE,
    lr_factor: float = C.LR_FACTOR,
    lr_min: float = C.LR_MIN,
    batch_size: int = C.BATCH_SIZE,
    verbose: bool = False,
) -> nn.Module:
    """
    Train one backbone for ``epochs`` epochs.
    Checkpoints the best model by F1 on the training set (consistent
    with the paper; for stricter protocol use a separate validation set).
    """
    y_np   = train_loader.dataset.y.numpy()
    pos_w  = torch.tensor([_compute_pos_weight(y_np)], device=device)
    crit   = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optim  = torch.optim.Adam(model.parameters(), lr=lr)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, factor=lr_factor, patience=lr_patience, min_lr=lr_min,
    )

    best_f1    = -1.0
    best_state: Optional[dict] = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optim.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            optim.step()
            total_loss += loss.item() * len(yb)

        # Evaluate on training set for checkpoint selection
        probs, labels = _get_probs(model, train_loader, device)
        f1 = f1_score(labels, (probs > 0.5).astype(int), zero_division=0)
        sched.step(1 - f1)

        if f1 > best_f1:
            best_f1    = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            logger.debug(
                "  Epoch %02d  loss=%.4f  train-F1=%.4f  lr=%.2e",
                epoch + 1, total_loss / len(train_loader.dataset), f1,
                optim.param_groups[0]["lr"],
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_probs(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return sigmoid(logits) probabilities and ground-truth labels."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            p = torch.sigmoid(model(Xb.to(device))).cpu().numpy()
            all_probs.extend(p)
            all_labels.extend(yb.numpy())
    return np.array(all_probs, dtype=np.float32), np.array(all_labels, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    n_latency_passes: int = 100,
    latency_batch_size: int = 64,
) -> Dict[str, float]:
    """
    Compute all metrics reported in Table 6 / Table 11.

    Returns
    -------
    dict with keys: f1, roc_auc, precision, recall, far, latency_ms
    """
    probs, labels = _get_probs(model, loader, device)
    preds = (probs > 0.5).astype(int)

    cm = confusion_matrix(labels, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")

    far = fp / (fp + tn + 1e-8) * 100.0

    # Latency measurement
    lat_ms = _measure_latency(model, loader, device,
                              n_passes=n_latency_passes,
                              batch_size=latency_batch_size)

    return dict(
        f1          = float(f1_score(labels, preds, zero_division=0)),
        roc_auc     = float(auc),
        precision   = float(precision_score(labels, preds, zero_division=0)),
        recall      = float(recall_score(labels, preds, zero_division=0)),
        far         = float(far),
        latency_ms  = float(lat_ms),
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
    )


def _measure_latency(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    n_passes: int = 100,
    batch_size: int = 64,
) -> float:
    """Mean forward-pass time in ms/batch (batch_size=64)."""
    model.eval()
    batch = next(iter(loader))[0][:batch_size].to(device)

    # Warmup
    for _ in range(10):
        model(batch)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_passes):
        model(batch)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) / n_passes * 1000.0   # ms/batch


# ─────────────────────────────────────────────────────────────────────────────
# Per-class OOD recall  (Table 12)
# ─────────────────────────────────────────────────────────────────────────────

def ood_per_class_recall(
    model: nn.Module,
    X_all: np.ndarray,
    y_all: np.ndarray,
    types_all: np.ndarray,
    ood_classes: set,
    window_len: int,
    device: str,
    mu: np.ndarray,
    std: np.ndarray,
    batch_size: int = 64,
) -> Dict[str, float]:
    """
    Compute per-OOD-class recall (Table 12 / Table 20).

    For each OOD class, builds windows from the FULL class block
    using the same normalisation statistics fitted on the training split.
    """
    from dyra_iiot.data.partitioning import apply_normalizer

    results: Dict[str, float] = {}
    model.eval()

    for cls in np.unique(types_all):
        if str(cls).lower() not in {c.lower() for c in ood_classes}:
            continue

        idx = np.where(types_all == cls)[0]
        if len(idx) < window_len:
            results[str(cls)] = float("nan")
            continue

        wins, labs = [], []
        for i in range(window_len - 1, len(idx)):
            wins.append(X_all[idx[i - window_len + 1: i + 1]])
            labs.append(y_all[idx[i]])

        Xc_norm = apply_normalizer(np.stack(wins), mu, std)
        loader  = make_loader(Xc_norm, np.array(labs), batch_size, shuffle=False)
        probs, labels = _get_probs(model, loader, device)
        results[str(cls)] = float(
            recall_score(labels, (probs > 0.5).astype(int), zero_division=0)
        )

    return results
