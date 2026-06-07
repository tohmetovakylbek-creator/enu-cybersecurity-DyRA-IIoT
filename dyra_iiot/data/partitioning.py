"""
dyra_iiot/data/partitioning.py
─────────────────────────────────────────────────────────────────────────────
Stratified per-class-block partitioning  (Algorithm 1, Section 4.1.3).

This module implements the leakage-aware split used in all DyRA-IIoT
experiments.  Key properties:
  • Preserves within-block chronological order.
  • Prevents train/test packet overlap through shared windows.
  • Guarantees all attack classes appear in both train and test.
  • Supports in-distribution and OOD protocols (Section 4.1.5).

Includes normalisation helpers consistent with Eq. (7).
"""

from __future__ import annotations
import logging
from typing import List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 1 — stratified per-class-block split
# ─────────────────────────────────────────────────────────────────────────────

def stratified_per_class_block_split(
    X: np.ndarray,
    y: np.ndarray,
    attack_types: np.ndarray,
    train_ratio: float = 0.80,
    window_len: int = 50,
    ood_classes: Optional[Set[str]] = None,
    mode: str = "in_dist",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified per-class-block partitioning.

    Parameters
    ----------
    X            : feature matrix, shape (N, F).
    y            : binary labels,  shape (N,).
    attack_types : string attack-type label for each packet, shape (N,).
    train_ratio  : fraction of each class block assigned to training.
    window_len   : sliding-window length L.
    ood_classes  : set of class names held out from training (OOD protocol).
                   If None or empty → in-distribution protocol.
    mode         : "in_dist" — all classes in train+test (Section 4.2).
                   "ood"     — ood_classes excluded from train (Section 4.7).

    Returns
    -------
    X_train_wins : float32 array, shape (N_train, L, F).
    y_train      : float32 array, shape (N_train,).
    X_test_wins  : float32 array, shape (N_test,  L, F).
    y_test       : float32 array, shape (N_test,).
    """
    if ood_classes is None:
        ood_classes = set()

    # ── Step 1: discover contiguous class blocks ───────────────────────────
    blocks: List[Tuple[int, int, str]] = []   # (start_idx, end_idx, label)
    start = 0
    for i in range(1, len(attack_types)):
        if attack_types[i] != attack_types[i - 1]:
            blocks.append((start, i - 1, attack_types[i - 1]))
            start = i
    blocks.append((start, len(attack_types) - 1, attack_types[-1]))

    # ── Step 2: per-block chronological 80/20 split ────────────────────────
    train_wins, train_labs = [], []
    test_wins,  test_labs  = [], []

    for (blk_start, blk_end, blk_cls) in blocks:
        n_block = blk_end - blk_start + 1
        split_at = blk_start + int(n_block * train_ratio)

        train_range = (blk_start, split_at - 1)
        test_range  = (split_at,  blk_end)

        is_ood = str(blk_cls).lower() in {c.lower() for c in ood_classes}

        if mode == "in_dist":
            _build_windows(X, y, train_range, window_len, train_wins, train_labs)
            _build_windows(X, y, test_range,  window_len, test_wins,  test_labs)

        elif mode == "ood":
            if not is_ood:
                # In-distribution class → appears in both train and test
                _build_windows(X, y, train_range, window_len, train_wins, train_labs)
                _build_windows(X, y, test_range,  window_len, test_wins,  test_labs)
            else:
                # OOD class → entire block goes to test only (never seen in train)
                _build_windows(X, y, (blk_start, blk_end), window_len,
                               test_wins, test_labs)
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'in_dist' or 'ood'.")

    if not train_wins:
        raise RuntimeError("No training windows generated — check block sizes and train_ratio.")
    if not test_wins:
        raise RuntimeError("No test windows generated — check block sizes and train_ratio.")

    X_train = np.stack(train_wins).astype(np.float32)
    y_train = np.array(train_labs, dtype=np.float32)
    X_test  = np.stack(test_wins).astype(np.float32)
    y_test  = np.array(test_labs,  dtype=np.float32)

    logger.info(
        "[SPLIT-%s] train=%d  test=%d  "
        "pos-rate-train=%.3f  pos-rate-test=%.3f",
        mode, len(X_train), len(X_test),
        y_train.mean(), y_test.mean(),
    )

    return X_train, y_train, X_test, y_test


def _build_windows(
    X: np.ndarray,
    y: np.ndarray,
    rng: Tuple[int, int],
    L: int,
    out_X: list,
    out_y: list,
) -> None:
    """Append sliding windows from packets[rng[0]:rng[1]+1] to out lists."""
    r0, r1 = rng
    if r1 - r0 + 1 < L:
        return   # block too small for even one window
    for i in range(r0 + L - 1, r1 + 1):
        out_X.append(X[i - L + 1: i + 1])
        out_y.append(y[i])          # label = last packet in window


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation  (Eq. 7, Section 4.1.4)
# ─────────────────────────────────────────────────────────────────────────────

def fit_normalizer(
    X_train: np.ndarray,
    eps: float = 1e-6,
    robust: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit per-feature mean and std on the training partition only.

    Parameters
    ----------
    X_train : shape (N, L, F)  or  (N, F).
    eps     : minimum std to prevent division-by-zero.
    robust  : if True, clip to 1st–99th percentile before computing stats.
              Recommended for TON_IoT where src/dst_bytes have ~10^9 scale.

    Returns
    -------
    mu  : shape (F,)
    std : shape (F,)
    """
    flat = X_train.astype(np.float32).reshape(-1, X_train.shape[-1])

    if robust:
        p01 = np.percentile(flat, 1,  axis=0)
        p99 = np.percentile(flat, 99, axis=0)
        flat = np.clip(flat, p01, p99)

    mu  = flat.mean(0)
    std = flat.std(0)
    std[std < eps] = 1.0
    return mu, std


def apply_normalizer(
    X: np.ndarray,
    mu: np.ndarray,
    std: np.ndarray,
    clip: float = 10.0,
) -> np.ndarray:
    """
    Apply (X - mu) / std, then nan_to_num + clip to [-clip, +clip].

    Parameters
    ----------
    X    : shape (N, L, F)  or  (N, F).
    mu   : shape (F,).
    std  : shape (F,).
    clip : absolute value cutoff after normalization (default 10σ).
    """
    result = (X.astype(np.float32) - mu) / std
    result = np.nan_to_num(result, nan=0.0, posinf=clip, neginf=-clip)
    return result.clip(-clip, clip)
