"""
Dataset loader v2 for TiDE training on DNN-EdgeIIoT-dataset.csv.

Replaces the buggy original dataset_loader.py + train_and_evaluate.py logic
with a methodologically clean pipeline:

1. Loads only the 41 features declared in feature_list.py (no implicit
   inclusion of high-cardinality identifier columns).
2. Splits each class-block independently (stratified 80/20 within class)
   instead of naive global chronological split — necessary because the
   public CSV is concatenated per-class and frame.time is unreliable.
3. Builds sliding windows STRICTLY inside one class-block — no window
   spans a class boundary.
4. Fits StandardScaler on TRAIN windows only, then applies to test.
5. Stores tensors in float16 for memory efficiency (~9 GB peak vs ~18 GB).

Public API:
    build_split_and_windows() -> dict with X_train, y_train, X_test, y_test,
                                 attack_train, attack_test, mean, std, etc.

This module does NOT train anything; training is in train_v2.py.
"""

import os
import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from feature_list import (
    FEATURES,
    BINARY_LABEL_COL,
    MULTICLASS_LABEL_COL,
)


# ============================================================================
# CSV LOADING
# ============================================================================

def find_csv() -> Path:
    """Find DNN-EdgeIIoT-dataset.csv by walking up from project root."""
    for root, _, files in os.walk(cfg.PROJECT_ROOT):
        for f in files:
            if f.lower() == cfg.CSV_NAME.lower():
                return Path(root) / f
    raise FileNotFoundError(
        f"{cfg.CSV_NAME} not found under {cfg.PROJECT_ROOT}"
    )


def safe_numeric(series: pd.Series) -> np.ndarray:
    """Convert column to float32, coercing non-numeric and NaN to 0."""
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(np.float32).values


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    """Load CSV with only the columns we need; convert features to float32."""
    cols_needed = list(set(FEATURES + [BINARY_LABEL_COL, MULTICLASS_LABEL_COL]))
    print(f"  Loading {len(cols_needed)} columns from CSV...")
    t0 = time.time()
    df = pd.read_csv(csv_path, usecols=cols_needed, low_memory=False)
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f} sec")

    # Convert features
    for f in FEATURES:
        df[f] = safe_numeric(df[f])

    # Label
    df[BINARY_LABEL_COL] = (
        pd.to_numeric(df[BINARY_LABEL_COL], errors="coerce")
        .fillna(0).astype(np.int8).values
    )

    # Attack_type stays as string
    df[MULTICLASS_LABEL_COL] = df[MULTICLASS_LABEL_COL].astype(str)

    return df


# ============================================================================
# CLASS-BLOCK SPLIT (stratified per-class chronological-within-block)
# ============================================================================

def find_class_blocks(attack_types: np.ndarray) -> list[tuple[int, int, str]]:
    """
    Find contiguous blocks of identical Attack_type values.

    Since the CSV is concatenated per-class file, blocks are large. Returns
    list of (start_idx, end_idx, class_name) tuples.

    end_idx is EXCLUSIVE.
    """
    blocks = []
    n = len(attack_types)
    if n == 0:
        return blocks
    start = 0
    current_class = attack_types[0]
    for i in range(1, n):
        if attack_types[i] != current_class:
            blocks.append((start, i, current_class))
            start = i
            current_class = attack_types[i]
    blocks.append((start, n, current_class))
    return blocks


def split_block(
    start: int, end: int, train_ratio: float
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Split a single class-block into train and test ranges.

    Within-block order is preserved (not shuffled). The first
    `train_ratio` of the block goes to train, the rest to test.
    This gives us chronological splitting INSIDE each class-block
    (the only level at which the public CSV has meaningful order),
    while ensuring class representation in both splits (stratified).

    Returns ((train_start, train_end), (test_start, test_end)),
    half-open intervals.
    """
    n = end - start
    split_at = start + int(n * train_ratio)
    return (start, split_at), (split_at, end)


def build_class_block_splits(
    attack_types: np.ndarray, train_ratio: float
) -> dict:
    """
    Returns dict with:
        blocks: list of (start, end, class_name)
        train_ranges: list of (start, end) tuples — disjoint train ranges
        test_ranges:  list of (start, end) tuples — disjoint test ranges
        class_counts_train: {class_name: n_packets}
        class_counts_test:  {class_name: n_packets}
    """
    blocks = find_class_blocks(attack_types)
    train_ranges = []
    test_ranges = []
    class_counts_train = {}
    class_counts_test = {}

    for start, end, cls in blocks:
        (tr_s, tr_e), (te_s, te_e) = split_block(start, end, train_ratio)
        train_ranges.append((tr_s, tr_e))
        test_ranges.append((te_s, te_e))
        class_counts_train[cls] = class_counts_train.get(cls, 0) + (tr_e - tr_s)
        class_counts_test[cls] = class_counts_test.get(cls, 0) + (te_e - te_s)

    return {
        "blocks": blocks,
        "train_ranges": train_ranges,
        "test_ranges": test_ranges,
        "class_counts_train": class_counts_train,
        "class_counts_test": class_counts_test,
    }


# ============================================================================
# WINDOW CONSTRUCTION (strictly within class-blocks)
# ============================================================================

def build_windows_from_ranges(
    features_matrix: np.ndarray,
    binary_labels: np.ndarray,
    attack_types: np.ndarray,
    ranges: list[tuple[int, int]],
    window_len: int,
    stride: int,
    dtype: np.dtype = np.float16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding windows inside each range.

    A window at position end (inclusive) covers rows [end-L+1 .. end].
    Both endpoints must lie inside the SAME range — no cross-range windows.

    Returns:
        X: (N, L, F) array of dtype
        y: (N,) int8 — binary label of the LAST packet in window
        attack: (N,) object array — Attack_type of the LAST packet in window
    """
    L = window_len
    n_features = features_matrix.shape[1]

    # Pre-compute total number of windows
    total_windows = 0
    for start, end in ranges:
        n_in_range = end - start
        if n_in_range < L:
            continue
        # last window's end index is (end - 1); first is (start + L - 1)
        # number of windows = floor((n_in_range - L) / stride) + 1
        total_windows += (n_in_range - L) // stride + 1

    print(f"    Preallocating tensor of shape ({total_windows}, {L}, {n_features}) "
          f"dtype={dtype}...")

    X = np.empty((total_windows, L, n_features), dtype=dtype)
    y = np.empty(total_windows, dtype=np.int8)
    attack = np.empty(total_windows, dtype=object)

    w_idx = 0
    skipped_short = 0
    for start, end in ranges:
        n_in_range = end - start
        if n_in_range < L:
            skipped_short += 1
            continue
        # Window end positions inside this range (inclusive)
        first_end = start + L - 1
        last_end = end - 1
        for win_end in range(first_end, last_end + 1, stride):
            win_start = win_end - L + 1
            X[w_idx] = features_matrix[win_start:win_end + 1].astype(dtype, copy=False)
            y[w_idx] = binary_labels[win_end]
            attack[w_idx] = attack_types[win_end]
            w_idx += 1

    assert w_idx == total_windows, f"Window count mismatch: {w_idx} vs {total_windows}"
    return X, y, attack, skipped_short


# ============================================================================
# NORMALIZATION
# ============================================================================

def fit_normalization(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-feature mean and std on TRAIN windows only.

    Uses chunked accumulation in float64 for numerical stability, without
    materializing a full fp64 copy of the data (which would require ~24 GB
    for 1.8M windows × 50 timesteps × 36 features).

    Algorithm: streaming sum and sum-of-squares per feature, then
        mean = sum / N
        std  = sqrt(sum_sq / N - mean^2)

    Returns (mean, std) as float32 arrays of shape (n_features,).
    """
    print("    Computing train mean/std (chunked, memory-efficient)...")
    n_features = X_train.shape[2]
    # Flat view: (N*L, F) — this is a VIEW, no copy
    flat = X_train.reshape(-1, n_features)
    n_rows = flat.shape[0]

    # Chunk size — 1M rows × 36 features × 8 bytes = ~290 MB per chunk in fp64
    chunk_size = 1_000_000

    sum_per_feature = np.zeros(n_features, dtype=np.float64)
    sum_sq_per_feature = np.zeros(n_features, dtype=np.float64)
    total_n = 0

    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        chunk = flat[start:end].astype(np.float64)  # ~290 MB temp
        sum_per_feature += chunk.sum(axis=0)
        sum_sq_per_feature += (chunk ** 2).sum(axis=0)
        total_n += chunk.shape[0]
        del chunk

    mean = (sum_per_feature / total_n).astype(np.float32)
    # var = E[X^2] - (E[X])^2; clip to >=0 to avoid sqrt(negative) from numerical error
    variance = sum_sq_per_feature / total_n - (sum_per_feature / total_n) ** 2
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance).astype(np.float32)

    # Constant features: replace std=0 with 1.0 to avoid NaN on normalization
    n_constant = (std < 1e-6).sum()
    if n_constant > 0:
        print(f"    [INFO] {n_constant} feature(s) have near-zero std "
              f"(constant in train); std set to 1.0 for them")
    std[std < 1e-6] = 1.0
    return mean, std


def apply_normalization(
    X: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """
    Apply (X - mean) / std in place.

    Works for float16 or float32 X. Broadcasting handles (N, L, F) - (F,).
    """
    # In-place to save memory; cast mean/std to X dtype implicitly
    X -= mean.astype(X.dtype)
    X /= std.astype(X.dtype)
    return X


# ============================================================================
# CLASS WEIGHTS (sklearn 'balanced' formula)
# ============================================================================

def compute_class_weights_balanced(y: np.ndarray) -> dict:
    """
    Sklearn 'balanced' formula:
        w_class = N_total / (N_classes * N_class)

    Returns dict {class_label: weight} for use in weighted BCE.
    """
    from collections import Counter
    counts = Counter(y.tolist())
    n_total = len(y)
    n_classes = len(counts)
    return {cls: n_total / (n_classes * cnt) for cls, cnt in counts.items()}


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def build_split_and_windows(verbose: bool = True) -> dict:
    """
    Full pipeline: load CSV -> class-block split -> windows -> normalize.

    Returns dict with all artifacts needed for training:
        X_train, y_train, attack_train,
        X_test, y_test, attack_test,
        mean, std,
        class_weights,
        split_stats (for logging and paper)
    """
    cfg.make_dirs()
    t_start = time.time()

    if verbose:
        print(cfg.summary())
        print()

    # ---------- 1. Load CSV ----------
    print("[1/5] Loading CSV...")
    csv_path = find_csv()
    print(f"  Path: {csv_path}")
    df = load_dataframe(csv_path)
    print()

    # ---------- 2. Extract numpy arrays once ----------
    print("[2/5] Extracting feature matrix and labels...")
    features_matrix = df[FEATURES].to_numpy(dtype=np.float32, copy=False)
    binary_labels = df[BINARY_LABEL_COL].to_numpy(dtype=np.int8, copy=False)
    attack_types = df[MULTICLASS_LABEL_COL].to_numpy(dtype=object, copy=False)
    n_packets = len(features_matrix)
    print(f"  Feature matrix: {features_matrix.shape} dtype={features_matrix.dtype}")
    print(f"  Memory: {features_matrix.nbytes / 1e6:.0f} MB")
    print()

    # ---------- 3. Class-block stratified split ----------
    print("[3/5] Stratified per-class-block split (80/20 within each block)...")
    split_info = build_class_block_splits(attack_types, cfg.TRAIN_RATIO)
    n_blocks = len(split_info["blocks"])
    print(f"  Class-blocks found: {n_blocks}")
    print(f"\n  Class-block breakdown:")
    print(f"  {'#':<4} {'Range':<25} {'Class':<25} {'Train':>10} {'Test':>10}")
    for i, ((start, end, cls), (tr_s, tr_e), (te_s, te_e)) in enumerate(
        zip(split_info["blocks"], split_info["train_ranges"], split_info["test_ranges"])
    ):
        print(f"  {i:<4} [{start:>9,}:{end:>9,}] {cls:<25} "
              f"{tr_e-tr_s:>10,} {te_e-te_s:>10,}")

    print(f"\n  Per-class totals after split:")
    print(f"  {'Class':<25} {'Train pkts':>12} {'Test pkts':>12} {'Test share':>12}")
    for cls in sorted(set(split_info["class_counts_train"].keys()) |
                      set(split_info["class_counts_test"].keys())):
        tr = split_info["class_counts_train"].get(cls, 0)
        te = split_info["class_counts_test"].get(cls, 0)
        share = te / (tr + te) if (tr + te) > 0 else 0
        print(f"  {cls:<25} {tr:>12,} {te:>12,} {share*100:>11.2f}%")

    n_train_pkts = sum(e - s for s, e in split_info["train_ranges"])
    n_test_pkts = sum(e - s for s, e in split_info["test_ranges"])
    print(f"\n  Total: train={n_train_pkts:,} ({n_train_pkts/n_packets*100:.2f}%) | "
          f"test={n_test_pkts:,} ({n_test_pkts/n_packets*100:.2f}%)")
    print()

    # ---------- 4. Build windows (separately for train and test) ----------
    print(f"[4/6] Building windows (L={cfg.WINDOW_LEN}, stride={cfg.WINDOW_STRIDE})...")
    # IMPORTANT: Build in float32 first, normalize, THEN cast to fp16.
    # Casting raw features to fp16 before normalization causes overflow
    # (e.g. tcp.checksum max=65535 > float16 max=65504 → inf → NaN cascade).

    print("  Train windows:")
    t0 = time.time()
    X_train, y_train, attack_train, skipped_train = build_windows_from_ranges(
        features_matrix, binary_labels, attack_types,
        split_info["train_ranges"],
        cfg.WINDOW_LEN, cfg.WINDOW_STRIDE, dtype=np.float32,  # always float32 here
    )
    print(f"    Built {len(X_train):,} windows in {time.time()-t0:.1f} sec "
          f"(skipped {skipped_train} blocks < L={cfg.WINDOW_LEN})")
    print(f"    Memory: {X_train.nbytes / 1e6:.0f} MB")

    print("  Test windows:")
    t0 = time.time()
    X_test, y_test, attack_test, skipped_test = build_windows_from_ranges(
        features_matrix, binary_labels, attack_types,
        split_info["test_ranges"],
        cfg.WINDOW_LEN, cfg.WINDOW_STRIDE, dtype=np.float32,  # always float32 here
    )
    print(f"    Built {len(X_test):,} windows in {time.time()-t0:.1f} sec "
          f"(skipped {skipped_test} blocks < L={cfg.WINDOW_LEN})")
    print(f"    Memory: {X_test.nbytes / 1e6:.0f} MB")
    print()

    # Free the original feature matrix — we already have windows
    del features_matrix

    # ---------- 5. Fit and apply normalization on TRAIN only (in float32) ----------
    print("[5/6] Normalization (fit on TRAIN, apply to both)...")
    mean, std = fit_normalization(X_train)

    print("  Applying to train...")
    X_train = apply_normalization(X_train, mean, std)
    print("  Applying to test...")
    X_test = apply_normalization(X_test, mean, std)
    print(f"  Mean range: [{mean.min():.3g}, {mean.max():.3g}]")
    print(f"  Std  range: [{std.min():.3g}, {std.max():.3g}]")
    print(f"  Train post-norm: mean={X_train.mean():.3g}, std={X_train.std():.3g}")
    print(f"  Test  post-norm: mean={X_test.mean():.3g}, std={X_test.std():.3g}")

    # ---------- 6. Downcast to float16 AFTER normalization ----------
    use_fp16 = (cfg.WINDOW_STRATEGY == "in_memory_fp16")
    if use_fp16:
        print("\n[6/6] Downcasting to float16 (post-normalization — safe, values ≈ ±5)...")
        X_train = X_train.astype(np.float16, copy=False)
        X_test = X_test.astype(np.float16, copy=False)
        print(f"  Train memory after fp16: {X_train.nbytes / 1e6:.0f} MB")
        print(f"  Test  memory after fp16: {X_test.nbytes / 1e6:.0f} MB")
        # Sanity check: no inf/nan after cast (normalized values should be ±5 max)
        assert np.isfinite(X_train).all(), "NaN/Inf in X_train after fp16 cast!"
        assert np.isfinite(X_test).all(), "NaN/Inf in X_test after fp16 cast!"
        print("  Sanity check: no inf/nan — OK")
    else:
        print("\n[6/6] Keeping float32 (no downcast).")
    print()

    # ---------- Class weights from train ----------
    class_weights = compute_class_weights_balanced(y_train)
    print(f"Class weights (balanced, computed on train):")
    for cls, w in sorted(class_weights.items()):
        n = int((y_train == cls).sum())
        print(f"  class {cls}: weight={w:.4f}  (n={n:,})")
    print()

    # Diagnostic stats for paper / Section 4.1
    split_stats = {
        "n_packets_total": int(n_packets),
        "n_packets_train": int(n_train_pkts),
        "n_packets_test": int(n_test_pkts),
        "n_windows_train": int(len(X_train)),
        "n_windows_test": int(len(X_test)),
        "n_blocks": int(n_blocks),
        "train_ratio": cfg.TRAIN_RATIO,
        "window_len": cfg.WINDOW_LEN,
        "window_stride": cfg.WINDOW_STRIDE,
        "n_features": cfg.NUM_FEATURES,
        "binary_balance_train": {
            "normal": int((y_train == 0).sum()),
            "attack": int((y_train == 1).sum()),
        },
        "binary_balance_test": {
            "normal": int((y_test == 0).sum()),
            "attack": int((y_test == 1).sum()),
        },
        "class_counts_train": split_info["class_counts_train"],
        "class_counts_test": split_info["class_counts_test"],
        "class_weights": class_weights,
    }
    stats_path = cfg.METRICS_DIR / "split_stats.json"
    with open(stats_path, "w") as f:
        # Convert int keys to str for JSON
        cw_str = {str(k): v for k, v in class_weights.items()}
        stats_serializable = {**split_stats, "class_weights": cw_str}
        json.dump(stats_serializable, f, indent=2)
    print(f"Split stats written to: {stats_path}")
    print()
    print(f"Total time: {time.time()-t_start:.1f} sec")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "attack_train": attack_train,
        "X_test": X_test,
        "y_test": y_test,
        "attack_test": attack_test,
        "mean": mean,
        "std": std,
        "class_weights": class_weights,
        "split_stats": split_stats,
    }


def save_windows(artifacts: dict, out_dir: Path = None) -> None:
    """Persist windows to disk for reuse by training scripts."""
    if out_dir is None:
        out_dir = cfg.WINDOWS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "train.npz",
        X=artifacts["X_train"],
        y=artifacts["y_train"],
        attack=artifacts["attack_train"],
    )
    np.savez_compressed(
        out_dir / "test.npz",
        X=artifacts["X_test"],
        y=artifacts["y_test"],
        attack=artifacts["attack_test"],
    )
    np.savez(
        out_dir / "norm_stats.npz",
        mean=artifacts["mean"],
        std=artifacts["std"],
        features=np.array(FEATURES),
    )
    print(f"Windows saved to: {out_dir}")


if __name__ == "__main__":
    artifacts = build_split_and_windows(verbose=True)
    save_windows(artifacts)
