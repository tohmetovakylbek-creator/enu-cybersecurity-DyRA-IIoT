"""
ablation_leakage.py — Quantitative decomposition of F1 inflation from
three independently controllable leakage mechanisms identified in the
preliminary draft of this paper. Produces Table 19 of Section 4.8.1.

This script runs FOUR pipeline configurations of the TiDE backbone,
each on 3 random seeds (42, 123, 456):

    (revised)   stratified per-class-block + train-only norm + 36 features
    (a)         random_split   + train-only norm + 36 features
    (b)         stratified     + global norm    + 36 features
    (c)         stratified     + train-only norm + 46 features (incl. identifiers)
    (preliminary) random_split + global norm    + 46 features (all three)

Each run trains TiDE for the standard 10 epochs and reports test F1.
Output: artifacts/metrics/ablation_leakage_results.json

USAGE:
    python ablation_leakage.py                # all 4 variants × 3 seeds
    python ablation_leakage.py --variant a    # only variant (a)
    python ablation_leakage.py --variant c --seed 42  # one specific run

NOTE: This script BORROWS the pipeline from dataset_loader_v2.py but
applies controlled modifications per variant. The (revised) baseline
result is loaded from existing seed_NN.json files (no retraining).
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

import config as cfg
from feature_list import (
    FEATURES as FEATURES_36,
    BINARY_LABEL_COL,
    MULTICLASS_LABEL_COL,
    DROPPED_FEATURES,
)
from dataset_loader_v2 import (
    find_csv, load_dataframe, find_class_blocks,
    apply_normalization, build_windows_from_ranges,
)
from tide_model import TiDEAnomalyDetector

# ============================================================================
# Configuration
# ============================================================================

SEEDS_FOR_ABLATION = [42, 123, 456]  # 3 seeds is enough to detect the effect

# 46-feature schema = 36 baseline + 5 dropped + 5 high-cardinality identifiers
# that the preliminary draft included. We restore the 5 identifier-like
# features that were excluded in Stage 2 of Section 4.1.2.
IDENTIFIER_LIKE = [
    "tcp.options", "tcp.payload", "http.request.full_uri",
    "http.request.method",        # already in 36? — feature_list.py: yes, kept
    # We restore only those that aren't already in FEATURES_36:
]
# Build the 46-feature schema for variant (c): 36 baseline + 5 dropped + 5 identifiers
FEATURES_46_RESTORE = DROPPED_FEATURES + [
    "tcp.options", "tcp.payload", "http.request.full_uri",
    "tcp.dstport", "tcp.srcport",  # were also excluded in original
]
FEATURES_46 = FEATURES_36 + [f for f in FEATURES_46_RESTORE
                              if f not in FEATURES_36]


# ============================================================================
# Variant-specific pipeline modifications
# ============================================================================

def make_random_split_ranges(n_packets, train_ratio=0.80, seed=42):
    """
    Variant (a): RANDOM window split — instead of class-block-stratified,
    we randomly assign each PACKET to train/test. This is the LEAKY version:
    overlapping windows that cross the train/test boundary share packets.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(n_packets)
    rng.shuffle(indices)
    split_at = int(n_packets * train_ratio)
    train_idx = np.sort(indices[:split_at])
    test_idx = np.sort(indices[split_at:])
    return train_idx, test_idx


def fit_normalization_global(X_train, X_test):
    """
    Variant (b): GLOBAL normalization — mean/std fitted on ENTIRE dataset
    (train + test), not on train only. Leaky.

    Chunked streaming version that does NOT concatenate X_train and X_test
    in memory (which would require ~19 GB for the 46-feature schema).
    Processes each tensor's chunks in turn against shared accumulators.
    """
    n_features = X_train.shape[2]
    chunk_size = 1_000_000

    sum_pf = np.zeros(n_features, dtype=np.float64)
    sum_sq_pf = np.zeros(n_features, dtype=np.float64)
    total_n = 0

    for X in (X_train, X_test):
        flat = X.reshape(-1, n_features)
        for start in range(0, flat.shape[0], chunk_size):
            end = min(start + chunk_size, flat.shape[0])
            chunk = flat[start:end].astype(np.float64)
            sum_pf += chunk.sum(axis=0)
            sum_sq_pf += (chunk ** 2).sum(axis=0)
            total_n += chunk.shape[0]
            del chunk

    mean = (sum_pf / total_n).astype(np.float32)
    var = sum_sq_pf / total_n - (sum_pf / total_n) ** 2
    std = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def safe_label_encode(series):
    """Label-encode a string column to integer codes."""
    if series.dtype == object:
        codes, _ = pd.factorize(series.fillna("__NA__").astype(str))
        return codes.astype(np.float32)
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(np.float32)


def load_dataframe_with_schema(csv_path, schema_features):
    """Load CSV with the specified feature schema (36 or 46 features)."""
    cols_needed = schema_features + [BINARY_LABEL_COL, MULTICLASS_LABEL_COL]
    print(f"  Loading {len(schema_features)} features + 2 label columns...")
    df = pd.read_csv(csv_path, usecols=cols_needed, low_memory=False)
    print(f"  Loaded {len(df):,} rows")
    # Label-encode any string columns
    for f in schema_features:
        df[f] = safe_label_encode(df[f])
    return df


# ============================================================================
# Variant runners (each returns final test F1)
# ============================================================================

def build_windows_variant(variant, seed):
    """Build train/test windows according to the variant."""
    csv_path = find_csv()

    # Select feature schema
    if variant == "c" or variant == "preliminary":
        schema = FEATURES_46
        n_features_label = "46 (incl. identifiers)"
    else:
        schema = FEATURES_36
        n_features_label = "36"
    print(f"  Schema: {n_features_label} features")

    df = load_dataframe_with_schema(csv_path, schema)
    feats = df[schema].to_numpy(dtype=np.float32)
    labels = df[BINARY_LABEL_COL].to_numpy(dtype=np.int8)
    attacks = df[MULTICLASS_LABEL_COL].to_numpy(dtype=object)

    # Select split type
    if variant == "a" or variant == "preliminary":
        # RANDOM split (LEAKY)
        print(f"  Split: RANDOM 80/20 (LEAKY — variant {variant})")
        train_idx, test_idx = make_random_split_ranges(
            len(df), cfg.TRAIN_RATIO, seed=seed
        )
        # Build windows from index sets
        # Each window of length L=50 around index i
        L = cfg.WINDOW_LEN
        train_windows = []
        train_labels = []
        train_attacks = []
        for i in train_idx:
            if i >= L - 1 and i < len(df):
                train_windows.append(feats[i - L + 1 : i + 1])
                train_labels.append(labels[i])
                train_attacks.append(attacks[i])
        test_windows = []
        test_labels = []
        test_attacks = []
        for i in test_idx:
            if i >= L - 1 and i < len(df):
                test_windows.append(feats[i - L + 1 : i + 1])
                test_labels.append(labels[i])
                test_attacks.append(attacks[i])
        X_train = np.stack(train_windows).astype(np.float32)
        y_train = np.array(train_labels, dtype=np.float32)
        X_test = np.stack(test_windows).astype(np.float32)
        y_test = np.array(test_labels, dtype=np.float32)
    else:
        # Stratified per-class-block (revised, b, c)
        print(f"  Split: stratified per-class-block")
        blocks = find_class_blocks(attacks)
        train_ranges = []
        test_ranges = []
        for start, end, _ in blocks:
            split_at = start + int((end - start) * cfg.TRAIN_RATIO)
            train_ranges.append((start, split_at))
            test_ranges.append((split_at, end))
        X_train, y_train, _, _ = build_windows_from_ranges(
            feats, labels, attacks, train_ranges,
            cfg.WINDOW_LEN, cfg.WINDOW_STRIDE, dtype=np.float32,
        )
        X_test, y_test, _, _ = build_windows_from_ranges(
            feats, labels, attacks, test_ranges,
            cfg.WINDOW_LEN, cfg.WINDOW_STRIDE, dtype=np.float32,
        )

    # Normalization
    if variant == "b" or variant == "preliminary":
        # GLOBAL normalization (LEAKY)
        print(f"  Normalization: GLOBAL (LEAKY — variant {variant})")
        mean, std = fit_normalization_global(X_train, X_test)
    else:
        # Train-only normalization (revised, a, c)
        print(f"  Normalization: train-only")
        from dataset_loader_v2 import fit_normalization
        mean, std = fit_normalization(X_train)

    X_train = apply_normalization(X_train, mean, std)
    X_test = apply_normalization(X_test, mean, std)
    return X_train, y_train, X_test, y_test, schema


# ============================================================================
# Training (single seed, single variant)
# ============================================================================

def train_one_seed(variant, seed):
    print(f"\n{'='*70}")
    print(f"ABLATION: variant={variant}  seed={seed}")
    print(f"{'='*70}")

    X_train, y_train, X_test, y_test, schema = build_windows_variant(variant, seed)
    n_features = X_train.shape[2]
    print(f"  Train windows: {len(X_train):,}  Test windows: {len(X_test):,}")
    print(f"  Num features: {n_features}")

    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")

    model = TiDEAnomalyDetector(
        seq_len=cfg.WINDOW_LEN,
        num_features=n_features,
        hidden_dim=cfg.MODEL_HIDDEN_DIM,
        num_layers=cfg.MODEL_NUM_RESBLOCKS,
        dropout=cfg.MODEL_DROPOUT,
    ).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_ds = TensorDataset(torch.from_numpy(X_train),
                              torch.from_numpy(y_train.astype(np.float32)))
    test_ds = TensorDataset(torch.from_numpy(X_test),
                             torch.from_numpy(y_test.astype(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                               drop_last=True, num_workers=0, pin_memory=cfg.PIN_MEMORY)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                              drop_last=False, num_workers=0, pin_memory=cfg.PIN_MEMORY)

    n_pos = (y_train == 1).sum()
    pos_weight = (len(y_train) - n_pos) / max(n_pos, 1)
    base_loss = nn.BCELoss(reduction='none')
    def loss_fn(pred, target):
        raw = base_loss(pred, target)
        weights = torch.where(target == 1.0, pos_weight, 1.0)
        return (raw * weights).mean()

    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE,
                            weight_decay=cfg.WEIGHT_DECAY)

    best_f1 = -1.0
    for epoch in range(cfg.EPOCHS):
        t0 = time.time()
        model.train()
        total_loss = 0.0; n = 0
        for X, y in train_loader:
            X = X.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward(); optimizer.step()
            total_loss += loss.item(); n += 1
        # Eval
        model.eval()
        probs_all, targets_all = [], []
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                pred = model(X)
                probs_all.append(pred.cpu().numpy())
                targets_all.append(y.cpu().numpy())
        probs = np.concatenate(probs_all); targets = np.concatenate(targets_all)
        preds = (probs > cfg.DECISION_THRESHOLD).astype(np.float32)
        f1 = float(f1_score(targets, preds, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
        print(f"  Epoch {epoch+1:>2}/{cfg.EPOCHS} | F1={f1:.4f} | {time.time()-t0:.0f}s")

    print(f"  BEST F1: {best_f1:.4f}")
    return best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant",
        choices=["a", "b", "c", "preliminary", "all"], default="all")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg.make_dirs()
    print(cfg.summary())

    variants = (["a", "b", "c", "preliminary"]
                if args.variant == "all" else [args.variant])
    seeds = [args.seed] if args.seed is not None else SEEDS_FOR_ABLATION

    results = {}  # {variant: [f1_per_seed]}
    for variant in variants:
        results[variant] = []
        for seed in seeds:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            f1 = train_one_seed(variant, seed)
            results[variant].append(f1)

    # Aggregate
    print(f"\n{'='*70}")
    print(f"ABLATION SUMMARY")
    print(f"{'='*70}\n")
    print(f"{'Variant':<15} {'Seeds':<8} {'F1 mean':<12} {'F1 std':<12} F1 values")
    print(f"{'-'*70}")

    # Pull the revised baseline from existing seed_NN.json
    baseline_f1 = []
    for seed in seeds:
        p = cfg.METRICS_DIR / f"seed_{seed}.json"
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            baseline_f1.append(d["final_metrics"]["f1"])
    if baseline_f1:
        print(f"{'revised':<15} {len(baseline_f1):<8} "
              f"{np.mean(baseline_f1):<12.4f} {np.std(baseline_f1):<12.4f} "
              f"{baseline_f1}")

    for variant in variants:
        vals = results[variant]
        if vals:
            print(f"{variant:<15} {len(vals):<8} {np.mean(vals):<12.4f} "
                  f"{np.std(vals):<12.4f} {vals}")

    # Save
    out = {
        "seeds_used": seeds,
        "revised_baseline_f1": baseline_f1,
        "variants": {
            v: {"f1_values": results[v],
                "f1_mean": float(np.mean(results[v])) if results[v] else None,
                "f1_std": float(np.std(results[v])) if results[v] else None}
            for v in variants
        },
    }
    out_path = cfg.METRICS_DIR / "ablation_leakage_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
