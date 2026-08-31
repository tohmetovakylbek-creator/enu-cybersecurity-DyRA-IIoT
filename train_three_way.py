"""
train_three_way.py — Three-way split (train / val / test) with TRUE
held-out validation set for best-epoch checkpoint selection.

This addresses the methodological concern raised by reviewers:
the main pipeline (train_v2.py + baselines_v2.py) uses test-set F1
for checkpoint selection, which is a mild form of test-set leakage
into model selection. This script implements an honest alternative:

PROTOCOL:
    Within each class-block, split CHRONOLOGICALLY:
        - First 60% of packets  -> TRAIN (model fitting)
        - Next 20% of packets   -> VAL   (best-epoch selection)
        - Last 20% of packets   -> TEST  (final reporting, untouched until end)

    This is the same stratified-per-class-block logic as the main
    pipeline, refined into three partitions per block instead of two.

OUTPUT:
    artifacts/metrics/threeway_{model}_seed{N}.json
    artifacts/metrics/threeway_aggregate.json

USAGE:
    python train_three_way.py                       # all 5 models, all 5 seeds
    python train_three_way.py --model transformer   # one model, all seeds
    python train_three_way.py --model tide --seed 42  # one specific run
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
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
    FEATURES,
    BINARY_LABEL_COL,
    MULTICLASS_LABEL_COL,
)
from dataset_loader_v2 import (
    find_csv, load_dataframe, find_class_blocks,
    fit_normalization, apply_normalization, build_windows_from_ranges,
)
from tide_model import TiDEAnomalyDetector
from baselines_v2 import MODEL_REGISTRY, make_weighted_bce_loss


# ============================================================================
# THREE-WAY SPLIT
# ============================================================================

# Within each class-block: first 60% -> train, next 20% -> val, last 20% -> test
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
# TEST_RATIO = 0.20 (implicit)


def build_three_way_split(df, attack_types):
    """Per-class-block chronological 60/20/20 split."""
    blocks = find_class_blocks(attack_types)
    train_ranges, val_ranges, test_ranges = [], [], []
    for start, end, cls in blocks:
        n = end - start
        train_end = start + int(n * TRAIN_RATIO)
        val_end = start + int(n * (TRAIN_RATIO + VAL_RATIO))
        train_ranges.append((start, train_end))
        val_ranges.append((train_end, val_end))
        test_ranges.append((val_end, end))
    return {
        "train_ranges": train_ranges,
        "val_ranges": val_ranges,
        "test_ranges": test_ranges,
        "blocks": blocks,
    }


def build_three_way_windows():
    """Load CSV, three-way split, build windows, train-only normalization."""
    print("[3WAY] Loading CSV...")
    csv_path = find_csv()
    df = load_dataframe(csv_path)

    feats = df[FEATURES].to_numpy(dtype=np.float32, copy=False)
    labels = df[BINARY_LABEL_COL].to_numpy(dtype=np.int8, copy=False)
    attacks = df[MULTICLASS_LABEL_COL].to_numpy(dtype=object, copy=False)

    print("[3WAY] Building 60/20/20 split per class-block...")
    split_info = build_three_way_split(df, attacks)
    n_train_pkts = sum(e - s for s, e in split_info["train_ranges"])
    n_val_pkts = sum(e - s for s, e in split_info["val_ranges"])
    n_test_pkts = sum(e - s for s, e in split_info["test_ranges"])
    print(f"  Train packets: {n_train_pkts:,}")
    print(f"  Val packets:   {n_val_pkts:,}")
    print(f"  Test packets:  {n_test_pkts:,}")

    print("\n[3WAY] Building windows (each partition strictly within blocks)...")
    X_train, y_train, attack_train, _ = build_windows_from_ranges(
        feats, labels, attacks, split_info["train_ranges"],
        cfg.WINDOW_LEN, cfg.WINDOW_STRIDE, dtype=np.float32,
    )
    X_val, y_val, attack_val, _ = build_windows_from_ranges(
        feats, labels, attacks, split_info["val_ranges"],
        cfg.WINDOW_LEN, cfg.WINDOW_STRIDE, dtype=np.float32,
    )
    X_test, y_test, attack_test, _ = build_windows_from_ranges(
        feats, labels, attacks, split_info["test_ranges"],
        cfg.WINDOW_LEN, cfg.WINDOW_STRIDE, dtype=np.float32,
    )
    print(f"  Train windows: {len(X_train):,}")
    print(f"  Val windows:   {len(X_val):,}")
    print(f"  Test windows:  {len(X_test):,}")
    del feats

    print("\n[3WAY] Normalization (train-only stats)...")
    mean, std = fit_normalization(X_train)
    X_train = apply_normalization(X_train, mean, std)
    X_val = apply_normalization(X_val, mean, std)
    X_test = apply_normalization(X_test, mean, std)

    return {
        "X_train": X_train, "y_train": y_train, "attack_train": attack_train,
        "X_val": X_val, "y_val": y_val, "attack_val": attack_val,
        "X_test": X_test, "y_test": y_test, "attack_test": attack_test,
    }


# ============================================================================
# TRAINING (best-epoch selection on VAL, final reporting on TEST)
# ============================================================================

def make_model(model_name):
    if model_name == "tide":
        return TiDEAnomalyDetector(
            seq_len=cfg.WINDOW_LEN, num_features=cfg.NUM_FEATURES,
            hidden_dim=cfg.MODEL_HIDDEN_DIM,
            num_layers=cfg.MODEL_NUM_RESBLOCKS,
            dropout=cfg.MODEL_DROPOUT,
        )
    return MODEL_REGISTRY[model_name](
        seq_len=cfg.WINDOW_LEN, num_features=cfg.NUM_FEATURES,
    )


def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test):
    def _loader(X, y, shuffle, drop_last):
        ds = TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y.astype(np.float32)),
        )
        return DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=shuffle,
                          drop_last=drop_last, num_workers=0,
                          pin_memory=cfg.PIN_MEMORY)
    return (
        _loader(X_train, y_train, shuffle=True,  drop_last=True),
        _loader(X_val,   y_val,   shuffle=False, drop_last=False),
        _loader(X_test,  y_test,  shuffle=False, drop_last=False),
    )


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total = 0.0; n = 0
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total += loss.item(); n += 1
    return total / n


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total = 0.0; n_b = 0
    probs_all, targets_all = [], []
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(X)
        loss = loss_fn(pred, y)
        total += loss.item(); n_b += 1
        probs_all.append(pred.cpu().numpy())
        targets_all.append(y.cpu().numpy())
    probs = np.concatenate(probs_all)
    targets = np.concatenate(targets_all)
    preds = (probs > cfg.DECISION_THRESHOLD).astype(np.float32)
    cm = confusion_matrix(targets, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "loss": float(total / n_b),
        "accuracy": float(accuracy_score(targets, preds)),
        "precision": float(precision_score(targets, preds, zero_division=0)),
        "recall": float(recall_score(targets, preds, zero_division=0)),
        "f1": float(f1_score(targets, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(targets, probs)) if len(set(targets)) > 1 else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "far": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
    }


def run_one_seed(model_name, seed, data):
    print(f"\n{'='*60}")
    print(f"3WAY: {model_name.upper()}  |  SEED: {seed}")
    print(f"{'='*60}")
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")

    model = make_model(model_name).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    train_loader, val_loader, test_loader = make_loaders(
        data["X_train"], data["y_train"],
        data["X_val"],   data["y_val"],
        data["X_test"],  data["y_test"],
    )
    loss_fn = make_weighted_bce_loss(data["y_train"])
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE,
                            weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=cfg.LR_SCHEDULER_FACTOR,
        patience=cfg.LR_SCHEDULER_PATIENCE,
        min_lr=cfg.LR_SCHEDULER_MIN_LR,
    )

    best_val_f1 = -1.0; best_epoch = -1
    ckpt_path = cfg.CHECKPOINTS_DIR / f"3way_{model_name}_seed{seed}_best.pt"
    epoch_log = []

    for epoch in range(cfg.EPOCHS):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_m = evaluate(model, val_loader, loss_fn, device)
        scheduler.step(val_m["loss"])
        dt = time.time() - t0
        epoch_log.append({
            "epoch": epoch + 1, "train_loss": tr_loss,
            "val_loss": val_m["loss"], "val_f1": val_m["f1"],
            "val_roc_auc": val_m["roc_auc"], "val_far": val_m["far"],
            "lr": optimizer.param_groups[0]["lr"], "time_sec": dt,
        })
        # CHECKPOINT SELECTION ON VAL (not test!)
        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]; best_epoch = epoch + 1
            torch.save(model.state_dict(), ckpt_path)
        print(f"  Epoch {epoch+1:>2}/{cfg.EPOCHS} | tr_loss={tr_loss:.4f} | "
              f"val_loss={val_m['loss']:.4f} | val_F1={val_m['f1']:.4f} | "
              f"val_AUC={val_m['roc_auc']:.4f} | {dt:.1f}s")

    print(f"\n  Best epoch by VAL F1: {best_epoch} (val_F1={best_val_f1:.4f})")
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))

    # FINAL TEST EVALUATION (only now, after model is locked)
    test_m = evaluate(model, test_loader, loss_fn, device)
    print(f"  FINAL TEST F1: {test_m['f1']:.4f} | AUC={test_m['roc_auc']:.4f} | "
          f"FAR={test_m['far']:.4f}")

    result = {
        "model": model_name, "seed": seed,
        "best_epoch_by_val": best_epoch,
        "n_params": n_params,
        "val_metrics_at_best_epoch": {
            "f1": float(best_val_f1),
        },
        "test_metrics_final": test_m,
        "epoch_log": epoch_log,
        "protocol": "three_way_split_60_20_20",
    }
    out_path = cfg.METRICS_DIR / f"threeway_{model_name}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")
    return result


def aggregate(results, model_name):
    print(f"\n{'='*60}")
    print(f"3WAY AGGREGATE: {model_name.upper()}  ({len(results)} seeds)")
    print(f"{'='*60}")
    f1s = [r["test_metrics_final"]["f1"] for r in results]
    aucs = [r["test_metrics_final"]["roc_auc"] for r in results]
    fars = [r["test_metrics_final"]["far"] for r in results]
    precs = [r["test_metrics_final"]["precision"] for r in results]
    recs = [r["test_metrics_final"]["recall"] for r in results]
    print(f"  TEST F1:        {np.mean(f1s):.4f} ± {np.std(f1s):.4f}  "
          f"[{np.min(f1s):.4f}, {np.max(f1s):.4f}]")
    print(f"  TEST ROC-AUC:   {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  TEST Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
    print(f"  TEST Recall:    {np.mean(recs):.4f} ± {np.std(recs):.4f}")
    print(f"  TEST FAR:       {np.mean(fars):.4f} ± {np.std(fars):.4f}")
    summary = {
        "model": model_name, "n_seeds": len(results),
        "test_f1": {"mean": float(np.mean(f1s)), "std": float(np.std(f1s)),
                    "values": f1s},
        "test_roc_auc": {"mean": float(np.mean(aucs)), "std": float(np.std(aucs))},
        "test_precision": {"mean": float(np.mean(precs)), "std": float(np.std(precs))},
        "test_recall": {"mean": float(np.mean(recs)), "std": float(np.std(recs))},
        "test_far": {"mean": float(np.mean(fars)), "std": float(np.std(fars))},
    }
    out = cfg.METRICS_DIR / f"threeway_{model_name}_aggregate.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
        choices=["tide", "cnn", "lstm", "dlinear", "transformer", "all"],
        default="all")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg.make_dirs()
    print(cfg.summary())
    print(f"\nTHREE-WAY SPLIT: {TRAIN_RATIO:.0%} train / {VAL_RATIO:.0%} val / "
          f"{1-TRAIN_RATIO-VAL_RATIO:.0%} test (per class-block)\n")

    print("Building three-way windows (one-time)...")
    data = build_three_way_windows()
    print()

    models = (["tide", "cnn", "lstm", "dlinear", "transformer"]
              if args.model == "all" else [args.model])
    seeds = [args.seed] if args.seed is not None else cfg.SEEDS

    for m in models:
        print(f"\n{'#'*60}\n# 3WAY MODEL: {m.upper()}\n{'#'*60}")
        results = []
        for s in seeds:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            results.append(run_one_seed(m, s, data))
        if len(results) > 1:
            aggregate(results, m)
    print("\nThree-way experiments done.")


if __name__ == "__main__":
    main()
