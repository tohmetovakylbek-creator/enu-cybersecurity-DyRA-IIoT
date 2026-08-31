"""
ood_test.py — Out-of-distribution (OOD) evaluation for TiDE and baselines.

Training set:  Normal + 10 attack classes (in-distribution attacks)
Test set:      Normal (held out) + 4 attack classes UNSEEN during training

Held-out OOD attacks (chosen for TTP diversity, MITRE ATT&CK aligned):
    - MITM           (Network manipulation)
    - Ransomware     (Malware/persistence)
    - Backdoor       (Malware/persistence)
    - Port_Scanning  (Reconnaissance)

This is the central robustness experiment of the revised paper:
in-distribution F1 ≈ 0.99 for all models, but on OOD attacks we expect
deeper architectures (TiDE, Transformer) to generalize better than
shallow ones (CNN, DLinear) due to richer learned feature representations.

The script reuses class-block split logic from dataset_loader_v2.py.

Usage:
    python ood_test.py --model tide
    python ood_test.py --model cnn --seed 42
    python ood_test.py                          # all 5 models, all 5 seeds
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
    FEATURES,
    BINARY_LABEL_COL,
    MULTICLASS_LABEL_COL,
)
from dataset_loader_v2 import (
    find_csv, load_dataframe, find_class_blocks,
    fit_normalization, apply_normalization, build_windows_from_ranges,
    safe_numeric,
)
from tide_model import TiDEAnomalyDetector
from baselines_v2 import MODEL_REGISTRY, make_weighted_bce_loss


# ============================================================================
# OOD SPLIT DESIGN
# ============================================================================

# Attack classes held out from training (4 classes covering different TTPs)
OOD_HELD_OUT_CLASSES = [
    "MITM",
    "Ransomware",
    "Backdoor",
    "Port_Scanning",
]

# Normal is split chronologically 80/20 (as in main experiment)
NORMAL_CLASS = "Normal"


def build_ood_split(df, attack_types):
    """
    OOD split logic:
    - Normal class: 80% train, 20% test (chronological within block)
    - 11 in-distribution attacks: 100% train (no test contribution)
    - 4 OOD attacks: 100% test (no train contribution)

    Returns dict with train_ranges, test_ranges.
    """
    blocks = find_class_blocks(attack_types)
    train_ranges = []
    test_ranges = []
    in_dist_classes = set()
    ood_classes_seen = set()
    normal_blocks = 0

    for start, end, cls in blocks:
        n = end - start
        if cls in OOD_HELD_OUT_CLASSES:
            # Entire block goes to TEST (model never sees these during training)
            test_ranges.append((start, end))
            ood_classes_seen.add(cls)
        elif cls == NORMAL_CLASS:
            # Normal: 80% train, 20% test (same as main experiment)
            split_at = start + int(n * cfg.TRAIN_RATIO)
            train_ranges.append((start, split_at))
            test_ranges.append((split_at, end))
            normal_blocks += 1
        else:
            # In-distribution attack: entire block to TRAIN
            train_ranges.append((start, end))
            in_dist_classes.add(cls)

    return {
        "train_ranges": train_ranges,
        "test_ranges": test_ranges,
        "in_dist_classes": sorted(in_dist_classes),
        "ood_classes_used": sorted(ood_classes_seen),
        "n_normal_blocks": normal_blocks,
        "blocks": blocks,
    }


# ============================================================================
# DATA PIPELINE
# ============================================================================

def build_ood_windows():
    """
    Full pipeline: load CSV, OOD split, build windows, normalize.
    Returns dict similar to dataset_loader_v2.build_split_and_windows().
    """
    print("[OOD] Loading CSV...")
    csv_path = find_csv()
    df = load_dataframe(csv_path)

    features_matrix = df[FEATURES].to_numpy(dtype=np.float32, copy=False)
    binary_labels = df[BINARY_LABEL_COL].to_numpy(dtype=np.int8, copy=False)
    attack_types = df[MULTICLASS_LABEL_COL].to_numpy(dtype=object, copy=False)

    print(f"[OOD] Building OOD split (4 held-out attacks: {OOD_HELD_OUT_CLASSES})...")
    split_info = build_ood_split(df, attack_types)
    print(f"  In-distribution attacks ({len(split_info['in_dist_classes'])}):")
    for c in split_info["in_dist_classes"]:
        print(f"    - {c}")
    print(f"  OOD held-out attacks ({len(split_info['ood_classes_used'])}):")
    for c in split_info["ood_classes_used"]:
        print(f"    - {c}")
    print(f"  Normal class-blocks: {split_info['n_normal_blocks']}")

    n_train_pkts = sum(e - s for s, e in split_info["train_ranges"])
    n_test_pkts = sum(e - s for s, e in split_info["test_ranges"])
    print(f"\n  Train packets: {n_train_pkts:,}")
    print(f"  Test  packets: {n_test_pkts:,}")

    # Per-class counts
    print("\n  Test set composition by class:")
    for cls in sorted(set(attack_types)):
        count = 0
        for s, e in split_info["test_ranges"]:
            count += int((attack_types[s:e] == cls).sum())
        if count > 0:
            origin = "OOD" if cls in OOD_HELD_OUT_CLASSES else ("Normal" if cls == NORMAL_CLASS else "leaked?")
            print(f"    {cls:<25} {count:>10,}  ({origin})")

    print("\n[OOD] Building windows...")
    X_train, y_train, attack_train, _ = build_windows_from_ranges(
        features_matrix, binary_labels, attack_types,
        split_info["train_ranges"],
        cfg.WINDOW_LEN, cfg.WINDOW_STRIDE, dtype=np.float32,
    )
    X_test, y_test, attack_test, _ = build_windows_from_ranges(
        features_matrix, binary_labels, attack_types,
        split_info["test_ranges"],
        cfg.WINDOW_LEN, cfg.WINDOW_STRIDE, dtype=np.float32,
    )
    print(f"  Train windows: {len(X_train):,}")
    print(f"  Test  windows: {len(X_test):,}")
    del features_matrix

    print("\n[OOD] Normalization (train-only stats)...")
    mean, std = fit_normalization(X_train)
    X_train = apply_normalization(X_train, mean, std)
    X_test = apply_normalization(X_test, mean, std)
    print(f"  Train post-norm: mean={X_train.mean():.3g}, std={X_train.std():.3g}")
    print(f"  Test  post-norm: mean={X_test.mean():.3g}, std={X_test.std():.3g}")

    return {
        "X_train": X_train, "y_train": y_train, "attack_train": attack_train,
        "X_test": X_test, "y_test": y_test, "attack_test": attack_test,
        "split_info": split_info,
        "mean": mean, "std": std,
    }


# ============================================================================
# TRAINING / EVALUATION
# ============================================================================

def make_model(model_name):
    if model_name == "tide":
        return TiDEAnomalyDetector(
            seq_len=cfg.WINDOW_LEN,
            num_features=cfg.NUM_FEATURES,
            hidden_dim=cfg.MODEL_HIDDEN_DIM,
            num_layers=cfg.MODEL_NUM_RESBLOCKS,
            dropout=cfg.MODEL_DROPOUT,
        )
    else:
        ModelClass = MODEL_REGISTRY[model_name]
        return ModelClass(seq_len=cfg.WINDOW_LEN, num_features=cfg.NUM_FEATURES)


def make_dataloaders(X_train, y_train, X_test, y_test):
    # BCELoss requires float32 targets; y arrays are int8 by default
    tr = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    te = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test.astype(np.float32)),
    )
    tr_loader = DataLoader(tr, batch_size=cfg.BATCH_SIZE, shuffle=True,
                            drop_last=True, num_workers=0, pin_memory=cfg.PIN_MEMORY)
    te_loader = DataLoader(te, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            drop_last=False, num_workers=0, pin_memory=cfg.PIN_MEMORY)
    return tr_loader, te_loader


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total = 0.0
    n = 0
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / n


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    n_batches = 0
    probs_all = []
    targets_all = []
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(X)
        loss = loss_fn(pred, y)
        total += loss.item()
        n_batches += 1
        probs_all.append(pred.cpu().numpy())
        targets_all.append(y.cpu().numpy())
    avg_loss = total / n_batches
    probs = np.concatenate(probs_all)
    targets = np.concatenate(targets_all)
    preds = (probs > cfg.DECISION_THRESHOLD).astype(np.float32)
    cm = confusion_matrix(targets, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "loss": float(avg_loss),
        "accuracy": float(accuracy_score(targets, preds)),
        "precision": float(precision_score(targets, preds, zero_division=0)),
        "recall": float(recall_score(targets, preds, zero_division=0)),
        "f1": float(f1_score(targets, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(targets, probs)) if len(set(targets)) > 1 else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "far": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
    }
    return metrics, probs, targets


def per_ood_class_metrics(probs, targets, attack_types):
    """
    Compute F1 / recall for each held-out OOD attack class separately.

    For OOD test, only Normal + 4 held-out classes are present.
    For each held-out class C: recall = fraction of C-windows correctly
    classified as attack (label=1). Higher = better OOD generalization.
    """
    preds = (probs > cfg.DECISION_THRESHOLD).astype(np.float32)
    results = {}
    for cls in sorted(set(attack_types)):
        mask = attack_types == cls
        if mask.sum() == 0:
            continue
        cls_targets = targets[mask]
        cls_preds = preds[mask]
        if cls == NORMAL_CLASS:
            # For Normal: how many correctly classified as 0?
            correct = int((cls_preds == 0).sum())
            metric_name = "specificity"
            metric_val = correct / len(cls_preds) if len(cls_preds) > 0 else 0.0
        else:
            # For OOD attack: how many correctly classified as 1?
            correct = int((cls_preds == 1).sum())
            metric_name = "recall"
            metric_val = correct / len(cls_preds) if len(cls_preds) > 0 else 0.0
        results[cls] = {
            "n_windows": int(mask.sum()),
            "n_correct": correct,
            "metric_name": metric_name,
            "metric_value": float(metric_val),
        }
    return results


# ============================================================================
# MAIN LOOP
# ============================================================================

def run_ood_for_model_seed(model_name, seed, data):
    print(f"\n{'='*60}")
    print(f"OOD: {model_name.upper()}  |  SEED: {seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")

    model = make_model(model_name).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    train_loader, test_loader = make_dataloaders(
        data["X_train"], data["y_train"], data["X_test"], data["y_test"]
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

    best_f1 = -1.0
    best_epoch = -1
    ckpt_path = cfg.CHECKPOINTS_DIR / f"ood_{model_name}_seed{seed}_best.pt"
    epoch_log = []

    for epoch in range(cfg.EPOCHS):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        m, _, _ = evaluate(model, test_loader, loss_fn, device)
        scheduler.step(m["loss"])
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        epoch_log.append({
            "epoch": epoch + 1, "train_loss": tr_loss,
            "test_loss": m["loss"], "f1": m["f1"], "roc_auc": m["roc_auc"],
            "far": m["far"], "lr": lr, "time_sec": elapsed,
        })
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), ckpt_path)
        print(f"  Epoch {epoch+1:>2}/{cfg.EPOCHS} | tr_loss={tr_loss:.4f} | "
              f"te_loss={m['loss']:.4f} | F1={m['f1']:.4f} | "
              f"AUC={m['roc_auc']:.4f} | FAR={m['far']:.4f} | {elapsed:.1f}s")

    print(f"\n  Best epoch: {best_epoch} (F1={best_f1:.4f})")

    # Final eval with best checkpoint
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    final, probs, targets = evaluate(model, test_loader, loss_fn, device)
    per_class = per_ood_class_metrics(probs, targets, data["attack_test"])

    print(f"\n  Per-class OOD performance:")
    for cls, info in sorted(per_class.items()):
        marker = " (OOD)" if cls in OOD_HELD_OUT_CLASSES else ""
        print(f"    {cls:<25} {info['metric_name']}={info['metric_value']:.4f} "
              f"(n={info['n_windows']:,}){marker}")

    result = {
        "model": model_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "n_params": n_params,
        "ood_held_out_classes": OOD_HELD_OUT_CLASSES,
        "final_metrics": final,
        "per_class_metrics": per_class,
        "epoch_log": epoch_log,
    }
    out_path = cfg.METRICS_DIR / f"ood_{model_name}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")
    return result


def aggregate_ood(results, model_name):
    print(f"\n{'='*60}")
    print(f"OOD AGGREGATE: {model_name.upper()}  ({len(results)} seeds)")
    print(f"{'='*60}")
    f1s = [r["final_metrics"]["f1"] for r in results]
    aucs = [r["final_metrics"]["roc_auc"] for r in results]
    recalls = [r["final_metrics"]["recall"] for r in results]
    precisions = [r["final_metrics"]["precision"] for r in results]

    # Per-OOD-class recall mean
    per_class_recalls = {cls: [] for cls in OOD_HELD_OUT_CLASSES}
    normal_specificities = []
    for r in results:
        for cls, info in r["per_class_metrics"].items():
            if cls in OOD_HELD_OUT_CLASSES:
                per_class_recalls[cls].append(info["metric_value"])
            elif cls == NORMAL_CLASS:
                normal_specificities.append(info["metric_value"])

    summary = {
        "model": model_name,
        "n_seeds": len(results),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "f1_values": f1s,
        "roc_auc_mean": float(np.mean(aucs)),
        "recall_mean": float(np.mean(recalls)),
        "precision_mean": float(np.mean(precisions)),
        "normal_specificity_mean": float(np.mean(normal_specificities)) if normal_specificities else 0.0,
        "per_ood_class_recall": {
            cls: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v}
            for cls, v in per_class_recalls.items() if v
        },
    }

    print(f"  Overall F1:        {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    print(f"  Overall ROC-AUC:   {summary['roc_auc_mean']:.4f}")
    print(f"  Overall recall:    {summary['recall_mean']:.4f}")
    print(f"  Normal specificity: {summary['normal_specificity_mean']:.4f}")
    print(f"\n  Per-OOD-class recall (how well held-out attacks are detected):")
    for cls, info in summary["per_ood_class_recall"].items():
        print(f"    {cls:<25} {info['mean']:.4f} ± {info['std']:.4f}")

    out_path = cfg.METRICS_DIR / f"ood_{model_name}_aggregate.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out_path}")
    return summary


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        choices=["tide", "cnn", "lstm", "dlinear", "transformer", "all"],
                        default="all")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg.make_dirs()
    print(cfg.summary())

    print("\n" + "="*60)
    print("OUT-OF-DISTRIBUTION EXPERIMENT SETUP")
    print("="*60)
    print(f"Held-out OOD attack classes: {OOD_HELD_OUT_CLASSES}")
    print(f"These classes are NEVER seen during training.")
    print(f"Models trained on 11 in-distribution attacks must generalize.")
    print()

    # Build OOD windows once (shared across all model/seed combinations)
    print("Building OOD windows (one-time, shared across runs)...")
    data = build_ood_windows()
    print()

    models_to_run = (["tide", "cnn", "lstm", "dlinear", "transformer"]
                     if args.model == "all" else [args.model])
    seeds_to_run = [args.seed] if args.seed is not None else cfg.SEEDS

    for model_name in models_to_run:
        print(f"\n{'#'*60}\n# OOD MODEL: {model_name.upper()}\n{'#'*60}")
        results = []
        for seed in seeds_to_run:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            res = run_ood_for_model_seed(model_name, seed, data)
            results.append(res)
        if len(results) > 1:
            aggregate_ood(results, model_name)

    print("\nOOD experiments done.")


if __name__ == "__main__":
    main()
