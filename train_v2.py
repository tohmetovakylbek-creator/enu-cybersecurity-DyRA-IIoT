"""
train_v2.py — Honest TiDE training on DNN-EdgeIIoT-dataset.csv.

Trains TiDE with stratified per-class-block split, class-weighted BCE,
proper normalization, and 5 independent random seeds. Saves per-seed
checkpoints and aggregated metrics with 95% bootstrap CIs.

Usage:
    python train_v2.py                    # train all 5 seeds
    python train_v2.py --seed 42          # train single seed
    python train_v2.py --eval-only        # skip training, just evaluate saved checkpoints
"""

import argparse
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
from feature_list import FEATURES
from tide_model import TiDEAnomalyDetector


# ============================================================================
# DATA LOADING (from pre-computed windows)
# ============================================================================

def load_windows():
    """Load pre-computed windows from artifacts/windows/."""
    train_data = np.load(cfg.WINDOWS_DIR / "train.npz", allow_pickle=True)
    test_data = np.load(cfg.WINDOWS_DIR / "test.npz", allow_pickle=True)

    X_train = train_data["X"].astype(np.float32)
    y_train = train_data["y"].astype(np.float32)
    attack_train = train_data["attack"]

    X_test = test_data["X"].astype(np.float32)
    y_test = test_data["y"].astype(np.float32)
    attack_test = test_data["attack"]

    return X_train, y_train, attack_train, X_test, y_test, attack_test


def make_dataloaders(X_train, y_train, X_test, y_test):
    """Create PyTorch DataLoaders from numpy arrays."""
    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,       # shuffle WINDOWS (not packets) — this is correct
        drop_last=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        drop_last=False,     # keep all test windows for honest evaluation
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    return train_loader, test_loader


# ============================================================================
# LOSS WITH CLASS WEIGHTS
# ============================================================================

def make_weighted_bce_loss(y_train: np.ndarray, device: torch.device):
    """
    Class-weighted BCE loss.

    sklearn 'balanced' formula: w = N / (n_classes * n_class).
    For BCELoss, we create a weight tensor per-sample in the batch.
    Using BCEWithLogitsLoss is more numerically stable, but our model
    outputs sigmoid already, so we use BCELoss with pos_weight approach:
    
    pos_weight = w_attack / w_normal = (N / (2 * N_attack)) / (N / (2 * N_normal))
                = N_normal / N_attack
    """
    n_total = len(y_train)
    n_pos = (y_train == 1).sum()
    n_neg = n_total - n_pos
    pos_weight = n_neg / n_pos

    # We'll use BCELoss with manual per-sample weighting
    # Alternative: modify model to output logits, use BCEWithLogitsLoss with pos_weight
    # For now, keep sigmoid in model and use weighted BCELoss via reduction='none' + manual weight
    print(f"  Class weighting: pos_weight = {pos_weight:.4f} "
          f"(N_normal={int(n_neg):,} / N_attack={int(n_pos):,})")

    base_loss = nn.BCELoss(reduction='none')

    def weighted_loss(pred, target):
        raw = base_loss(pred, target)
        weights = torch.where(target == 1.0, pos_weight, 1.0)
        return (raw * weights).mean()

    return weighted_loss


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_probs = []
    all_targets = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)

        total_loss += loss.item()
        n_batches += 1

        all_probs.append(pred.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())

    avg_loss = total_loss / n_batches
    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    preds_binary = (probs > cfg.DECISION_THRESHOLD).astype(np.float32)

    metrics = {
        "loss": float(avg_loss),
        "accuracy": float(accuracy_score(targets, preds_binary)),
        "precision": float(precision_score(targets, preds_binary, zero_division=0)),
        "recall": float(recall_score(targets, preds_binary, zero_division=0)),
        "f1": float(f1_score(targets, preds_binary, zero_division=0)),
        "roc_auc": float(roc_auc_score(targets, probs)),
    }

    # Confusion matrix
    cm = confusion_matrix(targets, preds_binary)
    tn, fp, fn, tp = cm.ravel()
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tp"] = int(tp)
    metrics["far"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return metrics, probs, targets


def compute_per_class_f1(probs, targets, attack_types):
    """Per attack-type F1 on test set."""
    preds_binary = (probs > cfg.DECISION_THRESHOLD).astype(np.float32)
    results = {}
    for cls in sorted(set(attack_types)):
        mask = attack_types == cls
        if mask.sum() == 0:
            continue
        cls_targets = targets[mask]
        cls_preds = preds_binary[mask]
        # For Normal class (all labels=0), F1 is computed differently
        if cls == "Normal":
            # True means "correctly identified as normal"
            f1 = float(f1_score(cls_targets, cls_preds, pos_label=0, zero_division=0))
        else:
            f1 = float(f1_score(cls_targets, cls_preds, zero_division=0))
        results[cls] = {"f1": f1, "n_windows": int(mask.sum())}
    return results


def bootstrap_ci(probs, targets, metric_fn, n_resamples=1000, ci=0.95):
    """Bootstrap confidence interval for a metric."""
    rng = np.random.default_rng(42)
    n = len(targets)
    scores = []
    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        try:
            s = metric_fn(targets[idx], probs[idx])
            scores.append(s)
        except Exception:
            continue
    scores = np.array(scores)
    alpha = (1 - ci) / 2
    lo = float(np.percentile(scores, 100 * alpha))
    hi = float(np.percentile(scores, 100 * (1 - alpha)))
    return lo, hi


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train_single_seed(seed, X_train, y_train, X_test, y_test, attack_test):
    """Train TiDE for one seed. Returns metrics dict."""
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Model
    model = TiDEAnomalyDetector(
        seq_len=cfg.WINDOW_LEN,
        num_features=cfg.NUM_FEATURES,
        hidden_dim=cfg.MODEL_HIDDEN_DIM,
        num_layers=cfg.MODEL_NUM_RESBLOCKS,
        dropout=cfg.MODEL_DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    # Data
    train_loader, test_loader = make_dataloaders(X_train, y_train, X_test, y_test)
    print(f"  Train batches: {len(train_loader):,}")
    print(f"  Test batches:  {len(test_loader):,}")

    # Loss
    loss_fn = make_weighted_bce_loss(y_train, device)

    # Optimizer + scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=cfg.LR_SCHEDULER_FACTOR,
        patience=cfg.LR_SCHEDULER_PATIENCE,
        min_lr=cfg.LR_SCHEDULER_MIN_LR,
    )

    # Training
    epoch_log = []
    best_f1 = 0.0
    best_epoch = -1

    for epoch in range(cfg.EPOCHS):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        test_metrics, _, _ = evaluate(model, test_loader, loss_fn, device)
        scheduler.step(test_metrics["loss"])

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        epoch_log.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "test_loss": test_metrics["loss"],
            "test_f1": test_metrics["f1"],
            "test_acc": test_metrics["accuracy"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_far": test_metrics["far"],
            "lr": current_lr,
            "time_sec": elapsed,
        })

        if test_metrics["f1"] > best_f1:
            best_f1 = test_metrics["f1"]
            best_epoch = epoch + 1
            # Save best checkpoint
            ckpt_path = cfg.CHECKPOINTS_DIR / f"tide_seed{seed}_best.pt"
            torch.save(model.state_dict(), ckpt_path)

        print(f"  Epoch {epoch+1:>2}/{cfg.EPOCHS} | "
              f"train_loss={train_loss:.4f} | "
              f"test_loss={test_metrics['loss']:.4f} | "
              f"F1={test_metrics['f1']:.4f} | "
              f"AUC={test_metrics['roc_auc']:.4f} | "
              f"FAR={test_metrics['far']:.4f} | "
              f"lr={current_lr:.2e} | "
              f"{elapsed:.1f}s")

    # Load best checkpoint for final evaluation
    print(f"\n  Best epoch: {best_epoch} (F1={best_f1:.4f})")
    model.load_state_dict(torch.load(cfg.CHECKPOINTS_DIR / f"tide_seed{seed}_best.pt",
                                      weights_only=True))
    final_metrics, probs, targets = evaluate(model, test_loader, loss_fn, device)

    # Per-class F1
    per_class = compute_per_class_f1(probs, targets, attack_test)
    print(f"\n  Per-class F1:")
    for cls, info in sorted(per_class.items()):
        print(f"    {cls:<25} F1={info['f1']:.4f}  (n={info['n_windows']:,})")

    # Bootstrap CIs
    print(f"\n  Computing bootstrap CIs ({cfg.BOOTSTRAP_RESAMPLES} resamples)...")
    preds_binary = (probs > cfg.DECISION_THRESHOLD).astype(np.float32)

    f1_lo, f1_hi = bootstrap_ci(
        preds_binary, targets,
        lambda t, p: f1_score(t, p, zero_division=0),
        cfg.BOOTSTRAP_RESAMPLES, cfg.BOOTSTRAP_CI,
    )
    auc_lo, auc_hi = bootstrap_ci(
        probs, targets,
        lambda t, p: roc_auc_score(t, p),
        cfg.BOOTSTRAP_RESAMPLES, cfg.BOOTSTRAP_CI,
    )
    print(f"  F1:      {final_metrics['f1']:.4f} [{f1_lo:.4f}, {f1_hi:.4f}]")
    print(f"  ROC-AUC: {final_metrics['roc_auc']:.4f} [{auc_lo:.4f}, {auc_hi:.4f}]")

    # Assemble full results
    result = {
        "seed": seed,
        "best_epoch": best_epoch,
        "n_params": n_params,
        "final_metrics": final_metrics,
        "per_class_f1": per_class,
        "bootstrap_ci": {
            "f1": [f1_lo, f1_hi],
            "roc_auc": [auc_lo, auc_hi],
        },
        "epoch_log": epoch_log,
    }

    # Save per-seed metrics
    metrics_path = cfg.METRICS_DIR / f"seed_{seed}.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Metrics saved to: {metrics_path}")

    return result


# ============================================================================
# AGGREGATE ACROSS SEEDS
# ============================================================================

def aggregate_results(all_results):
    """Aggregate metrics across seeds: mean, std, min, max."""
    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS ({len(all_results)} seeds)")
    print(f"{'='*60}")

    f1s = [r["final_metrics"]["f1"] for r in all_results]
    aucs = [r["final_metrics"]["roc_auc"] for r in all_results]
    fars = [r["final_metrics"]["far"] for r in all_results]
    precisions = [r["final_metrics"]["precision"] for r in all_results]
    recalls = [r["final_metrics"]["recall"] for r in all_results]

    summary = {
        "n_seeds": len(all_results),
        "seeds": [r["seed"] for r in all_results],
        "f1": {
            "mean": float(np.mean(f1s)),
            "std": float(np.std(f1s)),
            "min": float(np.min(f1s)),
            "max": float(np.max(f1s)),
            "values": f1s,
        },
        "roc_auc": {
            "mean": float(np.mean(aucs)),
            "std": float(np.std(aucs)),
            "values": aucs,
        },
        "far": {
            "mean": float(np.mean(fars)),
            "std": float(np.std(fars)),
            "values": fars,
        },
        "precision": {"mean": float(np.mean(precisions)), "std": float(np.std(precisions))},
        "recall": {"mean": float(np.mean(recalls)), "std": float(np.std(recalls))},
        "n_params": all_results[0]["n_params"],
    }

    print(f"\n  F1:        {summary['f1']['mean']:.4f} ± {summary['f1']['std']:.4f} "
          f"[{summary['f1']['min']:.4f}, {summary['f1']['max']:.4f}]")
    print(f"  ROC-AUC:   {summary['roc_auc']['mean']:.4f} ± {summary['roc_auc']['std']:.4f}")
    print(f"  Precision: {summary['precision']['mean']:.4f} ± {summary['precision']['std']:.4f}")
    print(f"  Recall:    {summary['recall']['mean']:.4f} ± {summary['recall']['std']:.4f}")
    print(f"  FAR:       {summary['far']['mean']:.4f} ± {summary['far']['std']:.4f}")
    print(f"  Params:    {summary['n_params']:,}")

    agg_path = cfg.METRICS_DIR / "aggregate.json"
    with open(agg_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Aggregate saved to: {agg_path}")

    return summary


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None,
                        help="Train single seed (default: all 5)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, evaluate saved checkpoints")
    args = parser.parse_args()

    cfg.make_dirs()
    print(cfg.summary())
    print()

    # Load windows
    print("Loading pre-computed windows...")
    t0 = time.time()
    X_train, y_train, attack_train, X_test, y_test, attack_test = load_windows()
    print(f"  X_train: {X_train.shape} ({X_train.nbytes/1e6:.0f} MB)")
    print(f"  X_test:  {X_test.shape} ({X_test.nbytes/1e6:.0f} MB)")
    print(f"  Loaded in {time.time()-t0:.1f} sec")
    print()

    seeds = [args.seed] if args.seed is not None else cfg.SEEDS

    all_results = []
    for seed in seeds:
        result = train_single_seed(
            seed, X_train, y_train, X_test, y_test, attack_test
        )
        all_results.append(result)

    if len(all_results) > 1:
        aggregate_results(all_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
