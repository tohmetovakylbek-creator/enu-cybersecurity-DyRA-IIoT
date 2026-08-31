"""
baselines_v2.py — Train four baseline architectures on the same honest
stratified per-class-block split used for TiDE.

Architectures (per paper Section 4.1):
    1. 1D-CNN:     2 conv layers (64 filters, k=3) + maxpool + linear classifier
    2. LSTM:       2 recurrent layers, hidden=128, sequential processing
    3. DLinear:    trend/seasonal decomposition + linear projection
    4. Vanilla-Transformer: 1 encoder block, 4 heads, d_model=128, mean-pool

All baselines:
- Use the same 36 features (input_dim varies per architecture)
- Same train/test split from artifacts/windows/
- Same class-weighted BCE, Adam, ReduceLROnPlateau, batch=64, 10 epochs
- Same 5 random seeds: 42, 123, 456, 789, 2024

Usage:
    python baselines_v2.py                       # train all 4 baselines, all 5 seeds
    python baselines_v2.py --model cnn           # one baseline, all seeds
    python baselines_v2.py --model lstm --seed 42  # one baseline, one seed
"""

import argparse
import json
import gc
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


# ============================================================================
# BASELINE ARCHITECTURES
# ============================================================================

class CNN1D(nn.Module):
    """1D-CNN baseline: 2 conv layers + maxpool + linear classifier."""
    def __init__(self, seq_len=50, num_features=36, dropout=0.1):
        super().__init__()
        # Input: (B, seq_len, num_features) -> permute to (B, num_features, seq_len)
        self.conv1 = nn.Conv1d(num_features, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        # After 2 maxpools: seq_len 50 -> 25 -> 12
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64 * 12, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, L, F) -> (B, F, L) for Conv1d
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return self.sigmoid(x).squeeze(-1)


class LSTMModel(nn.Module):
    """LSTM baseline: 2 recurrent layers, hidden=128."""
    def __init__(self, seq_len=50, num_features=36, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, L, F)
        out, _ = self.lstm(x)
        # Use last timestep
        last = out[:, -1, :]
        logit = self.fc(last)
        return self.sigmoid(logit).squeeze(-1)


class DLinear(nn.Module):
    """DLinear baseline: trend + seasonal decomposition + linear."""
    def __init__(self, seq_len=50, num_features=36, kernel_size=25):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        # Trend extraction via moving average
        self.kernel_size = kernel_size
        # Two linear branches: trend and seasonal
        self.linear_trend = nn.Linear(seq_len * num_features, 64)
        self.linear_seasonal = nn.Linear(seq_len * num_features, 64)
        self.classifier = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def moving_avg(self, x):
        """Moving average along time dimension. x: (B, L, F) -> (B, L, F)."""
        # Reflective padding to keep length
        pad = (self.kernel_size - 1) // 2
        x_perm = x.permute(0, 2, 1)  # (B, F, L)
        x_padded = nn.functional.pad(x_perm, (pad, pad), mode='replicate')
        avg = nn.functional.avg_pool1d(x_padded, kernel_size=self.kernel_size, stride=1)
        return avg.permute(0, 2, 1)  # back to (B, L, F)

    def forward(self, x):
        # Decomposition
        trend = self.moving_avg(x)
        seasonal = x - trend
        # Flatten and project
        trend_flat = trend.reshape(trend.size(0), -1)
        season_flat = seasonal.reshape(seasonal.size(0), -1)
        t_proj = torch.relu(self.linear_trend(trend_flat))
        s_proj = torch.relu(self.linear_seasonal(season_flat))
        combined = torch.cat([t_proj, s_proj], dim=-1)
        logit = self.classifier(combined)
        return self.sigmoid(logit).squeeze(-1)


class VanillaTransformer(nn.Module):
    """Vanilla-Transformer baseline: 1 encoder block, 4 heads, d_model=128."""
    def __init__(self, seq_len=50, num_features=36, d_model=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, L, F) -> (B, L, d_model)
        x = self.input_proj(x)
        x = self.encoder(x)
        # Mean-pool over sequence
        pooled = x.mean(dim=1)
        logit = self.fc(pooled)
        return self.sigmoid(logit).squeeze(-1)


MODEL_REGISTRY = {
    "cnn": CNN1D,
    "lstm": LSTMModel,
    "dlinear": DLinear,
    "transformer": VanillaTransformer,
}


# ============================================================================
# SHARED INFRASTRUCTURE (similar to train_v2.py but parameterized by model)
# ============================================================================

def load_windows():
    train_data = np.load(cfg.WINDOWS_DIR / "train.npz", allow_pickle=True)
    test_data = np.load(cfg.WINDOWS_DIR / "test.npz", allow_pickle=True)
    return (
        train_data["X"].astype(np.float32),
        train_data["y"].astype(np.float32),
        train_data["attack"],
        test_data["X"].astype(np.float32),
        test_data["y"].astype(np.float32),
        test_data["attack"],
    )


def make_dataloaders(X_train, y_train, X_test, y_test):
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=cfg.PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, drop_last=False,
        num_workers=0, pin_memory=cfg.PIN_MEMORY,
    )
    return train_loader, test_loader


def make_weighted_bce_loss(y_train):
    n_total = len(y_train)
    n_pos = (y_train == 1).sum()
    n_neg = n_total - n_pos
    pos_weight = n_neg / n_pos
    print(f"  Class weighting: pos_weight={pos_weight:.4f}")

    base_loss = nn.BCELoss(reduction='none')

    def weighted_loss(pred, target):
        raw = base_loss(pred, target)
        weights = torch.where(target == 1.0, pos_weight, 1.0)
        return (raw * weights).mean()

    return weighted_loss


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / n


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
    preds = (probs > cfg.DECISION_THRESHOLD).astype(np.float32)

    metrics = {
        "loss": float(avg_loss),
        "accuracy": float(accuracy_score(targets, preds)),
        "precision": float(precision_score(targets, preds, zero_division=0)),
        "recall": float(recall_score(targets, preds, zero_division=0)),
        "f1": float(f1_score(targets, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(targets, probs)),
    }
    cm = confusion_matrix(targets, preds)
    tn, fp, fn, tp = cm.ravel()
    metrics["tn"] = int(tn); metrics["fp"] = int(fp)
    metrics["fn"] = int(fn); metrics["tp"] = int(tp)
    metrics["far"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return metrics, probs, targets


def compute_per_class_f1(probs, targets, attack_types):
    preds = (probs > cfg.DECISION_THRESHOLD).astype(np.float32)
    results = {}
    for cls in sorted(set(attack_types)):
        mask = attack_types == cls
        if mask.sum() == 0:
            continue
        if cls == "Normal":
            f1 = float(f1_score(targets[mask], preds[mask], pos_label=0, zero_division=0))
        else:
            f1 = float(f1_score(targets[mask], preds[mask], zero_division=0))
        results[cls] = {"f1": f1, "n_windows": int(mask.sum())}
    return results


def measure_inference_latency(model, device, batch_size=64, n_warmup=10, n_runs=100):
    """Average ms per batch for a forward pass."""
    model.eval()
    dummy = torch.randn(batch_size, cfg.WINDOW_LEN, cfg.NUM_FEATURES, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_runs):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.time() - t0) / n_runs * 1000
    return elapsed_ms


# ============================================================================
# TRAIN ONE BASELINE FOR ONE SEED
# ============================================================================

def train_baseline_seed(model_name, seed, X_train, y_train, X_test, y_test, attack_test):
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name.upper()}  |  SEED: {seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    ModelClass = MODEL_REGISTRY[model_name]
    model = ModelClass(seq_len=cfg.WINDOW_LEN, num_features=cfg.NUM_FEATURES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    train_loader, test_loader = make_dataloaders(X_train, y_train, X_test, y_test)
    loss_fn = make_weighted_bce_loss(y_train)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE,
                           weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=cfg.LR_SCHEDULER_FACTOR,
        patience=cfg.LR_SCHEDULER_PATIENCE,
        min_lr=cfg.LR_SCHEDULER_MIN_LR,
    )

    best_f1 = 0.0
    best_epoch = -1
    ckpt_path = cfg.CHECKPOINTS_DIR / f"{model_name}_seed{seed}_best.pt"
    epoch_log = []

    for epoch in range(cfg.EPOCHS):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        m, _, _ = evaluate(model, test_loader, loss_fn, device)
        scheduler.step(m["loss"])
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        epoch_log.append({
            "epoch": epoch + 1, "train_loss": tr_loss, "test_loss": m["loss"],
            "f1": m["f1"], "roc_auc": m["roc_auc"], "far": m["far"], "lr": lr,
            "time_sec": elapsed,
        })
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), ckpt_path)
        print(f"  Epoch {epoch+1:>2}/{cfg.EPOCHS} | tr_loss={tr_loss:.4f} | "
              f"te_loss={m['loss']:.4f} | F1={m['f1']:.4f} | "
              f"AUC={m['roc_auc']:.4f} | FAR={m['far']:.4f} | {elapsed:.1f}s")

    print(f"\n  Best epoch: {best_epoch} (F1={best_f1:.4f})")

    # Load best and evaluate
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    final, probs, targets = evaluate(model, test_loader, loss_fn, device)
    per_class = compute_per_class_f1(probs, targets, attack_test)

    # Inference latency
    latency_ms = measure_inference_latency(model, device, batch_size=cfg.BATCH_SIZE)
    print(f"  Inference latency: {latency_ms:.2f} ms/batch (batch={cfg.BATCH_SIZE})")

    result = {
        "model": model_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "n_params": n_params,
        "inference_ms_per_batch": float(latency_ms),
        "final_metrics": final,
        "per_class_f1": per_class,
        "epoch_log": epoch_log,
    }
    out_path = cfg.METRICS_DIR / f"{model_name}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to: {out_path}")
    return result


def aggregate(results, model_name):
    print(f"\n{'='*60}")
    print(f"AGGREGATE: {model_name.upper()}  ({len(results)} seeds)")
    print(f"{'='*60}")
    f1s = [r["final_metrics"]["f1"] for r in results]
    aucs = [r["final_metrics"]["roc_auc"] for r in results]
    fars = [r["final_metrics"]["far"] for r in results]
    precs = [r["final_metrics"]["precision"] for r in results]
    recs = [r["final_metrics"]["recall"] for r in results]
    latencies = [r["inference_ms_per_batch"] for r in results]

    summary = {
        "model": model_name,
        "n_seeds": len(results),
        "n_params": results[0]["n_params"],
        "inference_ms_per_batch": {"mean": float(np.mean(latencies)),
                                    "std": float(np.std(latencies))},
        "f1": {"mean": float(np.mean(f1s)), "std": float(np.std(f1s)),
               "min": float(np.min(f1s)), "max": float(np.max(f1s)),
               "values": f1s},
        "roc_auc": {"mean": float(np.mean(aucs)), "std": float(np.std(aucs))},
        "far": {"mean": float(np.mean(fars)), "std": float(np.std(fars))},
        "precision": {"mean": float(np.mean(precs)), "std": float(np.std(precs))},
        "recall": {"mean": float(np.mean(recs)), "std": float(np.std(recs))},
    }
    print(f"  F1:        {summary['f1']['mean']:.4f} ± {summary['f1']['std']:.4f} "
          f"[{summary['f1']['min']:.4f}, {summary['f1']['max']:.4f}]")
    print(f"  ROC-AUC:   {summary['roc_auc']['mean']:.4f} ± {summary['roc_auc']['std']:.4f}")
    print(f"  Precision: {summary['precision']['mean']:.4f} ± {summary['precision']['std']:.4f}")
    print(f"  Recall:    {summary['recall']['mean']:.4f} ± {summary['recall']['std']:.4f}")
    print(f"  FAR:       {summary['far']['mean']:.4f} ± {summary['far']['std']:.4f}")
    print(f"  Params:    {summary['n_params']:,}")
    print(f"  Latency:   {summary['inference_ms_per_batch']['mean']:.2f} ± "
          f"{summary['inference_ms_per_batch']['std']:.2f} ms/batch")

    out_path = cfg.METRICS_DIR / f"{model_name}_aggregate.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out_path}")
    return summary


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()) + ["all"],
                        default="all")
    parser.add_argument("--seed", type=int, default=None,
                        help="Train one seed (default: all 5)")
    args = parser.parse_args()

    cfg.make_dirs()
    print(cfg.summary())
    print()

    print("Loading windows...")
    t0 = time.time()
    X_train, y_train, attack_train, X_test, y_test, attack_test = load_windows()
    print(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"  Loaded in {time.time()-t0:.1f} sec\n")

    models_to_train = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    seeds_to_run = [args.seed] if args.seed is not None else cfg.SEEDS

    for model_name in models_to_train:
        print(f"\n{'#'*60}")
        print(f"# STARTING MODEL: {model_name.upper()}")
        print(f"{'#'*60}")
        results = []
        for seed in seeds_to_run:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            res = train_baseline_seed(
                model_name, seed,
                X_train, y_train, X_test, y_test, attack_test,
            )
            results.append(res)
        if len(results) > 1:
            aggregate(results, model_name)

    print("\nAll baselines done.")


if __name__ == "__main__":
    main()
