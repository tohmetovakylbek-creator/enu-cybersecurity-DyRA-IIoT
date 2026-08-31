"""
measure_tide_latency.py — Standalone tool to measure inference latency
for all 5 TiDE seeds WITHOUT retraining.

Loads each saved checkpoint, runs warmup + 100 timed forward passes
on synthetic input (batch=64), and updates the existing seed_NN.json
with an `inference_ms_per_batch` field.

This avoids the 4-5 hour cost of retraining TiDE from scratch.

Usage:
    python measure_tide_latency.py
"""

import json
import time
from pathlib import Path

import torch
import numpy as np

import config as cfg
from tide_model import TiDEAnomalyDetector


def measure_inference_latency(model, device, batch_size=64, n_warmup=10, n_runs=100):
    """Same methodology as baselines_v2.py."""
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


def main():
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Measuring inference latency for TiDE (5 seeds)...\n")

    latencies = []
    for seed in cfg.SEEDS:
        ckpt_path = cfg.CHECKPOINTS_DIR / f"tide_seed{seed}_best.pt"
        json_path = cfg.METRICS_DIR / f"seed_{seed}.json"

        if not ckpt_path.exists():
            print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
            continue
        if not json_path.exists():
            print(f"  [SKIP] Metrics JSON not found: {json_path}")
            continue

        # Rebuild model and load weights
        model = TiDEAnomalyDetector(
            seq_len=cfg.WINDOW_LEN,
            num_features=cfg.NUM_FEATURES,
            hidden_dim=cfg.MODEL_HIDDEN_DIM,
            num_layers=cfg.MODEL_NUM_RESBLOCKS,
            dropout=cfg.MODEL_DROPOUT,
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))

        # Measure
        lat_ms = measure_inference_latency(model, device, batch_size=cfg.BATCH_SIZE)
        latencies.append(lat_ms)
        print(f"  seed={seed}: {lat_ms:.3f} ms/batch (batch={cfg.BATCH_SIZE})")

        # Update existing JSON with latency field
        with open(json_path, "r") as f:
            data = json.load(f)
        data["inference_ms_per_batch"] = float(lat_ms)
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        # Free memory before next iteration
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if latencies:
        mean_lat = np.mean(latencies)
        std_lat = np.std(latencies)
        print(f"\nTiDE latency over {len(latencies)} seeds:")
        print(f"  mean: {mean_lat:.3f} ms/batch")
        print(f"  std:  {std_lat:.3f} ms/batch")
        print(f"\nAll seed_*.json files updated with `inference_ms_per_batch` field.")
        print(f"Now re-run: python build_tables.py")
    else:
        print("\n[ERROR] No latency measurements taken — check checkpoint paths.")


if __name__ == "__main__":
    main()
