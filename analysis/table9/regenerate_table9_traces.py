#!/usr/bin/env python3
"""Regenerate Table 9 per-window probability traces from seed-42 checkpoints.

Run this file from the DyRA-IIoT project root. The input test.npz already
contains train-only-normalized windows, so normalization MUST NOT be applied
again. The model classes used for the original training return probabilities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import config as cfg
from baselines_v2 import CNN1D, DLinear, LSTMModel, VanillaTransformer
from tide_model import TiDEAnomalyDetector


MODEL_BUILDERS = {
    "cnn": lambda: CNN1D(seq_len=50, num_features=36),
    "lstm": lambda: LSTMModel(seq_len=50, num_features=36),
    "dlinear": lambda: DLinear(seq_len=50, num_features=36),
    "transformer": lambda: VanillaTransformer(seq_len=50, num_features=36),
    "tide": lambda: TiDEAnomalyDetector(
        seq_len=50,
        num_features=36,
        hidden_dim=cfg.MODEL_HIDDEN_DIM,
        num_layers=cfg.MODEL_NUM_RESBLOCKS,
        dropout=cfg.MODEL_DROPOUT,
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_state(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch versions without weights_only
        return torch.load(path, map_location=device)


@torch.inference_mode()
def infer(model, X, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    out = np.empty(len(X), dtype=np.float32)
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        # Saved windows are float16 after normalization; models expect float32.
        xb = torch.from_numpy(X[start:stop].astype(np.float32)).to(device)
        pred = model(xb).detach().cpu().numpy().reshape(-1)
        out[start:stop] = pred
        if start == 0 or stop == len(X) or stop % 50000 < batch_size:
            print(f"    {stop:,}/{len(X):,}")
    return out


def confusion(y: np.ndarray, p: np.ndarray):
    pred = p > 0.5
    yb = y.astype(bool)
    return {
        "tn": int((~yb & ~pred).sum()),
        "fp": int((~yb & pred).sum()),
        "fn": int((yb & ~pred).sum()),
        "tp": int((yb & pred).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--out", type=Path, default=Path("results/table9_traces"))
    args = ap.parse_args()

    root = args.root.resolve()
    windows_path = root / "artifacts/windows/test.npz"
    checkpoint_dir = root / "artifacts/checkpoints"
    out_dir = args.out if args.out.is_absolute() else root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but CUDA is unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    print(f"Device: {device}")
    print(f"Loading {windows_path}")
    z = np.load(windows_path, allow_pickle=True)
    X = z["X"]
    y = np.asarray(z["y"], dtype=np.int8)
    attack = np.asarray(z["attack"], dtype=str)

    expected_shape = (443111, 50, 36)
    if X.shape != expected_shape:
        raise SystemExit(f"Unexpected X shape: {X.shape}; expected {expected_shape}")
    if len(y) != len(X) or len(attack) != len(X):
        raise SystemExit("X, y and attack arrays are not aligned")
    print(f"Windows: {len(X):,}; normal: {(y == 0).sum():,}; attack: {(y == 1).sum():,}")

    validation_rows = []
    manifest = {
        "windows": str(windows_path.relative_to(root)),
        "windows_sha256": sha256(windows_path),
        "n_windows": len(X),
        "seed": 42,
        "threshold": 0.5,
        "outputs": {},
    }

    for name, build in MODEL_BUILDERS.items():
        ckpt = checkpoint_dir / f"{name}_seed42_best.pt"
        if not ckpt.exists():
            raise SystemExit(f"Missing checkpoint: {ckpt}")
        print(f"\n{name}: {ckpt.name}")
        model = build().to(device)
        model.load_state_dict(load_state(ckpt, device), strict=True)
        p = infer(model, X, device, args.batch_size)

        if not np.isfinite(p).all():
            raise SystemExit(f"{name}: NaN/Inf in predictions")
        if p.min() < 0.0 or p.max() > 1.0:
            raise SystemExit(
                f"{name}: output range [{p.min()}, {p.max()}] is not probability; "
                "do not apply sigmoid blindly—check the model class"
            )

        trace_path = out_dir / f"{name}_seed42_table9.npz"
        np.savez_compressed(trace_path, p=p, y=y, attack_class=attack)
        cm = confusion(y, p)
        row = {
            "model": name,
            "windows": len(p),
            "p_min": float(p.min()),
            "p_max": float(p.max()),
            "p_mean": float(p.mean()),
            **cm,
        }
        validation_rows.append(row)
        manifest["outputs"][name] = {
            "trace": str(trace_path.relative_to(root)),
            "trace_sha256": sha256(trace_path),
            "checkpoint": str(ckpt.relative_to(root)),
            "checkpoint_sha256": sha256(ckpt),
            **row,
        }
        print(f"  P range: [{p.min():.8f}, {p.max():.8f}]")
        print(f"  confusion @ 0.5: {cm}")
        print(f"  saved: {trace_path}")
        del model, p
        if device.type == "cuda":
            torch.cuda.empty_cache()

    validation_path = out_dir / "trace_validation.csv"
    with validation_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(validation_rows[0]))
        writer.writeheader()
        writer.writerows(validation_rows)

    manifest_path = out_dir / "trace_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nValidation: {validation_path}")
    print(f"Manifest:   {manifest_path}")
    print("Done. Do not edit or reorder rows in the generated traces.")


if __name__ == "__main__":
    main()
