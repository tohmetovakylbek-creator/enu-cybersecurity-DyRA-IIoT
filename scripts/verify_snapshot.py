#!/usr/bin/env python3
"""Lightweight integrity checks for the tagged submission snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 456, 789, 2024)
MODELS = ("tide", "cnn", "lstm", "dlinear", "transformer")


def require(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"MISSING: {path.relative_to(ROOT)}")


def load_json(path: Path) -> dict:
    require(path)
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def verify_metrics() -> None:
    metrics = ROOT / "artifacts" / "metrics"
    for seed in SEEDS:
        load_json(metrics / f"seed_{seed}.json")
        for model in MODELS[1:]:
            load_json(metrics / f"{model}_seed{seed}.json")
        for model in MODELS:
            load_json(metrics / f"ood_{model}_seed{seed}.json")
            load_json(metrics / f"threeway_{model}_seed{seed}.json")
    for model in MODELS:
        load_json(metrics / f"threeway_{model}_aggregate.json")


def verify_onnx_pairs() -> None:
    onnx_dir = ROOT / "artifacts" / "onnx"
    for stem in ("tide_seed42_fp32", "transformer_seed42_best_fp32"):
        require(onnx_dir / f"{stem}.onnx")
        require(onnx_dir / f"{stem}.onnx.data")


def verify_manifest() -> None:
    manifest = ROOT / "ARTIFACT_MANIFEST.sha256"
    require(manifest)
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        require(path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"HASH MISMATCH: {relative}")


def optional_checkpoint_inventory() -> None:
    checkpoint_dir = ROOT / "artifacts" / "checkpoints"
    if not checkpoint_dir.is_dir():
        print("INFO: release checkpoints not unpacked; checkpoint checks skipped")
        return
    expected = []
    for seed in SEEDS:
        for model in MODELS:
            expected.extend((
                f"{model}_seed{seed}_best.pt",
                f"ood_{model}_seed{seed}_best.pt",
                f"3way_{model}_seed{seed}_best.pt",
            ))
    missing = [name for name in expected if not (checkpoint_dir / name).is_file()]
    if missing:
        raise SystemExit(f"MISSING CHECKPOINTS: {', '.join(missing)}")
    print(f"OK: {len(expected)} core protocol checkpoints found")


def main() -> None:
    verify_metrics()
    verify_onnx_pairs()
    verify_manifest()
    optional_checkpoint_inventory()
    print("OK: snapshot metrics, ONNX pairs, and checksums verified")


if __name__ == "__main__":
    main()
