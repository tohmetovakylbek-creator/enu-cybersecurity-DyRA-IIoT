#!/usr/bin/env python3
"""
scripts/quantize_edge.py
─────────────────────────────────────────────────────────────────────────────
Export a trained TiDE checkpoint to INT8 ONNX and benchmark latency.
Reproduces Table 17 (Section 4.9).

Usage:
  python scripts/quantize_edge.py \\
      --checkpoint results/edge_iiotset/TiDE_seed42.pt \\
      --data       /path/to/DNN-EdgeIIoT-dataset.csv   \\
      --out        results/quantized
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dyra_iiot.config as C
from dyra_iiot.data.features import select_features_edge
from dyra_iiot.data.partitioning import (
    apply_normalizer, fit_normalizer, stratified_per_class_block_split,
)
from dyra_iiot.deployment.quantize import (
    benchmark_onnx, export_to_onnx, full_quantize_pipeline, quantize_int8,
)
from dyra_iiot.models.backbones import build_backbone

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="DyRA-IIoT INT8 quantization")
    p.add_argument("--checkpoint", required=True, help=".pt file with model state_dict")
    p.add_argument("--data",       required=True, help="Edge-IIoTset CSV path")
    p.add_argument("--backbone",   default="TiDE", help="Backbone name")
    p.add_argument("--out",        default="results/quantized")
    p.add_argument("--n-calib",    type=int, default=C.INT8_CALIBRATION_SAMPLES,
                   help="Number of calibration windows")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and preprocess data ──────────────────────────────────────────
    logger.info("Loading %s", args.data)
    df   = pd.read_csv(args.data, low_memory=False)
    X, y, _ = select_features_edge(df)
    types   = df["Attack_type"].values

    Xtr, ytr, Xte, yte = stratified_per_class_block_split(
        X, y, types, C.TRAIN_RATIO, C.WINDOW_LEN)
    mu, std   = fit_normalizer(Xtr)
    Xtr_n     = apply_normalizer(Xtr, mu, std)

    # Calibration subset (stratified random sample)
    rng   = np.random.default_rng(42)
    idx   = rng.choice(len(Xtr_n), size=min(args.n_calib, len(Xtr_n)), replace=False)
    calib = Xtr_n[idx]

    # ── Load model ────────────────────────────────────────────────────────
    logger.info("Loading checkpoint %s", args.checkpoint)
    model = build_backbone(args.backbone, C.WINDOW_LEN, X.shape[1])
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # ── Export + quantize ─────────────────────────────────────────────────
    fp32_path, int8_path, fp32_bench, int8_bench = full_quantize_pipeline(
        model, calib, C.WINDOW_LEN, X.shape[1],
        output_dir=out_dir, backbone_name=args.backbone,
    )

    # ── Print Table 17 summary ────────────────────────────────────────────
    logger.info("\n%s\n  Table 17 (CPU benchmark)\n%s", "─"*55, "─"*55)
    logger.info("%-20s  %8s  %12s  %s",
                "Model", "Size(MB)", "Lat(ms/win)", "Throughput(win/s)")
    for name, bench, path in [("FP32", fp32_bench, fp32_path),
                               ("INT8", int8_bench, int8_path)]:
        logger.info("%-20s  %8.2f  %12.3f  %.0f",
                    name,
                    bench["model_size_mb"],
                    bench["latency_ms_mean"],
                    bench["throughput_wps"])

    results = {
        "fp32": {**fp32_bench, "path": str(fp32_path)},
        "int8": {**int8_bench, "path": str(int8_path)},
        "speedup_x":     fp32_bench["latency_ms_mean"] / int8_bench["latency_ms_mean"],
        "size_reduction_x": fp32_bench["model_size_mb"] / int8_bench["model_size_mb"],
    }
    with open(out_dir / "quantization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\n[DONE] Results → %s", out_dir)


if __name__ == "__main__":
    main()
