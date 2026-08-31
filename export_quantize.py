#!/usr/bin/env python3
"""Export final PyTorch checkpoints to ONNX and apply static INT8 quantization.

The calibration samples are read only from artifacts/windows/train.npz.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from onnxruntime.quantization import (
    CalibrationMethod,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

import config as cfg
from baselines_v2 import MODEL_REGISTRY
from tide_model import TiDEAnomalyDetector


class WindowCalibrationReader(CalibrationDataReader):
    def __init__(self, samples: np.ndarray):
        self._iterator = iter({"input": x[None].astype(np.float32)} for x in samples)

    def get_next(self):
        return next(self._iterator, None)


def build_model(name: str):
    if name == "tide":
        return TiDEAnomalyDetector(
            seq_len=cfg.WINDOW_LEN,
            num_features=cfg.NUM_FEATURES,
            hidden_dim=cfg.MODEL_HIDDEN_DIM,
            num_layers=cfg.MODEL_NUM_RESBLOCKS,
            dropout=cfg.MODEL_DROPOUT,
        )
    return MODEL_REGISTRY[name](seq_len=cfg.WINDOW_LEN, num_features=cfg.NUM_FEATURES)


def default_checkpoint(name: str) -> Path:
    if name == "tide":
        return cfg.CHECKPOINTS_DIR / "tide_seed42_best.pt"
    return cfg.CHECKPOINTS_DIR / f"best_{name}_edge.pt"


def load_state(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return payload
    for key in ("model_state_dict", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return payload


def export_and_quantize(name: str, checkpoint: Path, output_dir: Path,
                        calibration_samples: np.ndarray) -> None:
    model = build_model(name)
    model.load_state_dict(load_state(checkpoint), strict=True)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = output_dir / f"{name}_fp32.onnx"
    int8_path = output_dir / f"{name}_int8.onnx"
    dummy = torch.zeros(1, cfg.WINDOW_LEN, cfg.NUM_FEATURES, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        fp32_path,
        input_names=["input"],
        output_names=["probability"],
        dynamic_axes={"input": {0: "batch"}, "probability": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=WindowCalibrationReader(calibration_samples),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=True,
        calibrate_method=CalibrationMethod.Entropy,
    )
    print(f"{name}: {checkpoint} -> {fp32_path} -> {int8_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["tide", *MODEL_REGISTRY, "all"], default="all")
    parser.add_argument("--checkpoint", type=Path,
                        help="Override checkpoint; valid only for one --model")
    parser.add_argument("--calibration-windows", type=Path,
                        default=cfg.WINDOWS_DIR / "train.npz")
    parser.add_argument("--calibration-count", type=int, default=2048)
    parser.add_argument("--calibration-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path,
                        default=cfg.ARTIFACTS_DIR / "quantized")
    args = parser.parse_args()
    if args.model == "all" and args.checkpoint:
        parser.error("--checkpoint cannot be combined with --model all")

    with np.load(args.calibration_windows, allow_pickle=False) as data:
        all_windows = data["X"]
        if args.calibration_count > len(all_windows):
            parser.error("--calibration-count exceeds the number of training windows")
        rng = np.random.default_rng(args.calibration_seed)
        indices = rng.choice(len(all_windows), size=args.calibration_count, replace=False)
        windows = all_windows[indices].astype(np.float32)
    names = ["tide", *MODEL_REGISTRY] if args.model == "all" else [args.model]
    for name in names:
        export_and_quantize(
            name,
            args.checkpoint or default_checkpoint(name),
            args.output_dir,
            windows,
        )


if __name__ == "__main__":
    main()
