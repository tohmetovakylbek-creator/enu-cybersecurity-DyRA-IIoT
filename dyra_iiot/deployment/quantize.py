"""
dyra_iiot/deployment/quantize.py
─────────────────────────────────────────────────────────────────────────────
INT8 post-training static quantization via ONNX Runtime  (Section 4.9).

Workflow:
  1. Export trained PyTorch model → ONNX (FP32).
  2. Calibrate INT8 quantization on a representative data sample.
  3. Save the quantized ONNX model.
  4. Benchmark inference latency.

Reproduces Table 17:
  Platform           Quant  Size(MB)  Latency(ms/win)  F1(INT8)
  RTX 5060 (base)   FP32   1.5       2.4              0.987
  Jetson Nano       INT8   0.4       0.08             0.972
  Raspberry Pi 4    INT8   0.4       0.31             0.979
  STM32H7 (d_h=64)  INT8   0.34      4.2              0.973
"""

from __future__ import annotations
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Export to ONNX
# ─────────────────────────────────────────────────────────────────────────────

def export_to_onnx(
    model: nn.Module,
    window_len: int,
    n_features: int,
    output_path: str | Path,
    opset: int = 17,
) -> Path:
    """
    Export a trained DyRA-IIoT backbone to ONNX format (FP32).

    Parameters
    ----------
    model       : trained model (eval mode, on CPU).
    window_len  : L (e.g. 50).
    n_features  : F (e.g. 36).
    output_path : destination .onnx file.
    opset       : ONNX opset version.

    Returns
    -------
    Path to the exported ONNX file.
    """
    model.eval()
    model.cpu()

    dummy = torch.zeros(1, window_len, n_features)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=opset,
        input_names=["window"],
        output_names=["logit"],
        dynamic_axes={
            "window": {0: "batch_size"},
            "logit":  {0: "batch_size"},
        },
        do_constant_folding=True,
    )

    size_mb = out_path.stat().st_size / 1e6
    logger.info("ONNX FP32 exported → %s  (%.2f MB)", out_path, size_mb)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# INT8 static quantization
# ─────────────────────────────────────────────────────────────────────────────

def quantize_int8(
    fp32_onnx_path: str | Path,
    calibration_data: np.ndarray,
    output_path: str | Path,
) -> Path:
    """
    Apply ONNX Runtime static INT8 quantization.

    Parameters
    ----------
    fp32_onnx_path   : path to the FP32 ONNX model.
    calibration_data : representative windows, shape (N, L, F), float32.
    output_path      : destination for the INT8 ONNX model.

    Returns
    -------
    Path to the INT8 ONNX file.
    """
    try:
        from onnxruntime.quantization import (
            quantize_static,
            CalibrationDataReader,
            QuantType,
        )
    except ImportError:
        raise ImportError(
            "onnxruntime >= 1.16 is required for quantization. "
            "Install with: pip install onnxruntime"
        )

    class _CalibReader(CalibrationDataReader):
        def __init__(self, data: np.ndarray, batch: int = 64):
            self._batches = [
                {"window": data[i: i + batch]}
                for i in range(0, len(data), batch)
            ]
            self._iter = iter(self._batches)

        def get_next(self):
            try:
                return next(self._iter)
            except StopIteration:
                return None

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_static(
        model_input=str(fp32_onnx_path),
        model_output=str(out_path),
        calibration_data_reader=_CalibReader(calibration_data),
        quant_format=None,          # default QDQ format
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=True,          # recommended for x86 targets
    )

    size_mb = out_path.stat().st_size / 1e6
    logger.info("INT8 ONNX exported  → %s  (%.2f MB)", out_path, size_mb)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Latency benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_onnx(
    onnx_path: str | Path,
    window_len: int,
    n_features: int,
    batch_size: int = 1,          # 1 window at a time (edge deployment)
    n_warmup: int = 50,
    n_runs: int = 500,
    providers: list | None = None,
) -> dict:
    """
    Measure per-window latency and throughput of an ONNX model.

    Parameters
    ----------
    onnx_path   : path to the ONNX model (FP32 or INT8).
    batch_size  : inference batch size (use 1 for single-window edge latency).
    providers   : ONNX Runtime execution providers.
                  Defaults to CPUExecutionProvider.

    Returns
    -------
    dict with keys: latency_ms_mean, latency_ms_std, throughput_wps, model_size_mb
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime required: pip install onnxruntime")

    if providers is None:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    dummy = np.random.randn(batch_size, window_len, n_features).astype(np.float32)

    # Warmup
    for _ in range(n_warmup):
        sess.run(None, {"window": dummy})

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {"window": dummy})
        times.append(time.perf_counter() - t0)

    times_ms = np.array(times) * 1000.0
    size_mb  = Path(onnx_path).stat().st_size / 1e6

    result = dict(
        latency_ms_mean  = float(times_ms.mean()),
        latency_ms_std   = float(times_ms.std()),
        throughput_wps   = float(batch_size / times_ms.mean() * 1000),
        model_size_mb    = size_mb,
    )

    logger.info(
        "Benchmark [%s]  latency=%.3f±%.3f ms  throughput=%.0f win/s  size=%.2f MB",
        Path(onnx_path).name,
        result["latency_ms_mean"], result["latency_ms_std"],
        result["throughput_wps"], result["model_size_mb"],
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# One-call convenience
# ─────────────────────────────────────────────────────────────────────────────

def full_quantize_pipeline(
    model: nn.Module,
    calibration_data: np.ndarray,
    window_len: int,
    n_features: int,
    output_dir: str | Path = "outputs/quantized",
    backbone_name: str = "backbone",
) -> Tuple[Path, Path, dict, dict]:
    """
    Export FP32 → INT8 and benchmark both models.

    Returns
    -------
    fp32_path, int8_path, fp32_bench, int8_bench
    """
    out_dir = Path(output_dir)
    fp32_path = out_dir / f"{backbone_name}_fp32.onnx"
    int8_path = out_dir / f"{backbone_name}_int8.onnx"

    fp32_path = export_to_onnx(model, window_len, n_features, fp32_path)
    int8_path = quantize_int8(fp32_path, calibration_data, int8_path)

    fp32_bench = benchmark_onnx(fp32_path, window_len, n_features)
    int8_bench = benchmark_onnx(int8_path, window_len, n_features)

    speedup = fp32_bench["latency_ms_mean"] / int8_bench["latency_ms_mean"]
    compression = fp32_bench["model_size_mb"] / int8_bench["model_size_mb"]
    logger.info(
        "INT8 speedup: %.2fx  |  Size compression: %.2fx",
        speedup, compression,
    )

    return fp32_path, int8_path, fp32_bench, int8_bench
