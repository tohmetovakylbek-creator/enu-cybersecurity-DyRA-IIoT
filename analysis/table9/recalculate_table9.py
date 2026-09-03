#!/usr/bin/env python3
"""Recompute DyRA-IIoT Table 9 from saved per-window probability traces.

The script deliberately separates two quantities:
  1. false-positive windows: every Normal window for which alert state is active;
  2. false-alert onsets: transitions of that state from 0 to 1.

Run from the project root:
  python analysis/table9/recalculate_table9.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Routing used by the manuscript. The nine mappings present in
# run_replay_real.py are retained; MITM and Fingerprinting are assigned to the
# low-impact sensor as stated explicitly in Section 4.6.1. Classes previously
# handled by the script's gateway default are written explicitly here.
ATTACK_TO_ASSET = {
    "Backdoor": ("N2_scada", 0.90),
    "DDoS_HTTP": ("N1_gateway", 0.81),
    "DDoS_ICMP": ("N1_gateway", 0.81),
    "DDoS_TCP": ("N1_gateway", 0.81),
    "DDoS_UDP": ("N1_gateway", 0.81),
    "Fingerprinting": ("N4_sensor", 0.21),
    "MITM": ("N4_sensor", 0.21),
    "Password": ("N3_hmi", 0.58),
    "Port_Scanning": ("N1_gateway", 0.81),
    "Ransomware": ("N5_plc", 0.81),
    "SQL_injection": ("N2_scada", 0.90),
    "Uploading": ("N3_hmi", 0.58),
    "Vulnerability_scanner": ("N1_gateway", 0.81),
    "XSS": ("N1_gateway", 0.81),
}

MODEL_NAMES = {
    "tide": "TiDE",
    "transformer": "Vanilla-Transformer",
    "cnn": "1D-CNN",
    "lstm": "LSTM",
    "dlinear": "DLinear",
}

CONFIGS = (
    ("B", False, False),
    ("B+K", False, True),
    ("B+A", True, False),
    ("B+G", False, False),  # gamma == 1.0, hence identical to B
    ("FULL", True, True),
)


def contiguous_segments(labels: np.ndarray):
    """Yield half-open intervals [start, stop) without crossing class borders."""
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    stops = np.r_[starts[1:], len(labels)]
    for start, stop in zip(starts, stops):
        yield int(start), int(stop), str(labels[start])


def apply_k(exceed: np.ndarray, k: int) -> np.ndarray:
    """Active state after k consecutive exceedances; no periodic reset."""
    active = np.zeros(len(exceed), dtype=bool)
    run = 0
    for i, value in enumerate(exceed):
        run = run + 1 if bool(value) else 0
        active[i] = run >= k
    return active


def evaluate_trace(
    probabilities: np.ndarray,
    labels: np.ndarray,
    config: str,
    use_asset: bool,
    use_k: bool,
    tau: float,
    k: int,
    normal_impact: float,
    step_ms: float,
):
    active_all = np.zeros(len(probabilities), dtype=bool)
    per_class = []

    for start, stop, label in contiguous_segments(labels):
        p = probabilities[start:stop]
        if label == "Normal":
            impact = normal_impact if use_asset else 1.0
            asset = "N2_scada (worst case)" if use_asset else "not applicable"
        else:
            if label not in ATTACK_TO_ASSET:
                raise KeyError(f"No asset mapping for attack class: {label}")
            asset, mapped_impact = ATTACK_TO_ASSET[label]
            impact = mapped_impact if use_asset else 1.0

        # gamma is 1.0 in every Table 9 configuration.
        score = p * impact
        exceed = score > tau
        active = apply_k(exceed, k) if use_k else exceed
        active_all[start:stop] = active

        if label != "Normal":
            locations = np.flatnonzero(active)
            detected = len(locations) > 0
            # One-based count: an immediate decision is 1 window; K=3 normally
            # produces a delay of 3 windows.
            delay_windows = int(locations[0] + 1) if detected else np.nan
            per_class.append(
                {
                    "Config": config,
                    "Attack class": label,
                    "Asset": asset,
                    "Impact used": impact,
                    "Windows": stop - start,
                    "Detected": detected,
                    "Delay (windows)": delay_windows,
                }
            )

    normal = labels == "Normal"
    false_positive_windows = int(np.sum(active_all & normal))

    # Count 0->1 transitions only within each Normal segment. Class/asset
    # boundaries reset the state and cannot create cross-boundary onsets.
    false_alert_onsets = 0
    for start, stop, label in contiguous_segments(labels):
        if label != "Normal":
            continue
        state = active_all[start:stop]
        false_alert_onsets += int(np.sum(state & np.r_[True, ~state[:-1]]))

    normal_windows = int(np.sum(normal))
    normal_hours = normal_windows * step_ms / 1000.0 / 3600.0
    detected_delays = [r["Delay (windows)"] for r in per_class if r["Detected"]]
    detected_classes = sum(bool(r["Detected"]) for r in per_class)
    total_classes = len(per_class)

    summary = {
        "Config": config,
        "Detection (%)": 100.0 * detected_classes / total_classes,
        "Detected classes": detected_classes,
        "Attack classes": total_classes,
        "Delay (windows)": float(np.mean(detected_delays)),
        "False-positive windows": false_positive_windows,
        "Window FAR (%)": 100.0 * false_positive_windows / normal_windows,
        "False-alert onsets": false_alert_onsets,
        "False alerts/hour": false_alert_onsets / normal_hours,
        "Outbound messages/s": false_alert_onsets / (normal_hours * 3600.0),
        "Normal windows": normal_windows,
        "Normal duration (h)": normal_hours,
    }
    return summary, per_class


def model_key(path: Path) -> str:
    stem = path.stem.lower()
    for key in MODEL_NAMES:
        if stem.startswith(key + "_"):
            return key
    raise ValueError(f"Cannot infer backbone from filename: {path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=Path("artifacts/table9_traces"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/table9"),
    )
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--step-ms", type=float, default=66.7)
    parser.add_argument("--normal-impact", type=float, default=0.90)
    args = parser.parse_args()

    paths = sorted(args.traces_dir.glob("*_seed42_table9.npz"))
    if len(paths) != 5:
        raise SystemExit(f"Expected 5 trace files, found {len(paths)} in {args.traces_dir}")

    summaries = []
    class_rows = []
    reference_y = reference_labels = None

    for path in paths:
        data = np.load(path, allow_pickle=False)
        required = {"p", "y", "attack_class"}
        if not required.issubset(data.files):
            raise SystemExit(f"{path.name}: expected arrays {sorted(required)}")
        p = np.asarray(data["p"], dtype=np.float64).reshape(-1)
        y = np.asarray(data["y"]).reshape(-1)
        labels = np.asarray(data["attack_class"]).astype(str).reshape(-1)
        if not (len(p) == len(y) == len(labels) == 443111):
            raise SystemExit(f"{path.name}: unexpected array lengths")
        if np.any((p < 0) | (p > 1)):
            raise SystemExit(f"{path.name}: p contains values outside [0, 1]")
        if not np.array_equal(y, (labels != "Normal").astype(y.dtype)):
            raise SystemExit(f"{path.name}: y and attack_class disagree")
        if reference_y is None:
            reference_y, reference_labels = y.copy(), labels.copy()
        elif not np.array_equal(y, reference_y) or not np.array_equal(labels, reference_labels):
            raise SystemExit(f"{path.name}: window order differs from the other traces")

        key = model_key(path)
        backbone = MODEL_NAMES[key]
        for config, use_asset, use_k in CONFIGS:
            summary, details = evaluate_trace(
                p, labels, config, use_asset, use_k,
                args.tau, args.k, args.normal_impact, args.step_ms,
            )
            summary = {"Backbone": backbone, **summary}
            summaries.append(summary)
            class_rows.extend({"Backbone": backbone, **row} for row in details)

    # Manuscript order: five configurations, then five backbones in each.
    config_order = {name: i for i, (name, _, _) in enumerate(CONFIGS)}
    model_order = {name: i for i, name in enumerate(MODEL_NAMES.values())}
    summaries.sort(key=lambda r: (config_order[r["Config"]], model_order[r["Backbone"]]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "table9_recomputed.csv"
    detail_path = args.out_dir / "table9_per_class_audit.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False, float_format="%.6f")
    pd.DataFrame(class_rows).to_csv(detail_path, index=False, float_format="%.6f")

    shown = pd.DataFrame(summaries)[[
        "Config", "Backbone", "Detection (%)", "Delay (windows)",
        "False-positive windows", "Window FAR (%)", "False-alert onsets",
        "False alerts/hour",
    ]]
    print(shown.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nWritten: {summary_path}")
    print(f"Written: {detail_path}")


if __name__ == "__main__":
    main()
