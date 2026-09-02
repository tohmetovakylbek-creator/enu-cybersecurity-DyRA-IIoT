#!/usr/bin/env python3
"""Recalculate Table 9 occupancy and distinct false-alert onsets.

Input: the directory produced by regenerate_table9_traces.py.
The calculation preserves the original chronological order and resets the
K-consecutive state at every class-block boundary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


IMPACT = {
    "DDoS_ICMP": 0.81, "DDoS_UDP": 0.81, "DDoS_HTTP": 0.81,
    "DDoS_TCP": 0.81, "Port_Scanning": 0.81,
    "Vulnerability_scanner": 0.81, "SQL_injection": 0.90, "XSS": 0.90,
    "Password": 0.58, "Uploading": 0.58, "Backdoor": 0.58,
    "MITM": 0.21, "Fingerprinting": 0.21, "Ransomware": 0.81,
    "Normal": 0.90,
}

CONFIGS = {
    "B": (False, False),
    "B+K": (False, True),
    "B+A": (True, False),
    "B+G": (False, False),  # gamma = 1.0 by design
    "FULL": (True, True),
}

MODEL_ORDER = {"tide": 0, "transformer": 1, "cnn": 2, "lstm": 3, "dlinear": 4}
CONFIG_ORDER = {name: i for i, name in enumerate(CONFIGS)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def k_state(exceed: np.ndarray, k: int) -> np.ndarray:
    state = np.zeros_like(exceed, dtype=bool)
    run = 0
    for i, value in enumerate(exceed):
        run = run + 1 if value else 0
        state[i] = run >= k
    return state


def count_onsets(state: np.ndarray) -> int:
    return int(np.sum(state & ~np.r_[False, state[:-1]]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=Path, default=Path("results/table9_traces"))
    ap.add_argument("--out", type=Path, default=Path("results/table9_corrected.csv"))
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--step-ms", type=float, default=66.7)
    args = ap.parse_args()

    root = args.traces.resolve()
    manifest = json.loads((root / "trace_manifest.json").read_text(encoding="utf-8"))
    rows = []
    reference_y = reference_classes = None

    for model, item in manifest["outputs"].items():
        filename = item["trace"].replace("\\", "/").split("/")[-1]
        path = root / filename
        if sha256(path) != item["trace_sha256"]:
            raise SystemExit(f"SHA-256 mismatch: {path}")
        z = np.load(path, allow_pickle=False)
        p = np.asarray(z["p"], dtype=np.float64)
        y = np.asarray(z["y"], dtype=np.int8)
        classes = np.asarray(z["attack_class"], dtype=str)
        if len(p) != 443111 or not (len(y) == len(classes) == len(p)):
            raise SystemExit(f"Invalid or unaligned trace: {path}")
        if reference_y is None:
            reference_y, reference_classes = y, classes
        elif not (np.array_equal(reference_y, y) and
                  np.array_equal(reference_classes, classes)):
            raise SystemExit(f"Label order differs: {path}")

        starts = np.r_[0, np.where(classes[1:] != classes[:-1])[0] + 1]
        ends = np.r_[starts[1:], len(classes)]
        normal = classes == "Normal"
        normal_windows = int(normal.sum())
        normal_hours = normal_windows * args.step_ms / 3_600_000.0

        for config, (use_asset, use_k) in CONFIGS.items():
            multiplier = (np.array([IMPACT[c] for c in classes])
                          if use_asset else np.ones(len(classes)))
            exceed = p * multiplier > args.tau
            alert_state = np.zeros(len(p), dtype=bool)
            detected, delays = [], []

            for start, end in zip(starts, ends):
                state = (k_state(exceed[start:end], args.K)
                         if use_k else exceed[start:end])
                alert_state[start:end] = state
                if classes[start] != "Normal":
                    found = bool(state.any())
                    detected.append(found)
                    if found:
                        delays.append(int(np.argmax(state)) + 1)

            normal_state = alert_state[normal]
            occupancy = int(normal_state.sum())
            onsets = count_onsets(normal_state)
            rows.append({
                "config": config,
                "backbone": model,
                "detection_percent": 100.0 * sum(detected) / len(detected),
                "mean_delay_windows": float(np.mean(delays)),
                "false_alert_occupancy_windows": occupancy,
                "occupancy_FAR_percent": 100.0 * occupancy / normal_windows,
                "distinct_false_alert_onsets": onsets,
                "false_alerts_per_normal_hour": onsets / normal_hours,
                "normal_windows": normal_windows,
                "normal_exposure_hours": normal_hours,
            })

    rows.sort(key=lambda r: (CONFIG_ORDER[r["config"]], MODEL_ORDER[r["backbone"]]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written: {args.out.resolve()}")


if __name__ == "__main__":
    main()
