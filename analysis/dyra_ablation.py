#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dyra_ablation.py -- component-level ablation of the DyRA-IIoT risk pipeline.

Answers Reviewer 2, comment "A fairer ablation would compare: raw backbone IDS,
backbone + K rule, backbone + asset gate, backbone + gamma(t), and the full
DyRA-IIoT pipeline under the same traces and alert metrics."

The script does NOT retrain anything. It replays a saved per-window probability
trace P(t) through five configurations of the downstream risk pipeline and
reports identical alert metrics for each, so that any difference is attributable
to the pipeline components alone.

Configurations
--------------
  B        raw backbone IDS      alert  <=>  P(t) > tau                (per window)
  B+K      + K-consecutive rule  alert  <=>  P(t) > tau on K windows
  B+A      + asset gate          alert  <=>  P(t)*Impact(A) > tau
  B+G      + context factor      alert  <=>  P(t)*gamma(phi) > tau
  FULL     full DyRA-IIoT        alert  <=>  P(t)*Impact(A)*gamma(phi) > tau on K windows

Metrics (identical for every configuration)
-------------------------------------------
  detection rate per attack class          fraction of attack episodes that raise an alert
  attack-to-alert delay                    windows from episode start to first alert
  alert volume per 1000 normal windows
  false-alarm rate on Normal windows       evaluated on the worst-case (highest-impact) node

Input
-----
A CSV or NPZ with one row per test window, in chronological order:

    p              float   backbone probability P(t)
    y              int     1 = attack window, 0 = normal window
    attack_class   str     class label ("Normal" for benign windows)
    node           str     optional; if absent, taken from the routing map

and a JSON config (see --write-config to emit a template) holding the asset
inventory (Impact per node), the attack-class -> node routing, and the gamma
schedule.

Usage
-----
    python dyra_ablation.py --write-config config_ablation.json
    python dyra_ablation.py --trace results/replay_tide.csv --config config_ablation.json \
                            --tau 0.5 --K 3 --out tables/table7_ablation.csv
    python dyra_ablation.py --demo            # synthetic self-test, no data needed

Author: DyRA-IIoT authors. Released under the licence of the parent repository.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict

import numpy as np

try:
    import pandas as pd
except ImportError:  # pandas is optional for CSV input only
    pd = None


# --------------------------------------------------------------------------- #
# default configuration (Table 2 and Table 3 of the manuscript)
# --------------------------------------------------------------------------- #

DEFAULT_CONFIG = OrderedDict([
    ("assets", OrderedDict([
        ("N1_gateway", {"impact": 0.810, "phase": "standard_production"}),
        ("N2_scada",   {"impact": 0.900, "phase": "standard_production"}),
        ("N3_hmi",     {"impact": 0.580, "phase": "standard_production"}),
        ("N4_sensor",  {"impact": 0.210, "phase": "standard_production"}),
        ("N5_plc",     {"impact": 0.810, "phase": "standard_production"}),
    ])),
    ("gamma_schedule", OrderedDict([
        ("scheduled_maintenance", 0.4),
        ("standby",               0.6),
        ("standard_production",   1.0),
        ("shift_changeover",      1.2),
        ("critical_window",       1.5),
    ])),
    ("routing", OrderedDict([
        ("DDoS_ICMP",     "N1_gateway"),
        ("DDoS_UDP",      "N1_gateway"),
        ("DDoS_HTTP",     "N1_gateway"),
        ("DDoS_TCP",      "N1_gateway"),
        ("Port_Scanning", "N1_gateway"),
        ("Vulnerability_scanner", "N1_gateway"),
        ("SQL_injection", "N2_scada"),
        ("XSS",           "N2_scada"),
        ("Password",      "N3_hmi"),
        ("Uploading",     "N3_hmi"),
        ("Backdoor",      "N3_hmi"),
        ("MITM",          "N4_sensor"),
        ("Fingerprinting", "N4_sensor"),
        ("Ransomware",    "N5_plc"),
    ])),
    ("worst_case_node_for_far", "N2_scada"),
])


# --------------------------------------------------------------------------- #
# core pipeline
# --------------------------------------------------------------------------- #

def k_consecutive(exceed: np.ndarray, K: int) -> np.ndarray:
    """Boolean array: True where `exceed` has been True on K consecutive windows."""
    if K <= 1:
        return exceed.copy()
    out = np.zeros_like(exceed, dtype=bool)
    run = 0
    for i, e in enumerate(exceed):
        run = run + 1 if e else 0
        out[i] = run >= K
    return out


def run_pipeline(p, impact, gamma, tau, K, use_gate, use_gamma, use_K):
    """Return the boolean alert trace for one pipeline configuration."""
    r = np.asarray(p, dtype=float)
    if use_gate:
        r = r * float(impact)
    if use_gamma:
        r = r * float(gamma)
    exceed = r > tau
    return k_consecutive(exceed, K) if use_K else exceed


CONFIGS = OrderedDict([
    #  label            gate   gamma   K
    ("B",     dict(use_gate=False, use_gamma=False, use_K=False)),
    ("B+K",   dict(use_gate=False, use_gamma=False, use_K=True)),
    ("B+A",   dict(use_gate=True,  use_gamma=False, use_K=False)),
    ("B+G",   dict(use_gate=False, use_gamma=True,  use_K=False)),
    ("FULL",  dict(use_gate=True,  use_gamma=True,  use_K=True)),
])

CONFIG_NAMES = {
    "B":    "Backbone only (P(t) > tau)",
    "B+K":  "Backbone + K-consecutive rule",
    "B+A":  "Backbone + asset gate",
    "B+G":  "Backbone + context factor gamma(t)",
    "FULL": "Full DyRA-IIoT pipeline",
}


# --------------------------------------------------------------------------- #
# episode handling
# --------------------------------------------------------------------------- #

def find_episodes(labels):
    """Contiguous runs of an identical non-Normal class -> [(class, start, end), ...]."""
    episodes = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            if labels[start] != "Normal":
                episodes.append((labels[start], start, i - 1))
            start = i
    return episodes


def evaluate(df, cfg, tau, K, min_windows):
    """Evaluate all five configurations; return (per_class_rows, summary_rows)."""
    p = df["p"].to_numpy(dtype=float)
    cls = df["attack_class"].astype(str).to_numpy()
    nodes = df["node"].astype(str).to_numpy()

    assets = cfg["assets"]
    gsched = cfg["gamma_schedule"]
    episodes = find_episodes(cls)

    per_class, summary = [], []

    for label, opts in CONFIGS.items():
        det_num = det_den = 0
        delays = []
        skipped = []

        for aclass, s, e in episodes:
            n_win = e - s + 1
            if n_win < min_windows:
                skipped.append(aclass)
                continue
            node = nodes[s]
            asset = assets.get(node)
            if asset is None:
                raise KeyError("node %r is not in the asset inventory" % node)
            gamma = gsched[asset["phase"]]
            alerts = run_pipeline(p[s:e + 1], asset["impact"], gamma, tau, K, **opts)
            det_den += 1
            first = int(np.argmax(alerts)) + 1 if alerts.any() else None
            if first is not None:
                det_num += 1
                delays.append(first)
            per_class.append(OrderedDict([
                ("config", label),
                ("attack_class", aclass),
                ("node", node),
                ("impact", asset["impact"]),
                ("gamma", gamma),
                ("windows", n_win),
                ("mean_P", round(float(p[s:e + 1].mean()), 4)),
                ("detected", int(first is not None)),
                ("attack_to_alert_windows", first if first is not None else ""),
            ]))

        # false alarms on Normal windows, worst case = highest-impact node
        wc_node = cfg["worst_case_node_for_far"]
        wc = assets[wc_node]
        normal_mask = cls == "Normal"
        n_normal = int(normal_mask.sum())
        if n_normal:
            fa = run_pipeline(p[normal_mask], wc["impact"], gsched[wc["phase"]],
                              tau, K, **opts)
            n_fa = int(fa.sum())
            far = 100.0 * n_fa / n_normal
            per_1000 = 1000.0 * n_fa / n_normal
        else:
            n_fa, far, per_1000 = 0, float("nan"), float("nan")

        summary.append(OrderedDict([
            ("config", label),
            ("description", CONFIG_NAMES[label]),
            ("episodes_evaluated", det_den),
            ("detection_rate_%", round(100.0 * det_num / det_den, 2) if det_den else float("nan")),
            ("mean_attack_to_alert_windows", round(float(np.mean(delays)), 2) if delays else ""),
            ("normal_windows", n_normal),
            ("false_alerts", n_fa),
            ("FAR_%", round(far, 4)),
            ("false_alerts_per_1000_normal", round(per_1000, 3)),
        ]))

    if skipped:
        sys.stderr.write("note: %d episode(s) skipped (< %d windows): %s\n"
                         % (len(skipped), min_windows, ", ".join(sorted(set(skipped)))))
    return per_class, summary


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #

def load_trace(path, cfg):
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        data = {k: z[k] for k in z.files}
        if pd is None:
            raise SystemExit("pandas is required for NPZ input")
        df = pd.DataFrame({
            "p": data["p"].astype(float),
            "y": data.get("y", np.zeros(len(data["p"]))).astype(int),
            "attack_class": data["attack_class"].astype(str),
        })
        if "node" in data:
            df["node"] = data["node"].astype(str)
    else:
        if pd is None:
            raise SystemExit("pandas is required for CSV input")
        df = pd.read_csv(path)

    missing = {"p", "attack_class"} - set(df.columns)
    if missing:
        raise SystemExit("trace is missing column(s): %s" % ", ".join(sorted(missing)))
    if "node" not in df.columns:
        routing = cfg["routing"]
        unknown = sorted(set(df["attack_class"]) - set(routing) - {"Normal"})
        if unknown:
            raise SystemExit("no routing entry for attack class(es): %s" % ", ".join(unknown))
        df["node"] = [routing.get(c, cfg["worst_case_node_for_far"]) for c in df["attack_class"]]
    return df


def make_demo(n_normal=6000, seed=42):
    """Synthetic trace: benign windows plus five attack episodes."""
    rng = np.random.default_rng(seed)
    p = list(rng.beta(1.2, 40.0, n_normal))
    cls = ["Normal"] * n_normal
    episodes = [("DDoS_ICMP", 400), ("SQL_injection", 150),
                ("Uploading", 60), ("Ransomware", 40), ("MITM", 80)]
    for name, n in episodes:
        p += list(np.clip(rng.beta(9.0, 1.2, n), 0, 1))
        cls += [name] * n
        p += list(rng.beta(1.2, 40.0, 300))
        cls += ["Normal"] * 300
    if pd is None:
        raise SystemExit("pandas is required for --demo")
    return pd.DataFrame({"p": p, "y": [0 if c == "Normal" else 1 for c in cls],
                         "attack_class": cls})


def to_markdown(rows):
    if not rows:
        return ""
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def write_csv(rows, path):
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", help="CSV or NPZ with columns p, attack_class [, node, y]")
    ap.add_argument("--config", help="JSON config (assets, gamma_schedule, routing)")
    ap.add_argument("--write-config", metavar="PATH", help="write a template config and exit")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--min-windows", type=int, default=None,
                    help="skip episodes shorter than this (default: K)")
    ap.add_argument("--out", default="table7_ablation.csv", help="summary CSV path")
    ap.add_argument("--out-per-class", default=None, help="optional per-episode CSV path")
    ap.add_argument("--demo", action="store_true", help="run on a synthetic trace")
    a = ap.parse_args()

    if a.write_config:
        with open(a.write_config, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        print("template written to %s" % a.write_config)
        return

    cfg = DEFAULT_CONFIG
    if a.config:
        with open(a.config, encoding="utf-8") as f:
            cfg = json.load(f, object_pairs_hook=OrderedDict)

    if a.demo:
        df = make_demo()
        routing = cfg["routing"]
        df["node"] = [routing.get(c, cfg["worst_case_node_for_far"]) for c in df["attack_class"]]
    elif a.trace:
        df = load_trace(a.trace, cfg)
    else:
        raise SystemExit("give --trace PATH (or --demo)")

    min_windows = a.K if a.min_windows is None else a.min_windows
    per_class, summary = evaluate(df, cfg, a.tau, a.K, min_windows)

    print("\nDyRA-IIoT component ablation  (tau = %.2f, K = %d, windows = %d)\n"
          % (a.tau, a.K, len(df)))
    print(to_markdown(summary))
    print()

    write_csv(summary, a.out)
    print("summary written to %s" % a.out)
    if a.out_per_class:
        write_csv(per_class, a.out_per_class)
        print("per-episode detail written to %s" % a.out_per_class)


if __name__ == "__main__":
    main()
