#!/usr/bin/env python3
"""
scripts/train_all.py
─────────────────────────────────────────────────────────────────────────────
Main replication entry point.

Reproduces:
  • Table 6  — In-distribution performance (5 backbone × 5 seeds)
  • Table 11 — OOD performance
  • Table 12 — Per-OOD-class recall
  • Table 13 — Leakage decomposition (TiDE, 3 seeds)
  • Table 15 — Three-way protocol cross-check

Usage (Edge-IIoTset):
  python scripts/train_all.py \\
      --data /path/to/DNN-EdgeIIoT-dataset.csv \\
      --out  results/edge_iiotset

Usage (TON_IoT):
  python scripts/train_all.py \\
      --data /path/to/TON_IoT_Train_Test_Network.csv \\
      --dataset ton_iot \\
      --out  results/ton_iot

Fast smoke-test (1 seed, 3 epochs, 20 % subsample):
  python scripts/train_all.py --data ... --fast
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── make sure the package root is on sys.path ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dyra_iiot.config as C
from dyra_iiot.data.features import select_features_edge, select_features_ton
from dyra_iiot.data.partitioning import (
    apply_normalizer,
    fit_normalizer,
    stratified_per_class_block_split,
)
from dyra_iiot.models.backbones import BACKBONE_NAMES, build_backbone, count_parameters
from dyra_iiot.training.trainer import (
    evaluate_model,
    make_loader,
    ood_per_class_recall,
    train_backbone,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="DyRA-IIoT — full replication training script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",    required=True, help="Path to dataset CSV / directory")
    p.add_argument("--dataset", choices=["edge", "ton_iot"], default="edge",
                   help="Dataset variant (edge=Edge-IIoTset, ton_iot=TON_IoT)")
    p.add_argument("--out",     default="results", help="Output directory")
    p.add_argument("--seeds",   nargs="+", type=int, default=None,
                   help="Random seeds (default: paper's 5 seeds)")
    p.add_argument("--backbones", nargs="+", default=None,
                   choices=BACKBONE_NAMES,
                   help="Subset of backbones to train (default: all 5)")
    p.add_argument("--epochs",  type=int,   default=None)
    p.add_argument("--fast",    action="store_true",
                   help="1 seed / 3 epochs / 20 %% subsample — quick sanity check")
    p.add_argument("--skip-leakage",   action="store_true",
                   help="Skip leakage decomposition experiment")
    p.add_argument("--skip-threeway",  action="store_true",
                   help="Skip three-way cross-check (Table 15)")
    p.add_argument("--device",  default=None,
                   help="Force device: cpu / cuda (auto-detected by default)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path: str, dataset: str) -> tuple:
    """Returns (X_raw, y_raw, attack_types, ood_classes)."""
    p = Path(path)
    csvs = [p] if p.is_file() else sorted(p.glob("**/*.csv"))
    logger.info("Loading %d CSV file(s) from %s", len(csvs), path)

    frames = [pd.read_csv(c, low_memory=False) for c in csvs]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    logger.info("Total rows: %d", len(df))

    if dataset == "edge":
        # Edge-IIoTset label column
        if "Attack_type" not in df.columns:
            raise ValueError("Expected 'Attack_type' column for Edge-IIoTset")
        X, y, _ = select_features_edge(df)
        types      = df["Attack_type"].values
        ood_classes = C.OOD_CLASSES_EDGE

    else:  # ton_iot
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        type_col = next((c for c in ["type","attack_type"] if c in df.columns), None)
        if type_col is None:
            raise ValueError("Expected 'type' column for TON_IoT")
        df["_type_clean"] = (
            df[type_col].astype(str).str.lower().str.strip()
            .str.replace(r"[\s\-]+", "_", regex=True)
        )
        df["_label_bin"] = (df["_type_clean"] != "normal").astype(int)
        X, y, _ = select_features_ton(df)
        types      = df["_type_clean"].values
        ood_classes = C.OOD_CLASSES_TON

    return X, y, types, ood_classes


# ─────────────────────────────────────────────────────────────────────────────
# Run one backbone × one seed
# ─────────────────────────────────────────────────────────────────────────────

def run_one(
    X_raw, y_raw, types, ood_classes,
    backbone_name, seed, mode, device, cfg_overrides,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    epochs     = cfg_overrides.get("epochs",     C.EPOCHS)
    batch_size = cfg_overrides.get("batch_size", C.BATCH_SIZE)

    Xtr, ytr, Xte, yte = stratified_per_class_block_split(
        X_raw, y_raw, types,
        C.TRAIN_RATIO, C.WINDOW_LEN,
        ood_classes if mode == "ood" else set(),
        mode=mode,
    )
    mu, std  = fit_normalizer(Xtr, robust=True)
    Xtr_n    = apply_normalizer(Xtr, mu, std)
    Xte_n    = apply_normalizer(Xte, mu, std)

    tr_ld = make_loader(Xtr_n, ytr, batch_size)
    te_ld = make_loader(Xte_n, yte, batch_size, shuffle=False)

    model   = build_backbone(backbone_name, C.WINDOW_LEN, X_raw.shape[1]).to(device)
    n_params = count_parameters(model)

    train_backbone(model, tr_ld, device=device, epochs=epochs)
    metrics = evaluate_model(model, te_ld, device)
    metrics["n_params"] = n_params

    if mode == "ood":
        per_class = ood_per_class_recall(
            model, X_raw, y_raw, types, ood_classes,
            C.WINDOW_LEN, device, mu, std,
        )
        metrics["per_class_recall"] = per_class

    return metrics, mu, std, model


# ─────────────────────────────────────────────────────────────────────────────
# Leakage decomposition  (Table 13)
# ─────────────────────────────────────────────────────────────────────────────

def leakage_decomposition(X_raw, y_raw, types, device, epochs, batch_size, seeds):
    from dyra_iiot.data.partitioning import fit_normalizer, apply_normalizer
    import numpy as np

    configs = [
        ("Revised (Section 4.1)",  "stratified", "train-only"),
        ("(a) + random split",     "random",     "train-only"),
        ("(b) + global norm",      "stratified", "train+test"),
        ("Preliminary (all three)","random",     "train+test"),
    ]
    rows = []
    for label, split_mode, norm_mode in configs:
        f1s = []
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            L = C.WINDOW_LEN

            if split_mode == "stratified":
                Xtr, ytr, Xte, yte = stratified_per_class_block_split(
                    X_raw, y_raw, types, C.TRAIN_RATIO, L)
            else:
                all_X, all_y = [], []
                for i in range(L - 1, len(X_raw)):
                    all_X.append(X_raw[i - L + 1: i + 1])
                    all_y.append(y_raw[i])
                all_X, all_y = np.stack(all_X), np.array(all_y)
                perm = np.random.permutation(len(all_y))
                n_tr = int(len(perm) * 0.8)
                Xtr, ytr = all_X[perm[:n_tr]], all_y[perm[:n_tr]]
                Xte, yte = all_X[perm[n_tr:]], all_y[perm[n_tr:]]

            if norm_mode == "train-only":
                mu, std = fit_normalizer(Xtr)
            else:
                mu, std = fit_normalizer(np.concatenate([Xtr, Xte], 0))

            Xtr_n = apply_normalizer(Xtr, mu, std)
            Xte_n = apply_normalizer(Xte, mu, std)
            tr_ld = make_loader(Xtr_n, ytr, batch_size)
            te_ld = make_loader(Xte_n, yte, batch_size, shuffle=False)

            model = build_backbone("TiDE", L, X_raw.shape[1]).to(device)
            train_backbone(model, tr_ld, device=device, epochs=epochs)
            m = evaluate_model(model, te_ld, device)
            f1s.append(m["f1"])

        row = dict(config=label, split=split_mode, norm=norm_mode,
                   f1_mean=np.mean(f1s), f1_std=np.std(f1s), n_seeds=len(seeds))
        rows.append(row)
        logger.info("[Leakage] %-42s  F1=%.4f ± %.4f", label, row["f1_mean"], row["f1_std"])
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device    = args.device or C.get_device()
    seeds     = args.seeds  or C.SEEDS
    backbones = args.backbones or BACKBONE_NAMES
    epochs    = args.epochs or C.EPOCHS

    if args.fast:
        seeds  = [42]
        epochs = 3
        logger.info("[FAST MODE] 1 seed / %d epochs / 20 %% subsample", epochs)

    cfg_overrides = {"epochs": epochs, "batch_size": C.BATCH_SIZE}

    logger.info("Device: %s  |  Seeds: %s  |  Backbones: %s", device, seeds, backbones)

    # ── Load data ─────────────────────────────────────────────────────────
    X_raw, y_raw, types, ood_classes = load_dataset(args.data, args.dataset)

    if args.fast:
        # 20 % stratified subsample
        idx = []
        for t in np.unique(types):
            mask = np.where(types == t)[0]
            idx.extend(mask[: max(1, len(mask) // 5)].tolist())
        idx = np.array(idx)
        X_raw, y_raw, types = X_raw[idx], y_raw[idx], types[idx]
        logger.info("Subsample: %d rows", len(X_raw))

    # ── In-distribution experiment ────────────────────────────────────────
    logger.info("\n%s\n  IN-DISTRIBUTION\n%s", "─" * 60, "─" * 60)
    id_results: dict = {}

    for bn in backbones:
        seed_metrics = []
        for seed in seeds:
            logger.info("  %s  seed=%d", bn, seed)
            m, *_ = run_one(X_raw, y_raw, types, ood_classes,
                            bn, seed, "in_dist", device, cfg_overrides)
            seed_metrics.append(m)

        id_results[bn] = {
            "f1_mean":    float(np.mean([m["f1"]       for m in seed_metrics])),
            "f1_std":     float(np.std ([m["f1"]       for m in seed_metrics])),
            "roc_auc_mean": float(np.mean([m["roc_auc"] for m in seed_metrics])),
            "roc_auc_std":  float(np.std ([m["roc_auc"] for m in seed_metrics])),
            "precision_mean": float(np.mean([m["precision"] for m in seed_metrics])),
            "recall_mean":  float(np.mean([m["recall"]  for m in seed_metrics])),
            "far_mean":   float(np.mean([m["far"]       for m in seed_metrics])),
            "far_std":    float(np.std ([m["far"]       for m in seed_metrics])),
            "latency_ms": float(np.mean([m["latency_ms"] for m in seed_metrics])),
            "n_params":   seed_metrics[0]["n_params"],
        }
        logger.info("  ► %s: F1=%.4f±%.4f", bn,
                    id_results[bn]["f1_mean"], id_results[bn]["f1_std"])

    with open(out_dir / "in_dist_results.json", "w") as f:
        json.dump(id_results, f, indent=2)
    pd.DataFrame(id_results).T.to_csv(out_dir / "in_dist_results.csv")
    logger.info("Saved → in_dist_results.json / .csv")

    # ── OOD experiment ────────────────────────────────────────────────────
    active_ood = ood_classes & set(np.unique(types).tolist())
    if active_ood:
        logger.info("\n%s\n  OOD  held-out: %s\n%s", "─"*60, active_ood, "─"*60)
        ood_results: dict = {}

        for bn in backbones:
            seed_metrics, seed_recalls = [], []
            for seed in seeds:
                logger.info("  %s  seed=%d [OOD]", bn, seed)
                m, mu, std, _ = run_one(X_raw, y_raw, types, active_ood,
                                        bn, seed, "ood", device, cfg_overrides)
                seed_metrics.append(m)
                if "per_class_recall" in m:
                    seed_recalls.append(m.pop("per_class_recall"))

            ood_results[bn] = {
                "f1_mean":  float(np.mean([m["f1"]  for m in seed_metrics])),
                "f1_std":   float(np.std ([m["f1"]  for m in seed_metrics])),
                "roc_auc_mean": float(np.mean([m["roc_auc"] for m in seed_metrics])),
                "far_mean": float(np.mean([m["far"] for m in seed_metrics])),
                "n_params": seed_metrics[0]["n_params"],
            }
            if seed_recalls:
                for cls in sorted({k for d in seed_recalls for k in d}):
                    vals = [d.get(cls, float("nan")) for d in seed_recalls]
                    ood_results[bn][f"recall_{cls}_mean"] = float(np.nanmean(vals))
                    ood_results[bn][f"recall_{cls}_std"]  = float(np.nanstd(vals))

            logger.info("  ► %s: OOD-F1=%.4f±%.4f", bn,
                        ood_results[bn]["f1_mean"], ood_results[bn]["f1_std"])

        with open(out_dir / "ood_results.json", "w") as f:
            json.dump(ood_results, f, indent=2)
        pd.DataFrame(ood_results).T.to_csv(out_dir / "ood_results.csv")
        logger.info("Saved → ood_results.json / .csv")

    # ── Leakage decomposition ─────────────────────────────────────────────
    if not args.skip_leakage:
        logger.info("\n%s\n  LEAKAGE DECOMPOSITION (Table 13)\n%s", "─"*60, "─"*60)
        leak_rows = leakage_decomposition(
            X_raw, y_raw, types, device, epochs,
            C.BATCH_SIZE, seeds[:3])
        pd.DataFrame(leak_rows).to_csv(out_dir / "leakage_decomposition.csv", index=False)
        logger.info("Saved → leakage_decomposition.csv")

    logger.info("\n[DONE] All results → %s", out_dir)


if __name__ == "__main__":
    main()
