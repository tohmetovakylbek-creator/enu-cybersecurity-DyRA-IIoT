"""
build_tables.py — Aggregates per-seed JSON metrics into final paper tables.

Reads:
    artifacts/metrics/seed_NN.json                  — TiDE in-distribution (5 seeds)
    artifacts/metrics/{cnn,lstm,dlinear,transformer}_seedNN.json — baselines in-dist
    artifacts/metrics/ood_{model}_seedNN.json       — OOD for all 5 models

Outputs:
    artifacts/metrics/table6_indist.csv             — Table 6 (in-distribution)
    artifacts/metrics/table10_ood.csv               — Table 10 (OOD comparison)
    Prints all tables to stdout in markdown format ready for the paper.

Usage:
    python build_tables.py
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np

METRICS_DIR = Path("artifacts/metrics")

SEEDS = [42, 123, 456, 789, 2024]
MODELS_IN_ORDER = ["tide", "cnn", "lstm", "dlinear", "transformer"]
OOD_HELD_OUT = ["MITM", "Ransomware", "Backdoor", "Port_Scanning"]


def load_indist_for_model(model: str) -> list[dict]:
    """Returns list of per-seed dicts for in-distribution metrics."""
    results = []
    for seed in SEEDS:
        # TiDE uses base seed_NN.json naming (from train_v2.py)
        if model == "tide":
            path = METRICS_DIR / f"seed_{seed}.json"
        else:
            path = METRICS_DIR / f"{model}_seed{seed}.json"
        if not path.exists():
            print(f"  [MISSING] {path}")
            continue
        with open(path, "r") as f:
            d = json.load(f)
        results.append(d)
    return results


def load_ood_for_model(model: str) -> list[dict]:
    results = []
    for seed in SEEDS:
        path = METRICS_DIR / f"ood_{model}_seed{seed}.json"
        if not path.exists():
            print(f"  [MISSING] {path}")
            continue
        with open(path, "r") as f:
            d = json.load(f)
        results.append(d)
    return results


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def build_indist_table():
    print("\n" + "=" * 80)
    print("TABLE 6: IN-DISTRIBUTION EVALUATION (stratified per-class-block)")
    print("=" * 80)

    rows = []
    for model in MODELS_IN_ORDER:
        runs = load_indist_for_model(model)
        if not runs:
            continue
        f1s = [r["final_metrics"]["f1"] for r in runs]
        aucs = [r["final_metrics"]["roc_auc"] for r in runs]
        fars = [r["final_metrics"]["far"] for r in runs]
        precs = [r["final_metrics"]["precision"] for r in runs]
        recs = [r["final_metrics"]["recall"] for r in runs]

        # TiDE results saved by train_v2.py have inference latency separately
        # Baselines save it in each per-seed JSON
        latencies = []
        for r in runs:
            if "inference_ms_per_batch" in r:
                latencies.append(r["inference_ms_per_batch"])

        n_params = runs[0]["n_params"]
        f1_m, f1_s = mean_std(f1s)
        auc_m, auc_s = mean_std(aucs)
        far_m, far_s = mean_std(fars)
        prec_m, prec_s = mean_std(precs)
        rec_m, rec_s = mean_std(recs)
        lat_m, lat_s = mean_std(latencies) if latencies else (0.0, 0.0)

        rows.append({
            "model": model,
            "n_seeds": len(runs),
            "n_params": n_params,
            "f1_mean": f1_m, "f1_std": f1_s,
            "auc_mean": auc_m, "auc_std": auc_s,
            "far_mean": far_m, "far_std": far_s,
            "prec_mean": prec_m, "prec_std": prec_s,
            "rec_mean": rec_m, "rec_std": rec_s,
            "lat_mean": lat_m, "lat_std": lat_s,
        })

    # Print markdown table
    print()
    print("| Model | Params | F1 (mean ± std) | ROC-AUC | Precision | Recall | FAR | Latency (ms/batch) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        params_str = f"{r['n_params']/1000:.0f}K"
        f1_str = f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}"
        auc_str = f"{r['auc_mean']:.4f} ± {r['auc_std']:.4f}"
        prec_str = f"{r['prec_mean']:.4f} ± {r['prec_std']:.4f}"
        rec_str = f"{r['rec_mean']:.4f} ± {r['rec_std']:.4f}"
        far_str = f"{r['far_mean']*100:.2f}% ± {r['far_std']*100:.2f}%"
        lat_str = f"{r['lat_mean']:.2f}" if r['lat_mean'] > 0 else "—"
        print(f"| {r['model'].upper()} | {params_str} | {f1_str} | {auc_str} | "
              f"{prec_str} | {rec_str} | {far_str} | {lat_str} |")

    # CSV out
    csv_path = METRICS_DIR / "table6_indist.csv"
    with open(csv_path, "w") as f:
        f.write("model,n_seeds,n_params,f1_mean,f1_std,auc_mean,auc_std,"
                "prec_mean,prec_std,rec_mean,rec_std,far_mean,far_std,lat_mean,lat_std\n")
        for r in rows:
            f.write(f"{r['model']},{r['n_seeds']},{r['n_params']},"
                    f"{r['f1_mean']:.6f},{r['f1_std']:.6f},"
                    f"{r['auc_mean']:.6f},{r['auc_std']:.6f},"
                    f"{r['prec_mean']:.6f},{r['prec_std']:.6f},"
                    f"{r['rec_mean']:.6f},{r['rec_std']:.6f},"
                    f"{r['far_mean']:.6f},{r['far_std']:.6f},"
                    f"{r['lat_mean']:.4f},{r['lat_std']:.4f}\n")
    print(f"\nCSV saved: {csv_path}")
    return rows


def build_ood_table():
    print("\n" + "=" * 80)
    print("TABLE 10: OUT-OF-DISTRIBUTION EVALUATION")
    print("Held-out attacks: " + ", ".join(OOD_HELD_OUT))
    print("=" * 80)

    rows = []
    perclass_rows = []
    for model in MODELS_IN_ORDER:
        runs = load_ood_for_model(model)
        if not runs:
            continue
        f1s = [r["final_metrics"]["f1"] for r in runs]
        aucs = [r["final_metrics"]["roc_auc"] for r in runs]
        fars = [r["final_metrics"]["far"] for r in runs]
        precs = [r["final_metrics"]["precision"] for r in runs]
        recs = [r["final_metrics"]["recall"] for r in runs]

        # Per-class recall on OOD classes; specificity for Normal
        per_class_recall = defaultdict(list)
        normal_spec = []
        for r in runs:
            for cls, info in r["per_class_metrics"].items():
                if cls == "Normal":
                    normal_spec.append(info["metric_value"])
                elif cls in OOD_HELD_OUT:
                    per_class_recall[cls].append(info["metric_value"])

        n_params = runs[0]["n_params"]
        f1_m, f1_s = mean_std(f1s)
        auc_m, auc_s = mean_std(aucs)
        far_m, _ = mean_std(fars)
        spec_m, _ = mean_std(normal_spec)

        row = {
            "model": model,
            "n_seeds": len(runs),
            "n_params": n_params,
            "f1_mean": f1_m, "f1_std": f1_s,
            "auc_mean": auc_m, "auc_std": auc_s,
            "far_mean": far_m,
            "normal_spec_mean": spec_m,
        }
        for cls in OOD_HELD_OUT:
            m, s = mean_std(per_class_recall.get(cls, []))
            row[f"{cls}_mean"] = m
            row[f"{cls}_std"] = s
        rows.append(row)

    print()
    print("| Model | Params | OOD F1 (mean ± std) | ROC-AUC | Normal spec. | FAR |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        params_str = f"{r['n_params']/1000:.0f}K"
        f1_str = f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}"
        auc_str = f"{r['auc_mean']:.4f} ± {r['auc_std']:.4f}"
        spec_str = f"{r['normal_spec_mean']:.4f}"
        far_str = f"{r['far_mean']*100:.2f}%"
        print(f"| {r['model'].upper()} | {params_str} | {f1_str} | {auc_str} | "
              f"{spec_str} | {far_str} |")

    print()
    print("Per-class OOD recall (held-out attacks):")
    print()
    header = "| Model | " + " | ".join(OOD_HELD_OUT) + " |"
    sep = "|---|" + "|".join(["---"] * len(OOD_HELD_OUT)) + "|"
    print(header)
    print(sep)
    for r in rows:
        cells = [r['model'].upper()]
        for cls in OOD_HELD_OUT:
            m = r.get(f"{cls}_mean", 0)
            s = r.get(f"{cls}_std", 0)
            cells.append(f"{m:.3f} ± {s:.3f}")
        print("| " + " | ".join(cells) + " |")

    # CSV out
    csv_path = METRICS_DIR / "table10_ood.csv"
    with open(csv_path, "w") as f:
        cols = ["model", "n_seeds", "n_params", "f1_mean", "f1_std",
                "auc_mean", "auc_std", "far_mean", "normal_spec_mean"]
        for cls in OOD_HELD_OUT:
            cols.append(f"{cls}_mean")
            cols.append(f"{cls}_std")
        f.write(",".join(cols) + "\n")
        for r in rows:
            vals = [str(r.get(c, "")) for c in cols]
            f.write(",".join(vals) + "\n")
    print(f"\nCSV saved: {csv_path}")
    return rows


def main():
    if not METRICS_DIR.exists():
        print(f"[ERROR] {METRICS_DIR} not found. Run training first.")
        return

    build_indist_table()
    build_ood_table()

    print("\n" + "=" * 80)
    print("DONE. CSV files saved to artifacts/metrics/")
    print("=" * 80)


if __name__ == "__main__":
    main()
