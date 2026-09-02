# DyRA-IIoT

Code and compact result artifacts for **“DyRA-IIoT: A Hybrid Framework for
Asset-Aware Dynamic Risk Assessment in IIoT Networks.”** This submission
snapshot is intentionally small: it contains experiment code, reported
per-seed metrics, normalization statistics, and verified FP32 ONNX exports.
Large PyTorch checkpoints are attached to the GitHub release instead of being
kept in Git history.

## Release assets

For tag `v1.0.0-submission`, download:

- `DyRA-IIoT-checkpoints-v1.0.0-submission.zip` — all supplied PyTorch
  checkpoints, including the no-prefix, `ood_`, and `3way_` families;
- `DyRA-IIoT-windows-v1.0.0-submission.zip` — the exact precomputed 80/20
  windows used by the Section 4.2 scripts.

Their SHA-256 values are recorded in `RELEASE_ASSETS.sha256`.

## Protocol-to-artifact map

| Filename pattern | Evaluation protocol | Manuscript |
|---|---|---|
| `<model>_seed<seed>_best.pt` (TiDE metrics use `seed_<seed>.json`) | Stratified per-class-block 80/20 protocol; checkpoint selected on the test partition, as disclosed | Section 4.2, Table 6 |
| `ood_<model>_seed<seed>_best.pt` | Train on Normal + 10 known attacks; evaluate on Normal + 4 held-out attack classes | Section 4.7, Tables 10–11 |
| `3way_<model>_seed<seed>_best.pt` | Chronological 60/20/20 split within every class block; validation-only checkpoint selection | Section 4.8.3, Table 12 |
| `best_<model>_edge.pt` | Edge export/profiling checkpoint for CNN, LSTM, DLinear, and Transformer | Section 4.9, Table 13 |

The reference TiDE export was produced from `tide_seed42_best.pt`; there is no
separate `best_tide_edge.pt`. The supplied `tide_seed42_fp32.onnx` was checked
against that checkpoint at the tensor-storage level. The release asset contains
all three checkpoint families.

## INT8 export scope

Legacy `*_int8.onnx` files whose model-definition provenance could not be
verified are not included. `export_quantize.py` performs a fresh export from the
final checkpoint definitions and applies static INT8 quantization using training
windows only. The two supplied FP32 ONNX models are retained because their
checkpoint provenance was verified from the submitted materials.

## Repository layout

```text
.
├── dataset_loader_v2.py      # leakage-aware preprocessing and windows
├── train_v2.py               # Section 4.2 protocol (TiDE)
├── baselines_v2.py           # Section 4.2 baseline backbones
├── ood_test.py               # Section 4.7 held-out-class protocol
├── train_three_way.py        # Section 4.8.3 validation protocol
├── export_quantize.py        # final-checkpoint ONNX/INT8 export
├── build_tables.py           # rebuild Tables 6 and 10 from JSON metrics
├── analysis/                 # risk, ablation, robustness analyses
├── artifacts/metrics/        # reported per-seed and aggregate results
├── artifacts/onnx/           # verified FP32 exports
├── artifacts/windows/        # normalization statistics only
└── scripts/verify_snapshot.py
```

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

The Edge-IIoTset CSV is not redistributed. Download
`DNN-EdgeIIoT-dataset.csv` from its official distribution and place it in the
repository root. Then build the leakage-aware windows:

```bash
python dataset_loader_v2.py
```

## Reproduce the three reported protocols

```bash
# Section 4.2 / Table 6
python train_v2.py --seed 42
python baselines_v2.py --model all --seed 42

# Section 4.7 / Tables 10–11
python ood_test.py --model all --seed 42

# Section 4.8.3 / Table 12
python train_three_way.py --model all --seed 42

# Re-aggregate saved Table 6 and Table 10 metrics
python build_tables.py
```

Run each command for seeds `42`, `123`, `456`, `789`, and `2024` to reproduce
the five-seed summaries. Use `--help` for the exact command-line options.

## Reproduce Table 9

Table 9 was recomputed from the window-level probability traces produced
by the five seed-42 backbone checkpoints. The reconstructed traces are
provided in:

`artifacts/table9_traces/`

This directory contains one NPZ file for each backbone:

- `cnn_seed42_table9.npz`
- `dlinear_seed42_table9.npz`
- `lstm_seed42_table9.npz`
- `tide_seed42_table9.npz`
- `transformer_seed42_table9.npz`

It also contains:

- `trace_manifest.json`, which records the provenance and structure of
  the reconstructed traces;
- `trace_validation.csv`, which reports the validation checks performed
  on the five traces.

To recompute the aggregate and class-level Table 9 results, run from the
repository root:

```bash
python analysis/table9/recalculate_table9.py

## Verification

After downloading and unpacking the release checkpoint asset into
`artifacts/checkpoints/`, run:

```bash
python scripts/verify_snapshot.py
python -m compileall -q .
```

`ARTIFACT_MANIFEST.sha256` records checksums of compact artifacts in this tagged
snapshot. The checkpoint release archive has its own checksum.

## Citation and license

Please cite the manuscript using [`CITATION.cff`](CITATION.cff). Code is
released under the MIT License; dataset licensing remains with its authors.
