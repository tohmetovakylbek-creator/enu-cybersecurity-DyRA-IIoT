# DyRA-IIoT

**A Hybrid Framework for Asset-Aware Dynamic Risk Assessment in IIoT Networks**

[![CI](https://github.com/enu-cybersec/DyRA-IIoT/actions/workflows/ci.yml/badge.svg)](https://github.com/enu-cybersec/DyRA-IIoT/actions)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official replication code for the paper:

> **DyRA-IIoT: A Hybrid Framework for Asset-Aware Dynamic Risk Assessment in IIoT Networks**  
> Akylbek Tokhmetov, Liliya Tanchenko, Mansiya Kantureyeva, Ainagul Alimagambetova  
> *Journal of Information Security and Applications*, 2026  
> DOI: [pending]

---

## Overview

DyRA-IIoT continuously updates a risk signal

```
R(t) = P(t) × Impact(A) × γ(t)
```

by combining three components:

| Component | Description | Section |
|-----------|-------------|---------|
| **P(t)** | Per-window attack probability from any sequence-aware backbone | §3.1 |
| **Impact(A)** | Fuzzy SAW asset-criticality score | §3.2 |
| **γ(t)** | ISA-95 operational context factor ∈ [0.4, 1.5] | §3.3 |

Five backbone architectures are evaluated: TiDE, 1D-CNN, LSTM, DLinear, Vanilla-Transformer.

---

## Repository structure

```
DyRA-IIoT/
├── dyra_iiot/
│   ├── config.py              # All hyperparameters (mirrors paper Tables 1–3, 6)
│   ├── data/
│   │   ├── features.py        # 36-feature schema (Table 5) + TON_IoT auto-select
│   │   └── partitioning.py    # Algorithm 1 — stratified per-class-block split
│   ├── models/
│   │   └── backbones.py       # TiDE, 1D-CNN, LSTM, DLinear, Vanilla-Transformer
│   ├── training/
│   │   └── trainer.py         # Training loop, evaluation, per-class OOD recall
│   ├── risk/
│   │   └── pipeline.py        # Fuzzy SAW, γ(t), R(t) = P×Impact×γ, K-alerting
│   └── deployment/
│       └── quantize.py        # ONNX export + INT8 static quantization
├── scripts/
│   ├── train_all.py           # ★ Main entry point — reproduces Tables 6, 11–13
│   └── quantize_edge.py       # INT8 export + benchmark (Table 17)
├── tests/
│   ├── test_partitioning.py   # Algorithm 1 unit tests (8 cases)
│   └── test_models_and_risk.py# Backbone + risk pipeline tests (20 cases)
├── configs/
│   ├── edge_iiotset.yaml      # Dataset-specific overrides
│   └── ton_iot.yaml
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Datasets

| Dataset | Source | Size | Used for |
|---------|--------|------|----------|
| **Edge-IIoTset** | [IEEE DataPort](https://ieee-dataport.org/open-access/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot) | 2.2 M packets | Main experiments (Tables 6, 11–17) |
| **TON_IoT** | [UNSW](https://research.unsw.edu.au/projects/toniot-datasets) | 461 K records | Cross-dataset validation (Tables 18–22) |

Download `DNN-EdgeIIoT-dataset.csv` and `TON_IoT_Train_Test_Network.csv` and place them anywhere accessible.

---

## Quick start

### Option A — pip (local)

```bash
git clone https://github.com/enu-cybersec/DyRA-IIoT.git
cd DyRA-IIoT
pip install -e .

# Smoke test (1 seed, 3 epochs, 20 % subsample — ~5 min on CPU)
python scripts/train_all.py \
    --data /path/to/DNN-EdgeIIoT-dataset.csv \
    --fast

# Full replication — 5 seeds × 5 backbones (~24 h on RTX-class GPU)
python scripts/train_all.py \
    --data /path/to/DNN-EdgeIIoT-dataset.csv \
    --out  results/edge_iiotset
```

### Option B — Docker (GPU)

```bash
# Build image
docker build -t dyra-iiot .

# Run with GPU
docker run --gpus all \
    -v /your/data:/data \
    -v $(pwd)/results:/results \
    dyra-iiot \
    python scripts/train_all.py --data /data/DNN-EdgeIIoT-dataset.csv --out /results/edge

# Or use docker compose
DATA_DIR=/your/data RESULTS_DIR=$(pwd)/results \
    docker compose run --rm dyra-gpu \
    python scripts/train_all.py --data /data/DNN-EdgeIIoT-dataset.csv
```

### Option C — CPU-only Docker

```bash
# Build CPU image (see docker-compose.yml for Dockerfile.cpu template)
docker build -f Dockerfile.cpu -t dyra-iiot-cpu .
docker run -v /your/data:/data dyra-iiot-cpu \
    python scripts/train_all.py --data /data/DNN-EdgeIIoT-dataset.csv --fast
```

---

## Reproducing paper tables

### Table 6 — In-distribution performance (Edge-IIoTset)

```bash
python scripts/train_all.py \
    --data  /path/to/DNN-EdgeIIoT-dataset.csv \
    --out   results/edge_iiotset \
    --skip-threeway    # optional: skip 3-way cross-check
```

Output: `results/edge_iiotset/in_dist_results.csv`

### Tables 11–12 — OOD evaluation

Automatically generated alongside in-distribution results (no extra flags needed).  
Output: `results/edge_iiotset/ood_results.csv`

### Table 13 — Leakage decomposition

```bash
python scripts/train_all.py \
    --data /path/to/DNN-EdgeIIoT-dataset.csv \
    --out  results/edge_iiotset
# leakage_decomposition.csv is produced by default
```

To skip: add `--skip-leakage`.

### Table 17 — INT8 edge deployment

```bash
# First, train and save a checkpoint
python scripts/train_all.py --data ... --out results/edge_iiotset

# Then quantize
python scripts/quantize_edge.py \
    --checkpoint results/edge_iiotset/TiDE_seed42.pt \
    --data       /path/to/DNN-EdgeIIoT-dataset.csv \
    --out        results/quantized
```

### Tables 18–22 — Cross-dataset validation (TON_IoT)

```bash
python scripts/train_all.py \
    --data    /path/to/TON_IoT_Train_Test_Network.csv \
    --dataset ton_iot \
    --out     results/ton_iot
```

---

## Selected results

### In-distribution F1 (Table 6, 5 seeds, Edge-IIoTset)

| Backbone | F1 | ROC-AUC | FAR | Lat (ms/batch) |
|---|---|---|---|---|
| 1D-CNN | 0.9905 ± 0.0057 | 0.9965 | 0.69% | **0.17** |
| DLinear | 0.9956 ± 0.0015 | 0.9988 | 0.29% | 0.22 |
| TiDE | 0.9867 ± 0.0047 | **0.9992** | 0.95% | 0.41 |
| Vanilla-Transformer | **0.9959 ± 0.0018** | 0.9990 | **0.18%** | 0.46 |
| LSTM | 0.9914 ± 0.0029 | 0.9988 | 0.63% | 1.01 |

### OOD F1 — held-out MITM, Ransomware, Backdoor, Port_Scanning (Table 11)

| Backbone | OOD F1 | Normal specificity |
|---|---|---|
| **Vanilla-Transformer** | **0.9869 ± 0.0024** | 0.9981 |
| 1D-CNN | 0.9804 ± 0.0041 | 0.9982 |
| LSTM | 0.9447 ± 0.0177 | 0.9942 |
| TiDE | 0.9071 ± 0.0147 | 0.9981 |
| DLinear | 0.8802 ± 0.0220 | 0.9982 |

### Edge deployment — INT8 (Table 17)

| Platform | Quantization | Lat (ms/win) | Size (MB) | F1 |
|---|---|---|---|---|
| RTX 5060 (baseline) | FP32 | 2.4 | 1.5 | 0.987 |
| Jetson Nano | INT8 | **0.08** | 0.4 | 0.972 |
| Raspberry Pi 4 | INT8 | 0.31 | 0.4 | 0.979 |
| STM32H7 (d_h=64) | INT8 | 4.2 | 0.34 | 0.973 |

---

## Risk pipeline API

```python
from dyra_iiot.risk.pipeline import DyRAPipeline, compute_impact, get_gamma

# Asset impact: SCADA server (hw=VH, sw=VH, comm=VH) → 0.90
impact = compute_impact("VH", "VH", "VH")

# Context factor during shift change-over
gamma = get_gamma("shift_change")   # 1.2

# Build pipeline for this asset
pipeline = DyRAPipeline(impact=impact, gamma=gamma, K=3, tau=0.5)

# Apply to a threat-probability trace from any backbone
import numpy as np
P_trace = np.random.rand(300).astype(np.float32)
alerts  = pipeline(P_trace)          # 0/1 alert array, shape (300,)
```

---

## Running tests

```bash
pytest tests/ -v --tb=short

# With coverage
pytest tests/ --cov=dyra_iiot --cov-report=html
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Citation

```bibtex
@article{tokhmetov2025dyra,
  title   = {{DyRA-IIoT}: A Hybrid Framework for Asset-Aware Dynamic Risk
             Assessment in {IIoT} Networks},
  author  = {Tokhmetov, Akylbek and Tanchenko, Liliya and
Kantureyeva, Mansiya  and Alimagambetova, Ainagul},
  journal = {Journal of Information Security and Applications},
  year    = {2026},
  doi     = {pending}
}
```

---

## Contact

Akylbek Tokhmetov · Department of Information Systems · L.N. Gumilyov ENU · Astana, Kazakhstan  
tokhmetov_ab@enu.kz
