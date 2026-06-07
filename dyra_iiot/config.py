"""
dyra_iiot/config.py
─────────────────────────────────────────────────────────────────────────────
Central configuration.  Every hyperparameter used in the paper is defined
here with a reference to the section/table where it appears.

Import pattern:
    from dyra_iiot.config import CFG
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Tuple
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Experiment protocol  (Section 4.1.6)
# ─────────────────────────────────────────────────────────────────────────────
SEEDS: List[int] = [42, 123, 456, 789, 2024]

# OOD held-out attack categories (Section 4.1.5 / Table 11)
OOD_CLASSES_EDGE: Set[str] = {"MITM", "Ransomware", "Backdoor", "Port_Scanning"}
# Identical category names in TON_IoT (Section 4.10)
OOD_CLASSES_TON: Set[str]  = {"mitm", "ransomware", "backdoor", "scanning"}


# ─────────────────────────────────────────────────────────────────────────────
# Partitioning & windowing  (Section 4.1.3–4.1.4)
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_RATIO:  float = 0.80
WINDOW_LEN:   int   = 50    # L = 50 packets per window
STRIDE:       int   = 1


# ─────────────────────────────────────────────────────────────────────────────
# Training protocol  (Section 4.1.6)
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE:   int   = 64
EPOCHS:       int   = 10
LR:           float = 5e-4
LR_PATIENCE:  int   = 2
LR_FACTOR:    float = 0.5
LR_MIN:       float = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Backbone hyperparameters  (Section 3.1.3)
# ─────────────────────────────────────────────────────────────────────────────
# TiDE — 627 K parameters (with F=36, L=50)
TIDE_HIDDEN:   int   = 256
TIDE_N_BLOCKS: int   = 2
TIDE_DROPOUT:  float = 0.1

# 1D-CNN — 20 K parameters
CNN_FILTERS:   int   = 64
CNN_KERNEL:    int   = 3
CNN_DROPOUT:   float = 0.1

# LSTM — 217 K parameters
LSTM_HIDDEN:   int   = 128
LSTM_LAYERS:   int   = 2
LSTM_DROPOUT:  float = 0.1

# DLinear — 231 K parameters
DLINEAR_DIM:    int  = 64
DLINEAR_KERNEL: int  = 25   # moving-average kernel size
DLINEAR_DROPOUT: float = 0.1

# Vanilla-Transformer — 203 K parameters
VT_DMODEL:   int   = 128
VT_HEADS:    int   = 4
VT_FF_DIM:   int   = 512
VT_DROPOUT:  float = 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Risk pipeline defaults  (Section 3.4–3.5)
# ─────────────────────────────────────────────────────────────────────────────
RISK_K:     int   = 3    # K-consecutive-exceedance windows
RISK_TAU:   float = 0.5  # critical threshold τ_critical


# ─────────────────────────────────────────────────────────────────────────────
# Fuzzy SAW weights  (Section 3.2, Table 2)
# ─────────────────────────────────────────────────────────────────────────────
FUZZY_WEIGHTS: Dict[str, float] = {
    "hardware":      0.3,
    "software":      0.3,
    "communication": 0.4,
}

# Linguistic term → defuzzified value (Table 1)
FUZZY_CRISP: Dict[str, float] = {
    "VL": 0.10,   # Very Low
    "L":  0.30,   # Low
    "M":  0.50,   # Medium
    "H":  0.70,   # High
    "VH": 0.90,   # Very High
}

# Five-node simulation topology  (Section 4.4, Table 2)
# Each node: {name, hw_score, sw_score, comm_score}
# hw/sw/comm values are the exact defuzzified scores from Table 2
# (hw_label etc. stored for reference; the crisp scores are used directly)
NODE_TOPOLOGY = [
    {"id": "N1", "name": "Edge Gateway",   "hw": 0.80, "sw": 0.70, "comm": 0.90},  # VH, H,  VH
    {"id": "N2", "name": "SCADA Server",   "hw": 0.90, "sw": 0.90, "comm": 0.90},  # VH, VH, VH
    {"id": "N3", "name": "HMI Terminal",   "hw": 0.50, "sw": 0.50, "comm": 0.70},  # M,  M,  H
    {"id": "N4", "name": "Sensor Array",   "hw": 0.20, "sw": 0.10, "comm": 0.30},  # L,  VL, L  (expert score 0.20 for hw)
    {"id": "N5", "name": "Actuator PLC",   "hw": 0.80, "sw": 0.70, "comm": 0.90},  # VH, H,  VH
]


# ─────────────────────────────────────────────────────────────────────────────
# Operational context factor γ(t)  (Section 3.3, Table 3)
# ─────────────────────────────────────────────────────────────────────────────
GAMMA_MAP: Dict[str, float] = {
    "maintenance": 0.4,
    "standby":     0.6,
    "production":  1.0,
    "shift_change": 1.2,
    "critical_process": 1.5,
}


# ─────────────────────────────────────────────────────────────────────────────
# Edge deployment  (Section 4.9)
# ─────────────────────────────────────────────────────────────────────────────
QUANTIZE_BACKBONE: str = "TiDE"   # reference backbone for edge export
INT8_CALIBRATION_SAMPLES: int = 512


# ─────────────────────────────────────────────────────────────────────────────
# Convenience accessor
# ─────────────────────────────────────────────────────────────────────────────
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
