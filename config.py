"""
Centralized configuration for TiDE-SAW experiments.

All hyperparameters and paths live here. Importing from this module is the
ONLY way other scripts should access these values — no hardcoded constants
elsewhere. This is what makes Methods reproducible.
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
# Project root — assumes config.py sits in the project root or one level deep.
PROJECT_ROOT = Path(__file__).resolve().parent
# CSV path will be auto-discovered by walking the project; см. utils.find_csv()
CSV_NAME = "DNN-EdgeIIoT-dataset.csv"

# Output directories
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
WINDOWS_DIR = ARTIFACTS_DIR / "windows"        # pre-computed window tensors
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"  # trained models
METRICS_DIR = ARTIFACTS_DIR / "metrics"          # JSON metrics per seed
LOGS_DIR = ARTIFACTS_DIR / "logs"                # training logs


# ============================================================================
# DATA PROTOCOL — must match what is written in Methods
# ============================================================================

# Chronological split (no shuffling at packet level).
# First TRAIN_RATIO of packets in time order -> train. Rest -> test.
TRAIN_RATIO = 0.80

# Sliding window
WINDOW_LEN = 50          # L = number of packets per window
WINDOW_STRIDE = 1        # stride=1 means every packet becomes the end of one window

# Label assignment: target = label of LAST packet in window
# (as opposed to predicting the next packet's label; see Methods)
LABEL_POSITION = "last"  # "last" or "next"

# Boundary handling: any window that spans the train/test split boundary
# must be discarded (no look-ahead leakage). See Methods 4.1.
PURGE_BOUNDARY_WINDOWS = True


# ============================================================================
# MODEL — TiDE architecture (must match Methods Section 3.2.1, Table 2)
# ============================================================================

MODEL_HIDDEN_DIM = 256       # d_h
MODEL_NUM_RESBLOCKS = 2      # number of residual MLP blocks
MODEL_DROPOUT = 0.1


# ============================================================================
# TRAINING — must match Methods Table 2
# ============================================================================

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.0           # Methods doesn't mention L2 reg; keep at 0

# Adam optimizer is used (Methods Table 2)

# LR scheduler: ReduceLROnPlateau
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 2
LR_SCHEDULER_MIN_LR = 1e-6

# Class-weighted BCE loss (Methods Section 3.2.1)
# Weights are computed automatically from train distribution as
# w_class = N_total / (N_classes * N_class)
USE_CLASS_WEIGHTED_BCE = True

# Seeds for multi-seed evaluation (Methods Section 4.2)
SEEDS = [42, 123, 456, 789, 2024]


# ============================================================================
# EVALUATION
# ============================================================================

DECISION_THRESHOLD = 0.5     # P(t) > 0.5 -> attack

# Bootstrap confidence intervals (Methods Table 6)
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_CI = 0.95


# ============================================================================
# HARDWARE / EFFICIENCY
# ============================================================================

# Strategy for handling 2.2M windows in 32 GB RAM:
# "in_memory"      -> pre-compute all windows, hold tensors in RAM (~18 GB float32)
# "in_memory_fp16" -> same but float16 (~9 GB)
# "on_the_fly"     -> generate windows on-demand in __getitem__ (~2 GB)
WINDOW_STRATEGY = "in_memory_fp16"

DEVICE = "cuda"              # "cuda" or "cpu"
NUM_WORKERS = 4              # DataLoader workers
PIN_MEMORY = True


# ============================================================================
# DERIVED VALUES (do not edit)
# ============================================================================

from feature_list import FEATURES   # noqa: E402

NUM_FEATURES = len(FEATURES)
INPUT_DIM = WINDOW_LEN * NUM_FEATURES


def make_dirs():
    """Create all artifact directories. Call once at script entry."""
    for d in (ARTIFACTS_DIR, WINDOWS_DIR, CHECKPOINTS_DIR, METRICS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def summary() -> str:
    """Human-readable config summary for logs."""
    lines = [
        "=" * 60,
        "EXPERIMENT CONFIG",
        "=" * 60,
        f"  CSV:                {CSV_NAME}",
        f"  Features:           {NUM_FEATURES}",
        f"  Window length L:    {WINDOW_LEN}",
        f"  Stride:             {WINDOW_STRIDE}",
        f"  Train ratio:        {TRAIN_RATIO}",
        f"  Label position:     {LABEL_POSITION}",
        f"  Purge boundary:     {PURGE_BOUNDARY_WINDOWS}",
        "",
        f"  Model hidden dim:   {MODEL_HIDDEN_DIM}",
        f"  Model ResBlocks:    {MODEL_NUM_RESBLOCKS}",
        f"  Model dropout:      {MODEL_DROPOUT}",
        f"  Input dim:          {INPUT_DIM}  (= {WINDOW_LEN} * {NUM_FEATURES})",
        "",
        f"  Batch size:         {BATCH_SIZE}",
        f"  Epochs:             {EPOCHS}",
        f"  Learning rate:      {LEARNING_RATE}",
        f"  Weight decay:       {WEIGHT_DECAY}",
        f"  Class-weighted BCE: {USE_CLASS_WEIGHTED_BCE}",
        f"  Seeds:              {SEEDS}",
        "",
        f"  Decision threshold: {DECISION_THRESHOLD}",
        f"  Bootstrap resamples:{BOOTSTRAP_RESAMPLES}",
        "",
        f"  Window strategy:    {WINDOW_STRATEGY}",
        f"  Device:             {DEVICE}",
        "=" * 60,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
