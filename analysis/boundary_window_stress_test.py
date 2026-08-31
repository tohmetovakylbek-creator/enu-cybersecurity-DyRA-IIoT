#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boundary_window_stress_test.py
===============================
DyRA-IIoT — Section 4.8.5 (Boundary-Window Stress Test).

Tests how the reference TiDE instantiation and the K-consecutive-exceedance
alerting rule behave on HETEROGENEOUS sliding windows that straddle a
Normal->attack class-block boundary, as opposed to the homogeneous windows
used everywhere else in the paper (Algorithm 1, windows never cross block
boundaries).

DESIGN (mirrors Element 11 of the roadmap)
-------------------------------------------
  * Backbone: reference TiDE instantiation (loaded from a trained checkpoint,
    NOT retrained here -- this measures the SAME model used in Sections 4.2-4.5).
  * Data: DNN-EdgeIIoT-dataset.csv, restricted to the TEST partition only
    (the chronological last 20% of each class block, per Algorithm 1), so
    leakage control is preserved -- we never touch train rows.
  * Boundary selection (REVISED): Edge-IIoTset is dominated by a single large
    Normal block, so restricting to Normal->attack adjacencies (the original
    design) yields only 1 usable boundary -- too few for any per-fraction
    statistic. We therefore use ALL inter-block adjacencies in the test tail
    (Normal->attack, attack->Normal, and attack_i->attack_j alike; up to 14
    for Edge-IIoTset's 15 class blocks). For every such boundary, we build
    heterogeneous windows of length L=50 in which the transition point falls
    INSIDE the window, at controlled right-side-fraction targets
    {10%, 30%, 50%, 70%, 90%}. Window label = label of the last packet (same
    convention as Section 4.1.4). Recall and attack-to-alert delay (Table 28)
    are computed only over boundaries whose resulting window label is attack
    (1), mirroring Table 6's "attack recall" convention; Normal-labelled
    heterogeneous windows are still logged in the raw JSON for audit but
    excluded from the headline statistic.
  * Caveat made explicit in code and in the printed output: block boundaries in
    this CSV are session joins (different attack-type capture sessions
    concatenated), not organic in-stream transitions. This script is therefore
    a controlled stress test of attack-signal dilution, not a replication of
    natural transition dynamics. See manuscript Section 4.8.5 wording.
  * Metrics: recall on attack-labelled heterogeneous windows (vs. the
    homogeneous in-distribution baseline of Table 6, ~0.999), and
    attack-to-alert delay under the K-consecutive-exceedance rule
    (K=3, tau_critical=0.5, gamma=1.0 -- i.e. R(t) = P(t) since Impact x gamma
    is not applied here; this isolates the predictor + alerting-rule behavior,
    consistent with the gamma=1.0 convention used in Sections 4.3-4.4.1).

USAGE
-----
    python boundary_window_stress_test.py \
        --csv path/to/DNN-EdgeIIoT-dataset.csv \
        --checkpoint path/to/tide_checkpoint.pt \
        --out_dir ./boundary_results

If you do not have a saved checkpoint and trained TiDE in-memory in another
script, see `load_external_model()` below -- swap in your own loader (the only
contract is: returns an nn.Module that maps [B, L, F] -> [B] logits).

OUTPUT
------
    boundary_results/table_28_boundary_stress_test.csv   (paste-ready for Table 28)
    boundary_results/boundary_windows_raw.json            (per-window predictions, for audit)
    Console summary matching the Table 28 layout in the manuscript.
"""

import os
import sys
import json
import argparse
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# ----------------------------------------------------------------------------- #
#  CONFIG -- mirrors Sections 3.1.1, 4.1, 4.3 of the manuscript                 #
# ----------------------------------------------------------------------------- #
WINDOW_LEN     = 50            # L  (Eq. 1)
TRAIN_RATIO    = 0.80          # rho (Algorithm 1) -- used to locate the test tail
K_CONSECUTIVE  = 3              # K  (Eq. 6)
TAU_CRITICAL   = 0.5            # tau_critical (Section 3.5)
GAMMA          = 1.0            # gamma held at 1.0, as in Sections 4.3-4.4.1
ATTACK_FRACTIONS = [0.10, 0.30, 0.50, 0.70, 0.90]
EPS            = 1e-6
WINDOW_STEP_MS = 66.7           # one window step (Section 4.3), for reporting

# Section 4.1.1 / 4.1.2: 36-feature schema (Table 5), label column, attack-type
# column. Adjust LABEL_COL / TYPE_COL only if your CSV uses different headers.
LABEL_COL = "Attack_label"      # binary 0/1 (Normal vs Attack)
TYPE_COL  = "Attack_type"       # 15-way class, used to find block boundaries

FEATURE_SCHEMA_36 = [
    # Header features (13)
    "arp.opcode", "arp.hw.size", "icmp.checksum", "icmp.seq_le", "icmp.unused",
    "http.content_length", "http.response", "http.request.method",
    "tcp.connection.fin", "tcp.connection.rst", "tcp.connection.syn",
    "tcp.connection.synack", "tcp.checksum",
    # Flow features (15)
    "tcp.flags", "tcp.flags.ack", "tcp.len", "udp.port", "udp.time_delta",
    "mqtt.conflag.cleansess", "mqtt.conflags", "mqtt.hdrflags", "mqtt.len",
    "mqtt.msg_decoded_as", "mqtt.proto_len", "mqtt.protoname", "mqtt.ver",
    "mbtcp.len", "mbtcp.trans_id",
    # Payload-derived features (8)
    "http.request.uri.query", "http.request.version", "http.referer",
    "dns.qry.name.len", "dns.qry.qu", "dns.qry.type", "dns.retransmission",
    "dns.retransmit_request",
]

# --------------------------------------------------------------------------- #
#  CATEGORICAL / STRING FIELD HANDLING                                        #
#                                                                             #
#  A subset of the 36 features arrive as raw strings in DNN-EdgeIIoT-         #
#  dataset.csv (e.g. http.request.method = 'GET','POST',...) and must be      #
#  label-encoded to floats before normalization/inference, exactly as the     #
#  original training pipeline does (see feature_schema.py: CATEGORICAL_      #
#  ENCODERS, PRESENCE_FIELDS, decode_value()). The mappings below are ported  #
#  from that module.                                                          #
#                                                                             #
#  IMPORTANT CAVEAT (carried over verbatim from feature_schema.py): these are #
#  PLACEHOLDER encodings with a plausible common ordering. If your original   #
#  training run used a persisted sklearn LabelEncoder (classes_ saved to      #
#  JSON) rather than this fixed mapping, replace CATEGORICAL_ENCODERS below   #
#  with that exact persisted mapping -- otherwise category indices may not    #
#  align with what the checkpoint was trained on, silently corrupting the     #
#  three affected columns (the other 33 numeric features are unaffected).     #
# --------------------------------------------------------------------------- #
CATEGORICAL_ENCODERS = {
    "http.request.method": {
        "": 0, "GET": 1, "POST": 2, "PUT": 3, "DELETE": 4,
        "HEAD": 5, "OPTIONS": 6, "PATCH": 7, "TRACE": 8, "CONNECT": 9,
    },
    "http.request.version": {
        "": 0, "HTTP/1.0": 1, "HTTP/1.1": 2, "HTTP/2": 3,
    },
    "mqtt.protoname": {
        "": 0, "MQTT": 1, "MQIsdp": 2,
    },
}

# Fields where the CSV stores a binary presence/absence indicator rather than
# the raw string value (1 if the field is present/non-empty, else 0).
PRESENCE_FIELDS = {"http.request.uri.query", "http.referer"}


def decode_categorical_columns(df, feature_cols, verbose=True):
    """
    Converts every column in `feature_cols` to a numeric dtype in place,
    applying CATEGORICAL_ENCODERS / PRESENCE_FIELDS to the handful of string
    columns and leaving already-numeric columns untouched. Must be called
    once on the full DataFrame before any .astype(float) on feature_cols.
    """
    decoded_any = False
    for col in feature_cols:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue  # already numeric -- nothing to decode
        decoded_any = True
        s = df[col].astype(str).str.strip()
        if col in PRESENCE_FIELDS:
            df[col] = (s != "").astype(np.float64)
            if verbose:
                print(f"[decode] '{col}': presence-encoded "
                      f"({int((df[col]==1).sum())} non-empty / {len(df)})")
        elif col in CATEGORICAL_ENCODERS:
            mapping = CATEGORICAL_ENCODERS[col]
            unseen = sorted(set(s.unique()) - set(mapping.keys()))
            if unseen and verbose:
                print(f"[decode][warn] '{col}': {len(unseen)} value(s) not in "
                      f"the placeholder mapping (mapped to 0): {unseen[:5]}"
                      f"{'...' if len(unseen) > 5 else ''}")
            df[col] = s.map(mapping).fillna(0).astype(np.float64)
            if verbose:
                print(f"[decode] '{col}': label-encoded via "
                      f"CATEGORICAL_ENCODERS ({len(mapping)} known classes)")
        else:
            # Generic fallback: numeric coercion, non-parseable -> 0
            # (mirrors decode_value()'s hex/comma-joined handling for the
            # remaining edge-case string columns, if any).
            def _coerce(v):
                v = v.strip()
                if v == "":
                    return 0.0
                if "," in v:
                    v = v.split(",", 1)[0]
                try:
                    if v.lower().startswith("0x"):
                        return float(int(v, 16))
                    return float(v)
                except ValueError:
                    return 0.0
            df[col] = s.apply(_coerce)
            if verbose:
                print(f"[decode] '{col}': generic numeric coercion applied "
                      f"(not in CATEGORICAL_ENCODERS/PRESENCE_FIELDS)")
    if not decoded_any and verbose:
        print("[decode] all feature columns already numeric -- no decoding needed")
    return df


# ----------------------------------------------------------------------------- #
#  TiDE -- faithful to Section 3.1.3 (must match the checkpoint's architecture)  #
# ----------------------------------------------------------------------------- #
if _HAS_TORCH:
    class ResidualBlock(nn.Module):
        """Matches the user's main-pipeline ResidualBlock exactly."""
        def __init__(self, hidden_dim, dropout=0.2):
            super(ResidualBlock, self).__init__()
            self.linear = nn.Linear(hidden_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, x):
            identity = x
            out = self.linear(x)
            out = self.relu(out)
            out = self.dropout(out)
            return self.norm(identity + out)

    class TiDE(nn.Module):
        """
        Matches the user's main-pipeline TiDEAnomalyDetector exactly (same
        layer names / order, so a state_dict trained with that class loads
        here without renaming). NOTE: forward() returns a PROBABILITY (Sigmoid
        is the last layer of `classifier`), NOT logits -- downstream code must
        NOT re-apply sigmoid (see predict_proba()).
        """
        def __init__(self, seq_len=WINDOW_LEN, num_features=36, hidden_dim=256,
                    num_layers=2, dropout=0.2):
            super(TiDE, self).__init__()
            self.flatten = nn.Flatten()
            input_dim = seq_len * num_features
            self.feature_projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            )
            self.encoder = nn.Sequential(
                *[ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)]
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )

        def forward(self, x):                       # x: [B, L, F]
            x_flat = self.flatten(x)
            projected = self.feature_projection(x_flat)
            encoded = self.encoder(projected)
            prob = self.classifier(encoded)
            return prob.squeeze(-1)                  # PROBABILITY, not logits


def load_external_model(checkpoint_path, L, F, device):
    """
    Loads a trained reference TiDE checkpoint.
    Expects a state_dict saved via torch.save(model.state_dict(), path), with
    layer names matching TiDEAnomalyDetector (feature_projection / encoder /
    classifier). If your checkpoint was saved differently (full model object,
    Lightning checkpoint, etc.), adapt this function -- the only contract
    downstream is: returns an nn.Module in eval() mode on `device`.
    """
    model = TiDE(seq_len=L, num_features=F)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, nn.Module):
        model = state
    else:
        model.load_state_dict(state)
    model.to(device).eval()
    return model


# ----------------------------------------------------------------------------- #
#  DATA LOADING + TEST-PARTITION RECONSTRUCTION (Algorithm 1, test tail only)    #
# ----------------------------------------------------------------------------- #
def load_test_tail(df, feature_cols, rho=TRAIN_RATIO, verbose=True):
    """
    Re-derives the per-class-block TEST tail exactly as Algorithm 1 would,
    without touching train rows. Takes an ALREADY-LOADED and
    ALREADY-DECODED DataFrame (see decode_categorical_columns()) to avoid
    re-reading the CSV and to guarantee categorical columns are numeric
    before any downstream astype(float). Returns a DataFrame restricted to
    the test tail of every class block.
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected feature columns: {missing}")
    if LABEL_COL not in df.columns or TYPE_COL not in df.columns:
        raise ValueError(f"Expected label columns '{LABEL_COL}'/'{TYPE_COL}' not found; "
                          f"available columns: {list(df.columns)[:10]}...")

    n = len(df)
    df["__row_idx__"] = np.arange(n)

    # Step 1: discover contiguous class blocks by Attack_type, exactly as
    # Algorithm 1 (lines 1-5).
    change = df[TYPE_COL].ne(df[TYPE_COL].shift()).cumsum()
    blocks = df.groupby(change)
    block_ranges = []  # list of (start_idx, end_idx_inclusive, type)
    for _, g in blocks:
        block_ranges.append((g["__row_idx__"].iloc[0], g["__row_idx__"].iloc[-1],
                             g[TYPE_COL].iloc[0]))
    if verbose:
        print(f"[load] found {len(block_ranges)} contiguous class blocks")

    # Step 2: test tail of each block (Algorithm 1, lines 7-10)
    test_frames = []
    for (b_s, b_e, btype) in block_ranges:
        length = b_e - b_s + 1
        cut = b_s + int(np.floor(rho * length))
        test_frames.append(df.loc[cut:b_e])  # test tail: rows [cut, b_e]
    test_df = pd.concat(test_frames).sort_values("__row_idx__").reset_index(drop=True)
    test_df["__type__"] = test_df[TYPE_COL]
    test_df["__y__"] = test_df[LABEL_COL].astype(int)

    if verbose:
        print(f"[load] test-tail rows: {len(test_df)} "
              f"({test_df['__y__'].mean()*100:.1f}% attack)")
    return test_df, block_ranges


def find_all_block_boundaries(test_df, block_ranges):
    """
    Within the TEST tail, locate ALL adjacencies between consecutive class
    blocks (not just Normal->attack). Edge-IIoTset is dominated by a single
    large Normal block, so restricting to Normal->attack adjacencies yields
    only 1 usable boundary in practice -- far too few for any meaningful
    per-fraction statistic. Using every one of the 14 inter-block transitions
    (whatever the class on either side) is still a faithful stress test of
    attack-signal dilution in heterogeneous windows: each boundary still
    produces a window whose first part is drawn from one class-block and
    whose last part is drawn from the next, with the window label inherited
    from the LAST packet (Section 4.1.4 convention), exactly as before. Only
    the *direction-agnostic* relaxation changes; the underlying construction
    (real CSV block adjacency, train/test-tail leakage control, label
    convention) is unchanged.

    A boundary is usable only if BOTH adjacent blocks have at least
    WINDOW_LEN rows available in the test tail (so a full L=50 window with a
    controlled split point can be built without crossing into a THIRD block).

    Returns a list of dicts:
        {left_type, right_type, left_rows, right_rows,
         left_label, right_label}
    where *_rows are row-index arrays (original CSV order) restricted to the
    test tail of each adjacent block, and *_label is the binary Attack_label
    of that side (0=Normal, 1=attack) -- both sides are kept regardless of
    which is Normal, so downstream code no longer assumes a fixed direction.
    """
    boundaries = []
    test_idx_set = set(test_df["__row_idx__"].values)
    row_to_label = dict(zip(test_df["__row_idx__"].values, test_df["__y__"].values))

    for i in range(len(block_ranges) - 1):
        b_s, b_e, btype = block_ranges[i]
        nb_s, nb_e, nbtype = block_ranges[i + 1]
        left_rows = np.array([r for r in range(b_s, b_e + 1) if r in test_idx_set])
        right_rows = np.array([r for r in range(nb_s, nb_e + 1) if r in test_idx_set])
        if len(left_rows) >= WINDOW_LEN and len(right_rows) >= WINDOW_LEN:
            boundaries.append(dict(
                left_type=btype, right_type=nbtype,
                left_rows=left_rows, right_rows=right_rows,
                left_label=int(row_to_label[left_rows[-1]]),
                right_label=int(row_to_label[right_rows[0]]),
            ))
    return boundaries


# ----------------------------------------------------------------------------- #
#  NORMALIZATION -- train-only statistics REQUIRED                              #
# ----------------------------------------------------------------------------- #
def fit_or_load_normalizer(df_full, feature_cols, rho, norm_stats_path, verbose=True):
    """
    Per Eq. (7), normalization must be fit on TRAIN-partition packets only.
    If you already have the mu/sigma from your main training run, point
    --norm_stats to that JSON (recommended -- guarantees identical
    preprocessing to the original TiDE training). Otherwise this function
    recomputes mu/sigma from the train head of each block, replicating
    Section 4.1.4 exactly.
    """
    if norm_stats_path and os.path.exists(norm_stats_path):
        with open(norm_stats_path) as f:
            stats = json.load(f)
        mu = np.array([stats["mu"][c] for c in feature_cols])
        sd = np.array([stats["sd"][c] for c in feature_cols])
        if verbose:
            print(f"[norm] loaded train-only normalizer from {norm_stats_path}")
        return mu, sd

    if verbose:
        print("[norm][warn] no --norm_stats provided; recomputing train-only "
              "mu/sigma from this CSV's train heads (Section 4.1.4). For exact "
              "reproducibility with the original TiDE training run, prefer "
              "passing the saved normalizer via --norm_stats.")
    change = df_full[TYPE_COL].ne(df_full[TYPE_COL].shift()).cumsum()
    train_frames = []
    for _, g in df_full.groupby(change):
        b_s, b_e = g.index[0], g.index[-1]
        cut = b_s + int(np.floor(rho * (b_e - b_s + 1)))
        train_frames.append(df_full.loc[b_s:cut - 1])
    train_df = pd.concat(train_frames)
    X_train = train_df[feature_cols].astype(np.float64).values
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def normalize(X, mu, sd):
    return (X - mu) / (sd + EPS)


# ----------------------------------------------------------------------------- #
#  HETEROGENEOUS WINDOW CONSTRUCTION                                            #
# ----------------------------------------------------------------------------- #
def build_heterogeneous_windows(df_full, feature_cols, mu, sd, boundary,
                                L=WINDOW_LEN, fractions=ATTACK_FRACTIONS):
    """
    For one inter-block boundary (left_type | right_type, in original CSV
    order), build one heterogeneous window per target fraction f in
    `fractions`: the window's last k = round(f*L) rows are drawn from the
    START of the right block's test rows, and the first L-k rows are drawn
    from the END of the left block's test rows (immediately preceding the
    boundary, preserving original adjacency/order). Window label = label of
    the LAST row (Section 4.1.4 convention) = right_label by construction.

    `f` is therefore the fraction of the window drawn from the RIGHT side of
    the boundary, not specifically "attack fraction" -- the boundary can be
    Normal->attack, attack->Normal, or attack_i->attack_j. Recall is only
    meaningful for windows whose resulting label is 1 (attack); windows
    landing on a Normal-labelled boundary (right_label=0) are still returned
    (for delay-sequence bookkeeping) but excluded from the recall aggregate
    by the caller, consistent with Table 6's "attack recall" convention.

    Returns: dict[fraction] -> dict(window: np.ndarray [L,F], label: int,
                                    left_type: str, right_type: str,
                                    n_right_packets: int)
    """
    left_rows = boundary["left_rows"]
    right_rows = boundary["right_rows"]
    out = {}
    for f in fractions:
        k = max(1, round(f * L))
        k = min(k, L, len(right_rows))
        n_left = L - k
        if n_left > len(left_rows):
            continue  # not enough left-side context available for this boundary
        sel_left = left_rows[-n_left:] if n_left > 0 else np.array([], dtype=int)
        sel_right = right_rows[:k]
        sel = np.concatenate([sel_left, sel_right])
        if len(sel) != L:
            continue
        Xw = df_full.loc[sel, feature_cols].astype(np.float64).values
        Xw = normalize(Xw, mu, sd)
        last_label = int(df_full.loc[sel[-1], LABEL_COL])
        out[f] = dict(window=Xw.astype(np.float32), label=last_label,
                      left_type=boundary["left_type"],
                      right_type=boundary["right_type"],
                      n_right_packets=k)
    return out


# ----------------------------------------------------------------------------- #
#  INFERENCE + K-CONSECUTIVE ALERTING                                          #
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def predict_proba(model, X, device):
    """
    X: [N, L, F] -> probabilities [N].
    NOTE: the TiDE/TiDEAnomalyDetector class used here ends with nn.Sigmoid()
    inside `classifier`, so model(x) already returns a probability in [0,1].
    Do NOT re-apply torch.sigmoid() here -- that would double-squash the
    output and silently corrupt every downstream recall/delay computation.
    """
    xb = torch.from_numpy(X).to(device)
    prob = model(xb)
    return prob.detach().cpu().numpy()


def k_consecutive_delay(prob_sequence, K=K_CONSECUTIVE, tau=TAU_CRITICAL, gamma=GAMMA):
    """
    Given a chronological sequence of per-window risk scores R(t) = P(t)*gamma
    (Impact(A) omitted / set to 1, isolating predictor+rule behavior as in
    Sections 4.3-4.4.1), returns the 1-indexed window at which the
    K-consecutive-exceedance rule (Eq. 6) first fires, or None if it never
    fires within the sequence.
    """
    R = np.asarray(prob_sequence) * gamma
    exceed = R > tau
    run = 0
    for i, e in enumerate(exceed):
        run = run + 1 if e else 0
        if run >= K:
            return i + 1  # window index (1-indexed) at which alert fires
    return None


# ----------------------------------------------------------------------------- #
#  MAIN EXPERIMENT                                                              #
# ----------------------------------------------------------------------------- #
def run_stress_test(csv_path, checkpoint_path, norm_stats_path, out_dir,
                    device, max_boundaries_per_type=None):
    os.makedirs(out_dir, exist_ok=True)
    feature_cols = FEATURE_SCHEMA_36

    print(f"[load] reading {csv_path} ...")
    df_full = pd.read_csv(csv_path, low_memory=False)
    missing = [c for c in feature_cols if c not in df_full.columns]
    if missing:
        raise ValueError(f"CSV missing expected 36-feature columns: {missing}")

    print("[decode] checking feature columns for non-numeric (categorical) "
          "values and decoding them (see CATEGORICAL_ENCODERS / "
          "PRESENCE_FIELDS)...")
    df_full = decode_categorical_columns(df_full, feature_cols)

    test_df, block_ranges = load_test_tail(df_full, feature_cols)
    mu, sd = fit_or_load_normalizer(df_full, feature_cols, TRAIN_RATIO,
                                    norm_stats_path)

    boundaries = find_all_block_boundaries(test_df, block_ranges)
    print(f"[boundary] usable inter-block boundaries found: {len(boundaries)} "
          f"(out of {len(block_ranges)-1} total block adjacencies; direction-"
          f"agnostic, i.e. not restricted to Normal->attack)")
    n_normal_to_attack = sum(1 for b in boundaries
                             if b["left_label"] == 0 and b["right_label"] == 1)
    n_attack_to_normal = sum(1 for b in boundaries
                             if b["left_label"] == 1 and b["right_label"] == 0)
    n_attack_to_attack = sum(1 for b in boundaries
                             if b["left_label"] == 1 and b["right_label"] == 1)
    print(f"[boundary] composition: {n_normal_to_attack} Normal->attack, "
          f"{n_attack_to_normal} attack->Normal, "
          f"{n_attack_to_attack} attack->attack")
    for b in boundaries:
        print(f"[boundary]   {b['left_type']!r} -> {b['right_type']!r}  "
              f"(left n={len(b['left_rows'])}, right n={len(b['right_rows'])})")
    if not boundaries:
        print("[boundary][error] no usable boundaries -- check that the test "
              "tail of every block has >= {WINDOW_LEN} rows.")
        return

    if max_boundaries_per_type:
        by_type = {}
        filtered = []
        for b in boundaries:
            t = b["right_type"]
            by_type.setdefault(t, 0)
            if by_type[t] < max_boundaries_per_type:
                filtered.append(b)
                by_type[t] += 1
        boundaries = filtered
        print(f"[boundary] capped to {len(boundaries)} boundaries "
              f"(<= {max_boundaries_per_type} per right-side block type)")

    model = load_external_model(checkpoint_path, WINDOW_LEN, len(feature_cols), device)

    # ---- per-fraction aggregation ----
    per_fraction_probs = {f: [] for f in ATTACK_FRACTIONS}
    per_fraction_labels = {f: [] for f in ATTACK_FRACTIONS}
    raw_records = []

    for bi, boundary in enumerate(boundaries):
        wins = build_heterogeneous_windows(df_full, feature_cols, mu, sd, boundary)
        for f, w in wins.items():
            p = predict_proba(model, w["window"][None, :, :], device)[0]
            raw_records.append(dict(boundary_idx=bi, left_type=w["left_type"],
                                    right_type=w["right_type"], fraction=f,
                                    n_right_packets=w["n_right_packets"],
                                    prob=float(p), label=w["label"]))
            # Recall (Table 28) is defined on ATTACK-labelled windows only,
            # mirroring Table 6's "attack recall" convention. Boundaries whose
            # last packet is Normal (right_label=0) still contribute a record
            # above for audit, but are excluded from the recall aggregate.
            if w["label"] == 1:
                per_fraction_probs[f].append(p)
                per_fraction_labels[f].append(w["label"])
        if (bi + 1) % 50 == 0:
            print(f"[boundary] processed {bi+1}/{len(boundaries)} boundaries")

    # ---- homogeneous baseline recall, recomputed on this same test_df for a
    #      fair within-run comparison (Table 6 reports 0.999 on the full test
    #      partition; this is the matched-conditions reference) ----
    print("[baseline] computing homogeneous in-distribution recall on test tail "
          "(reference point for Delta column)...")
    homog_probs, homog_labels = [], []
    rng = np.random.default_rng(42)
    attack_row_pool = test_df.loc[test_df["__y__"] == 1, "__row_idx__"].values
    sample_n = min(2000, len(attack_row_pool))
    sampled_attack_rows = rng.choice(attack_row_pool, size=sample_n, replace=False)
    # build homogeneous windows ending at each sampled attack row, using only
    # rows from the SAME contiguous block (skip if not enough preceding rows
    # in-block -- mirrors Algorithm 1's within-block windowing).
    row_to_block = {}
    for (b_s, b_e, btype) in block_ranges:
        for r in range(b_s, b_e + 1):
            row_to_block[r] = (b_s, b_e)
    for r in sampled_attack_rows:
        b_s, b_e = row_to_block.get(r, (None, None))
        if b_s is None or r - WINDOW_LEN + 1 < b_s:
            continue
        sel = np.arange(r - WINDOW_LEN + 1, r + 1)
        Xw = normalize(df_full.loc[sel, feature_cols].astype(np.float64).values, mu, sd)
        p = predict_proba(model, Xw.astype(np.float32)[None, :, :], device)[0]
        homog_probs.append(p)
        homog_labels.append(1)
    homog_probs = np.array(homog_probs)
    homog_recall = float((homog_probs > TAU_CRITICAL).mean()) if len(homog_probs) else float("nan")
    print(f"[baseline] homogeneous recall on {len(homog_probs)} sampled attack "
          f"windows (matched test tail): {homog_recall:.4f}")

    # ---- Table 28 rows ----
    rows = []
    for f in ATTACK_FRACTIONS:
        probs = np.array(per_fraction_probs[f])
        labels = np.array(per_fraction_labels[f])
        n = len(probs)
        if n == 0:
            rows.append(dict(fraction=f, n=0, recall=float("nan"),
                             delta=float("nan"), delay_windows=None))
            continue
        recall = float((probs > TAU_CRITICAL).mean())
        delta = recall - homog_recall

        # attack-to-alert delay: meaningful only for boundaries whose
        # heterogeneous window is itself attack-labelled (right_label=1),
        # mirroring the recall filter above. For each such boundary, build
        # the short chronological sub-sequence [this heterogeneous window +
        # subsequent homogeneous windows sliding 1 step at a time within the
        # right-side block (stride=1, Section 4.1.4)] and find when K=3
        # first fires. We summarize the MEDIAN delay across boundaries that
        # have enough surrounding context to evaluate Eq. (6).
        delays = []
        for bi, boundary in enumerate(boundaries):
            wins = build_heterogeneous_windows(df_full, feature_cols, mu, sd, boundary)
            if f not in wins or wins[f]["label"] != 1:
                continue
            right_rows = boundary["right_rows"]
            k = wins[f]["n_right_packets"]
            # Sequence for the K-consecutive rule: the heterogeneous window's
            # own probability, then subsequent homogeneous windows sliding
            # 1 step at a time within the same right-side block (stride=1,
            # Section 4.1.4), up to K_CONSECUTIVE+1 extra steps.
            het_window = wins[f]["window"]
            p0 = predict_proba(model, het_window[None, :, :], device)[0]
            seq = [p0]
            extra_needed = max(0, K_CONSECUTIVE + 2 - 1)
            for step in range(min(extra_needed, max(0, len(right_rows) - k - WINDOW_LEN + 1))):
                sel = right_rows[k + step: k + step + WINDOW_LEN] \
                      if k + step + WINDOW_LEN <= len(right_rows) else None
                if sel is None or len(sel) != WINDOW_LEN:
                    break
                Xw = normalize(df_full.loc[sel, feature_cols].astype(np.float64).values, mu, sd)
                p_next = predict_proba(model, Xw.astype(np.float32)[None, :, :], device)[0]
                seq.append(p_next)
            delay = k_consecutive_delay(seq)
            if delay is not None:
                delays.append(delay)
        delay_summary = (f"{int(np.median(delays))} windows (n={len(delays)})"
                         if delays else "not reached")

        rows.append(dict(fraction=f, n=n, recall=round(recall, 4),
                         delta=round(delta, 4), delay_windows=delay_summary))
        print(f"[result] fraction={int(f*100)}%  n={n:4d}  recall={recall:.4f}  "
              f"Delta_vs_homog={delta:+.4f}  delay={delay_summary}")

    # ---- write outputs ----
    out_csv = os.path.join(out_dir, "table_28_boundary_stress_test.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    with open(os.path.join(out_dir, "boundary_windows_raw.json"), "w") as f:
        json.dump(dict(homogeneous_baseline_recall=homog_recall,
                       n_boundaries=len(boundaries), records=raw_records),
                 f, indent=2, default=float)

    print("\n" + "=" * 70)
    print("TABLE 28 -- Boundary-window stress test "
          f"(reference TiDE, K={K_CONSECUTIVE}, tau={TAU_CRITICAL}, gamma={GAMMA})")
    print("=" * 70)
    print(f"{'Attack fraction':<18}{'Windows n':<12}{'Recall':<10}"
          f"{'Delta vs baseline':<20}{'Delay'}")
    for r in rows:
        frac_str = f"{int(r['fraction']*100)}%"
        print(f"{frac_str:<18}{r['n']:<12}{r['recall']:<10}"
              f"{r['delta']:<20}{r['delay_windows']}")
    print(f"{'100% (homogeneous)':<18}{'--':<12}{round(homog_recall,4):<10}"
          f"{'--':<20}{'3 windows (paper baseline)'}")
    print("=" * 70)
    print(f"\n[done] wrote {out_csv}")
    print("[done] send back this CSV + the console TABLE 28 block above "
          "for integration into manuscript Table 28 / Section 4.8.5.")


# ----------------------------------------------------------------------------- #
#  ENTRY POINT                                                                  #
# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True,
                    help="Path to DNN-EdgeIIoT-dataset.csv")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to the trained reference TiDE checkpoint "
                         "(state_dict via torch.save)")
    ap.add_argument("--norm_stats", default=None,
                    help="Optional JSON with {'mu': {...}, 'sd': {...}} train-only "
                         "normalizer from your original training run (recommended). "
                         "If omitted, recomputed from this CSV's train heads.")
    ap.add_argument("--out_dir", default="./boundary_results")
    ap.add_argument("--device", default=("cuda" if (_HAS_TORCH and torch.cuda.is_available())
                                          else "cpu"))
    ap.add_argument("--max_boundaries_per_type", type=int, default=None,
                    help="Optional cap on boundaries sampled per attack type "
                         "(speeds up the run on large CSVs; omit for full coverage)")
    args = ap.parse_args()

    if not _HAS_TORCH:
        print("[fatal] PyTorch is required to run this script.")
        sys.exit(1)

    print("=" * 70)
    print("DyRA-IIoT -- Section 4.8.5 Boundary-Window Stress Test")
    print(f"K={K_CONSECUTIVE}  tau_critical={TAU_CRITICAL}  gamma={GAMMA}  "
          f"L={WINDOW_LEN}  fractions={ATTACK_FRACTIONS}")
    print(f"device: {args.device}")
    print("CAVEAT: class-block boundaries in this CSV are session joins between "
          "different capture sessions, not organic in-stream transitions. This "
          "is a controlled stress test of attack-signal dilution (Section 4.8.5 "
          "wording), not a replication of natural transition dynamics.")
    print("=" * 70)

    run_stress_test(args.csv, args.checkpoint, args.norm_stats, args.out_dir,
                    args.device, args.max_boundaries_per_type)


if __name__ == "__main__":
    main()
