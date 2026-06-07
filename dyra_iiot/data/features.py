"""
dyra_iiot/data/features.py
─────────────────────────────────────────────────────────────────────────────
Five-step feature-selection procedure and the 36-feature schema for
Edge-IIoTset  (Section 4.1.2, Table 5).

The 36 features are grouped into three semantic categories matching Table 5:
  • Header features  (13) — layers 2–4 identifiers
  • Flow features    (15) — cross-packet aggregations
  • Payload features  (8) — layers 5–7 semantic indicators
"""

from __future__ import annotations
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Canonical 36-feature schema  (Table 5)
# ─────────────────────────────────────────────────────────────────────────────

HEADER_FEATURES: List[str] = [
    "arp.opcode",
    "arp.hw.size",
    "icmp.checksum",
    "icmp.seq_le",
    "icmp.unused",
    "http.content_length",
    "http.response",
    "http.request.method",
    "tcp.connection.fin",
    "tcp.connection.rst",
    "tcp.connection.syn",
    "tcp.connection.synack",
    "tcp.checksum",
]

FLOW_FEATURES: List[str] = [
    "tcp.flags",
    "tcp.flags.ack",
    "tcp.len",
    "udp.port",
    "udp.time_delta",
    "mqtt.conflag.cleansess",
    "mqtt.conflags",
    "mqtt.hdrflags",
    "mqtt.len",
    "mqtt.msg_decoded_as",
    "mqtt.proto_len",
    "mqtt.protoname",
    "mqtt.ver",
    "mbtcp.len",
    "mbtcp.trans_id",
]

PAYLOAD_FEATURES: List[str] = [
    "http.request.uri.query",
    "http.request.version",
    "http.referer",
    "dns.qry.name.len",
    "dns.qry.qu",
    "dns.qry.type",
    "dns.retransmission",
    "dns.retransmit_request",
]

SCHEMA_36: List[str] = HEADER_FEATURES + FLOW_FEATURES + PAYLOAD_FEATURES

# Features excluded in step 4 of the diagnostic  (Section 4.1.2, Table 14)
DIAGNOSTIC_EXCLUDED: List[str] = [
    "tcp.ack",
    "tcp.ack_raw",
    "tcp.seq",
    "udp.stream",
    "mqtt.conack.flags",
]

# Columns that are always dropped  (session identifiers / label columns)
_DROP_ALWAYS: set = {
    "ip.src", "ip.dst", "ip.src_host", "ip.dst_host",
    "eth.src", "eth.dst",
    "frame.time", "frame.time_delta", "frame.time_relative",
    "Attack_label", "Attack_type",         # Edge-IIoTset label columns
    "label", "type", "attack_type",        # generic label columns
}


# ─────────────────────────────────────────────────────────────────────────────
# Feature selection — Edge-IIoTset
# ─────────────────────────────────────────────────────────────────────────────
def select_features_edge(
    df: pd.DataFrame,
    schema: List[str] | None = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Apply the canonical 36-feature schema to an Edge-IIoTset DataFrame.

    Parameters
    ----------
    df : DataFrame with all CSV columns loaded.
    schema : feature list to use (defaults to SCHEMA_36).
    verbose : log missing features.

    Returns
    -------
    X : float32 array, shape (N, F)
    y : float32 binary labels  (0=Normal, 1=Attack)
    features : list of feature names used
    """
    if schema is None:
        schema = SCHEMA_36

    missing = [f for f in schema if f not in df.columns]
    if missing and verbose:
        logger.warning("Features absent from CSV (will be zero-filled): %s", missing)

    # Build feature matrix — zero-fill absent columns
    X_df = pd.DataFrame(index=df.index)
    for feat in schema:
        if feat in df.columns:
            X_df[feat] = pd.to_numeric(df[feat], errors="coerce").fillna(0.0)
        else:
            X_df[feat] = 0.0

    X = X_df.values.astype(np.float32)

    # Binary label
    if "Attack_label" in df.columns:
        y = df["Attack_label"].values.astype(np.float32)
    else:
        y = (df["Attack_type"].str.lower() != "normal").astype(np.float32).values

    return X, y, schema


# ─────────────────────────────────────────────────────────────────────────────
# Feature selection — TON_IoT (auto-detect)
# ─────────────────────────────────────────────────────────────────────────────

_TON_CATEGORICAL: List[str] = [
    "proto", "service", "conn_state",
    "ssl_version", "ssl_cipher",
    "http_method", "http_version",
]

_TON_EXCLUDE_ALWAYS: set = {
    "ts", "src_ip", "dst_ip",
    "dns_query", "ssl_subject", "ssl_issuer",
    "http_uri", "http_referrer", "http_user_agent",
    "http_orig_mime_types", "http_resp_mime_types",
    "weird_name", "weird_addl",
    "label", "type", "_type_clean", "_label_bin",
}


def select_features_ton(
    df: pd.DataFrame,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Auto-detect and encode features from TON_IoT Network CSV.
    Applies the same 5-step filtering logic as Section 4.1.2 but
    adapted to the TON_IoT feature space (31 features).

    Returns X (N, F), y (N,), feature_names.
    """
    drop_cols = _TON_EXCLUDE_ALWAYS.copy()

    # Step 1: drop high-cardinality string columns (>100 unique values)
    for col in df.columns:
        if col not in drop_cols and df[col].dtype == object:
            if df[col].nunique() > 100:
                drop_cols.add(col)

    feature_cols = [c for c in df.columns if c not in drop_cols]
    df_feat = df[feature_cols].copy()

    # Step 2: label-encode low-cardinality categoricals
    encoders: dict = {}
    for col in _TON_CATEGORICAL:
        if col in df_feat.columns:
            le = LabelEncoder()
            df_feat[col] = le.fit_transform(
                df_feat[col].astype(str).fillna("unknown"))
            encoders[col] = le

    # Step 3: encode any remaining object columns
    for col in df_feat.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df_feat[col] = le.fit_transform(
            df_feat[col].astype(str).fillna("unknown"))

    # Step 4: fill NaN → 0
    df_feat = df_feat.fillna(0)

    X = df_feat.values.astype(np.float32)

    if "_label_bin" in df.columns:
        y = df["_label_bin"].values.astype(np.float32)
    elif "label" in df.columns:
        y = df["label"].values.astype(np.float32)
    else:
        y = (df["type"].str.lower() != "normal").astype(np.float32).values

    if verbose:
        logger.info("TON_IoT feature schema: %d features", X.shape[1])
        # Warn about pathological scales
        for i, col in enumerate(df_feat.columns):
            if np.std(X[:, i]) > 1e6:
                logger.warning("  Pathological scale: %s (std=%.2e)", col, np.std(X[:, i]))

    return X, y, list(df_feat.columns)
