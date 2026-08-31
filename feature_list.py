"""
Explicit feature list for TiDE training on DNN-EdgeIIoT-dataset.csv.

FINAL VERSION after per-feature diagnostic (diagnose_features.py).

Of the 46 features declared in the original Methods Section 4.1.1:
- 4 timing features are unavailable in the public CSV (PCAP-only)
- 1 feature (udp.time_delta) is shared between Flow and Timing groups
  -> 41 unique features available in CSV
- 5 features removed after post-hoc analysis:
  * tcp.ack, tcp.ack_raw, tcp.seq: session-specific sequence numbers
    (mean 10^7-10^9; do not generalize across captures)
  * udp.stream: Wireshark-internal capture index (artifact)
  * mqtt.conack.flags: near-zero variance in train, pathological
    14386x distribution shift in test
  -> 36 final features

Additionally, 11 of the 36 features are constant (std < 1e-6) in the
public CSV and contribute zero to training; they are retained for
consistency with the published feature schema and protected by
std-clipping (std -> 1.0) during normalization.
"""

# ============================================================================
# FINAL FEATURE LIST - 36 features fed to TiDE
# ============================================================================

HEADER_FEATURES = [
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

FLOW_FEATURES = [
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

PAYLOAD_FEATURES = [
    "http.request.uri.query",
    "http.request.version",
    "http.referer",
    "dns.qry.name.len",
    "dns.qry.qu",
    "dns.qry.type",
    "dns.retransmission",
    "dns.retransmit_request",
]

TIMING_FEATURES_MISSING = [
    "frame.time_delta",
    "frame.time_relative",
    "tcp.time_delta",
    "tcp.time_relative",
]

FEATURES = HEADER_FEATURES + FLOW_FEATURES + PAYLOAD_FEATURES
assert len(FEATURES) == 36, f"Expected 36 features, got {len(FEATURES)}"
assert len(set(FEATURES)) == 36, "Duplicate feature names!"

DROPPED_FEATURES = [
    "tcp.ack",
    "tcp.ack_raw",
    "tcp.seq",
    "udp.stream",
    "mqtt.conack.flags",
]

EXCLUDED_IDENTIFIERS = [
    "frame.time", "ip.src_host", "ip.dst_host",
    "arp.src.proto_ipv4", "arp.dst.proto_ipv4", "icmp.transmit_timestamp",
]
EXCLUDED_PAYLOAD_FREEFORM = [
    "http.file_data", "http.request.full_uri",
    "tcp.options", "tcp.payload", "mqtt.msg", "mqtt.topic", "dns.qry.name",
]
EXCLUDED_HIGH_CARDINALITY = [
    "tcp.dstport", "tcp.srcport", "http.tls_port",
    "dns.retransmit_request_in", "mqtt.msgtype", "mqtt.topic_len", "mbtcp.unit_id",
]
EXCLUDED_TARGET_COLUMNS = ["Attack_label", "Attack_type"]

EXCLUDED_ALL = (
    EXCLUDED_IDENTIFIERS + EXCLUDED_PAYLOAD_FREEFORM
    + EXCLUDED_HIGH_CARDINALITY + EXCLUDED_TARGET_COLUMNS + DROPPED_FEATURES
)

BINARY_LABEL_COL = "Attack_label"
MULTICLASS_LABEL_COL = "Attack_type"
SORT_KEY_COL = "frame.time"


def summary() -> str:
    lines = [
        f"FEATURES used by TiDE: {len(FEATURES)}",
        f"  Header:             {len(HEADER_FEATURES)}",
        f"  Flow:               {len(FLOW_FEATURES)}",
        f"  Payload-derived:    {len(PAYLOAD_FEATURES)}",
        f"  Dropped (diagnostic): {len(DROPPED_FEATURES)}",
        f"  Missing (PCAP-only):  {len(TIMING_FEATURES_MISSING)}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
    print()
    for i, f in enumerate(FEATURES, 1):
        print(f"  {i:>3}. {f}")
