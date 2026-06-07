"""
dyra_iiot/risk/pipeline.py
─────────────────────────────────────────────────────────────────────────────
DyRA-IIoT risk pipeline (Section 3.2–3.5).

Implements:
  • Fuzzy SAW asset impact  Impact(A)      — Eq. (4)
  • Operational context     γ(t)           — Table 3
  • Dynamic risk signal     R(t)           — Eq. (5)
  • K-consecutive-exceedance alerting rule — Eq. (6)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np

import dyra_iiot.config as C


# ─────────────────────────────────────────────────────────────────────────────
# Fuzzy SAW  (Section 3.2, Tables 1–2)
# ─────────────────────────────────────────────────────────────────────────────

def compute_impact(
    hw_linguistic: str,
    sw_linguistic: str,
    comm_linguistic: str,
    weights: Dict[str, float] | None = None,
    crisp: Dict[str, float] | None = None,
) -> float:
    """
    Compute the defuzzified asset impact score.

        Impact(A) = Σ w_j · c_ij    (Eq. 4)

    Parameters
    ----------
    hw_linguistic   : linguistic term for Hardware damage (e.g. "VH").
    sw_linguistic   : linguistic term for Software damage.
    comm_linguistic : linguistic term for Communication damage.
    weights         : criterion weights; defaults to Table 2 values.
    crisp           : linguistic → crisp mapping; defaults to Table 1.

    Returns
    -------
    Impact(A) ∈ [0, 1]
    """
    w = weights if weights is not None else C.FUZZY_WEIGHTS
    c = crisp   if crisp   is not None else C.FUZZY_CRISP

    terms = {
        "hardware":      hw_linguistic.upper(),
        "software":      sw_linguistic.upper(),
        "communication": comm_linguistic.upper(),
    }

    for crit, term in terms.items():
        if term not in c:
            raise ValueError(f"Unknown linguistic term '{term}' for {crit}. "
                             f"Valid: {list(c.keys())}")

    return sum(w[crit] * c[term] for crit, term in terms.items())


def build_node_impacts(
    topology: List[Dict] | None = None,
) -> Dict[str, float]:
    """
    Compute Impact(A) for each node in the simulation topology.

    Parameters
    ----------
    topology : list of node dicts  (defaults to C.NODE_TOPOLOGY).
               Each dict must have keys: id, hw, sw, comm.

    Returns
    -------
    dict  {node_id: impact_score}
    """
    topology = topology if topology is not None else C.NODE_TOPOLOGY
    results = {}
    for node in topology:
        hw, sw, comm = node["hw"], node["sw"], node["comm"]
        # Support both float (exact Table 2 scores) and linguistic strings
        if isinstance(hw, float):
            w = C.FUZZY_WEIGHTS
            impact = w["hardware"]*hw + w["software"]*sw + w["communication"]*comm
        else:
            impact = compute_impact(hw, sw, comm)
        results[node["id"]] = impact
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Operational context factor γ(t)  (Section 3.3, Table 3)
# ─────────────────────────────────────────────────────────────────────────────

def get_gamma(operational_state: str) -> float:
    """
    Return γ for a given ISA-95 operational state (Section 3.3.1, Table 3).

    Recognised states:
        "maintenance"       → 0.4
        "standby"           → 0.6
        "production"        → 1.0
        "shift_change"      → 1.2
        "critical_process"  → 1.5
    """
    key = operational_state.lower().replace(" ", "_").replace("-", "_")
    if key not in C.GAMMA_MAP:
        raise ValueError(
            f"Unknown operational state '{operational_state}'. "
            f"Valid: {list(C.GAMMA_MAP.keys())}"
        )
    return C.GAMMA_MAP[key]


def gamma_schedule(
    n_windows: int,
    schedule: List[Tuple[int, int, str]] | None = None,
) -> np.ndarray:
    """
    Build a piecewise-constant γ(t) array of length n_windows.

    Parameters
    ----------
    n_windows : total number of time steps.
    schedule  : list of (start, end, state) tuples.
                Windows not covered default to "production" (γ=1.0).

    Example
    -------
    >>> g = gamma_schedule(500, [(0, 100, "maintenance"), (400, 500, "shift_change")])
    """
    g = np.full(n_windows, C.GAMMA_MAP["production"], dtype=np.float32)
    if schedule:
        for start, end, state in schedule:
            g[start:end] = get_gamma(state)
    return g


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic risk aggregation  R(t) = P(t) × Impact(A) × γ(t)  (Eq. 5)
# ─────────────────────────────────────────────────────────────────────────────

def compute_risk(
    P: np.ndarray,
    impact: float,
    gamma: np.ndarray | float = 1.0,
) -> np.ndarray:
    """
    Compute the dynamic risk signal for a single asset.

        R_i(t) = P_i(t) × Impact(A_i) × γ_i(t)        (Eq. 5)

    Parameters
    ----------
    P      : threat probability trace, shape (T,).
    impact : scalar Impact(A) for the asset.
    gamma  : scalar or array of length T.  Defaults to 1.0 (γ ≡ 1).

    Returns
    -------
    R : float32 array, shape (T,).  R ∈ [0, 1.5].
    """
    P = np.asarray(P, dtype=np.float32)
    g = np.asarray(gamma, dtype=np.float32)
    return (P * float(impact) * g).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# K-consecutive-exceedance alerting rule  (Eq. 6, Section 3.5)
# ─────────────────────────────────────────────────────────────────────────────

def k_exceedance_alerts(
    R: np.ndarray,
    K: int = C.RISK_K,
    tau: float = C.RISK_TAU,
) -> np.ndarray:
    """
    Generate discrete alerts from the continuous risk signal R(t).

        alert(t) = 1  ⟺  R(t-k) > τ  for all k ∈ {0, …, K-1}

    Parameters
    ----------
    R   : risk signal, shape (T,).
    K   : number of consecutive exceedances required.
    tau : critical threshold τ_critical.

    Returns
    -------
    alerts : int array, shape (T,).  1 = alert, 0 = no alert.
    """
    R = np.asarray(R, dtype=np.float32)
    exceeds = (R > tau).astype(np.int8)
    alerts  = np.zeros_like(exceeds)

    for t in range(K - 1, len(R)):
        if exceeds[t - K + 1: t + 1].sum() == K:
            alerts[t] = 1

    return alerts


def attack_to_alert_latency(
    alerts: np.ndarray,
    attack_onset: int,
) -> Optional[int]:
    """
    Number of windows from attack onset to first alert.
    Returns None if no alert was raised after onset.
    """
    post = np.where(alerts[attack_onset:] == 1)[0]
    return int(post[0]) if len(post) > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

class DyRAPipeline:
    """
    Combines the three risk components into a single callable.

    Usage
    -----
    >>> pipeline = DyRAPipeline(impact=0.81, gamma=1.0, K=3, tau=0.5)
    >>> alerts = pipeline(P_trace)   # np.ndarray of 0/1 alerts
    """

    def __init__(
        self,
        impact: float,
        gamma: np.ndarray | float = 1.0,
        K: int = C.RISK_K,
        tau: float = C.RISK_TAU,
    ):
        self.impact = impact
        self.gamma  = gamma
        self.K      = K
        self.tau    = tau

    def risk_signal(self, P: np.ndarray) -> np.ndarray:
        return compute_risk(P, self.impact, self.gamma)

    def __call__(self, P: np.ndarray) -> np.ndarray:
        R = self.risk_signal(P)
        return k_exceedance_alerts(R, self.K, self.tau)
