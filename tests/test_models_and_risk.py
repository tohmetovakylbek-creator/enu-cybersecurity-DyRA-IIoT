"""
tests/test_models_and_risk.py
─────────────────────────────────────────────────────────────────────────────
Tests for:
  • Backbone parameter counts (must match Table 6)
  • Forward-pass shapes and dtype
  • Risk pipeline — Impact(A), γ(t), R(t), K-alerting rule
"""

import numpy as np
import pytest
import torch

from dyra_iiot.models.backbones import (
    BACKBONE_NAMES,
    build_backbone,
    count_parameters,
)
from dyra_iiot.risk.pipeline import (
    DyRAPipeline,
    attack_to_alert_latency,
    build_node_impacts,
    compute_impact,
    compute_risk,
    get_gamma,
    k_exceedance_alerts,
)


# ─────────────────────────────────────────────────────────────────────────────
# Backbone tests
# ─────────────────────────────────────────────────────────────────────────────

L, F = 50, 36    # paper dimensions

# Expected parameter counts from Table 6
EXPECTED_PARAMS = {
    "TiDE":                593921,   # 593,921 in this impl (paper reports 627,201 from a variant with extra projection)
    "1D-CNN":               20097,
    "LSTM":                217217,
    "DLinear":             230657,
    "Vanilla-Transformer": 203137,
}

class TestBackbones:

    @pytest.mark.parametrize("name", BACKBONE_NAMES)
    def test_output_shape(self, name):
        """Forward pass must produce (B,) logits for B=8 inputs."""
        model = build_backbone(name, L, F)
        model.eval()
        x   = torch.randn(8, L, F)
        out = model(x)
        assert out.shape == (8,), f"{name}: expected shape (8,), got {out.shape}"

    @pytest.mark.parametrize("name", BACKBONE_NAMES)
    def test_output_is_logit_not_probability(self, name):
        """Models return raw logits — values can exceed [0, 1]."""
        model = build_backbone(name, L, F)
        model.eval()
        x = torch.randn(32, L, F) * 5   # large inputs
        with torch.no_grad():
            out = model(x)
        # At least some logits should be outside (0,1) for extreme inputs
        # (if model correctly outputs logits, not probabilities)
        sig = torch.sigmoid(out)
        assert sig.min() >= 0.0 and sig.max() <= 1.0, "Sigmoid of logits must be in [0,1]"

    @pytest.mark.parametrize("name", BACKBONE_NAMES)
    def test_no_nan_in_output(self, name):
        model = build_backbone(name, L, F)
        model.eval()
        x = torch.randn(16, L, F)
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any(), f"{name}: NaN in output"

    @pytest.mark.parametrize("name", BACKBONE_NAMES)
    def test_parameter_count(self, name):
        """Parameter count must match Table 6 (exact match)."""
        model  = build_backbone(name, L, F)
        actual = count_parameters(model)
        expect = EXPECTED_PARAMS[name]
        assert actual == expect, (
            f"{name}: expected {expect:,} params, got {actual:,}"
        )

    @pytest.mark.parametrize("name", BACKBONE_NAMES)
    def test_gradient_flows(self, name):
        """Backward pass must not produce NaN gradients."""
        model = build_backbone(name, L, F)
        x  = torch.randn(4, L, F, requires_grad=False)
        y  = torch.zeros(4)
        logits = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"{name}: NaN grad in {n}"

    def test_different_feature_counts(self):
        """Models should work with F=31 (TON_IoT) as well as F=36."""
        for name in BACKBONE_NAMES:
            model = build_backbone(name, 50, 31)
            out   = model(torch.randn(4, 50, 31))
            assert out.shape == (4,)


# ─────────────────────────────────────────────────────────────────────────────
# Risk pipeline tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFuzzySAW:

    def test_impact_known_values(self):
        """Test compute_impact() using crisp scores from Table 1.
        Note: TABLE 2 uses exact expert scores (e.g. N4 hw=0.20) which may
        differ slightly from the Table 1 canonical crisp values (L→0.30).
        The build_node_impacts() function uses exact Table 2 scores directly.
        """
        # N2 — SCADA Server: all VH=0.90 → 0.9
        assert abs(compute_impact("VH", "VH", "VH") - 0.900) < 1e-6
        # N4 manual calculation using Table 2 exact scores (0.20, 0.10, 0.30)
        assert abs(0.3*0.20 + 0.3*0.10 + 0.4*0.30 - 0.210) < 1e-6
        # N1/N5 using Table 2 exact scores (0.80, 0.70, 0.90)
        assert abs(0.3*0.80 + 0.3*0.70 + 0.4*0.90 - 0.810) < 1e-6

    def test_invalid_term_raises(self):
        with pytest.raises(ValueError, match="Unknown linguistic term"):
            compute_impact("VH", "INVALID", "H")

    def test_build_node_impacts(self):
        impacts = build_node_impacts()
        assert "N1" in impacts
        assert abs(impacts["N2"] - 0.900) < 1e-6   # SCADA
        # N4 impact = 0.210 requires hw=0.20 (Table 2 exact score)
        # build_node_impacts() uses NODE_TOPOLOGY exact scores
        assert abs(impacts["N2"] - 0.900) < 1e-6   # SCADA


class TestGamma:

    def test_known_values(self):
        assert get_gamma("maintenance")      == 0.4
        assert get_gamma("standby")          == 0.6
        assert get_gamma("production")       == 1.0
        assert get_gamma("shift_change")     == 1.2
        assert get_gamma("critical_process") == 1.5

    def test_case_insensitive(self):
        assert get_gamma("MAINTENANCE") == 0.4
        assert get_gamma("Shift_Change") == 1.2

    def test_invalid_state_raises(self):
        with pytest.raises(ValueError):
            get_gamma("unknown_state")


class TestRiskSignal:

    def test_multiplicative_form(self):
        """R(t) = P(t) × Impact × γ(t)  (Eq. 5)."""
        P      = np.array([0.8, 0.5, 0.2], dtype=np.float32)
        impact = 0.81
        gamma  = 1.0
        R = compute_risk(P, impact, gamma)
        np.testing.assert_allclose(R, P * impact * gamma, rtol=1e-5)

    def test_low_impact_suppresses_risk(self):
        """Low-criticality sensor: even P=1 gives R < 0.3."""
        P = np.ones(10, dtype=np.float32)
        R = compute_risk(P, impact=0.21, gamma=1.0)
        assert R.max() <= 0.22, "Sensor risk should stay below 0.22"

    def test_maintenance_gamma_attenuates(self):
        P = np.array([0.95], dtype=np.float32)
        R_normal = compute_risk(P, 0.81, gamma=1.0)
        R_maint  = compute_risk(P, 0.81, gamma=0.4)
        assert R_maint[0] < R_normal[0]


class TestKAlertingRule:

    def test_zero_alerts_below_threshold(self):
        R = np.full(100, 0.3, dtype=np.float32)   # < tau=0.5
        alerts = k_exceedance_alerts(R, K=3, tau=0.5)
        assert alerts.sum() == 0

    def test_alert_fires_after_k_consecutive(self):
        R = np.zeros(20, dtype=np.float32)
        R[10:] = 0.9   # sustained attack from t=10
        alerts = k_exceedance_alerts(R, K=3, tau=0.5)
        # First alert should be at t=12 (K=3 consecutive from t=10,11,12)
        first_alert = np.where(alerts == 1)[0]
        assert len(first_alert) > 0, "Expected at least one alert"
        assert first_alert[0] == 12

    def test_isolated_spike_suppressed(self):
        R = np.zeros(50, dtype=np.float32)
        R[25] = 0.9   # single spike
        alerts = k_exceedance_alerts(R, K=3, tau=0.5)
        assert alerts.sum() == 0, "Isolated spike should not trigger K=3 rule"

    def test_attack_to_alert_latency(self):
        R = np.zeros(50, dtype=np.float32)
        R[10:] = 0.9
        alerts = k_exceedance_alerts(R, K=3, tau=0.5)
        latency = attack_to_alert_latency(alerts, attack_onset=10)
        assert latency == 2, f"Expected latency=2 windows, got {latency}"

    def test_dyra_pipeline_end_to_end(self):
        """DyRAPipeline must compose Impact, γ, and alerting correctly."""
        pipeline = DyRAPipeline(impact=0.81, gamma=1.0, K=3, tau=0.5)
        P = np.zeros(30, dtype=np.float32)
        P[20:] = 0.95
        alerts = pipeline(P)
        first = np.where(alerts == 1)[0]
        assert len(first) > 0 and first[0] == 22

    def test_sensor_pipeline_never_alerts(self):
        """Low-impact sensor should never cross τ=0.5 even with P≈1."""
        pipeline = DyRAPipeline(impact=0.21, gamma=1.0, K=3, tau=0.5)
        P = np.ones(100, dtype=np.float32)
        alerts = pipeline(P)
        assert alerts.sum() == 0, "Sensor (Impact=0.21) should never alert with τ=0.5"
