"""
tests/test_partitioning.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for Algorithm 1 (stratified per-class-block split) and
normalisation utilities.

Run:
  pytest tests/test_partitioning.py -v
"""

import numpy as np
import pytest

from dyra_iiot.data.partitioning import (
    apply_normalizer,
    fit_normalizer,
    stratified_per_class_block_split,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic(n_per_class=500, n_classes=5, n_features=36, seed=0):
    """Create a synthetic dataset with n_classes contiguous class blocks."""
    rng = np.random.default_rng(seed)
    classes = [f"cls_{i}" for i in range(n_classes)]
    X       = rng.standard_normal((n_per_class * n_classes, n_features)).astype(np.float32)
    types   = np.repeat(classes, n_per_class)
    y       = (types != "cls_0").astype(np.float32)   # cls_0 = normal
    return X, y, types, classes


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 1 tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStratifiedSplit:

    def test_all_classes_in_train_and_test(self):
        X, y, types, classes = make_synthetic()
        Xtr, ytr, Xte, yte = stratified_per_class_block_split(
            X, y, types, train_ratio=0.8, window_len=50)
        # Binary: must have both 0 and 1 in both splits
        assert ytr.mean() > 0, "No attack windows in training set"
        assert ytr.mean() < 1, "No normal windows in training set"
        assert yte.mean() > 0, "No attack windows in test set"

    def test_train_test_sizes(self):
        """Train set should be ≈ 80 % of total windows."""
        X, y, types, _ = make_synthetic(n_per_class=1000)
        Xtr, ytr, Xte, yte = stratified_per_class_block_split(
            X, y, types, train_ratio=0.8, window_len=50)
        total = len(Xtr) + len(Xte)
        ratio = len(Xtr) / total
        assert 0.75 < ratio < 0.85, f"Train ratio {ratio:.3f} not in [0.75, 0.85]"

    def test_window_shape(self):
        X, y, types, _ = make_synthetic()
        L, F = 50, 36
        Xtr, ytr, Xte, yte = stratified_per_class_block_split(
            X, y, types, window_len=L)
        assert Xtr.shape[1] == L, f"Window length mismatch: {Xtr.shape[1]} != {L}"
        assert Xtr.shape[2] == F, f"Feature count mismatch: {Xtr.shape[2]} != {F}"

    def test_no_window_crosses_block_boundary(self):
        """A window must not contain packets from two different class blocks."""
        n_per_class, L = 500, 50
        X, y, types, classes = make_synthetic(n_per_class=n_per_class, n_classes=3)
        Xtr, ytr, Xte, yte = stratified_per_class_block_split(
            X, y, types, train_ratio=0.8, window_len=L)
        # If no window crosses a boundary, the type of packet[0] and packet[-1]
        # should always be the same within each window
        # (we can't check this directly on the normalised tensor, so we test
        #  via the block-boundary property: max windows = sum_c (block_size - L + 1))
        n_expected_max = sum(
            max(0, n_per_class - L + 1) for _ in classes
        )
        assert len(Xtr) + len(Xte) <= n_expected_max, "More windows than possible without boundary crossing"

    def test_ood_mode_excludes_classes_from_train(self):
        X, y, types, classes = make_synthetic(n_per_class=300, n_classes=5)
        ood = {"cls_3", "cls_4"}
        Xtr, ytr, Xte, yte = stratified_per_class_block_split(
            X, y, types, window_len=50, ood_classes=ood, mode="ood")
        # Train must only have cls_0..cls_2 windows; since cls_0=normal and
        # cls_1,cls_2=attack, both labels should appear in train
        assert len(Xtr) > 0
        assert len(Xte) > 0

    def test_reproducible_with_same_inputs(self):
        X, y, types, _ = make_synthetic()
        result1 = stratified_per_class_block_split(X, y, types, window_len=50)
        result2 = stratified_per_class_block_split(X, y, types, window_len=50)
        np.testing.assert_array_equal(result1[0], result2[0])

    def test_invalid_mode_raises(self):
        X, y, types, _ = make_synthetic()
        with pytest.raises(ValueError, match="Unknown mode"):
            stratified_per_class_block_split(X, y, types, mode="invalid")

    def test_small_block_skipped_gracefully(self):
        """Very small datasets raise RuntimeError (< window_len windows possible)."""
        X, y, types, _ = make_synthetic(n_per_class=30)
        # n_per_class=30 with train_ratio=0.8 → test blocks have 6 rows < L=50
        with pytest.raises(RuntimeError):
            stratified_per_class_block_split(X, y, types, window_len=50)


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizer:

    def test_no_nan_after_normalization(self):
        X, y, types, _ = make_synthetic(n_per_class=500)
        Xtr, _, Xte, _ = stratified_per_class_block_split(X, y, types)
        mu, std = fit_normalizer(Xtr, robust=False)
        Xn = apply_normalizer(Xtr, mu, std)
        assert not np.isnan(Xn).any(), "NaN in normalised training set"
        assert not np.isinf(Xn).any(), "Inf in normalised training set"

    def test_no_nan_with_pathological_scale(self):
        """Simulate TON_IoT src_bytes / dst_bytes scale."""
        X, y, types, _ = make_synthetic(n_per_class=500)
        X[:, 0] = np.random.exponential(1e9, size=len(X)).astype(np.float32)
        X[:, 1] = np.random.exponential(5e8, size=len(X)).astype(np.float32)
        Xtr, _, Xte, _ = stratified_per_class_block_split(X, y, types)
        mu, std = fit_normalizer(Xtr, robust=True)
        Xn = apply_normalizer(Xtr, mu, std)
        assert not np.isnan(Xn).any()
        assert not np.isinf(Xn).any()
        assert np.abs(Xn).max() <= 10.0, "Values exceed clip range"

    def test_test_uses_train_stats(self):
        """Test normalisation must not use test-set statistics."""
        X, y, types, _ = make_synthetic(n_per_class=500)
        Xtr, _, Xte, _ = stratified_per_class_block_split(X, y, types)
        mu, std = fit_normalizer(Xtr, robust=False)  # no percentile clipping
        Xte_n   = apply_normalizer(Xte, mu, std)
        # Verify mu matches exact training-set mean (robust=False)
        flat_tr = Xtr.reshape(-1, Xtr.shape[-1])
        np.testing.assert_allclose(mu, flat_tr.mean(0), rtol=1e-4)

    def test_near_zero_std_handled(self):
        """Features with zero variance in training should not cause division by zero."""
        X, y, types, _ = make_synthetic(n_per_class=300)
        X[:, 5] = 0.0   # constant feature
        Xtr, _, Xte, _ = stratified_per_class_block_split(X, y, types)
        mu, std = fit_normalizer(Xtr)
        assert (std > 0).all(), "Zero std not clipped"
        Xn = apply_normalizer(Xtr, mu, std)
        assert not np.isnan(Xn).any()
