# tests/test_tier_filters.py
"""
Tests for mirpv_ng/tier_filters.py
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from mirpv_ng.tier_filters import (
    TierConfig,
    tier1_energy_filter,
    GeometryConfig,
    tier2_geometry_filter,
    Tier2Config,
    tier2_soft_features,
)


class TestTier1EnergyFilter:
    """Tests for tier1_energy_filter function."""

    def test_tier1_filter_pass(self):
        """Test that a valid sequence passes Filter 1."""
        seq = "A" * 60  # 60nt sequence
        struct = "(" * 20 + "." * 20 + ")" * 20  # 20 pairs, 20 unpaired
        mfe = -20.0  # Good MFE
        cfg = TierConfig(min_len=40, max_len=120, min_pairs=18, min_mfe=-15.0)
        
        result = tier1_energy_filter(seq, struct, mfe, cfg)
        
        assert result is True

    def test_tier1_filter_fail_too_short(self):
        """Test that too-short sequence fails."""
        seq = "A" * 30  # Only 30nt
        struct = "(" * 10 + "." * 10 + ")" * 10
        mfe = -20.0
        cfg = TierConfig(min_len=40, max_len=120, min_pairs=18, min_mfe=-15.0)
        
        result = tier1_energy_filter(seq, struct, mfe, cfg)
        
        assert result is False

    def test_tier1_filter_fail_too_long(self):
        """Test that too-long sequence fails."""
        seq = "A" * 150  # 150nt
        struct = "(" * 50 + "." * 50 + ")" * 50
        mfe = -40.0
        cfg = TierConfig(min_len=40, max_len=120, min_pairs=18, min_mfe=-15.0)
        
        result = tier1_energy_filter(seq, struct, mfe, cfg)
        
        assert result is False

    def test_tier1_filter_fail_insufficient_pairs(self):
        """Test that sequence with too few pairs fails."""
        seq = "A" * 60
        struct = "(" * 10 + "." * 40 + ")" * 10  # Only 10 pairs
        mfe = -20.0
        cfg = TierConfig(min_len=40, max_len=120, min_pairs=18, min_mfe=-15.0)
        
        result = tier1_energy_filter(seq, struct, mfe, cfg)
        
        assert result is False

    def test_tier1_filter_fail_poor_mfe(self):
        """Test that sequence with insufficient MFE fails."""
        seq = "A" * 60
        struct = "(" * 20 + "." * 20 + ")" * 20
        mfe = -10.0  # Not negative enough (> -15)
        cfg = TierConfig(min_len=40, max_len=120, min_pairs=18, min_mfe=-15.0)
        
        result = tier1_energy_filter(seq, struct, mfe, cfg)
        
        assert result is False


class TestTier2GeometryFilter:
    """Tests for tier2_geometry_filter function."""

    def test_tier2_filter_all_none_passes(self):
        """Test that disabled geometry filter always passes."""
        cfg = GeometryConfig()  # All None by default
        
        # Create a mock object with some attributes
        class MockHP:
            num_loops = 10
            loop_size = 50
        
        result = tier2_geometry_filter(MockHP(), cfg)
        
        assert result is True

    def test_tier2_filter_fails_on_loop_count(self):
        """Test that too many loops causes failure."""
        cfg = GeometryConfig(max_num_loops=3)
        
        class MockHP:
            num_loops = 5  # > 3
        
        result = tier2_geometry_filter(MockHP(), cfg)
        
        assert result is False


class TestTier2SoftFeatures:
    """Tests for tier2_soft_features function."""

    def test_tier2_soft_disabled_returns_zeros(self):
        """Test that disabled tier2 returns zero penalties."""
        seq = "GGCCAUUAGGCC"
        struct = "((((....))))"
        mfe = -10.0
        
        feats = tier2_soft_features(seq, struct, mfe, Tier2Config(enabled=False))
        
        assert feats["tier2_enabled"] == 0.0
        assert feats["tier2_penalty"] == 0.0

    def test_tier2_soft_enabled_returns_values(self):
        """Test that enabled tier2 computes values."""
        seq = "GGCCAUUAGGCC"
        struct = "((((....))))"
        mfe = -10.0
        
        feats = tier2_soft_features(seq, struct, mfe, Tier2Config(enabled=True))
        
        assert feats["tier2_enabled"] == 1.0
        assert "tier2_mfe_per_nt" in feats
        assert "tier2_loop_len" in feats
        assert "tier2_unpaired_frac" in feats

    def test_tier2_soft_schema_stability(self):
        """Test that tier2 always returns the same keys."""
        seq = "ACGU"
        struct = "...."
        
        feats_disabled = tier2_soft_features(seq, struct, 0.0, Tier2Config(enabled=False))
        feats_enabled = tier2_soft_features(seq, struct, 0.0, Tier2Config(enabled=True))
        
        assert set(feats_disabled.keys()) == set(feats_enabled.keys())
