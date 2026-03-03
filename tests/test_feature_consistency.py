# tests/test_feature_consistency.py
"""
Tests for feature-column consistency, input validation, and edge cases.
Covers audit fixes H1 (missing feature warning), M1/M6 (ambiguous bases),
M2 (deterministic merge), and general edge-case hardening.
"""

import logging
import os
import tempfile

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from mirpv_ng.features import (
    core36_features,
    extended_features,
    read_fasta,
    classify_loops,
    bulge_stats,
    run_rnafold,
)
from mirpv_ng.tier_filters import Tier2Config
from mirpv_ng.classifier import generate_windows


# ==================== Feature Schema Tests ====================

class TestFeatureSchema:
    """Verify that feature extraction produces consistent keys and no NaNs."""

    HAIRPIN_SEQ = "CCUGCCGACUAUGCCAAUUGUCAGGUCCCAACCUGGGCUGGGGUCCGAGGGGAGUUGGUAGAUGGGUGG"
    HAIRPIN_STRUCT = "(((((((.((((((((((((((......))))))))...........))))))))))))......."
    HAIRPIN_MFE = -25.5

    def test_core36_no_nan_values(self):
        feats = core36_features(self.HAIRPIN_SEQ, self.HAIRPIN_STRUCT, self.HAIRPIN_MFE)
        for k, v in feats.items():
            assert v == v, f"NaN found in core36 feature: {k}"  # NaN != NaN

    def test_extended_no_nan_values(self):
        feats = extended_features(self.HAIRPIN_SEQ, self.HAIRPIN_STRUCT, self.HAIRPIN_MFE)
        for k, v in feats.items():
            assert v == v, f"NaN found in extended feature: {k}"

    def test_extended_superset_of_core36(self):
        core = core36_features(self.HAIRPIN_SEQ, self.HAIRPIN_STRUCT, self.HAIRPIN_MFE)
        ext = extended_features(self.HAIRPIN_SEQ, self.HAIRPIN_STRUCT, self.HAIRPIN_MFE)
        for k in core:
            assert k in ext, f"Core feature '{k}' missing from extended set"

    def test_extended_tier2_enabled_has_same_keys(self):
        off = extended_features(self.HAIRPIN_SEQ, self.HAIRPIN_STRUCT, self.HAIRPIN_MFE,
                                tier2_cfg=Tier2Config(enabled=False))
        on = extended_features(self.HAIRPIN_SEQ, self.HAIRPIN_STRUCT, self.HAIRPIN_MFE,
                               tier2_cfg=Tier2Config(enabled=True))
        assert set(off.keys()) == set(on.keys()), "Tier2 on/off should produce same feature keys"

    def test_core36_all_values_finite(self):
        feats = core36_features(self.HAIRPIN_SEQ, self.HAIRPIN_STRUCT, self.HAIRPIN_MFE)
        import math
        for k, v in feats.items():
            assert math.isfinite(v), f"Non-finite value in core36 feature '{k}': {v}"

    @pytest.mark.integration
    def test_model_columns_match_extraction(self, rf_model_path):
        """Verify all model feature_cols are produced by extraction."""
        if not rf_model_path.exists():
            pytest.skip(f"Model not found: {rf_model_path}")
        from mirpv_ng.classifier import load_rf_model
        model_info = load_rf_model(rf_model_path)
        feats = extended_features(
            self.HAIRPIN_SEQ, self.HAIRPIN_STRUCT, self.HAIRPIN_MFE,
            tier2_cfg=Tier2Config(enabled=model_info.tier2_enabled),
        )
        missing = [c for c in model_info.feature_cols if c not in feats]
        assert not missing, f"Model expects features not produced by extraction: {missing}"


# ==================== Feature Warning Tests ====================

class TestFeatureWarnings:
    """Test that the missing-feature warning fires correctly."""

    def test_vector_from_features_warns_on_missing(self, rf_model_path, caplog):
        """H1 fix: verify warning when model expects features not in feats dict."""
        if not rf_model_path.exists():
            pytest.skip(f"Model not found: {rf_model_path}")

        from mirpv_ng.classifier import HairpinClassifier
        import numpy as np

        clf = HairpinClassifier(model_path=rf_model_path)

        # Provide an intentionally incomplete feature dict
        incomplete_feats = {"len": 70.0, "gc_frac": 0.5}
        with caplog.at_level(logging.WARNING, logger="mirpv_ng.classifier"):
            clf._vector_from_features(incomplete_feats)
        assert any("missing" in r.message.lower() for r in caplog.records), \
            "Expected a warning about missing features"


# ==================== Input Validation Tests ====================

class TestInputValidation:
    """Test FASTA reader edge cases and ambiguous base warnings."""

    def test_empty_fasta(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
            f.write("")
            fasta_path = f.name
        try:
            records = read_fasta(fasta_path)
            assert records == []
        finally:
            os.unlink(fasta_path)

    def test_fasta_header_only(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
            f.write(">seq_with_no_body\n")
            fasta_path = f.name
        try:
            records = read_fasta(fasta_path)
            assert len(records) == 1
            assert records[0][0] == "seq_with_no_body"
            assert records[0][1] == ""
        finally:
            os.unlink(fasta_path)

    def test_fasta_ambiguous_bases_warns(self, caplog):
        """M1 fix: non-ACGTU characters should trigger a warning."""
        content = ">ambiguous\nACGTNRYSW\n"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
            f.write(content)
            fasta_path = f.name
        try:
            with caplog.at_level(logging.WARNING, logger="mirpv_ng.features"):
                records = read_fasta(fasta_path)
            assert len(records) == 1
            assert records[0][1] == "ACGTNRYSW"  # preserved, not stripped
            assert any("non-ACGTU" in r.message for r in caplog.records), \
                "Expected warning about non-ACGTU characters"
        finally:
            os.unlink(fasta_path)

    def test_fasta_gz_support(self):
        import gzip
        content = b">gzseq\nACGUACGU\n"
        with tempfile.NamedTemporaryFile(suffix='.fa.gz', delete=False) as f:
            gz_path = f.name
        try:
            with gzip.open(gz_path, 'wb') as gz:
                gz.write(content)
            records = read_fasta(gz_path)
            assert len(records) == 1
            assert records[0][0] == "gzseq"
            assert records[0][1] == "ACGUACGU"
        finally:
            os.unlink(gz_path)


# ==================== Edge Case Feature Tests ====================

class TestEdgeCaseFeatures:
    """Test feature extraction on boundary inputs."""

    def test_single_base_sequence(self):
        feats = core36_features("A", ".", 0.0)
        assert feats["len"] == 1.0
        assert feats["gc_frac"] == 0.0

    def test_all_paired_structure(self):
        seq = "GGGGCCCC"
        struct = "(((())))"
        feats = core36_features(seq, struct, -5.0)
        assert feats["num_pairs"] == 4.0
        assert feats["paired_frac"] == 1.0  # 4*2/8

    def test_zero_mfe(self):
        feats = core36_features("AAAA", "....", 0.0)
        assert feats["mfe"] == 0.0
        assert feats["mfe_per_nt"] == 0.0

    def test_classify_loops_single_dot(self):
        # Single dot flanked by brackets
        struct = "((.(()))" 
        num_loops, mean_loop, max_loop, loop_frac = classify_loops(struct)
        assert num_loops >= 0  # at least should not crash


# ==================== Determinism Tests ====================

class TestDeterminism:
    """Test that outputs are deterministic given the same input."""

    def test_generate_windows_deterministic(self):
        seq = "A" * 200
        w1 = generate_windows(seq, window_len=80, step=30)
        w2 = generate_windows(seq, window_len=80, step=30)
        assert w1 == w2

    def test_core36_deterministic(self):
        seq = "CCUGCCGACUAUGCCAAUUGUCAGGUCCCAACCUGGGCUGGGGUCCGAGGGGAGUUGGUAGAUGGGUGG"
        struct = "(((((((.((((((((((((((......))))))))...........))))))))))))......."
        f1 = core36_features(seq, struct, -25.5)
        f2 = core36_features(seq, struct, -25.5)
        assert f1 == f2

    def test_extended_deterministic(self):
        seq = "CCUGCCGACUAUGCCAAUUGUCAGGUCCCAACCUGGGCUGGGGUCCGAGGGGAGUUGGUAGAUGGGUGG"
        struct = "(((((((.((((((((((((((......))))))))...........))))))))))))......."
        f1 = extended_features(seq, struct, -25.5)
        f2 = extended_features(seq, struct, -25.5)
        assert f1 == f2
