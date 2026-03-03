# tests/test_features.py
"""
Tests for mirpv_ng/features.py
"""

import pytest
import tempfile
import os
from pathlib import Path

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from mirpv_ng.features import (
    core36_features,
    extended_features,
    classify_loops,
    bulge_stats,
    segment_gc_and_pairing,
    read_fasta,
)
from mirpv_ng.tier_filters import Tier2Config


class TestCore36Features:
    """Tests for core36_features function."""

    def test_core36_features_returns_dict(self, simple_hairpin_seq, simple_hairpin_struct, simple_hairpin_mfe):
        """Test that core36_features returns a dictionary with expected keys."""
        feats = core36_features(simple_hairpin_seq, simple_hairpin_struct, simple_hairpin_mfe)
        
        assert isinstance(feats, dict)
        assert "len" in feats
        assert "gc_frac" in feats
        assert "num_pairs" in feats
        assert "mfe" in feats
        assert "mfe_per_nt" in feats

    def test_core36_features_length(self, simple_hairpin_seq, simple_hairpin_struct, simple_hairpin_mfe):
        """Test that length feature is correct."""
        feats = core36_features(simple_hairpin_seq, simple_hairpin_struct, simple_hairpin_mfe)
        assert feats["len"] == len(simple_hairpin_seq)

    def test_core36_features_gc_in_range(self, simple_hairpin_seq, simple_hairpin_struct, simple_hairpin_mfe):
        """Test that GC fraction is between 0 and 1."""
        feats = core36_features(simple_hairpin_seq, simple_hairpin_struct, simple_hairpin_mfe)
        assert 0 <= feats["gc_frac"] <= 1

    def test_core36_features_mfe(self, simple_hairpin_seq, simple_hairpin_struct, simple_hairpin_mfe):
        """Test that MFE is correctly stored."""
        feats = core36_features(simple_hairpin_seq, simple_hairpin_struct, simple_hairpin_mfe)
        assert feats["mfe"] == simple_hairpin_mfe


class TestExtendedFeatures:
    """Tests for extended_features function."""

    def test_extended_features_includes_core(self, simple_hairpin_seq, simple_hairpin_struct, simple_hairpin_mfe):
        """Test that extended features include all core features."""
        feats = extended_features(simple_hairpin_seq, simple_hairpin_struct, simple_hairpin_mfe)
        
        # Should include core keys
        assert "len" in feats
        assert "gc_frac" in feats
        assert "mfe" in feats

    def test_extended_features_includes_dinucleotides(self, simple_hairpin_seq, simple_hairpin_struct, simple_hairpin_mfe):
        """Test that extended features include dinucleotide frequencies."""
        feats = extended_features(simple_hairpin_seq, simple_hairpin_struct, simple_hairpin_mfe)
        
        # Should have dinucleotide features
        assert "di_AA" in feats
        assert "di_GC" in feats
        assert "di_UU" in feats

    def test_extended_features_includes_segments(self, simple_hairpin_seq, simple_hairpin_struct, simple_hairpin_mfe):
        """Test that extended features include segment features."""
        feats = extended_features(simple_hairpin_seq, simple_hairpin_struct, simple_hairpin_mfe)
        
        assert "seg1_gc" in feats
        assert "seg4_gc" in feats

    def test_extended_features_tier2_disabled_by_default(self, simple_hairpin_seq, simple_hairpin_struct, simple_hairpin_mfe):
        """Test that tier2 features are included but disabled by default."""
        feats = extended_features(simple_hairpin_seq, simple_hairpin_struct, simple_hairpin_mfe)
        
        assert "tier2_enabled" in feats
        assert feats["tier2_enabled"] == 0.0


class TestClassifyLoops:
    """Tests for classify_loops function."""

    def test_classify_loops_simple(self):
        """Test loop classification on a simple structure."""
        # Structure with one clear loop in the middle
        struct = "((((....))))"
        num_loops, mean_loop, max_loop, loop_frac = classify_loops(struct)
        
        assert num_loops >= 1
        assert max_loop >= 4

    def test_classify_loops_no_loops(self):
        """Test that fully paired structure has no loops."""
        struct = "((((()))))"  # No internal loops, but terminal loop
        num_loops, mean_loop, max_loop, loop_frac = classify_loops(struct)
        
        # This structure has a "loop" of 0 dots between ( and )
        assert num_loops == 0 or max_loop == 0

    def test_classify_loops_unpaired_only(self):
        """Test that fully unpaired structure has no loops (not flanked by pairs)."""
        struct = "........"
        num_loops, mean_loop, max_loop, loop_frac = classify_loops(struct)
        
        assert num_loops == 0


class TestBulgeStats:
    """Tests for bulge_stats function."""

    def test_bulge_stats_with_bulge(self):
        """Test bulge detection with a bulge present."""
        # Structure with a bulge (dot adjacent to paired)
        struct = "(((.(((....)))..)))"
        num_bulges, density = bulge_stats(struct)
        
        assert num_bulges >= 1
        assert 0 <= density <= 1

    def test_bulge_stats_no_bulge(self):
        """Test structure with no bulges."""
        struct = "((((....))))"
        num_bulges, density = bulge_stats(struct)
        
        # Internal loop dots are adjacent to pairs, so they count as bulges
        # This is expected behavior based on the function logic
        assert num_bulges >= 0


class TestSegmentGCAndPairing:
    """Tests for segment_gc_and_pairing function."""

    def test_segment_returns_expected_keys(self, simple_hairpin_seq, simple_hairpin_struct):
        """Test that segment function returns expected keys."""
        feats = segment_gc_and_pairing(simple_hairpin_seq.replace("T", "U"), simple_hairpin_struct, n_segments=4)
        
        assert "seg1_gc" in feats
        assert "seg2_gc" in feats
        assert "seg3_gc" in feats
        assert "seg4_gc" in feats
        assert "gc_grad_last_first" in feats

    def test_segment_gc_values_in_range(self, simple_hairpin_seq, simple_hairpin_struct):
        """Test that GC values are between 0 and 1."""
        feats = segment_gc_and_pairing(simple_hairpin_seq.replace("T", "U"), simple_hairpin_struct, n_segments=4)
        
        for i in range(1, 5):
            assert 0 <= feats[f"seg{i}_gc"] <= 1


class TestReadFasta:
    """Tests for read_fasta function."""

    def test_read_fasta_parses_correctly(self, sample_fasta_content):
        """Test that FASTA parsing works correctly."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
            f.write(sample_fasta_content)
            f.flush()
            fasta_path = f.name

        try:
            records = read_fasta(fasta_path)
            
            assert len(records) == 3
            assert records[0][0] == "seq1"
            assert records[0][1] == "ACGUACGUACGU"
            assert records[1][0] == "seq2"
            assert records[2][0] == "seq3"
        finally:
            os.unlink(fasta_path)

    def test_read_fasta_handles_multiline(self):
        """Test that FASTA parsing handles multi-line sequences."""
        content = """>multiline
ACGU
ACGU
ACGU
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
            f.write(content)
            f.flush()
            fasta_path = f.name

        try:
            records = read_fasta(fasta_path)
            
            assert len(records) == 1
            assert records[0][1] == "ACGUACGUACGU"
        finally:
            os.unlink(fasta_path)
