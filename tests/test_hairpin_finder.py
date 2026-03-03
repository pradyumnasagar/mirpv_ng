# tests/test_hairpin_finder.py
"""
Tests for mirpv_ng/geom_hairpin_finder.py
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from mirpv_ng.geom_hairpin_finder import (
    find_primary_hairpin,
    find_hairpins,
    HairpinGeometry,
)


class TestFindPrimaryHairpin:
    """Tests for find_primary_hairpin function."""

    def test_find_primary_hairpin_valid_structure(self):
        """Test that a valid hairpin structure is detected."""
        seq = "GGCCAUUAGGCC"  # Simple stem-loop
        struct = "((((....))))"
        
        hp = find_primary_hairpin(seq, struct)
        
        assert hp is not None
        assert isinstance(hp, HairpinGeometry)
        assert hp.num_pairs > 0

    def test_find_primary_hairpin_returns_geometry(self):
        """Test that returned HairpinGeometry has all expected fields."""
        seq = "GGGCCCAAAGGGCCC"
        struct = "(((((.....)))))."
        
        hp = find_primary_hairpin(seq, struct)
        
        assert hp is not None
        assert hasattr(hp, "start")
        assert hasattr(hp, "end")
        assert hasattr(hp, "loop_start")
        assert hasattr(hp, "loop_end")
        assert hasattr(hp, "num_pairs")
        assert hasattr(hp, "helix_count")
        assert hasattr(hp, "anchor_stem_len")

    def test_find_primary_hairpin_empty_sequence(self):
        """Test that empty sequence returns None."""
        hp = find_primary_hairpin("", "")
        assert hp is None

    def test_find_primary_hairpin_no_pairs(self):
        """Test that unpaired structure returns None."""
        seq = "AAAAAAAA"
        struct = "........"
        
        hp = find_primary_hairpin(seq, struct)
        assert hp is None


class TestFindHairpins:
    """Tests for find_hairpins function."""

    def test_find_hairpins_returns_list(self):
        """Test that find_hairpins returns a list."""
        seq = "GGCCAUUAGGCC"
        struct = "((((....))))"
        
        hairpins = find_hairpins(seq, struct)
        
        assert isinstance(hairpins, list)
        assert len(hairpins) >= 1

    def test_find_hairpins_empty_for_no_pairs(self):
        """Test that unpaired structure returns empty list."""
        seq = "AAAAAAAA"
        struct = "........"
        
        hairpins = find_hairpins(seq, struct)
        
        assert hairpins == []


class TestHairpinGeometryAttributes:
    """Tests for HairpinGeometry dataclass attributes."""

    def test_hairpin_geometry_stem_runs(self):
        """Test that stem runs are populated."""
        seq = "GGCCAUUAGGCC"
        struct = "((((....))))"
        
        hp = find_primary_hairpin(seq, struct)
        
        assert hp is not None
        assert hasattr(hp, "stem_runs")
        assert len(hp.stem_runs) > 0

    def test_hairpin_geometry_loop_stats(self):
        """Test that loop statistics are computed."""
        seq = "GGGCCCAAAAGGGCCC"
        struct = "((((((....))).))"
        
        hp = find_primary_hairpin(seq, struct)
        
        assert hp is not None
        assert hp.num_loops >= 0
        assert hp.max_loop_size >= 0

    def test_hairpin_geometry_bulge_features(self):
        """Test that bulge features dict is populated."""
        seq = "GGCCAUUAGGCC"
        struct = "((((....))))"
        
        hp = find_primary_hairpin(seq, struct)
        
        assert hp is not None
        assert isinstance(hp.bulge_features, dict)
