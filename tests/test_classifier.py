# tests/test_classifier.py
"""
Tests for mirpv_ng/classifier.py
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from mirpv_ng.classifier import (
    load_rf_model,
    generate_windows,
    ModelInfo,
)


class TestLoadRFModel:
    """Tests for load_rf_model function."""

    def test_load_rf_model_returns_model_info(self, rf_model_path):
        """Test that load_rf_model returns a ModelInfo object."""
        if not rf_model_path.exists():
            pytest.skip(f"Model file not found: {rf_model_path}")
        
        model_info = load_rf_model(rf_model_path)
        
        assert isinstance(model_info, ModelInfo)
        assert model_info.model is not None

    def test_load_rf_model_has_feature_set(self, rf_model_path):
        """Test that loaded model has feature_set metadata."""
        if not rf_model_path.exists():
            pytest.skip(f"Model file not found: {rf_model_path}")
        
        model_info = load_rf_model(rf_model_path)
        
        assert model_info.feature_set is not None
        assert model_info.feature_set in ["extended", "core36"]

    def test_load_rf_model_has_feature_cols(self, rf_model_path):
        """Test that loaded model has feature columns."""
        if not rf_model_path.exists():
            pytest.skip(f"Model file not found: {rf_model_path}")
        
        model_info = load_rf_model(rf_model_path)
        
        assert isinstance(model_info.feature_cols, list)
        assert len(model_info.feature_cols) > 0


class TestGenerateWindows:
    """Tests for generate_windows function."""

    def test_generate_windows_short_seq(self):
        """Test that short sequence returns single window."""
        seq = "ACGUACGU"  # 8nt
        windows = generate_windows(seq, window_len=20, step=5)
        
        assert len(windows) == 1
        assert windows[0] == (0, seq)

    def test_generate_windows_exact_length(self):
        """Test that exact-length sequence returns single window."""
        seq = "A" * 20
        windows = generate_windows(seq, window_len=20, step=5)
        
        assert len(windows) == 1
        assert windows[0] == (0, seq)

    def test_generate_windows_long_seq(self):
        """Test that long sequence generates multiple windows."""
        seq = "A" * 50
        windows = generate_windows(seq, window_len=20, step=10)
        
        # Should have multiple windows
        assert len(windows) > 1
        
        # All windows should be same length
        for start, wseq in windows:
            assert len(wseq) == 20

    def test_generate_windows_coverage(self):
        """Test that windows cover the entire sequence."""
        seq = "A" * 35
        windows = generate_windows(seq, window_len=20, step=10)
        
        # First window starts at 0
        assert windows[0][0] == 0
        
        # Last window should reach the end
        last_start, last_seq = windows[-1]
        assert last_start + len(last_seq) >= len(seq)

    def test_generate_windows_step_smaller_than_window(self):
        """Test overlapping windows with step < window_len."""
        seq = "ACGUACGUACGUACGUACGUACGUACGUACGU"  # 32nt
        windows = generate_windows(seq, window_len=20, step=5)
        
        # Should have multiple overlapping windows
        assert len(windows) >= 2
        
        # Windows should overlap
        if len(windows) >= 2:
            first_end = windows[0][0] + 20
            second_start = windows[1][0]
            assert second_start < first_end  # Overlap


class TestHairpinClassifier:
    """Integration tests for HairpinClassifier."""

    @pytest.mark.integration
    def test_hairpin_classifier_init(self, rf_model_path):
        """Test that HairpinClassifier can be initialized."""
        if not rf_model_path.exists():
            pytest.skip(f"Model file not found: {rf_model_path}")
        
        from mirpv_ng.classifier import HairpinClassifier
        
        clf = HairpinClassifier(
            model_path=rf_model_path,
            species="hsa",
            feature_set="extended",
        )
        
        assert clf is not None
        assert clf.species == "hsa"

    @pytest.mark.integration
    def test_hairpin_classifier_score_returns_dict(self, rf_model_path, simple_hairpin_seq):
        """Test that scoring returns expected schema."""
        if not rf_model_path.exists():
            pytest.skip(f"Model file not found: {rf_model_path}")
        
        try:
            from mirpv_ng.classifier import HairpinClassifier
            
            clf = HairpinClassifier(model_path=rf_model_path)
            result = clf.score_hairpin("test_id", simple_hairpin_seq)
            
            assert isinstance(result, dict)
            assert "input_id" in result
            assert "rf_score" in result
            assert "pred_label" in result
            assert 0 <= result["rf_score"] <= 1
        except Exception as e:
            # RNAfold may not be available in test environment
            if "RNAfold" in str(e):
                pytest.skip("RNAfold not available")
            raise
