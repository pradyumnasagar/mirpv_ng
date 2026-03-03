# mirpv_ng/classifier.py
"""
Classifier module for miRPV-NG pre-miRNA prediction.

This module provides the main classification interface for scoring RNA sequences
as potential pre-miRNA candidates using trained Random Forest models.

Classes:
    HairpinClassifier: Main classifier for scoring sequences/hairpins.
    ModelInfo: Container for loaded model metadata.

Functions:
    load_rf_model: Load a trained Random Forest model from disk.
    compute_feature_vector: Compute features for a single sequence.
    generate_windows: Generate sliding windows for long sequence scanning.

Example:
    >>> from mirpv_ng.classifier import HairpinClassifier
    >>> clf = HairpinClassifier("models/hsa_premirna_rf_extended_tier2_v6.pkl")
    >>> results = clf.score_sequence_record("seq1", "GGCCAUUAGGCC...")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

import joblib
import numpy as np

# Import parallel module for batch RNAfold
try:
    from .parallel import (
        ParallelConfig, run_rnafold_parallel,
    )
    HAS_PARALLEL = True
except ImportError:
    HAS_PARALLEL = False
    ParallelConfig = None

from .features import extended_features, core36_features, run_rnafold
from .tier_filters import TierConfig, GeometryConfig, Tier2Config, tier1_energy_filter, tier2_geometry_filter
from .geom_hairpin_finder import find_hairpins


@dataclass
class ModelInfo:
    """
    Container for loaded machine learning model and its metadata.
    
    Attributes:
        model: The trained sklearn model object (e.g., CalibratedClassifierCV or RandomForestClassifier).
        feature_set: Name of the feature set used for training ("extended" or "core36").
        feature_cols: Ordered list of feature column names expected by the model.
        decision_threshold: Optimal decision threshold for classification (default 0.5).
        tier2_enabled: Whether tier-2 features were enabled during training.
    """
    model: object
    feature_set: str
    feature_cols: List[str]
    decision_threshold: float
    tier2_enabled: bool = False

def load_rf_model(model_path: str | Path) -> ModelInfo:
    """
    Load a trained Random Forest model from a pickle file.
    
    The model file should be a joblib-serialized dictionary containing:
        - 'model': The trained sklearn classifier
        - 'feature_set': Name of feature set ("extended" or "core36")
        - 'feature_cols': List of feature column names
    
    Args:
        model_path: Path to the .pkl model file.
    
    Returns:
        ModelInfo object containing the model and metadata.
    
    Example:
        >>> model_info = load_rf_model("models/hsa_premirna_rf_v6.pkl")
        >>> print(model_info.feature_set)  # "extended"
    """
    model_path = Path(model_path)
    payload = joblib.load(model_path)
    
    # Robust feature_cols loading with fallback chain
    feature_cols = (
        payload.get("feature_cols") or 
        payload.get("feature_names_") or 
        payload.get("feature_names") or 
        []
    )
    
    # Robust threshold loading: use is-not-None so a stored 0.0 is preserved.
    threshold = 0.5
    for _k in ("decision_threshold", "threshold", "f1_threshold"):
        _v = payload.get(_k)
        if _v is not None:
            threshold = float(_v)
            break
    
    # Load tier2_enabled from model metadata
    tier2_enabled = payload.get("tier2_enabled", False)
    
    return ModelInfo(
        model=payload["model"], 
        feature_set=payload.get("feature_set", "extended"), 
        feature_cols=feature_cols,
        decision_threshold=threshold,
        tier2_enabled=tier2_enabled,
    )

def compute_feature_vector(
    seq: str,
    feature_set: str = "extended",
    mfe: float = None,
    struct: str = None,
    tier2_enabled: bool = False,
    rnafold_bin: str = "RNAfold",
) -> Dict[str, float]:

    if struct is None or mfe is None:
        struct, mfe = run_rnafold(seq, rnafold_bin=rnafold_bin)
    
    if feature_set == "core36":
        feats = core36_features(seq, struct, mfe)
    else:
        tier2_cfg = Tier2Config(enabled=tier2_enabled)
        feats = extended_features(seq, struct, mfe, tier2_cfg=tier2_cfg)
        
    if "mfe" not in feats: 
        feats["mfe"] = mfe
    return feats

def generate_windows(seq: str, window_len: int, step: int) -> List[Tuple[int, str]]:
    """
    Generate sliding windows over a sequence for scanning.
    
    Windows are generated with the specified length and step size. If the
    sequence is shorter than window_len, a single window containing the
    entire sequence is returned. An extra window is added at the end if
    needed to cover the full sequence.
    
    Args:
        seq: Input sequence to window.
        window_len: Length of each window in nucleotides.
        step: Step size between window starts.
    
    Returns:
        List of (start_position, window_sequence) tuples.
    
    Example:
        >>> windows = generate_windows("ACGU" * 10, window_len=20, step=10)
        >>> len(windows)  # 3 overlapping windows
    """
    n = len(seq)
    if n <= window_len: return [(0, seq)]
    windows = []
    for start in range(0, n - window_len + 1, step):
        windows.append((start, seq[start : start + window_len]))
    if windows and windows[-1][0] + window_len < n:
        start = n - window_len
        windows.append((start, seq[start : start + window_len]))
    return windows

class HairpinClassifier:
    """
    Main classifier for pre-miRNA candidate scoring.
    
    This classifier loads a trained Random Forest model and provides methods
    to score RNA sequences for their likelihood of being pre-miRNAs. It supports
    both direct scoring of short hairpin sequences and sliding-window scanning
    of longer sequences.
    
    Attributes:
        species: Species code (e.g., "hsa" for human).
        max_hairpin_len: Maximum length for direct hairpin scoring (default 120).
        max_seq_only_len: Maximum length for window scanning (default 5000).
        window_len: Window size for scanning (default 100).
        step: Step size between windows (default 20).
        tier2_enabled: Whether tier-2 soft-gating features are enabled.
        feature_set: Feature set name from loaded model.
    
    Example:
        >>> clf = HairpinClassifier("models/hsa_premirna_rf_v6.pkl")
        >>> result = clf.score_hairpin("seq1", "GGCCAUUAGGCC")
        >>> print(result["rf_score"])  # 0.85
    """
    
    def __init__(
        self,
        model_path: str | Path,
        species: str = "hsa",
        feature_set: str = None,
        rnafold_bin: str = "RNAfold",
        max_hairpin_len: int = 120,
        max_seq_only_len: int = 5000,
        window_len: int = 100,
        step: int = 20,
        tier1_min_pairs: int = 18,
        tier1_min_mfe: float = -15.0,
        tier2_enabled: bool = None,  # None = use model's tier2_enabled
    ):
        """
        Initialize the HairpinClassifier with a trained model.
        
        Args:
            model_path: Path to the trained .pkl model file.
            species: Species code for annotation (default "hsa").
            feature_set: Override feature set (default uses model's config).
            rnafold_bin: Path to RNAfold executable.
            max_hairpin_len: Max length for direct scoring mode.
            max_seq_only_len: Max length for window scanning mode.
            window_len: Window size for long sequence scanning.
            step: Step between windows during scanning.
            tier1_min_pairs: Minimum base pairs for tier-1 filter.
            tier1_min_mfe: Minimum MFE threshold for tier-1 filter.
            tier2_enabled: Enable tier-2 soft-gating features (None = use model's setting).
        """
        self.species = species
        self.rnafold_bin = rnafold_bin
        self.max_hairpin_len = max_hairpin_len
        self.max_seq_only_len = max_seq_only_len
        self.window_len = window_len
        self.step = step
        self.parallel_cfg = None  # Set via set_parallel_config()
        
        self.tier_cfg = TierConfig(
            min_len=40, max_len=max_hairpin_len, min_pairs=tier1_min_pairs,
            min_mfe=tier1_min_mfe, max_unpaired_frac=0.8
        )
        
        # Tier 2 Filters: Disabled by default (all None)
        
        self.geom_cfg = GeometryConfig(
            max_num_loops=None, 
            max_loop_size=None,
            max_bulge_chain=None,
            max_bulge_density=None,
            max_cactus_score=None,
        )

        self.model_info = load_rf_model(model_path)
        
        # Use model's feature_set if available, otherwise fallback
        self.feature_set = self.model_info.feature_set if self.model_info.feature_set else (feature_set or "extended")
        
        self.feature_cols = self.model_info.feature_cols
        self.decision_threshold = self.model_info.decision_threshold
        
        # Tier2: use model's setting if not explicitly overridden
        if tier2_enabled is None:
            self.tier2_enabled = self.model_info.tier2_enabled
            tier2_source = "model"
        else:
            self.tier2_enabled = tier2_enabled
            tier2_source = "override"
        
        print(f"[classifier] Model loaded. Feature Set: {self.feature_set}, Threshold: {self.decision_threshold:.4f}")
        print(f"[classifier] tier2_enabled={self.tier2_enabled} (source: {tier2_source})")

    def set_parallel_config(self, cfg: ParallelConfig) -> None:
        """Set parallelism configuration for batch processing."""
        self.parallel_cfg = cfg

    def _vector_from_features(self, feats: Dict[str, float]) -> np.ndarray:
        if not self.feature_cols:
            self.feature_cols = sorted(feats.keys())
        missing = [col for col in self.feature_cols if col not in feats]
        if missing:
            logger.warning(
                "[classifier] %d feature(s) expected by model but missing from "
                "extraction (defaulting to 0.0): %s",
                len(missing), missing[:10],
            )
        x = np.array([feats.get(col, 0.0) for col in self.feature_cols], dtype=float)
        return x.reshape(1, -1)

    def score_hairpin(self, seq_id: str, seq: str) -> Dict:
        """
        Score a short sequence directly as a hairpin candidate.
        
        This method bypasses tier-2 geometry filters and scores the sequence
        directly using the loaded Random Forest model.
        
        Args:
            seq_id: Identifier for the sequence.
            seq: RNA sequence to score.
        
        Returns:
            Dictionary with scoring results:
                - input_id: The sequence identifier
                - mode: "hairpin"
                - start, end: Position coordinates (0, len)
                - length: Sequence length
                - rf_score: Random Forest probability score (0-1)
                - pred_label: Binary prediction (1 if rf_score >= 0.5)
        """
        feats = compute_feature_vector(
            seq, feature_set=self.feature_set,
            tier2_enabled=self.tier2_enabled,
            rnafold_bin=self.rnafold_bin,
        )
        x = self._vector_from_features(feats)
        proba = self.model_info.model.predict_proba(x)[0, 1]
        return {
            "input_id": seq_id, "mode": "hairpin", "start": 0, "end": len(seq),
            "length": len(seq), "rf_score": float(proba), "pred_label": int(proba >= self.decision_threshold)
        }

    def scan_long_sequence(self, seq_id: str, seq: str) -> List[Dict]:
        """Scan long sequence (Applies Tier 2 Filters)."""
        windows = generate_windows(seq, self.window_len, self.step)
        candidates: List[Dict] = []

        for win_start, wseq in windows:
            struct, mfe = run_rnafold(wseq, rnafold_bin=self.rnafold_bin)

            # Tier 1 Filter (Energy/Pairs) — keep scan sane for long sequences
            if not tier1_energy_filter(wseq, struct, mfe, self.tier_cfg):
                continue

            # Find Hairpins
            hairpins = find_hairpins(wseq, struct)
            if not hairpins:
                continue

            for hp in hairpins:
                # Tier-2 geometry as SOFT gate: do not drop candidates
                tier2_geom_pass = 1.0 if tier2_geometry_filter(hp, self.geom_cfg) else 0.0

                # Extract and Score
                hp_seq = wseq[hp.start:hp.end]

                # Re-compute features on the cropped hairpin
                feats = compute_feature_vector(
                    hp_seq,
                    feature_set=self.feature_set,
                    tier2_enabled=self.tier2_enabled,
                    rnafold_bin=self.rnafold_bin,
                )
                feats["tier2_geom_pass"] = tier2_geom_pass

                # Use hairpin MFE if re-fold happened, otherwise fallback to window MFE (proxy)
                if "mfe" not in feats:
                    feats["mfe"] = mfe

                proba = float(self.model_info.model.predict_proba(self._vector_from_features(feats))[0, 1])

                candidates.append({
                    "input_id": seq_id, "mode": "scan",
                    "start": win_start + hp.start, "end": win_start + hp.end,
                    "length": hp.end - hp.start, "rf_score": proba, "pred_label": int(proba >= self.decision_threshold)
                })

        return self._merge_overlapping_candidates(candidates)

    def scan_long_sequence_parallel(self, seq_id: str, seq: str) -> List[Dict]:
        """
        Scan long sequence with parallel batch RNAfold processing.
        
        Uses batch folding for efficiency when parallel_cfg.jobs >= 1.
        """
        if not HAS_PARALLEL or self.parallel_cfg is None:
            return self.scan_long_sequence(seq_id, seq)
        
        windows = generate_windows(seq, self.window_len, self.step)
        candidates: List[Dict] = []
        
        # Batch fold all windows at once
        window_seqs = [wseq for _, wseq in windows]
        fold_results = run_rnafold_parallel(window_seqs, self.parallel_cfg)
        
        for i, (win_start, wseq) in enumerate(windows):
            struct, mfe = fold_results[i]
            
            if not tier1_energy_filter(wseq, struct, mfe, self.tier_cfg):
                continue
            
            hairpins = find_hairpins(wseq, struct)
            if not hairpins:
                continue
            
            for hp in hairpins:
                tier2_geom_pass = 1.0 if tier2_geometry_filter(hp, self.geom_cfg) else 0.0
                hp_seq = wseq[hp.start:hp.end]
                
                feats = compute_feature_vector(
                    hp_seq,
                    feature_set=self.feature_set,
                    tier2_enabled=self.tier2_enabled,
                    rnafold_bin=self.rnafold_bin,
                )
                feats["tier2_geom_pass"] = tier2_geom_pass

                if "mfe" not in feats:
                    feats["mfe"] = mfe

                proba = float(self.model_info.model.predict_proba(self._vector_from_features(feats))[0, 1])

                candidates.append({
                    "input_id": seq_id, "mode": "scan",
                    "start": win_start + hp.start, "end": win_start + hp.end,
                    "length": hp.end - hp.start, "rf_score": proba,
                    "pred_label": int(proba >= self.decision_threshold)
                })

        return self._merge_overlapping_candidates(candidates)

    def _merge_overlapping_candidates(self, candidates: List[Dict], overlap_threshold: float = 0.7) -> List[Dict]:
        if not candidates: return []
        # Secondary sort on start position for deterministic output when scores tie
        sorted_cands = sorted(candidates, key=lambda d: (-d["rf_score"], d["start"]))
        kept: List[Dict] = []
        for cand in sorted_cands:
            c_s, c_e = cand["start"], cand["end"]
            discard = False
            for k in kept:
                k_s, k_e = k["start"], k["end"]
                inter_len = max(0, min(c_e, k_e) - max(c_s, k_s))
                shorter = float(min(c_e - c_s, k_e - k_s))
                if shorter > 0 and inter_len / shorter >= overlap_threshold:
                    discard = True; break
            if not discard: kept.append(cand)
        return kept

    def score_sequence_record(self, seq_id: str, seq: str) -> List[Dict]:
        """
        Score a sequence record, automatically choosing the appropriate method.
        
        Routing logic:
            - If length <= max_hairpin_len: Direct hairpin scoring
            - If length <= max_seq_only_len: Sliding window scanning
            - Otherwise: Returns "too_long" placeholder
        
        Args:
            seq_id: Identifier for the sequence.
            seq: RNA sequence to score.
        
        Returns:
            List of scoring result dictionaries. Each contains:
                - input_id, mode, start, end, length, rf_score, pred_label
        """
        L = len(seq)
        if L <= self.max_hairpin_len: return [self.score_hairpin(seq_id, seq)]
        if L <= self.max_seq_only_len:
            # Use parallel scanning if configured
            if HAS_PARALLEL and self.parallel_cfg is not None:
                return self.scan_long_sequence_parallel(seq_id, seq)
            return self.scan_long_sequence(seq_id, seq)
        return [{"input_id": seq_id, "mode": "too_long", "start": 0, "end": L, "length": L, "rf_score": float("nan"), "pred_label": -1}]
