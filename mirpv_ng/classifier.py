# mirpv_ng/classifier.py

"""
Classifier utilities for miRPV-NG.
(Fixed: Feature set detection and Filter Logic)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np

from .features import extended_features, core36_features, run_rnafold
from .tier_filters import TierConfig, GeometryConfig, Tier2Config, tier1_energy_filter, tier2_geometry_filter
from .geom_hairpin_finder import find_hairpins


@dataclass
class ModelInfo:
    model: object
    feature_set: str
    feature_cols: List[str]

def load_rf_model(model_path: str | Path) -> ModelInfo:
    model_path = Path(model_path)
    payload = joblib.load(model_path)
    return ModelInfo(
        model=payload["model"], 
        feature_set=payload.get("feature_set", "extended"), 
        feature_cols=payload.get("feature_cols", [])
    )

def compute_feature_vector(
    seq: str,
    feature_set: str = "extended",
    mfe: float = None,
    struct: str = None,
    tier2_enabled: bool = False,
) -> Dict[str, float]:
    
    if struct is None or mfe is None:
        struct, mfe = run_rnafold(seq)
    
    if feature_set == "core36":
        feats = core36_features(seq, struct, mfe)
    else:
        tier2_cfg = Tier2Config(enabled=tier2_enabled)
        feats = extended_features(seq, struct, mfe, tier2_cfg=tier2_cfg)
        
    if "mfe" not in feats: 
        feats["mfe"] = mfe
    return feats

def generate_windows(seq: str, window_len: int, step: int) -> List[Tuple[int, str]]:
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
    def __init__(
        self,
        model_path: str | Path,
        species: str = "hsa",
        feature_set: str = None,  # CHANGED: Default to None to prefer Model's config
        rnafold_bin: str = "RNAfold",
        max_hairpin_len: int = 120,
        max_seq_only_len: int = 5000,
        window_len: int = 100,
        step: int = 20,
        tier1_min_pairs: int = 18,
        tier1_min_mfe: float = -15.0,
    ):
        self.species = species
        self.max_hairpin_len = max_hairpin_len
        self.max_seq_only_len = max_seq_only_len
        self.window_len = window_len
        self.step = step
        
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
        
        # BUG FIX: Use model's feature_set if available, otherwise fallback
        self.feature_set = self.model_info.feature_set if self.model_info.feature_set else (feature_set or "extended")
        
        self.feature_cols = self.model_info.feature_cols
        
        print(f"[DEBUG] Model loaded. Feature Set: {self.feature_set}")

    def _vector_from_features(self, feats: Dict[str, float]) -> np.ndarray:
        if not self.feature_cols: self.feature_cols = sorted(feats.keys())
        x = np.array([feats.get(col, 0.0) for col in self.feature_cols], dtype=float)
        return x.reshape(1, -1)

    def score_hairpin(self, seq_id: str, seq: str) -> Dict:
        """Score a short sequence directly (Bypasses Tier 2 Filters)."""
        feats = compute_feature_vector(seq, feature_set=self.feature_set)
        x = self._vector_from_features(feats)
        proba = self.model_info.model.predict_proba(x)[0, 1]
        return {
            "input_id": seq_id, "mode": "hairpin", "start": 0, "end": len(seq),
            "length": len(seq), "rf_score": float(proba), "pred_label": int(proba >= 0.5)
        }

    def scan_long_sequence(self, seq_id: str, seq: str) -> List[Dict]:
        """Scan long sequence (Applies Tier 2 Filters)."""
        windows = generate_windows(seq, self.window_len, self.step)
        candidates: List[Dict] = []

        for win_start, wseq in windows:
            struct, mfe = run_rnafold(wseq)
            
            # Tier-2 geometry as SOFT gate: do not drop candidates
            tier2_geom_pass = 1.0 if tier2_geometry_filter(hp, self.geom_cfg) else 0.0
            
            # Find Hairpins
            hairpins = find_hairpins(wseq, struct)
            if not hairpins: continue
            
            for hp in hairpins:
                # Tier 2 Filter (Geometry)
                if not tier2_geometry_filter(hp, self.geom_cfg):
                    continue
                
                # Extract and Score
                hp_seq = wseq[hp.start:hp.end]
                # Re-compute features on the cropped hairpin
                feats = compute_feature_vector(
                    hp_seq,
                    feature_set=self.feature_set,
                    tier2_enabled=True,   # enables tier2_* features in extended_features
                )
                feats["tier2_geom_pass"] = tier2_geom_pass
                # Use hairpin MFE if re-fold happened, otherwise fallback to window MFE (proxy)
                if "mfe" not in feats: 
                    feats["mfe"] = mfe 

                proba = float(self.model_info.model.predict_proba(self._vector_from_features(feats))[0, 1])
                
                candidates.append({
                    "input_id": seq_id, "mode": "scan",
                    "start": win_start + hp.start, "end": win_start + hp.end,
                    "length": hp.end - hp.start, "rf_score": proba, "pred_label": int(proba >= 0.5)
                })

        return self._merge_overlapping_candidates(candidates)

    def _merge_overlapping_candidates(self, candidates: List[Dict], overlap_threshold: float = 0.7) -> List[Dict]:
        if not candidates: return []
        sorted_cands = sorted(candidates, key=lambda d: d["rf_score"], reverse=True)
        kept: List[Dict] = []
        for cand in sorted_cands:
            c_s, c_e = cand["start"], cand["end"]
            discard = False
            for k in kept:
                k_s, k_e = k["start"], k["end"]
                inter_len = max(0, min(c_e, k_e) - max(c_s, k_s))
                if inter_len / float(min(c_e - c_s, k_e - k_s)) >= overlap_threshold:
                    discard = True; break
            if not discard: kept.append(cand)
        return kept

    def score_sequence_record(self, seq_id: str, seq: str) -> List[Dict]:
        L = len(seq)
        if L <= self.max_hairpin_len: return [self.score_hairpin(seq_id, seq)]
        if L <= self.max_seq_only_len: return self.scan_long_sequence(seq_id, seq)
        return [{"input_id": seq_id, "mode": "too_long", "start": 0, "end": L, "length": L, "rf_score": float("nan"), "pred_label": -1}]