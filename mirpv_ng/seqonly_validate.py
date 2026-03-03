#!/usr/bin/env python3
"""
Sequence-only validation module for miRPV-NG.

Implements two-stage validation to handle miRBase-style truncated hairpins:
- Stage A: Mature-centric duplex evidence (tolerates boundary issues)
- Stage B: Precursor-centric RF scoring on extracted hairpin region

This module provides validation for sequence-only inputs where precursor
boundaries may be incorrect or truncated, common in miRBase annotations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import re

from .features import run_rnafold
from .mature_ranker import generate_duplex_candidates, dotbracket_pairs
from .geom_hairpin_finder import find_hairpins
from .classifier import compute_feature_vector


@dataclass
class StageAResult:
    """Results from Stage A (mature-centric evidence)."""
    mature_score: float  # Normalized 0-1 using percentile ranking
    mode: str  # "model" or "heuristic"
    best_5p_start: int
    best_5p_end: int
    best_3p_start: int
    best_3p_end: int
    notes: str


@dataclass
class StageBResult:
    """Results from Stage B (precursor-centric evidence)."""
    precursor_score: float
    threshold_q90: str  # "NA" or float
    threshold_q95: str
    threshold_q99: str
    extracted_start: int
    extracted_end: int
    notes: str


@dataclass
class ValidationResult:
    """Complete validation result for one sequence."""
    input_id: str
    seq_len: int
    stageA: StageAResult
    stageB: StageBResult
    verdict: str
    recommendation: str
    coords_parsed: bool
    reext_seq_len: Optional[int] = None
    reext_stageB_score: Optional[float] = None


def compute_heuristic_mature_score(seq: str, struct: str, window_start: int, window_len: int) -> float:
    """
    Compute heuristic mature score for a window (Mode A2, no mature model).
    
    Formula: 0.4*paired_frac + 0.3*max_helix + 0.3*(1-bulge_frac)
    
    Returns: Normalized score in [0, 1]
    """
    if window_start + window_len > len(struct):
        return 0.0
    
    window_struct = struct[window_start:window_start + window_len]
    
    # Paired fraction (FIXED with parentheses)
    paired_frac = (window_struct.count('(') + window_struct.count(')')) / window_len
    
    # Max contiguous helix length
    import re as regex_module
    helices = regex_module.findall(r'[\(\)]+', window_struct)
    max_helix = max((len(h) for h in helices), default=0) / window_len
    
    # Internal bulge fraction
    bulge_frac = window_struct.count('.') / window_len
    
    # Weighted combination (already normalized 0-1)
    score = 0.4 * paired_frac + 0.3 * max_helix + 0.3 * (1 - bulge_frac)
    return score


def check_arms_face_each_other(
    pair: List[int],
    start_5p: int, end_5p: int,
    start_3p: int, end_3p: int,
    min_frac: float = 0.5,
    min_absolute: int = 6
) -> bool:
    """
    Robust check that 5p and 3p arms face each other.
    
    Requirements:
    - Fraction of paired bases >= min_frac (in BOTH directions)
    - Absolute count of paired bases >= min_absolute
    """
    # Count 5p->3p pairing
    count_5p_to_3p = sum(1 for i in range(start_5p, end_5p)
                         if pair[i] != -1 and start_3p <= pair[i] < end_3p)
    
    # Count 3p->5p pairing
    count_3p_to_5p = sum(1 for i in range(start_3p, end_3p)
                         if pair[i] != -1 and start_5p <= pair[i] < end_5p)
    
    len_5p = end_5p - start_5p
    len_3p = end_3p - start_3p
    
    frac_5p = count_5p_to_3p / len_5p if len_5p > 0 else 0.0
    frac_3p = count_3p_to_5p / len_3p if len_3p > 0 else 0.0
    
    return (frac_5p >= min_frac and frac_3p >= min_frac and
            count_5p_to_3p >= min_absolute and count_3p_to_5p >= min_absolute)


def compute_percentile_score(best_score: float, all_scores: List[float]) -> float:
    """
    Convert raw ranking score to normalized percentile in [0, 1].
    
    percentile = (rank_of_best - 1) / (n - 1)
    Where rank is 1-indexed from lowest to highest.
    
    Returns: Normalized score in [0, 1]
    """
    n = len(all_scores)
    if n <= 1:
        return 1.0  # If only one candidate, it's the best
    
    # Count how many scores are less than best_score
    rank = sum(1 for s in all_scores if s <= best_score)  # 1-indexed rank
    
    # Normalize to [0, 1]
    percentile = (rank - 1) / (n - 1)
    return percentile


def run_stage_a(
    seq: str,
    struct: str,
    mature_model=None
) -> StageAResult:
    """
    Stage A: Mature-centric duplex evidence.
    
    Mode A1 (with model): Use generate_duplex_candidates + mature model scoring.
        Score is normalized to [0, 1] using percentile ranking within each arm.
    Mode A2 (heuristic): Enumerate windows with heuristic scoring (already [0, 1]).
    
    Returns: StageAResult with mature_score in [0, 1]
    """
    if mature_model:
        # Mode A1: Use mature model with percentile normalization
        cands = generate_duplex_candidates(
            seq, struct,
            lengths=(18, 19, 20, 21, 22, 23, 24, 25, 26),
            max_per_arm=30,
            min_paired_context=6,
            loop_buffer=0
        )
        
        if not cands:
            return StageAResult(
                mature_score=0.0, mode="model",
                best_5p_start=-1, best_5p_end=-1,
                best_3p_start=-1, best_3p_end=-1,
                notes="no_candidates_found"
            )
        
        # Score all candidates
        X = [[float(c.features.get(k, 0.0)) for k in mature_model.feature_cols] for c in cands]
        raw_scores = mature_model.ranker.predict(X)
        
        # Separate by arm
        cands_5p = [(i, c, raw_scores[i]) for i, c in enumerate(cands) if c.arm == "5p"]
        cands_3p = [(i, c, raw_scores[i]) for i, c in enumerate(cands) if c.arm == "3p"]
        
        if not cands_5p or not cands_3p:
            # Single arm only - use percentile within available arm
            all_scored = [(i, c, raw_scores[i]) for i, c in enumerate(cands)]
            best_i, best_c, best_score = max(all_scored, key=lambda x: x[2])
            
            arm_scores = [s for _, _, s in all_scored]
            pct = compute_percentile_score(best_score, arm_scores)
            
            if best_c.arm == "5p":
                return StageAResult(
                    mature_score=pct * 0.7,  # Penalty for single arm
                    mode="model",
                    best_5p_start=best_c.start, best_5p_end=best_c.end,
                    best_3p_start=-1, best_3p_end=-1,
                    notes="single_arm_only_5p"
                )
            else:
                return StageAResult(
                    mature_score=pct * 0.7,
                    mode="model",
                    best_5p_start=-1, best_5p_end=-1,
                    best_3p_start=best_c.start, best_3p_end=best_c.end,
                    notes="single_arm_only_3p"
                )
        
        # Get best candidates per arm
        best_5p_i, best_5p, score_5p = max(cands_5p, key=lambda x: x[2])
        best_3p_i, best_3p, score_3p = max(cands_3p, key=lambda x: x[2])
        
        # Compute percentiles within each arm
        scores_5p = [s for _, _, s in cands_5p]
        scores_3p = [s for _, _, s in cands_3p]
        pct_5p = compute_percentile_score(score_5p, scores_5p)
        pct_3p = compute_percentile_score(score_3p, scores_3p)
        
        # Check if arms face each other
        pair = dotbracket_pairs(struct)
        arms_face = check_arms_face_each_other(
            pair, best_5p.start, best_5p.end, best_3p.start, best_3p.end
        )
        
        if arms_face:
            # Conservative: both arms must rank well
            mature_score = min(pct_5p, pct_3p)
            notes = f"arms_face_each_other=True pct_5p={pct_5p:.3f} pct_3p={pct_3p:.3f}"
        else:
            # Penalty for non-facing arms
            mature_score = max(pct_5p, pct_3p) * 0.7
            notes = f"arms_face_each_other=False pct_5p={pct_5p:.3f} pct_3p={pct_3p:.3f}"
        
        return StageAResult(
            mature_score=mature_score,
            mode="model",
            best_5p_start=best_5p.start, best_5p_end=best_5p.end,
            best_3p_start=best_3p.start, best_3p_end=best_3p.end,
            notes=notes
        )
    
    else:
        # Mode A2: Heuristic fallback (scores already in [0, 1])
        pair = dotbracket_pairs(struct)
        best_5p_score, best_5p_win = -1.0, (-1, -1)
        best_3p_score, best_3p_win = -1.0, (-1, -1)
        
        # Determine loop region
        from .mature_ranker import identify_loop_region
        loop_start, loop_end = identify_loop_region(struct)
        
        # Scan windows
        for length in range(18, 27):
            for start in range(0, len(seq) - length + 1, 2):  # Step 2
                score = compute_heuristic_mature_score(seq, struct, start, length)
                
                # Determine arm (5p if before loop, 3p if after)
                if start < loop_start:
                    if score > best_5p_score:
                        best_5p_score = score
                        best_5p_win = (start, start + length)
                elif start >= loop_end:
                    if score > best_3p_score:
                        best_3p_score = score
                        best_3p_win = (start, start + length)
        
        # Check if arms face each other
        if best_5p_win[0] != -1 and best_3p_win[0] != -1:
            arms_face = check_arms_face_each_other(
                pair, best_5p_win[0], best_5p_win[1], best_3p_win[0], best_3p_win[1]
            )
            if arms_face:
                mature_score = min(best_5p_score, best_3p_score)
                notes = "heuristic_arms_face=True"
            else:
                mature_score = max(best_5p_score, best_3p_score) * 0.7
                notes = "heuristic_arms_face=False"
        else:
            mature_score = max(best_5p_score, best_3p_score, 0.0) * 0.7
            notes = "heuristic_single_arm"
        
        return StageAResult(
            mature_score=mature_score,
            mode="heuristic",
            best_5p_start=best_5p_win[0], best_5p_end=best_5p_win[1],
            best_3p_start=best_3p_win[0], best_3p_end=best_3p_win[1],
            notes=notes
        )


def select_best_hairpin(
    hairpins: List[Any],
    stageA: Optional[StageAResult],
    seq: str
) -> Tuple[int, int]:
    """
    Select best hairpin region for Stage B scoring.
    
    Preference:
    1. Hairpin that contains Stage A windows (maximize overlap)
    2. Tie-break by ((1 - unpaired_frac) * length)
    
    Returns: (extracted_start, extracted_end)
    """
    if not hairpins:
        return (0, len(seq))  # Use full sequence
    
    def compute_overlap(hp_start, hp_end, win_start, win_end):
        if win_start == -1:
            return 0
        overlap_start = max(hp_start, win_start)
        overlap_end = min(hp_end, win_end)
        return max(0, overlap_end - overlap_start)
    
    scores = []
    for hp in hairpins:
        # Containment score
        overlap_5p = 0
        overlap_3p = 0
        if stageA:
            overlap_5p = compute_overlap(hp.start, hp.end, stageA.best_5p_start, stageA.best_5p_end)
            overlap_3p = compute_overlap(hp.start, hp.end, stageA.best_3p_start, stageA.best_3p_end)
        
        containment = overlap_5p + overlap_3p
        # HairpinGeometry has unpaired_frac, so paired_frac = 1 - unpaired_frac
        paired_frac = 1.0 - hp.unpaired_frac
        structural = paired_frac * (hp.end - hp.start)
        
        scores.append((containment, structural, hp))
    
    # Sort by containment first, then structural
    best_hp = max(scores, key=lambda x: (x[0], x[1]))[2]
    return (best_hp.start, best_hp.end)


def load_reference_thresholds(model_path, model_info) -> Dict[str, str]:
    """
    Load reference thresholds from model payload.
    
    Tries keys in priority order:
    1. reference_thresholds
    2. ref_thresholds  
    3. training_metadata.reference_thresholds
    
    Returns: dict with:
        - q90, q95, q99 (as strings, "NA" if missing)
        - schema: "qXX", "neg_qXX", or "none" indicating which key pattern was used
    """
    import joblib
    
    # First try to get payload from model_info if available
    payload = getattr(model_info, "payload", None)
    
    # If not available, load from disk
    if payload is None:
        try:
            payload = joblib.load(model_path)
        except Exception:
            return {"q90": "NA", "q95": "NA", "q99": "NA", "schema": "none"}
    
    # Try different keys
    ref_thresholds = (
        payload.get("reference_thresholds") or 
        payload.get("ref_thresholds")
    )
    
    # Try nested in training_metadata
    if not ref_thresholds:
        train_meta = payload.get("training_metadata", {})
        if isinstance(train_meta, dict):
            ref_thresholds = train_meta.get("reference_thresholds")
    
    schema = "none"
    q90 = q95 = q99 = "NA"
    
    if ref_thresholds and isinstance(ref_thresholds, dict):
        # Schema 1: direct q90/q95/q99 keys
        if any(k in ref_thresholds for k in ("q90", "q95", "q99")):
            schema = "qXX"
            q90 = str(ref_thresholds.get("q90", "NA"))
            q95 = str(ref_thresholds.get("q95", "NA"))
            q99 = str(ref_thresholds.get("q99", "NA"))
        # Schema 2: neg_q90/neg_q95/neg_q99 keys from train_premirna_model.py
        elif any(k in ref_thresholds for k in ("neg_q90", "neg_q95", "neg_q99")):
            schema = "neg_qXX"
            q90 = str(ref_thresholds.get("neg_q90", "NA"))
            q95 = str(ref_thresholds.get("neg_q95", "NA"))
            q99 = str(ref_thresholds.get("neg_q99", "NA"))
    
    return {"q90": q90, "q95": q95, "q99": q99, "schema": schema}


def run_stage_b(
    seq: str,
    struct: str,
    mfe: float,
    model_info,
    model_path,
    stageA: Optional[StageAResult],
    tier2_enabled: bool = False
) -> StageBResult:
    """
    Stage B: Precursor-centric RF scoring.
    
    Extracts best hairpin region (or uses full seq) and scores with RF model.
    """
    # Find hairpins
    hairpins = find_hairpins(seq, struct)
    
    extracted_start, extracted_end = select_best_hairpin(hairpins, stageA, seq)
    
    # Extract region
    if extracted_start == 0 and extracted_end == len(seq):
        hp_seq = seq
        hp_struct = struct
        hp_mfe = mfe
        notes = "no_hairpin_found_scored_full_seq"
    else:
        hp_seq = seq[extracted_start:extracted_end]
        hp_struct, hp_mfe = run_rnafold(hp_seq)
        notes = f"scored_hairpin_{extracted_start}_{extracted_end}"
    
    # Compute features and score
    feats = compute_feature_vector(
        hp_seq,
        feature_set=model_info.feature_set,
        mfe=hp_mfe,
        struct=hp_struct,
        tier2_enabled=tier2_enabled
    )
    
    # Build feature vector in correct order
    x = [feats.get(col, 0.0) for col in model_info.feature_cols]
    import numpy as np
    x = np.array(x, dtype=float).reshape(1, -1)
    
    # Score
    proba = float(model_info.model.predict_proba(x)[0, 1])
    
    # Load thresholds using robust loader
    thresholds = load_reference_thresholds(model_path, model_info)
    schema = thresholds.get("schema", "none")
    
    # Append threshold schema to notes for downstream inspection / debugging
    if schema != "none":
        notes = f"{notes};threshold_schema={schema}"
    else:
        notes = f"{notes};threshold_schema=none"
    
    return StageBResult(
        precursor_score=proba,
        threshold_q90=thresholds["q90"],
        threshold_q95=thresholds["q95"],
        threshold_q99=thresholds["q99"],
        extracted_start=extracted_start,
        extracted_end=extracted_end,
        notes=notes
    )


def compute_verdict(
    stageB_score: float,
    stageA_score: float,
    q95: str,
    q99: str,
    mature_min: float = 0.8
) -> Tuple[str, str]:
    """
    Compute verdict and recommendation based on Stage A + B results.
    
    Stage A is normalized to [0, 1] using percentile ranking.
    
    Thresholds (with stageA now in [0,1]):
    - HIGH_CONFIDENCE_PREMIRNA: (StageB >= q99) AND (StageA >= mature_min)
    - MATURE_LIKE_BUT_TRUNCATED_PRECURSOR: (StageB < q95) AND (StageA >= mature_min)
    - WEAK_EVIDENCE: (q95 <= StageB < q99) OR (0.5 <= StageA < mature_min)
    - NO_SUPPORT: otherwise
    
    The mature_min parameter (default 0.6 from the CLI) controls the cutoff for
    "strong" mature evidence; scores between 0.5 and mature_min are treated as
    weak but non-zero evidence.
    
    Returns: (verdict, recommendation)
    """
    # Handle missing thresholds
    if q95 == "NA" or q99 == "NA":
        return ("WEAK_EVIDENCE",
                f"Reference thresholds unavailable in model; cannot determine confidence. "
                f"StageA={stageA_score:.3f}, StageB={stageB_score:.3f}.")
    
    q95_val = float(q95)
    q99_val = float(q99)
    
    # Verdict logic with updated thresholds
    if stageB_score >= q99_val and stageA_score >= mature_min:
        return ("HIGH_CONFIDENCE_PREMIRNA",
                f"Strong precursor (P={stageB_score:.3f}>=q99={q99_val:.3f}) + mature-like evidence "
                f"(M={stageA_score:.3f}>={mature_min:.2f}). Confirm with scan-mode locus context or sRNA support.")
    
    elif stageB_score < q95_val and stageA_score >= mature_min:
        return ("MATURE_LIKE_BUT_TRUNCATED_PRECURSOR",
                f"Mature-like duplex detected (M={stageA_score:.3f}>={mature_min:.2f}), but precursor context weak "
                f"(P={stageB_score:.3f}<q95={q95_val:.3f}); input may be truncated. "
                f"Recommended: re-extract ±30 nt flanks from genome locus and rescore.")
    
    elif (q95_val <= stageB_score < q99_val) or (0.5 <= stageA_score < mature_min):
        return ("WEAK_EVIDENCE",
                f"Some evidence (P={stageB_score:.3f}, M={stageA_score:.3f}), not definitive. "
                f"Provide genome coordinates for standardized re-extraction.")
    
    else:
        return ("NO_SUPPORT",
                f"No mature-like duplex and precursor score low (P={stageB_score:.3f}, M={stageA_score:.3f}). "
                f"Unlikely under current model assumptions.")


def parse_coords_from_header(header: str) -> Optional[Tuple[str, int, int]]:
    """
    Parse coordinates from FASTA header.
    
    Supported patterns:
    - chr:start-end
    - chr_start_end
    
    Returns: (chrom, start, end) or None
    """
    # Pattern 1: chr:start-end
    m = re.search(r'(chr\w+):(\d+)-(\d+)', header)
    if m:
        return (m.group(1), int(m.group(2)), int(m.group(3)))
    
    # Pattern 2: chr_start_end
    m = re.search(r'(chr\w+)_(\d+)_(\d+)', header)
    if m:
        return (m.group(1), int(m.group(2)), int(m.group(3)))
    
    return None
