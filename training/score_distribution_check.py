#!/usr/bin/env python3
"""
Score Distribution Check: Verify model scores on positives and scanner-like negatives.

Reports score quantiles and overlap to confirm calibration is working properly.

Usage:
    python training/score_distribution_check.py \
        --model models/hsa_premirna_rf_v7_negv2_calibrated.pkl \
        --positives data/train/hsa_mirgene_premirna.fa \
        --negatives data/train/negatives_v2.fa \
        --out analysis/score_check.txt

Author: miRPV-NG Team
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import subprocess
import numpy as np
import joblib
from typing import List, Dict, Tuple

from Bio import SeqIO

try:
    from mirpv_ng.classifier import compute_feature_vector
    HAS_CLASSIFIER = True
except ImportError:
    HAS_CLASSIFIER = False
    print("[WARNING] Could not import classifier module")


def run_rnafold(seq: str) -> Tuple[str, float]:
    """Run RNAfold and return (structure, MFE)."""
    seq_rna = seq.upper().replace("T", "U")
    try:
        p = subprocess.Popen(
            ["RNAfold", "--noPS"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        out, _ = p.communicate(seq_rna + "\n", timeout=30)
        lines = out.strip().splitlines()
        if len(lines) < 2:
            return "", 0.0
        struct_line = lines[1]
        struct = struct_line.split()[0]
        mfe_str = struct_line.split("(")[-1].strip(" )")
        mfe = float(mfe_str)
        return struct, mfe
    except Exception:
        return "", 0.0


def score_sequences(
    fasta_path: str,
    model_data: dict,
    max_seqs: int = 1000,
    seed: int = 42,
) -> List[float]:
    """Score sequences using the model's predict_proba directly."""
    import random
    
    model = model_data["model"]
    feature_cols = model_data["feature_cols"]
    feature_set = model_data.get("feature_set", "extended")
    tier2_enabled = model_data.get("tier2_enabled", False)
    
    rng = random.Random(seed)
    
    # Load sequences
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if len(records) > max_seqs:
        rng.shuffle(records)
        records = records[:max_seqs]
    
    scores = []
    for rec in records:
        seq = str(rec.seq).upper().replace("T", "U")
        
        if not HAS_CLASSIFIER:
            continue
        
        try:
            feats = compute_feature_vector(
                seq,
                feature_set=feature_set,
                tier2_enabled=tier2_enabled,
            )
            
            # Build feature vector
            x = np.array([[feats.get(col, 0.0) for col in feature_cols]])
            
            # Score using predict_proba directly (class 1)
            proba = float(model.predict_proba(x)[0, 1])
            scores.append(proba)
            
        except Exception as e:
            continue
    
    return scores


def debug_score_sequences(
    fasta_path: str,
    model_data: dict,
    max_seqs: int = 200,
    seed: int = 42,
) -> Tuple[List[float], List[float]]:
    """
    Score sequences and return BOTH raw RF probs and calibrated probs.
    Uses model internals to get raw RF.
    
    Returns: (raw_probs, calibrated_probs)
    """
    import random
    from sklearn.calibration import CalibratedClassifierCV
    
    model = model_data["model"]
    feature_cols = (
        model_data.get("feature_cols") or 
        model_data.get("feature_names_") or 
        model_data.get("feature_names") or []
    )
    feature_set = model_data.get("feature_set", "extended")
    tier2_enabled = model_data.get("tier2_enabled", False)
    
    # Check if model is calibrated
    is_calibrated = isinstance(model, CalibratedClassifierCV)
    
    # Try to get base estimator
    base_rf = None
    if is_calibrated:
        # CalibratedClassifierCV stores calibrated classifiers in calibrated_classifiers_
        # Each has a 'estimator' attribute (the base RF)
        try:
            if hasattr(model, 'calibrated_classifiers_') and model.calibrated_classifiers_:
                cc = model.calibrated_classifiers_[0]
                if hasattr(cc, 'estimator'):
                    base_rf = cc.estimator
        except Exception:
            pass
    else:
        base_rf = model
    
    rng = random.Random(seed)
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if len(records) > max_seqs:
        rng.shuffle(records)
        records = records[:max_seqs]
    
    raw_probs = []
    cal_probs = []
    
    for rec in records:
        seq = str(rec.seq).upper().replace("T", "U")
        
        if not HAS_CLASSIFIER:
            continue
        
        try:
            feats = compute_feature_vector(
                seq,
                feature_set=feature_set,
                tier2_enabled=tier2_enabled,
            )
            
            x = np.array([[feats.get(col, 0.0) for col in feature_cols]])
            
            # Get calibrated score
            cal_p = float(model.predict_proba(x)[0, 1])
            cal_probs.append(cal_p)
            
            # Get raw RF score (if available)
            if base_rf is not None:
                raw_p = float(base_rf.predict_proba(x)[0, 1])
                raw_probs.append(raw_p)
            else:
                raw_probs.append(cal_p)  # Fallback
                
        except Exception:
            continue
    
    return raw_probs, cal_probs


def parse_bucket_from_id(rec_id: str) -> str:
    """Parse bucket label from provenance-tagged FASTA ID."""
    if rec_id.startswith("N1|"):
        return "N1"
    elif rec_id.startswith("N2|"):
        return "N2"
    elif rec_id.startswith("N3|"):
        return "N3"
    # Legacy format fallback
    elif rec_id.startswith("N1_"):
        return "N1"
    elif rec_id.startswith("N2_") or rec_id.startswith("N2|"):
        return "N2"
    elif rec_id.startswith("N3_"):
        return "N3"
    return "unknown"


def score_sequences_with_provenance(
    fasta_path: str,
    model_data: dict,
    max_seqs: int = 1000,
    seed: int = 42,
) -> List[Tuple[str, str, float]]:
    """
    Score sequences with provenance tracking.
    
    Returns: List of (record_id, bucket_label, score)
    """
    import random
    
    model = model_data["model"]
    feature_cols = model_data["feature_cols"]
    feature_set = model_data.get("feature_set", "extended")
    tier2_enabled = model_data.get("tier2_enabled", False)
    
    rng = random.Random(seed)
    
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if len(records) > max_seqs:
        rng.shuffle(records)
        records = records[:max_seqs]
    
    results = []
    for rec in records:
        seq = str(rec.seq).upper().replace("T", "U")
        bucket = parse_bucket_from_id(rec.id)
        
        if not HAS_CLASSIFIER:
            continue
        
        try:
            feats = compute_feature_vector(
                seq,
                feature_set=feature_set,
                tier2_enabled=tier2_enabled,
            )
            x = np.array([[feats.get(col, 0.0) for col in feature_cols]])
            proba = float(model.predict_proba(x)[0, 1])
            results.append((rec.id, bucket, proba))
        except Exception:
            continue
    
    return results


def format_bucket_stats(scored_negs: List[Tuple[str, str, float]], ref_thresholds: Dict) -> str:
    """Format per-bucket statistics."""
    lines = []
    
    # Group by bucket
    buckets = {}
    for rec_id, bucket, score in scored_negs:
        if bucket not in buckets:
            buckets[bucket] = []
        buckets[bucket].append((rec_id, score))
    
    q95 = ref_thresholds.get("neg_q95", 0.5)
    q99 = ref_thresholds.get("neg_q99", 0.5)
    
    for bucket in sorted(buckets.keys()):
        items = buckets[bucket]
        scores = np.array([s for _, s in items])
        
        lines.append(f"\n  {bucket} (n={len(scores)})")
        lines.append(f"    Min:    {np.min(scores):.4f}")
        lines.append(f"    Q25:    {np.percentile(scores, 25):.4f}")
        lines.append(f"    Median: {np.percentile(scores, 50):.4f}")
        lines.append(f"    Q75:    {np.percentile(scores, 75):.4f}")
        lines.append(f"    Q95:    {np.percentile(scores, 95):.4f}")
        lines.append(f"    Max:    {np.max(scores):.4f}")
        lines.append(f"    Above q95 ({q95:.3f}): {np.sum(scores >= q95)} ({np.mean(scores >= q95):.1%})")
        lines.append(f"    Above q99 ({q99:.3f}): {np.sum(scores >= q99)} ({np.mean(scores >= q99):.1%})")
    
    return "\n".join(lines)


def format_top_k_negatives(scored_negs: List[Tuple[str, str, float]], k: int = 20) -> str:
    """Format top-K highest scoring negatives with provenance."""
    lines = []
    
    # Sort by score descending
    sorted_negs = sorted(scored_negs, key=lambda x: x[2], reverse=True)[:k]
    
    lines.append(f"\nTop {k} Highest Scoring Negatives:")
    lines.append("-" * 70)
    
    for i, (rec_id, bucket, score) in enumerate(sorted_negs):
        lines.append(f"  {i+1:2d}. score={score:.4f}  bucket={bucket}  id={rec_id}")
    
    return "\n".join(lines)


def print_quantiles(name: str, scores: List[float]) -> str:
    """Format score quantiles for display."""
    if not scores:
        return f"{name}: No scores"
    
    lines = [f"\n{name} (n={len(scores)})"]
    lines.append("-" * 50)
    
    arr = np.array(scores)
    lines.append(f"  Min:    {np.min(arr):.4f}")
    lines.append(f"  Q10:    {np.percentile(arr, 10):.4f}")
    lines.append(f"  Q25:    {np.percentile(arr, 25):.4f}")
    lines.append(f"  Median: {np.percentile(arr, 50):.4f}")
    lines.append(f"  Q75:    {np.percentile(arr, 75):.4f}")
    lines.append(f"  Q90:    {np.percentile(arr, 90):.4f}")
    lines.append(f"  Max:    {np.max(arr):.4f}")
    lines.append(f"  Mean:   {np.mean(arr):.4f}")
    lines.append(f"  Std:    {np.std(arr):.4f}")
    
    return "\n".join(lines)


def compute_overlap(pos_scores: List[float], neg_scores: List[float]) -> Dict:
    """Compute overlap metrics between positive and negative score distributions."""
    if not pos_scores or not neg_scores:
        return {}
    
    pos = np.array(pos_scores)
    neg = np.array(neg_scores)
    
    # Fraction of negatives above various thresholds
    pos_q10 = np.percentile(pos, 10)
    pos_q25 = np.percentile(pos, 25)
    pos_median = np.percentile(pos, 50)
    
    neg_above_pos_q10 = np.mean(neg >= pos_q10)
    neg_above_pos_q25 = np.mean(neg >= pos_q25)
    neg_above_pos_median = np.mean(neg >= pos_median)
    
    # Score gap (difference between pos_q25 and neg_q75)
    neg_q75 = np.percentile(neg, 75)
    score_gap = pos_q25 - neg_q75
    
    return {
        "pos_q10": pos_q10,
        "pos_q25": pos_q25,
        "pos_median": pos_median,
        "neg_q75": neg_q75,
        "neg_above_pos_q10": neg_above_pos_q10,
        "neg_above_pos_q25": neg_above_pos_q25,
        "neg_above_pos_median": neg_above_pos_median,
        "score_gap": score_gap,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Check score distribution on positives and negatives"
    )
    ap.add_argument("--model", required=True, help="Model pickle file")
    ap.add_argument("--positives", required=True, help="Positive FASTA")
    ap.add_argument("--negatives", required=True, help="Negative FASTA (ideally N2 scanner-matched)")
    ap.add_argument("--out", default=None, help="Output report file")
    ap.add_argument("--max-seqs", type=int, default=1000, help="Max sequences to score per set")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--histogram", default=None, help="Output histogram PNG path")
    ap.add_argument("--per-bucket", action="store_true", help="Enable per-bucket analysis")
    ap.add_argument("--top-k", type=int, default=20, help="Show top-K highest scoring negatives")
    ap.add_argument("--debug", action="store_true", help="Debug mode: show unique raw RF vs calibrated scores")
    
    args = ap.parse_args()
    
    print("=" * 70)
    print("SCORE DISTRIBUTION CHECK")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Positives: {args.positives}")
    print(f"Negatives: {args.negatives}")
    print("=" * 70)
    
    # Load model
    print("\n[1] Loading model...")
    model_data = joblib.load(args.model)
    
    if not isinstance(model_data, dict):
        print("ERROR: Model is not a dict")
        return 1
    
    print(f"    Feature set: {model_data.get('feature_set', 'unknown')}")
    print(f"    Feature count: {len(model_data.get('feature_cols', []))}")
    print(f"    F1 threshold: {model_data.get('f1_threshold', model_data.get('decision_threshold', 'N/A'))}")
    
    if "reference_thresholds" in model_data:
        refs = model_data["reference_thresholds"]
        print(f"    Reference thresholds: q90={refs.get('neg_q90', 'N/A'):.4f}, "
              f"q95={refs.get('neg_q95', 'N/A'):.4f}, q99={refs.get('neg_q99', 'N/A'):.4f}")
    
    if not HAS_CLASSIFIER:
        print("\nERROR: Cannot import classifier module for feature computation")
        return 1
    
    # Score positives
    print("\n[2] Scoring positives...")
    pos_scores = score_sequences(args.positives, model_data, args.max_seqs, args.seed)
    print(f"    Scored {len(pos_scores)} sequences")
    
    # Score negatives
    print("\n[3] Scoring negatives...")
    neg_scores = score_sequences(args.negatives, model_data, args.max_seqs, args.seed)
    print(f"    Scored {len(neg_scores)} sequences")
    
    # Debug mode: show raw RF vs calibrated scores
    if args.debug:
        print("\n" + "=" * 70)
        print("DEBUG: RAW vs CALIBRATED SCORE ANALYSIS")
        print("=" * 70)
        
        print("\n[debug] Scoring positives (raw RF vs calibrated)...")
        pos_raw, pos_cal = debug_score_sequences(args.positives, model_data, min(200, args.max_seqs), args.seed)
        
        print("\n[debug] Scoring negatives (raw RF vs calibrated)...")
        neg_raw, neg_cal = debug_score_sequences(args.negatives, model_data, min(200, args.max_seqs), args.seed)
        
        print(f"\n  POSITIVES ({len(pos_raw)} samples):")
        print(f"    Raw RF unique scores  : {len(set(pos_raw))}")
        print(f"    Calibrated unique scores: {len(set(pos_cal))}")
        print(f"    Raw RF sample         : {sorted(set(pos_raw))[:10]}")
        print(f"    Calibrated sample     : {sorted(set(pos_cal))[:10]}")
        
        print(f"\n  NEGATIVES ({len(neg_raw)} samples):")
        print(f"    Raw RF unique scores  : {len(set(neg_raw))}")
        print(f"    Calibrated unique scores: {len(set(neg_cal))}")
        print(f"    Raw RF sample         : {sorted(set(neg_raw))[:10]}")
        print(f"    Calibrated sample     : {sorted(set(neg_cal))[:10]}")
        
        # Diagnose issue
        if len(set(pos_raw)) < 10:
            print("\n⚠ ISSUE: Raw RF probs are highly quantized!")
            print("   Likely cause: feature pipeline mismatch (extra/missing feature, constant column, wrong ordering)")
        elif len(set(pos_cal)) < 10 and len(set(pos_raw)) > 20:
            print("\n⚠ ISSUE: Only calibrated scores are quantized, raw RF is fine!")
            print("   Likely cause: isotonic calibration fitted on too few samples or in-sample fitting")
            print("   FIX: Use CV calibration (out-of-fold predictions) instead of prefit")
        else:
            print("\n✓ OK: Score distributions appear normal")
    
    # Format output
    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append("SCORE DISTRIBUTION REPORT")
    output_lines.append("=" * 70)
    
    output_lines.append(print_quantiles("POSITIVE SCORES", pos_scores))
    output_lines.append(print_quantiles("NEGATIVE SCORES", neg_scores))
    
    # Overlap analysis
    output_lines.append("\n" + "=" * 70)
    output_lines.append("OVERLAP ANALYSIS")
    output_lines.append("=" * 70)
    
    overlap = compute_overlap(pos_scores, neg_scores)
    if overlap:
        output_lines.append(f"\nPositive score thresholds:")
        output_lines.append(f"  Q10: {overlap['pos_q10']:.4f}")
        output_lines.append(f"  Q25: {overlap['pos_q25']:.4f}")
        output_lines.append(f"  Median: {overlap['pos_median']:.4f}")
        output_lines.append(f"\nFraction of negatives above positive thresholds:")
        output_lines.append(f"  Above pos_Q10: {overlap['neg_above_pos_q10']:.2%}")
        output_lines.append(f"  Above pos_Q25: {overlap['neg_above_pos_q25']:.2%}")
        output_lines.append(f"  Above pos_median: {overlap['neg_above_pos_median']:.2%}")
        output_lines.append(f"\nScore gap (pos_Q25 - neg_Q75): {overlap['score_gap']:.4f}")
        
        # Interpretation
        output_lines.append("\n" + "-" * 50)
        if overlap["score_gap"] > 0.3:
            output_lines.append("✓ GOOD SEPARATION: Positive and negative scores are well separated")
        elif overlap["score_gap"] > 0.1:
            output_lines.append("⚠ MODERATE SEPARATION: Some overlap but reasonable discrimination")
        else:
            output_lines.append("❌ POOR SEPARATION: Significant overlap between positive and negative scores")
            output_lines.append("   Consider: retraining with scanner-matched negatives")
        
        # Check for crushed scores
        if np.mean(pos_scores) < 0.3:
            output_lines.append("❌ CRUSHED SCORES: Positive scores are very low (mean < 0.3)")
            output_lines.append("   This indicates calibration or distribution mismatch issues")
    
    output_lines.append("\n" + "=" * 70)
    
    # Per-bucket analysis (if enabled)
    if args.per_bucket:
        output_lines.append("\n" + "=" * 70)
        output_lines.append("PER-BUCKET NEGATIVE ANALYSIS")
        output_lines.append("=" * 70)
        
        print("\n[4] Running per-bucket analysis...")
        scored_negs = score_sequences_with_provenance(
            args.negatives, model_data, args.max_seqs, args.seed
        )
        print(f"    Scored {len(scored_negs)} negatives with provenance")
        
        ref_thresholds = model_data.get("reference_thresholds", {"neg_q95": 0.5, "neg_q99": 0.5})
        output_lines.append(format_bucket_stats(scored_negs, ref_thresholds))
        
        # Top-K highest scoring negatives
        output_lines.append("\n" + "-" * 50)
        output_lines.append(format_top_k_negatives(scored_negs, args.top_k))
        
        output_lines.append("\n" + "=" * 70)
    
    # Print and save
    report = "\n".join(output_lines)
    print(report)
    
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(report)
        print(f"\n[OUTPUT] Wrote report to {out_path}")
    
    # Optional histogram
    if args.histogram:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(neg_scores, bins=50, alpha=0.6, label=f"Negatives (n={len(neg_scores)})", color="red")
            ax.hist(pos_scores, bins=50, alpha=0.6, label=f"Positives (n={len(pos_scores)})", color="green")
            ax.set_xlabel("Score (predict_proba class 1)")
            ax.set_ylabel("Count")
            ax.set_title("Score Distribution: Positives vs Negatives")
            ax.legend()
            ax.set_xlim(0, 1)
            
            hist_path = Path(args.histogram)
            hist_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(hist_path, dpi=150)
            plt.close()
            print(f"[OUTPUT] Saved histogram to {hist_path}")
        except ImportError:
            print("[WARNING] matplotlib not available, skipping histogram")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
