#!/usr/bin/env python3
"""
Minimal reproduction script for v8 quantization diagnosis.

Scores 200 MirGeneDB positives and 200 negatives, prints unique score counts
before and after calibration.

Usage:
    python training/debug_quantization.py \
        --model models/hsa_premirna_rf_v8.pkl \
        --positives data/train/hsa_mirgene_premirna.fa \
        --negatives data/train/negatives_v2.fa

Author: miRPV-NG Team
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import joblib
import random
from collections import Counter

from Bio import SeqIO

try:
    from mirpv_ng.classifier import compute_feature_vector
    HAS_CLASSIFIER = True
except ImportError:
    HAS_CLASSIFIER = False
    print("[ERROR] Could not import classifier module")
    sys.exit(1)

try:
    from sklearn.calibration import CalibratedClassifierCV
except ImportError:
    CalibratedClassifierCV = None


def load_model_and_base_rf(model_path: str):
    """Load model and extract base RF if calibrated."""
    payload = joblib.load(model_path)
    
    model = payload["model"]
    feature_cols = (
        payload.get("feature_cols") or 
        payload.get("feature_names_") or 
        payload.get("feature_names") or []
    )
    feature_set = payload.get("feature_set", "extended")
    tier2_enabled = payload.get("tier2_enabled", False)
    
    # Try to extract base RF
    base_rf = None
    is_calibrated = CalibratedClassifierCV and isinstance(model, CalibratedClassifierCV)
    
    if is_calibrated:
        try:
            if hasattr(model, 'calibrated_classifiers_') and model.calibrated_classifiers_:
                cc = model.calibrated_classifiers_[0]
                if hasattr(cc, 'estimator'):
                    base_rf = cc.estimator
                elif hasattr(cc, 'base_estimator'):
                    base_rf = cc.base_estimator
        except Exception:
            pass
    else:
        base_rf = model
    
    return model, base_rf, feature_cols, feature_set, tier2_enabled, is_calibrated


def score_fasta(fasta_path, model, base_rf, feature_cols, feature_set, tier2_enabled, max_seqs=200, seed=42):
    """Score sequences and return (raw_probs, calibrated_probs)."""
    rng = random.Random(seed)
    
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if len(records) > max_seqs:
        rng.shuffle(records)
        records = records[:max_seqs]
    
    raw_probs = []
    cal_probs = []
    
    for rec in records:
        seq = str(rec.seq).upper().replace("T", "U")
        
        try:
            feats = compute_feature_vector(
                seq,
                feature_set=feature_set,
                tier2_enabled=tier2_enabled,
            )
            
            x = np.array([[feats.get(col, 0.0) for col in feature_cols]])
            
            # Calibrated score
            cal_p = float(model.predict_proba(x)[0, 1])
            cal_probs.append(cal_p)
            
            # Raw RF score
            if base_rf is not None:
                raw_p = float(base_rf.predict_proba(x)[0, 1])
                raw_probs.append(raw_p)
            else:
                raw_probs.append(cal_p)
                
        except Exception as e:
            continue
    
    return raw_probs, cal_probs


def main():
    ap = argparse.ArgumentParser(description="Debug v8 quantization issue")
    ap.add_argument("--model", required=True, help="Model pickle file")
    ap.add_argument("--positives", required=True, help="Positive FASTA")
    ap.add_argument("--negatives", required=True, help="Negative FASTA")
    ap.add_argument("--max-seqs", type=int, default=200, help="Max sequences per set")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = ap.parse_args()
    
    print("=" * 70)
    print("QUANTIZATION DIAGNOSTIC")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Positives: {args.positives}")
    print(f"Negatives: {args.negatives}")
    print(f"Max seqs: {args.max_seqs}")
    print("=" * 70)
    
    # Load model
    print("\n[1] Loading model...")
    model, base_rf, feature_cols, feature_set, tier2_enabled, is_calibrated = load_model_and_base_rf(args.model)
    
    print(f"    Feature set: {feature_set}")
    print(f"    Feature count: {len(feature_cols)}")
    print(f"    Is calibrated: {is_calibrated}")
    print(f"    Base RF available: {base_rf is not None}")
    
    # Score positives
    print("\n[2] Scoring positives...")
    pos_raw, pos_cal = score_fasta(
        args.positives, model, base_rf, feature_cols, feature_set, tier2_enabled,
        args.max_seqs, args.seed
    )
    
    # Score negatives
    print("\n[3] Scoring negatives...")
    neg_raw, neg_cal = score_fasta(
        args.negatives, model, base_rf, feature_cols, feature_set, tier2_enabled,
        args.max_seqs, args.seed
    )
    
    # Report
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nPOSITIVES (n={len(pos_raw)}):")
    print(f"  Raw RF:    unique={len(set(pos_raw)):4d}, min={min(pos_raw):.4f}, max={max(pos_raw):.4f}, median={np.median(pos_raw):.4f}")
    print(f"  Calibrated: unique={len(set(pos_cal)):4d}, min={min(pos_cal):.4f}, max={max(pos_cal):.4f}, median={np.median(pos_cal):.4f}")
    
    print(f"\nNEGATIVES (n={len(neg_raw)}):")
    print(f"  Raw RF:    unique={len(set(neg_raw)):4d}, min={min(neg_raw):.4f}, max={max(neg_raw):.4f}, median={np.median(neg_raw):.4f}")
    print(f"  Calibrated: unique={len(set(neg_cal)):4d}, min={min(neg_cal):.4f}, max={max(neg_cal):.4f}, median={np.median(neg_cal):.4f}")
    
    # Unique values sample
    print("\nSAMPLE UNIQUE VALUES:")
    print(f"  Pos Raw (first 10):  {sorted(set(pos_raw))[:10]}")
    print(f"  Pos Cal (first 10):  {sorted(set(pos_cal))[:10]}")
    print(f"  Neg Raw (first 10):  {sorted(set(neg_raw))[:10]}")
    print(f"  Neg Cal (first 10):  {sorted(set(neg_cal))[:10]}")
    
    # Most common values
    print("\nMOST COMMON VALUES:")
    pos_cal_counts = Counter(pos_cal).most_common(5)
    neg_cal_counts = Counter(neg_cal).most_common(5)
    print(f"  Pos Cal top 5: {pos_cal_counts}")
    print(f"  Neg Cal top 5: {neg_cal_counts}")
    
    # Diagnosis
    print("\n" + "-" * 50)
    print("DIAGNOSIS:")
    
    pos_raw_unique = len(set(pos_raw))
    pos_cal_unique = len(set(pos_cal))
    neg_raw_unique = len(set(neg_raw))
    neg_cal_unique = len(set(neg_cal))
    
    if pos_raw_unique < 10 or neg_raw_unique < 10:
        print("⚠ RAW RF PROBABILITIES ARE QUANTIZED!")
        print("   Likely cause: feature pipeline mismatch")
        print("   - Check if feature_cols count matches number of computed features")
        print("   - Check for constant/NaN columns in feature matrix")
        print("   - Verify feature ordering is consistent between train and inference")
    elif pos_cal_unique < 10 or neg_cal_unique < 10:
        print("⚠ ONLY CALIBRATED SCORES ARE QUANTIZED!")
        print("   Likely cause: isotonic calibration issue")
        print("   - Isotonic may have been fitted on too few samples")
        print("   - Or fitted on in-sample data (prefit mode)")
        print("   FIX: Use CV-based calibration (fit on out-of-fold predictions)")
    else:
        print("✓ Score distributions appear normal")
        print(f"   Raw RF: {pos_raw_unique} pos unique, {neg_raw_unique} neg unique")
        print(f"   Calibrated: {pos_cal_unique} pos unique, {neg_cal_unique} neg unique")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
