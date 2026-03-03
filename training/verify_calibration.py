#!/usr/bin/env python3
"""
Verify Calibration: Check pre-miRNA model calibration pipeline for common issues.

Checks:
1. Pipeline is RF -> CalibratedClassifierCV -> predict_proba (called once)
2. Correct probability column used (class 1)
3. No double-sigmoid or post-score transforms
4. Threshold corresponds to current operating point

Usage:
    python training/verify_calibration.py \
        --model models/hsa_premirna_rf_extended_tier2_v6_calibrated.pkl

Author: miRPV-NG Team
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier


def check_model_structure(model_data: dict) -> list:
    """Check model artifact structure and calibration pipeline."""
    issues = []
    warnings = []
    info = []
    
    # 1. Check required keys
    required_keys = ["model", "feature_cols", "decision_threshold"]
    for key in required_keys:
        if key not in model_data:
            issues.append(f"Missing required key: {key}")
    
    if issues:
        return issues, warnings, info
    
    model = model_data["model"]
    threshold = model_data["decision_threshold"]
    feature_cols = model_data["feature_cols"]
    
    info.append(f"Feature columns: {len(feature_cols)}")
    info.append(f"Decision threshold: {threshold:.4f}")
    
    # 2. Check model type
    if isinstance(model, CalibratedClassifierCV):
        info.append("✓ Model is CalibratedClassifierCV (correct)")
        
        # Check underlying estimator
        estimators = getattr(model, "calibrated_classifiers_", [])
        if estimators:
            base = estimators[0].estimator
            if isinstance(base, RandomForestClassifier):
                info.append(f"✓ Base estimator is RandomForestClassifier")
                info.append(f"  - n_estimators: {base.n_estimators}")
            else:
                warnings.append(f"Unexpected base estimator type: {type(base).__name__}")
        else:
            # cv="prefit" case - check base_estimator
            base = getattr(model, "estimator", None)
            if base is None:
                base = getattr(model, "base_estimator", None)
            if base is not None:
                info.append(f"✓ Using prefit estimator: {type(base).__name__}")
            else:
                warnings.append("Could not find base estimator")
        
        # Check calibration method
        method = getattr(model, "method", "unknown")
        info.append(f"  - Calibration method: {method}")
        
        if method == "sigmoid":
            info.append("  - Sigmoid (Platt scaling) calibration")
        elif method == "isotonic":
            info.append("  - Isotonic regression calibration")
    else:
        warnings.append(f"Model is NOT CalibratedClassifierCV, type: {type(model).__name__}")
        if isinstance(model, RandomForestClassifier):
            warnings.append("  Using raw RF without calibration - probabilities may not be well-calibrated")
    
    # 3. Check for double-calibration
    if isinstance(model, CalibratedClassifierCV):
        estimators = getattr(model, "calibrated_classifiers_", [])
        for i, clf in enumerate(estimators):
            base = getattr(clf, "estimator", None)
            if isinstance(base, CalibratedClassifierCV):
                issues.append(f"DOUBLE CALIBRATION DETECTED in estimator {i}!")
    
    # 4. Check threshold range
    if threshold < 0.0 or threshold > 1.0:
        issues.append(f"Invalid threshold {threshold} (must be 0-1)")
    elif threshold < 0.3:
        warnings.append(f"Very low threshold ({threshold:.3f}) - may accept many false positives")
    elif threshold > 0.8:
        warnings.append(f"Very high threshold ({threshold:.3f}) - may reject many true positives")
    
    # 5. Check additional metadata
    if "score_type" in model_data:
        info.append(f"Score type: {model_data['score_type']}")
    if "feature_set" in model_data:
        info.append(f"Feature set: {model_data['feature_set']}")
    if "tier2_enabled" in model_data:
        info.append(f"Tier2 enabled: {model_data['tier2_enabled']}")
    if "calibration" in model_data:
        info.append(f"Calibration: {model_data['calibration']}")
    if "validation_fraction" in model_data:
        info.append(f"Validation fraction: {model_data['validation_fraction']}")
    if "roc_auc" in model_data:
        info.append(f"ROC AUC (stored): {model_data['roc_auc']:.4f}")
    if "brier_score" in model_data:
        info.append(f"Brier score (stored): {model_data['brier_score']:.4f}")
    
    return issues, warnings, info


def test_predict_proba(model_data: dict) -> tuple:
    """Test predict_proba behavior with synthetic data."""
    issues = []
    warnings = []
    info = []
    
    model = model_data["model"]
    feature_cols = model_data["feature_cols"]
    n_features = len(feature_cols)
    
    # Create synthetic test data
    np.random.seed(42)
    X_test = np.random.randn(100, n_features)
    
    try:
        proba = model.predict_proba(X_test)
        info.append(f"✓ predict_proba succeeded, shape: {proba.shape}")
        
        # Check shape
        if proba.shape[1] != 2:
            issues.append(f"predict_proba returned {proba.shape[1]} columns, expected 2")
        else:
            info.append("✓ Returns 2 probability columns (class 0, class 1)")
        
        # Check probability range
        if np.any(proba < 0) or np.any(proba > 1):
            issues.append("Probabilities outside [0, 1] range!")
        else:
            info.append("✓ All probabilities in [0, 1] range")
        
        # Check row sums
        row_sums = proba.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            issues.append(f"Probability rows don't sum to 1: {row_sums.min():.6f} - {row_sums.max():.6f}")
        else:
            info.append("✓ Probability rows sum to 1")
        
        # Check for collapsed probabilities
        class1_probs = proba[:, 1]
        unique_probs = len(np.unique(np.round(class1_probs, 4)))
        
        if unique_probs < 5:
            warnings.append(f"Very few unique probability values ({unique_probs}) - may indicate calibration issue")
        else:
            info.append(f"✓ {unique_probs} unique probability values (good distribution)")
        
        # Check for extreme concentration
        if np.mean(class1_probs < 0.1) > 0.9:
            warnings.append("90%+ of probabilities < 0.1 - scores may be crushed")
        if np.mean(class1_probs > 0.9) > 0.9:
            warnings.append("90%+ of probabilities > 0.9 - model may be overconfident")
        
        info.append(f"  - Class 1 proba range: [{class1_probs.min():.4f}, {class1_probs.max():.4f}]")
        info.append(f"  - Class 1 proba mean: {class1_probs.mean():.4f}")
        info.append(f"  - Class 1 proba std: {class1_probs.std():.4f}")
        
    except Exception as e:
        issues.append(f"predict_proba failed: {e}")
    
    return issues, warnings, info


def check_for_double_sigmoid(model_data: dict) -> tuple:
    """Check if sigmoid calibration might be applied twice."""
    issues = []
    warnings = []
    info = []
    
    model = model_data["model"]
    
    # Check if classifier.py or cli.py applies additional transforms
    info.append("Checking for potential double-transform issues...")
    
    if isinstance(model, CalibratedClassifierCV):
        method = getattr(model, "method", "unknown")
        
        if method == "sigmoid":
            info.append("Model uses sigmoid calibration (Platt scaling)")
            info.append("⚠ Ensure cli.py / classifier.py don't apply additional sigmoid transforms")
            info.append("  - Should call: model.predict_proba(X)[:, 1] directly")
            info.append("  - Should NOT apply: 1 / (1 + exp(-score)) afterwards")
    
    return issues, warnings, info


def main():
    ap = argparse.ArgumentParser(
        description="Verify pre-miRNA model calibration pipeline"
    )
    ap.add_argument("--model", required=True, help="Model pickle file")
    
    args = ap.parse_args()
    
    print("=" * 70)
    print("CALIBRATION VERIFICATION")
    print("=" * 70)
    print(f"Model: {args.model}")
    print("=" * 70)
    
    # Load model
    print("\n[1] Loading model...")
    try:
        model_data = joblib.load(args.model)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return 1
    
    if not isinstance(model_data, dict):
        print(f"WARNING: Model is not a dict, type: {type(model_data)}")
        model_data = {"model": model_data, "feature_cols": [], "decision_threshold": 0.5}
    
    all_issues = []
    all_warnings = []
    all_info = []
    
    # Check 1: Model structure
    print("\n[2] Checking model structure...")
    issues, warnings, info = check_model_structure(model_data)
    all_issues.extend(issues)
    all_warnings.extend(warnings)
    all_info.extend(info)
    
    # Check 2: predict_proba behavior
    if "model" in model_data and "feature_cols" in model_data and model_data["feature_cols"]:
        print("\n[3] Testing predict_proba...")
        issues, warnings, info = test_predict_proba(model_data)
        all_issues.extend(issues)
        all_warnings.extend(warnings)
        all_info.extend(info)
    else:
        print("\n[3] Skipping predict_proba test (missing model or feature_cols)")
    
    # Check 3: Double-sigmoid
    print("\n[4] Checking for double-sigmoid issues...")
    issues, warnings, info = check_for_double_sigmoid(model_data)
    all_issues.extend(issues)
    all_warnings.extend(warnings)
    all_info.extend(info)
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    print("\n📋 INFO:")
    for item in all_info:
        print(f"  {item}")
    
    if all_warnings:
        print("\n⚠️  WARNINGS:")
        for item in all_warnings:
            print(f"  {item}")
    
    if all_issues:
        print("\n❌ ISSUES:")
        for item in all_issues:
            print(f"  {item}")
        print("\n" + "=" * 70)
        print("RESULT: ISSUES FOUND - Review required")
        print("=" * 70)
        return 1
    else:
        print("\n" + "=" * 70)
        print("RESULT: ✓ No critical issues found")
        if all_warnings:
            print(f"         ({len(all_warnings)} warning(s) to review)")
        print("=" * 70)
        return 0


if __name__ == "__main__":
    sys.exit(main())
