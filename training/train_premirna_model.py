# mirpv_ng/train_premirna_model.py

"""
Train a Random Forest pre-miRNA classifier with probability calibration.
Includes class balancing and feature set metadata preservation.
"""
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import argparse
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
import sklearn
from packaging.version import parse as parse_version

# Version-aware import for FrozenEstimator
try:
    from sklearn.utils.validation import check_is_fitted
    from sklearn.base import FrozenEstimator
except ImportError:
    FrozenEstimator = None

from mirpv_ng.features import read_fasta, compute_features_for_sequences


def load_and_featurize(
    fasta_path: str, label: int, feature_set: str, rnafold_bin: str, tier2_enabled: bool
) -> pd.DataFrame:
    records = read_fasta(fasta_path)
    print(f"[train] {fasta_path}: {len(records)} sequences (label={label})")
    df = compute_features_for_sequences(
        records,
        feature_set=feature_set,
        rnafold_bin=rnafold_bin,
        tier2_enabled=tier2_enabled,
    )
    df["label"] = label
    return df


def split_features_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    drop_cols = {"id", "seq", "struct", "mfe", "label"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].values
    y = df["label"].values
    return X, y, feature_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos-fasta", required=True)
    parser.add_argument("--neg-fasta", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--feature-set", default="extended")
    parser.add_argument("--rnafold-bin", default="RNAfold")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--tier2", action="store_true", help="Enable Tier-2 soft-gated features during training")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--model-version", default="v7",
                        help="Model version string for naming (default: v7)")
    # Scan-background injection (CRITICAL for scan-train consistency)
    parser.add_argument("--scan-background-fasta", default=None,
                        help="FASTA of scan-generated candidates to inject as negatives")
    parser.add_argument("--scan-background-limit", type=int, default=500,
                        help="Max scan-background sequences to inject (default: 500)")
    # Calibration strategy (choose ONE, avoid double calibration)
    parser.add_argument("--calibration", choices=["sigmoid", "isotonic", "none"],
                        default="isotonic",
                        help="Calibration method: sigmoid (Platt), isotonic (scan-friendly), none (default: isotonic)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--rng-seed", type=int, dest="seed",
                        help="Alias for --seed (deprecated)")
    parser.add_argument("--scan-background-weight", type=float, default=0.2,
                        help="Sample weight for injected scan-background sequences (default: 0.2)")
    
    # Use centralized parallel args
    if hasattr(sys.modules.get("mirpv_ng.parallel"), "add_parallel_args"):
        from mirpv_ng.parallel import add_parallel_args, ParallelConfig
        add_parallel_args(parser)
    else:
        # Fallback if module not found (shouldn't happen in repo)
        parser.add_argument("--threads", type=int, default=1, help="Number of threads")

    args = parser.parse_args()

    # Resolve threads using central logic
    if "ParallelConfig" in locals():
        pcfg = ParallelConfig.from_args(args)
        n_jobs = pcfg.jobs
        print(f"[config] Resolved parallel jobs: {n_jobs}")
    else:
        n_jobs = args.threads

    # 1. Load Data
    df_pos = load_and_featurize(args.pos_fasta, 1, args.feature_set, args.rnafold_bin, args.tier2)
    df_neg = load_and_featurize(args.neg_fasta, 0, args.feature_set, args.rnafold_bin, args.tier2)


    # Combine
    df = pd.concat([df_pos, df_neg], ignore_index=True)
    
    # Shuffle
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    
    # Scan-background injection (CRITICAL for scan-train consistency)
    # Teaching the model that scan-extracted hairpins are usually background
    df_scan = None
    if args.scan_background_fasta:
        print(f"[train] Loading scan-background for injection: {args.scan_background_fasta}")
        df_scan = load_and_featurize(
            args.scan_background_fasta, 0, args.feature_set, args.rnafold_bin, args.tier2
        )
        # Limit to avoid overwhelming the training set
        if len(df_scan) > args.scan_background_limit:
            df_scan = df_scan.sample(n=args.scan_background_limit, random_state=args.seed)
        
        # Enforce 5% limit (user requirement)
        # We want len(df_scan) / (len(df_neg) + len(df_scan)) <= 0.05
        # => len(df_scan) <= (0.05 / 0.95) * len(df_neg)
        max_scan = int((0.05 / 0.95) * len(df_neg))
        if len(df_scan) > max_scan:
            print(f"[train] LIMITING scan-background from {len(df_scan)} to {max_scan} (5% max)")
            df_scan = df_scan.sample(n=max_scan, random_state=args.seed)
        
        n_total_neg = len(df_neg) + len(df_scan)
        scan_frac = len(df_scan) / n_total_neg
        print(f"[train] Scan-background: {len(df_scan)} sequences ({scan_frac:.1%} of negatives)")
        
        # Add sample weights: 1.0 for normal, scan-background-weight for injected
        # This requires tracking which are which.
        # Simpler approach: Create a combined DataFrame with a weight column
        df["sample_weight"] = 1.0
        df_scan["sample_weight"] = args.scan_background_weight
        
        df = pd.concat([df, df_scan], ignore_index=True)
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    else:
        df["sample_weight"] = 1.0
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Split features / labels
    # We need to preserve weights if using them.
    # split_features_labels() returns X, y, feat_cols.
    # We'll extract weights manually before dropping columns?
    # Or just handle it here.
    weights = df["sample_weight"].values if "sample_weight" in df.columns else np.ones(len(df))
    
    X, y, feature_cols = split_features_labels(df)

    # Held-out validation split (CRITICAL)
    # Pass weights to split to keep them aligned
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X, y, weights,
        test_size=args.test_size,
        stratify=y,
        random_state=args.seed
    )

    print(f"[train] Total examples: {len(df)}")
    print(f"[train] Feature count: {len(feature_cols)}")
    print(f"[train] Train size: {len(X_train)}")
    print(f"[train] Validation size: {len(X_val)}")

    # 2. Initialize Model
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        class_weight="balanced_subsample",
        n_jobs=n_jobs,
        random_state=args.seed,
    )

    # 3. Cross-Validation
    if args.cv_folds > 1:
        print(f"[cv] Running {args.cv_folds}-fold stratified cross-validation...")
        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        # Pass X, y (numpy arrays)
        scores = cross_validate(
            clf, X, y, cv=cv, 
            scoring=["roc_auc", "accuracy", "precision", "recall", "f1"],
            n_jobs=n_jobs
        )
        
        auc_mean = scores["test_roc_auc"].mean()
        auc_std = scores["test_roc_auc"].std()
        rec_mean = scores["test_recall"].mean()
        
        print(f"[cv] AUC: {auc_mean:.4f} ± {auc_std:.4f}")
        print(f"[cv] Recall: {rec_mean:.4f}")

    # 4. Final Training and Calibration
    # Train base model on training set only (apply weights if distinct from 1.0)
    clf.fit(X_train, y_train, sample_weight=w_train)
    
    if args.calibration == "none":
        # No calibration - use raw RF probabilities
        print(f"[calibration] Using raw RF probabilities (no calibration)")
        final_model = clf
    else:
        print(f"[calibration] Applying {args.calibration} calibration on validation set...")
        
        # WARN if validation set is too small for isotonic calibration
        if args.calibration == "isotonic" and len(X_val) < 1000:
            print(f"[calibration] WARNING: Validation set size ({len(X_val)}) is small for isotonic calibration.")
            print(f"               Consider using --calibration sigmoid or increasing data.")
        
        # Version-aware calibration
        sklearn_version = parse_version(sklearn.__version__)
        calibration_impl = "prefit_legacy"
        
        if FrozenEstimator is not None and sklearn_version >= parse_version("1.6"):
            print(f"[calibration] Using FrozenEstimator (sklearn {sklearn.__version__})")
            frozen_clf = FrozenEstimator(clf)
            cal = CalibratedClassifierCV(
                estimator=frozen_clf,
                method=args.calibration,
            )
            cal.fit(X_val, y_val, sample_weight=w_val) 
            calibration_impl = "frozen_estimator"
        else:
            print(f"[calibration] Using cv='prefit' (sklearn {sklearn.__version__})")
            cal = CalibratedClassifierCV(
                estimator=clf,
                method=args.calibration,
                cv="prefit"
            )
            cal.fit(X_val, y_val, sample_weight=w_val)
            calibration_impl = "prefit_legacy"
            
        final_model = cal
    
    # Get validation probabilities from final model
    val_probs = final_model.predict_proba(X_val)[:, 1]
    
    # Determine optimal decision threshold by maximizing F1 score on validation data
    precision, recall, thresholds = precision_recall_curve(y_val, val_probs)
    
    f1 = (2 * precision * recall) / (precision + recall + 1e-9)
    best_idx = np.nanargmax(f1)
    best_threshold = float(thresholds[best_idx])
    
    print(f"[calibration] Selected decision threshold (F1-max, validation): {best_threshold:.4f}")
    
    # Validation metrics
    val_pred = (val_probs >= best_threshold).astype(int)
    acc = accuracy_score(y_val, val_pred)
    auc = roc_auc_score(y_val, val_probs)
    pr_auc = average_precision_score(y_val, val_probs)
    brier = brier_score_loss(y_val, val_probs)
    cm = confusion_matrix(y_val, val_pred)
    cr = classification_report(y_val, val_pred)

    print(f"[val] Accuracy: {acc:.4f}")
    print(f"[val] ROC AUC: {auc:.4f}")
    print(f"[val] PR AUC (Average Precision): {pr_auc:.4f}")
    print(f"[val] Brier Score: {brier:.4f} (lower is better)")
    print(f"[val] Confusion matrix:\n{cm}")

    # 5. Save Model
    # Saves a dictionary containing the calibrated model itself, along with metadata
    # ensuring reproducibility (feature names, threshold, etc.)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    
    # Compute percentile thresholds for reference
    q90 = float(np.percentile(val_probs[y_val == 0], 90))  # 90th pct of negatives
    q95 = float(np.percentile(val_probs[y_val == 0], 95))  # 95th pct of negatives
    q99 = float(np.percentile(val_probs[y_val == 0], 99))  # 99th pct of negatives
    
    print(f"[calibration] Reference thresholds:")
    print(f"  - F1-max threshold: {best_threshold:.4f}")
    print(f"  - 90th pct of negatives: {q90:.4f}")
    print(f"  - 95th pct of negatives: {q95:.4f}")
    print(f"  - 99th pct of negatives: {q99:.4f}")
    
    # Calculate Scan FPR (False Positive Rate) at these thresholds
    # "fraction of negatives above q95/q99 of negatives" 
    # By definition this is 5% / 1% on the validation set, but useful to verify calibration
    fpr_q95 = np.mean(val_probs[y_val == 0] >= q95)
    fpr_q99 = np.mean(val_probs[y_val == 0] >= q99)
    print(f"  - FPR at q95: {fpr_q95:.4f}")
    print(f"  - FPR at q99: {fpr_q99:.4f}")
    
    # If scan background was injected, compute its specific FPR (if trackable)
    # We don't track scan background distinct from other negatives easily here without ID tracking.
    # But since it's merged into negatives, it influences the thresholds.

    
    # Create reference thresholds dict (computed from NEGATIVE validation scores)
    ref_thresholds_dict = {
        "f1_max": best_threshold,
        "neg_q90": q90,
        "neg_q95": q95,
        "neg_q99": q99,
    }
    
    joblib.dump({
        "model": final_model,
        # Feature column names - save redundantly for CLI compatibility
        "feature_cols": feature_cols,
        "feature_names_": feature_cols,
        "feature_names": feature_cols,
        # Feature set and tier2
        "feature_set": args.feature_set,
        "tier2_enabled": args.tier2,
        # Decision threshold - save redundantly for CLI compatibility
        # NOTE: f1_threshold is for METRICS ONLY. CLI scoring should output raw scores
        # and let users filter with --min-score. Do NOT use as mandatory cutoff.
        "threshold": best_threshold,
        "f1_threshold": best_threshold,
        "decision_threshold": best_threshold,
        # Reference thresholds - save redundantly for CLI compatibility
        "reference_thresholds": ref_thresholds_dict,
        "ref_thresholds": ref_thresholds_dict,
        # Model metadata
        "score_type": f"calibrated_rf_{args.model_version}",
        "calibration": args.calibration,
        "calibration_impl": calibration_impl if args.calibration != "none" else "none",
        "validation_fraction": args.test_size,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "training_metadata": {
            "n_positives": len(df_pos),
            "n_negatives": len(df_neg),
            "n_scan_background": len(df_scan) if df_scan is not None else 0,
            "scan_background_weight": args.scan_background_weight,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "pos_fasta": args.pos_fasta,
            "neg_fasta": args.neg_fasta,
            "scan_background_fasta": args.scan_background_fasta,
            "n_estimators": args.n_estimators,
            "cv_folds": args.cv_folds,
            "seed": args.seed,
            "script_version": "train_premirna_model.py v3.2 (redundant keys)",
        },
        # Training args snapshot for reproducibility
        "training_args": vars(args),
    }, args.model_out)
    
    print(f"[train] Saved model to {args.model_out}")

    # 6. Save Metrics
    with open(args.metrics_out, "w") as f:
        f.write(f"Model Version: {args.model_version}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"ROC AUC: {auc:.4f}\n")
        f.write(f"PR AUC (Average Precision): {pr_auc:.4f}\n")
        f.write(f"Brier Score: {brier:.4f}\n")
        f.write(f"Decision Threshold: {best_threshold:.4f}\n\n")
        f.write(f"Confusion Matrix:\n{cm}\n\n")
        f.write(f"Classification Report:\n{cr}\n")
    print(f"[train] Wrote metrics to {args.metrics_out}")
    
    # 7. Feature Importance
    importances = clf.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    print("\n[analysis] Top 20 Most Important Features:")
    print(feat_imp.head(20))
    
    # Save importances
    imp_out = str(Path(args.model_out).with_suffix(".importances.csv"))
    feat_imp.to_csv(imp_out)
    print(f"[analysis] Saved full feature importances to {imp_out}")

if __name__ == "__main__":
    main()