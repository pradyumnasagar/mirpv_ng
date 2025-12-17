# mirpv_ng/train_premirna_model.py

"""
Train a RandomForest pre-miRNA classifier.
(FIXED: Includes class_weight and saves feature_set metadata)
"""

import argparse
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

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
    args = parser.parse_args()

    # 1. Load Data
    df_pos = load_and_featurize(args.pos_fasta, 1, args.feature_set, args.rnafold_bin, args.tier2)
    df_neg = load_and_featurize(args.neg_fasta, 0, args.feature_set, args.rnafold_bin, args.tier2)


    # Combine
    df = pd.concat([df_pos, df_neg], ignore_index=True)
    
    # Shuffle
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Split X, y
    X, y, feature_cols = split_features_labels(df)
    print(f"[train] Total examples: {len(df)}")
    print(f"[train] Feature count: {len(feature_cols)}")

    # 2. Initialize Model (CRITICAL FIXES HERE)
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        class_weight="balanced_subsample",  # <--- CRITICAL: Restores sensitivity!
        n_jobs=-1,
        random_state=42,
    )

    # 3. Cross-Validation
    if args.cv_folds > 1:
        print(f"[cv] Running {args.cv_folds}-fold stratified cross-validation...")
        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        # We can pass DataFrame directly to sklearn now, but X is numpy array
        # Just pass X, y
        scores = cross_validate(
            clf, X, y, cv=cv, 
            scoring=["roc_auc", "accuracy", "precision", "recall", "f1"],
            n_jobs=-1
        )
        
        auc_mean = scores["test_roc_auc"].mean()
        auc_std = scores["test_roc_auc"].std()
        rec_mean = scores["test_recall"].mean()
        
        print(f"[cv] AUC: {auc_mean:.4f} ± {auc_std:.4f}")
        print(f"[cv] Recall: {rec_mean:.4f}")

    # 4. Final Training on Full Dataset
    clf.fit(X, y)
    
    # Evaluate on training data (just for sanity check log)
    ypred = clf.predict(X)
    acc = accuracy_score(y, ypred)
    auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
    cm = confusion_matrix(y, ypred)
    cr = classification_report(y, ypred)

    print(f"[train] Accuracy: {acc:.4f}")
    print(f"[train] AUC: {auc:.4f}")
    print(f"[train] Confusion matrix:\n{cm}")

    # 5. Save Model (CRITICAL FIX: Save feature_set and feature_cols)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump({
        "model": clf, 
        "feature_cols": feature_cols,     # Saves the new 'pgs_' names
        "feature_set": args.feature_set   # Saves 'extended'
    }, args.model_out)
    
    print(f"[train] Saved model to {args.model_out}")

    # 6. Save Metrics
    with open(args.metrics_out, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n\n")
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