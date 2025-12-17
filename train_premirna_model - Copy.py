#!/usr/bin/env python3

"""
Train a RandomForest pre-miRNA classifier from:
    - positive FASTA (known pre-miRNAs)
    - negative FASTA (pseudo hairpins from CDS)

Uses features from mirpv_ng.features (core36 or extended).
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
from sklearn.model_selection import train_test_split

from mirpv_ng.features import read_fasta, compute_features_for_sequences


def load_and_featurize(
    fasta_path: str, label: int, feature_set: str, rnafold_bin: str
) -> pd.DataFrame:
    records = read_fasta(fasta_path)
    print(f"[train] {fasta_path}: {len(records)} sequences (label={label})")
    df = compute_features_for_sequences(
        records, feature_set=feature_set, rnafold_bin=rnafold_bin
    )
    df["label"] = label
    return df


def split_features_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    drop_cols = {"id", "seq", "struct", "mfe", "label"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].values.astype(float)
    y = df["label"].values.astype(int)
    return X, y, feature_cols


def main():
    ap = argparse.ArgumentParser(
        description="Train pre-miRNA RF classifier from positive and negative FASTA."
    )
    ap.add_argument("--pos-fasta", required=True)
    ap.add_argument("--neg-fasta", required=True)
    ap.add_argument(
        "--feature-set",
        choices=["core36", "extended"],
        default="core36",
    )
    ap.add_argument("--rnafold-bin", default="RNAfold")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--model-out", required=True)
    ap.add_argument("--metrics-out", required=True)
    args = ap.parse_args()

    # 1. Load Data
    df_pos = load_and_featurize(
        args.pos_fasta, label=1, feature_set=args.feature_set, rnafold_bin=args.rnafold_bin
    )
    df_neg = load_and_featurize(
        args.neg_fasta, label=0, feature_set=args.feature_set, rnafold_bin=args.rnafold_bin
    )

    df_all = pd.concat([df_pos, df_neg], ignore_index=True)
    print(f"[train] Total examples: {len(df_all)}")

    # 2. Prepare Features
    X, y, feature_cols = split_features_labels(df_all)
    print(f"[train] Feature count: {X.shape[1]}")

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    # 3. Train Model
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        n_jobs=-1,
    )
    clf.fit(Xtr, ytr)

    # 4. Evaluate
    ypred = clf.predict(Xte)
    yprob = clf.predict_proba(Xte)[:, 1]

    acc = accuracy_score(yte, ypred)
    auc = roc_auc_score(yte, yprob)
    cm = confusion_matrix(yte, ypred)
    cr = classification_report(yte, ypred)

    print(f"[train] Accuracy: {acc:.4f}")
    print(f"[train] AUC: {auc:.4f}")
    print(f"[train] Confusion matrix:\n{cm}")
    print(f"[train] Classification report:\n{cr}")

    # 5. Save Model
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "feature_cols": feature_cols}, args.model_out)
    print(f"[train] Saved model to {args.model_out}")

    # 6. Save Metrics
    with open(args.metrics_out, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n\n")
        f.write("Confusion matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification report:\n")
        f.write(cr + "\n")

    print(f"[train] Wrote metrics to {args.metrics_out}")

    # --- NEW: Feature Importance Analysis (Inside main) ---
    print("\n[analysis] Top 20 Most Important Features:")
    importances = clf.feature_importances_
    # Create a Series for easy sorting
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    print(feat_imp.head(20))
    
    # Save importances to a file for record keeping
    imp_out = str(args.metrics_out).replace(".metrics.txt", ".importances.csv")
    feat_imp.to_csv(imp_out)
    print(f"[analysis] Saved full feature importances to {imp_out}")
    # ------------------------------------------------------


if __name__ == "__main__":
    main()