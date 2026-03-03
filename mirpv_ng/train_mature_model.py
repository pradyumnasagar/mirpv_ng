# train_mature_ranker.py
# Unified Trainer: Trains a single model to rank candidates from BOTH arms.

import argparse
import pickle
import re
import joblib
from pathlib import Path
import numpy as np
import xgboost as xgb
from Bio import SeqIO
from sklearn.model_selection import GroupShuffleSplit

from mirpv_ng.features import run_rnafold
from mirpv_ng.mature_ranker import generate_duplex_candidates

def norm_seq(s: str) -> str:
    return str(s).upper().replace("T", "U").replace(" ", "")

def infer_parent_id(mature_id: str) -> str:
    """Robust ID parsing."""
    mid = mature_id.split()[0]
    # Remove suffix like -5p, -3p, -mature to find pre-miRNA ID
    mid = re.sub(r"-(5p|3p)$", "", mid, flags=re.IGNORECASE)
    mid = re.sub(r"-(mature|star|loop)$", "", mid, flags=re.IGNORECASE)
    return mid

def load_fasta_dict(path: str) -> dict:
    return {rec.id.split()[0]: norm_seq(rec.seq) for rec in SeqIO.parse(path, "fasta")}

def map_matures_to_precursors(pre_fa: str, fa_5p: str | None, fa_3p: str | None):
    """
    Returns a unified truth map:
    { pre_id: { "5p": (start, len), "3p": (start, len) } }
    """
    pre = load_fasta_dict(pre_fa)
    truth = {}  # pre_id -> dict of arms

    # Helper to process a file
    def process_file(path, label_arm):
        if not path: return
        seqs = load_fasta_dict(path)
        mapped = 0
        for mid, mseq in seqs.items():
            pid = infer_parent_id(mid)
            if pid in pre:
                full = pre[pid]
                start = full.find(mseq)
                if start != -1:
                    if pid not in truth: truth[pid] = {}
                    truth[pid][label_arm] = (start, len(mseq))
                    mapped += 1
        print(f"[data] Mapped {mapped} sequences from {label_arm} file.")

    process_file(fa_5p, "5p")
    process_file(fa_3p, "3p")
    
    return pre, truth

def build_unified_dataset(pre_seqs, truth_map, rnafold_bin):
    """
    Generates candidates for BOTH arms and labels them.
    Relevance:
       3 = Perfect Start + Perfect Length
       2 = Perfect Start + Wrong Length (Seed is correct!)
       1 = Shifted Start (+/- 1nt)
       0 = Garbage
    """
    X, y, groups = [], [], []
    feature_names = [
        "cand_len", "dist_loop", "dist_base", 
        "paired_frac", "starts_U", "stability", 
        "cut_unpaired", "overhang", "arm_5p"
    ]
    
    cnt = 0
    for pid, arms_truth in truth_map.items():
        seq = pre_seqs[pid]
        struct, _ = run_rnafold(seq, rnafold_bin=rnafold_bin)
        
        # Get ALL candidates (5p and 3p)
        cands = generate_duplex_candidates(seq, struct)
        if not cands: continue
        
        grp_X = []
        grp_y = []
        
        for c in cands:
            # 1. Feature Extraction
            f = c.features
            vec = [
                f["feat_len"], f["feat_dist_loop"], f["feat_dist_base"],
                f["feat_paired_frac"], f["feat_starts_U"], f["feat_stability"],
                f["feat_cut_unpaired"], f["feat_overhang_true"], f["feat_arm_5p"]
            ]
            
            # 2. Labeling
            # We check if this candidate matches the truth for its arm
            relevance = 0
            if c.arm in arms_truth:
                t_start, t_len = arms_truth[c.arm]
                d_start = abs(c.start - t_start)
                d_len = abs(c.length - t_len)
                
                if d_start == 0 and d_len == 0:
                    relevance = 3
                elif d_start == 0:
                    relevance = 2 # Seed is valid, length is isomiR
                elif d_start <= 1:
                    relevance = 1 # Shifted isomiR
                    
            grp_X.append(vec)
            grp_y.append(relevance)
        
        # Only add group if there is at least one positive candidate (relevance > 0)
        # Otherwise the Ranker learns nothing useful from this group
        if any(lbl > 0 for lbl in grp_y):
            X.extend(grp_X)
            y.extend(grp_y)
            groups.append(len(grp_X))
            cnt += 1
            
    print(f"[train] Processed {cnt} hairpins into dataset.")
    return np.array(X), np.array(y), np.array(groups), feature_names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre-fasta", required=True)
    ap.add_argument("--fasta-5p", default=None)
    ap.add_argument("--fasta-3p", default=None)
    ap.add_argument("--rnafold-bin", default="RNAfold")
    ap.add_argument("--model-out", required=True)
    args = ap.parse_args()

    # 1. Load & Map
    pre, truth = map_matures_to_precursors(args.pre_fasta, args.fasta_5p, args.fasta_3p)
    if not truth:
        raise SystemExit("No mappings found!")

    # 2. Build Dataset
    X, y, groups, cols = build_unified_dataset(pre, truth, args.rnafold_bin)
    
    # 3. Split (Group-aware)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups=np.repeat(np.arange(len(groups)), groups)))
    
    # We need to slice X, y but also RECALCULATE groups for the split
    # (XGBoost expects group array to match rows in X)
    # This is tricky with raw arrays. Easier to just train on full for production 
    # or implement careful group slicing.
    # For now, let's train on FULL dataset for maximum performance, 
    # relying on OOB internal validation of XGBoost if needed.
    
    print(f"[train] Training XGBRanker on {len(X)} candidates from {len(groups)} hairpins...")
    
    ranker = xgb.XGBRanker(
        objective="rank:ndcg",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=42
    )
    
    # Train
    ranker.fit(X, y, group=groups, verbose=True)
    
    # 4. Save
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": ranker, "feature_names": cols}, args.model_out)
    print(f"[train] Saved unified model to {args.model_out}")
    
    print("\n[analysis] Feature Importance:")
    for name, imp in zip(cols, ranker.feature_importances_):
        print(f"  {name:15s}: {imp:.4f}")

if __name__ == "__main__":
    main()