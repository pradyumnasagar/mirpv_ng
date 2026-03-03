#!/usr/bin/env python3
"""
Tail Risk Report: Lightweight guard against future regressions.

Scores a stratified sample of negatives from each bucket and reports:
- Max negative score per bucket
- Count of negatives exceeding q95/q99 thresholds

Usage:
    python training/tail_risk_report.py \\
        --model models/hsa_premirna_rf_v7.pkl \\
        --negatives data/train/negatives_v2.fa \\
        --out models/hsa_premirna_rf_v7.neg_tail_report.tsv \\
        --samples-per-bucket 300

Author: miRPV-NG Team
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import random
import numpy as np
import joblib
from typing import List, Tuple, Dict

from Bio import SeqIO

try:
    from mirpv_ng.classifier import compute_feature_vector
    HAS_CLASSIFIER = True
except ImportError:
    HAS_CLASSIFIER = False
    print("[ERROR] Cannot import classifier module")
    sys.exit(1)

try:
    from mirpv_ng.parallel import ParallelConfig, add_parallel_args, get_executor
    HAS_PARALLEL = True
except ImportError:
    HAS_PARALLEL = False


def parse_bucket_from_id(rec_id: str) -> str:
    """Parse bucket label from provenance-tagged FASTA ID."""
    if rec_id.startswith("N1|"):
        return "N1"
    elif rec_id.startswith("N2|"):
        return "N2"
    elif rec_id.startswith("N3|"):
        return "N3"
    elif rec_id.startswith("N1_"):
        return "N1"
    elif rec_id.startswith("N2_"):
        return "N2"
    elif rec_id.startswith("N3_"):
        return "N3"
    return "unknown"


def stratified_sample(records: List, samples_per_bucket: int, seed: int) -> Dict[str, List]:
    """Sample records stratified by bucket."""
    rng = random.Random(seed)
    
    buckets = {}
    for rec in records:
        bucket = parse_bucket_from_id(rec.id)
        if bucket not in buckets:
            buckets[bucket] = []
        buckets[bucket].append(rec)
    
    sampled = {}
    for bucket, recs in buckets.items():
        if len(recs) > samples_per_bucket:
            rng.shuffle(recs)
            sampled[bucket] = recs[:samples_per_bucket]
        else:
            sampled[bucket] = recs
    
    return sampled


def score_records(records: List, model_data: dict) -> List[Tuple[str, float]]:
    """Score records and return (id, score) pairs."""
    model = model_data["model"]
    feature_cols = model_data["feature_cols"]
    feature_set = model_data.get("feature_set", "extended")
    tier2_enabled = model_data.get("tier2_enabled", False)
    
    results = []
    for rec in records:
        seq = str(rec.seq).upper().replace("T", "U")
        try:
            feats = compute_feature_vector(
                seq, feature_set=feature_set, tier2_enabled=tier2_enabled
            )
            x = np.array([[feats.get(col, 0.0) for col in feature_cols]])
            proba = float(model.predict_proba(x)[0, 1])
            results.append((rec.id, proba))
        except Exception:
            continue
    
    return results


def parallel_score_records(records: List, model_data: dict, pcfg: ParallelConfig) -> List[Tuple[str, float]]:
    """Score records in parallel."""
    model = model_data["model"]
    feature_cols = model_data["feature_cols"]
    feature_set = model_data.get("feature_set", "extended")
    tier2_enabled = model_data.get("tier2_enabled", False)
    
    # We need a function that takes a batch of records and returns results
    # Ideally we pickle the model once per worker, but for simplicity here
    # we can process batches where each batch computes features and predicts.
    
    # Actually, predict_proba is fast. Compute_features is slow (RNAfold).
    # So we should parallelize the feature computation logic.
    
    from mirpv_ng.classifier import compute_feature_vector # Need to ensure imports in worker
    
    # Define worker function
    def process_batch(batch_recs):
        results = []
        # Pre-compute features (this is the expensive part with RNAfold)
        # However, compute_feature_vector uses RNAfold internally.
        # If we want to use batch folding, we need to rewrite compute_feature_vector or use
        # a version that accepts pre-folded structures.
        # For now, let's just use process parallelism and let each worker fold its own sequences sequentially.
        # Since we use ProcessPoolExecutor, this works fine.
        
        batch_results = []
        X_batch = []
        valid_indices = []
        
        for i, rec in enumerate(batch_recs):
            seq = str(rec.seq).upper().replace("T", "U")
            try:
                feats = compute_feature_vector(
                    seq, feature_set=feature_set, tier2_enabled=tier2_enabled
                )
                x = [feats.get(col, 0.0) for col in feature_cols]
                X_batch.append(x)
                valid_indices.append(i)
            except Exception:
                continue
        
        if not X_batch:
            return []
            
        # Bulk predict (fast for RF)
        X_arr = np.array(X_batch)
        probs = model.predict_proba(X_arr)[:, 1]
        
        for idx, prob in zip(valid_indices, probs):
            batch_results.append((batch_recs[idx].id, float(prob)))
            
        return batch_results
    
    all_results = []
    if HAS_PARALLEL and pcfg.jobs > 1:
        with get_executor(pcfg) as executor:
            batch_results = executor.map_batches(process_batch, records)
            all_results.extend(batch_results)
    else:
        # Serial fallback
        all_results = score_records(records, model_data)
        
    return all_results


def main():
    ap = argparse.ArgumentParser(description="Generate tail risk report for model")
    ap.add_argument("--model", required=True, help="Model pickle file")
    ap.add_argument("--negatives", required=True, help="Negative FASTA with provenance tags")
    ap.add_argument("--out", required=True, help="Output TSV path")
    ap.add_argument("--samples-per-bucket", type=int, default=300, help="Samples per bucket")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    if HAS_PARALLEL:
        add_parallel_args(ap)
    
    args = ap.parse_args()
    
    print("=" * 60)
    print("TAIL RISK REPORT")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Negatives: {args.negatives}")
    print(f"Samples per bucket: {args.samples_per_bucket}")
    
    # Load model
    print("\n[1] Loading model...")
    model_data = joblib.load(args.model)
    
    ref_thresholds = model_data.get("reference_thresholds", {})
    q95 = ref_thresholds.get("neg_q95", 0.5)
    q99 = ref_thresholds.get("neg_q99", 0.5)
    print(f"    Reference thresholds: q95={q95:.4f}, q99={q99:.4f}")
    
    # Load and stratify negatives
    print("\n[2] Loading and stratifying negatives...")
    all_records = list(SeqIO.parse(args.negatives, "fasta"))
    print(f"    Total records: {len(all_records)}")
    
    stratified = stratified_sample(all_records, args.samples_per_bucket, args.seed)
    for bucket, recs in stratified.items():
        print(f"    {bucket}: {len(recs)} samples")
    
    # Score each bucket
    print("\n[3] Scoring stratified samples...")
    results = []
    
    pcfg = ParallelConfig.from_args(args) if HAS_PARALLEL else None
    print(f"[config] Threads: {pcfg.jobs if pcfg else 1}")
    
    for bucket in sorted(stratified.keys()):
        recs = stratified[bucket]
        if pcfg and pcfg.jobs > 1:
            scored = parallel_score_records(recs, model_data, pcfg)
        else:
            scored = score_records(recs, model_data)
        scores = [s for _, s in scored]
        
        if not scores:
            continue
        
        max_score = max(scores)
        above_q95 = sum(1 for s in scores if s >= q95)
        above_q99 = sum(1 for s in scores if s >= q99)
        
        results.append({
            "bucket": bucket,
            "n_samples": len(scores),
            "max_score": max_score,
            "above_q95": above_q95,
            "above_q99": above_q99,
            "frac_above_q95": above_q95 / len(scores) if scores else 0,
            "frac_above_q99": above_q99 / len(scores) if scores else 0,
            "mean_score": np.mean(scores),
            "median_score": np.median(scores),
        })
        
        print(f"    {bucket}: max={max_score:.4f}, above_q95={above_q95}, above_q99={above_q99}")
    
    # Write TSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w") as f:
        headers = ["bucket", "n_samples", "max_score", "mean_score", "median_score",
                   "above_q95", "above_q99", "frac_above_q95", "frac_above_q99"]
        f.write("\t".join(headers) + "\n")
        for r in results:
            f.write("\t".join([
                r["bucket"],
                str(r["n_samples"]),
                f"{r['max_score']:.4f}",
                f"{r['mean_score']:.4f}",
                f"{r['median_score']:.4f}",
                str(r["above_q95"]),
                str(r["above_q99"]),
                f"{r['frac_above_q95']:.4f}",
                f"{r['frac_above_q99']:.4f}",
            ]) + "\n")
    
    print(f"\n[OUTPUT] Wrote tail risk report to {out_path}")
    
    # Summary with RATE-BASED thresholds (per specification)
    print("\n" + "=" * 60)
    print("TAIL-RISK GOVERNANCE REPORT")
    print("-" * 60)
    
    overall_max = max(r["max_score"] for r in results) if results else 0
    total_samples = sum(r["n_samples"] for r in results)
    total_above_q95 = sum(r["above_q95"] for r in results)
    total_above_q99 = sum(r["above_q99"] for r in results)
    
    # Rate-based metrics (per 1000 samples)
    rate_q95_per_k = (total_above_q95 / total_samples * 1000) if total_samples > 0 else 0
    rate_q99_per_k = (total_above_q99 / total_samples * 1000) if total_samples > 0 else 0
    
    print(f"Overall max negative score: {overall_max:.4f}")
    print(f"Total negatives above q95: {total_above_q95} (rate: {rate_q95_per_k:.1f}/1000)")
    print(f"Total negatives above q99: {total_above_q99} (rate: {rate_q99_per_k:.1f}/1000)")
    
    # Per-bucket rates for diagnosis
    print("\n" + "-" * 40)
    print("Per-bucket tail rates (above q99 per 1000):")
    for r in results:
        bucket_rate = r["frac_above_q99"] * 1000
        flag = "⚠" if bucket_rate > 50 else "✓"
        print(f"  {r['bucket']}: {bucket_rate:.1f}/1000 {flag}")
    
    # Governance flags (rate-based)
    print("\n" + "-" * 40)
    print("GOVERNANCE STATUS:")
    
    has_issues = False
    
    if overall_max > 0.9:
        print("⚠ FLAG: Max score > 0.9 - investigate for potential leakage")
        has_issues = True
    
    if rate_q99_per_k > 50:
        print(f"⚠ FLAG: q99 exceedance rate ({rate_q99_per_k:.1f}/1000) exceeds threshold (50/1000)")
        has_issues = True
    
    # Check if N2 tail exceeds N3 (unexpected - N3 should dominate tail)
    n2_rate = next((r["frac_above_q99"] * 1000 for r in results if r["bucket"] == "N2"), 0)
    n3_rate = next((r["frac_above_q99"] * 1000 for r in results if r["bucket"] == "N3"), 0)
    
    if n2_rate > n3_rate and n2_rate > 10:
        print(f"⚠ FLAG: N2 tail ({n2_rate:.1f}/1000) exceeds N3 ({n3_rate:.1f}/1000) - unexpected")
        print("   Expected: N3 decoys dominate extreme tail, not scan-matched background")
        has_issues = True
    
    if not has_issues:
        print("✓ All governance checks passed")
    
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
