#!/usr/bin/env python3
"""
N3 Overlap Sanity Check: Verify if query sequences are present in negatives.

This script performs two checks:
1. Exact-match search of query sequences inside negatives_v2.fa (especially N3)
2. Check whether the query is present in MirGeneDB positives

Usage:
    python training/check_n3_overlap.py \
        --query data/test/query_hairpins.fa \
        --negatives data/train/negatives_v2.fa \
        --positives data/train/hsa_mirgene_premirna.fa

    # Or check a single sequence:
    python training/check_n3_overlap.py \
        --seq "AUGCAUGCAUGCAUGCAUGC..." \
        --negatives data/train/negatives_v2.fa \
        --positives data/train/hsa_mirgene_premirna.fa

Author: miRPV-NG Team
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from typing import Dict, List, Tuple, Optional
from Bio import SeqIO


def normalize_seq(seq: str) -> str:
    """Normalize sequence for comparison (uppercase, T->U)."""
    return seq.upper().replace("T", "U").strip()


def load_fasta_dict(fasta_path: str) -> Dict[str, Tuple[str, str]]:
    """Load FASTA into dict: normalized_seq -> (original_id, original_seq)."""
    result = {}
    for rec in SeqIO.parse(fasta_path, "fasta"):
        seq = str(rec.seq)
        norm_seq = normalize_seq(seq)
        result[norm_seq] = (rec.id, seq)
    return result


def load_negatives_by_bucket(fasta_path: str) -> Dict[str, Dict[str, Tuple[str, str]]]:
    """
    Load negatives FASTA grouped by bucket (N1, N2, N3).
    Returns: {bucket: {normalized_seq: (id, original_seq)}}
    """
    buckets = {"N1": {}, "N2": {}, "N3": {}, "unknown": {}}
    
    for rec in SeqIO.parse(fasta_path, "fasta"):
        seq = str(rec.seq)
        norm_seq = normalize_seq(seq)
        rec_id = rec.id
        
        # Parse bucket from ID
        if rec_id.startswith("N1|") or rec_id.startswith("N1_"):
            bucket = "N1"
        elif rec_id.startswith("N2|") or rec_id.startswith("N2_"):
            bucket = "N2"
        elif rec_id.startswith("N3|") or rec_id.startswith("N3_"):
            bucket = "N3"
        else:
            bucket = "unknown"
        
        buckets[bucket][norm_seq] = (rec_id, seq)
    
    return buckets


def check_sequence(
    query_seq: str,
    query_id: Optional[str],
    positives: Dict[str, Tuple[str, str]],
    negatives_by_bucket: Dict[str, Dict[str, Tuple[str, str]]],
) -> Dict:
    """
    Check a single sequence against positives and negatives.
    
    Returns dict with:
        - in_positives: bool
        - positive_id: str or None
        - in_negatives: bool
        - negative_bucket: str or None
        - negative_id: str or None
    """
    norm_seq = normalize_seq(query_seq)
    
    result = {
        "query_id": query_id,
        "query_len": len(norm_seq),
        "in_positives": False,
        "positive_id": None,
        "in_negatives": False,
        "negative_bucket": None,
        "negative_id": None,
    }
    
    # Check positives
    if norm_seq in positives:
        result["in_positives"] = True
        result["positive_id"] = positives[norm_seq][0]
    
    # Check negatives by bucket
    for bucket, bucket_seqs in negatives_by_bucket.items():
        if norm_seq in bucket_seqs:
            result["in_negatives"] = True
            result["negative_bucket"] = bucket
            result["negative_id"] = bucket_seqs[norm_seq][0]
            break
    
    return result


def format_result(result: Dict) -> str:
    """Format a check result for display."""
    lines = []
    
    query_id = result.get("query_id", "unknown")
    lines.append(f"\nQuery: {query_id} (len={result['query_len']})")
    lines.append("-" * 60)
    
    if result["in_positives"]:
        lines.append(f"  ✓ FOUND in POSITIVES: {result['positive_id']}")
    else:
        lines.append(f"  ✗ NOT in positives (MirGeneDB)")
    
    if result["in_negatives"]:
        bucket = result["negative_bucket"]
        neg_id = result["negative_id"]
        
        if bucket == "N3":
            lines.append(f"  ⚠ FOUND in NEGATIVES bucket N3: {neg_id}")
            lines.append(f"    → This explains low scores: N3 contains miRBase-not-in-MirGeneDB decoys")
            lines.append(f"    → Model is trained to suppress these sequences")
        else:
            lines.append(f"  ⚠ FOUND in NEGATIVES bucket {bucket}: {neg_id}")
    else:
        lines.append(f"  ✗ NOT in negatives")
    
    # Interpretation
    lines.append("")
    if result["in_positives"] and result["in_negatives"]:
        lines.append("  ❌ ERROR: Sequence in BOTH positives AND negatives - data contamination!")
    elif result["in_positives"]:
        lines.append("  ✓ OK: Sequence is a known MirGeneDB positive")
    elif result["in_negatives"] and result["negative_bucket"] == "N3":
        lines.append("  ℹ NOTE: This is a miRBase-only hairpin labeled as negative (N3)")
        lines.append("         The model is calibrated for MirGeneDB gold-standard specificity.")
        lines.append("         miRBase-only hairpins are intentionally scored low.")
    elif result["in_negatives"]:
        lines.append(f"  ⚠ WARN: Sequence is in negatives bucket {result['negative_bucket']}")
    else:
        lines.append("  ℹ INFO: Novel sequence (not in training data)")
    
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Check if query sequences are in negatives (especially N3) or positives"
    )
    
    # Query input (one of --query or --seq)
    query_group = ap.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query", help="Query FASTA file")
    query_group.add_argument("--seq", help="Single query sequence string")
    
    ap.add_argument("--negatives", required=True, help="Negatives FASTA (e.g., negatives_v2.fa)")
    ap.add_argument("--positives", required=True, help="Positives FASTA (e.g., hsa_mirgene_premirna.fa)")
    ap.add_argument("--out", default=None, help="Output report file")
    
    args = ap.parse_args()
    
    print("=" * 70)
    print("N3 OVERLAP SANITY CHECK")
    print("=" * 70)
    print(f"Negatives: {args.negatives}")
    print(f"Positives: {args.positives}")
    print("=" * 70)
    
    # Load data
    print("\n[1] Loading positives...")
    positives = load_fasta_dict(args.positives)
    print(f"    Loaded {len(positives)} sequences")
    
    print("\n[2] Loading negatives by bucket...")
    negatives_by_bucket = load_negatives_by_bucket(args.negatives)
    for bucket, seqs in negatives_by_bucket.items():
        if seqs:
            print(f"    {bucket}: {len(seqs)} sequences")
    
    # Prepare queries
    queries = []
    if args.query:
        print(f"\n[3] Loading query sequences from {args.query}...")
        for rec in SeqIO.parse(args.query, "fasta"):
            queries.append((rec.id, str(rec.seq)))
        print(f"    Loaded {len(queries)} query sequences")
    else:
        queries.append(("cli_query", args.seq))
    
    # Check each query
    print("\n[4] Checking sequences...")
    results = []
    n3_matches = 0
    positive_matches = 0
    
    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append("N3 OVERLAP CHECK RESULTS")
    output_lines.append("=" * 70)
    
    for query_id, query_seq in queries:
        result = check_sequence(query_seq, query_id, positives, negatives_by_bucket)
        results.append(result)
        
        if result["in_negatives"] and result["negative_bucket"] == "N3":
            n3_matches += 1
        if result["in_positives"]:
            positive_matches += 1
        
        formatted = format_result(result)
        output_lines.append(formatted)
        print(formatted)
    
    # Summary
    summary_lines = []
    summary_lines.append("\n" + "=" * 70)
    summary_lines.append("SUMMARY")
    summary_lines.append("=" * 70)
    summary_lines.append(f"Total queries: {len(queries)}")
    summary_lines.append(f"Found in positives (MirGeneDB): {positive_matches}")
    summary_lines.append(f"Found in N3 (miRBase-only decoys): {n3_matches}")
    
    if n3_matches > 0:
        summary_lines.append("")
        summary_lines.append("⚠ IMPORTANT:")
        summary_lines.append("  The model is calibrated for MirGeneDB gold-standard specificity.")
        summary_lines.append("  miRBase-only hairpins (in N3) are intentionally scored low.")
        summary_lines.append("")
        summary_lines.append("  To train a miRBase-inclusive model, use one of:")
        summary_lines.append("    --n3-mode exclude-mirbase  (remove miRBase-only from N3)")
        summary_lines.append("    --n3-mode downweight       (include N3 with reduced weight)")
        summary_lines.append("    --n3-mode unlabeled        (exclude N3 from supervised loss)")
    
    summary_lines.append("=" * 70)
    
    for line in summary_lines:
        print(line)
        output_lines.append(line)
    
    # Save output
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("\n".join(output_lines))
        print(f"\n[OUTPUT] Wrote report to {out_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
