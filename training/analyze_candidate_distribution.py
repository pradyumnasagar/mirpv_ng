#!/usr/bin/env python3
"""
Analyze Candidate Distribution: Compare sequence-only scanning candidates 
to training positives/negatives.

Uses the SAME candidate generation logic as the scanner to detect distribution mismatch.

Usage:
    python training/analyze_candidate_distribution.py \
        --genome refs/hg38/chr22.fa \
        --positives data/train/hsa_mirgene_premirna.fa \
        --negatives data/train/hsa_FINAL_negatives.fa \
        --out analysis/candidate_distribution.txt

Author: miRPV-NG Team
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import subprocess
import random
import numpy as np
from typing import List, Tuple, Dict, Optional

from Bio import SeqIO

# Import scanner components to replicate exactly
try:
    from mirpv_ng.tier_filters import TierConfig, tier1_energy_filter
    from mirpv_ng.geom_hairpin_finder import find_hairpins
    HAS_SCANNER = True
except ImportError:
    HAS_SCANNER = False
    print("[WARNING] Could not import scanner components")


def get_clean_seq(seq) -> str:
    """Normalize sequence to DNA alphabet."""
    return str(seq).upper().replace("U", "T")


def compute_gc(seq: str) -> float:
    """Compute GC fraction."""
    seq = seq.upper()
    gc = sum(1 for c in seq if c in "GC")
    return gc / max(len(seq), 1)


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


# Try parallel module
try:
    from mirpv_ng.parallel import run_rnafold_batch, ParallelConfig, add_parallel_args, get_executor
    HAS_PARALLEL = True
except ImportError:
    HAS_PARALLEL = False


def generate_scanner_candidates(
    genome_fa: str,
    window_len: int = 100,
    step: int = 10,
    max_candidates: int = 500,
    seed: int = 42,
) -> List[str]:
    """
    Generate candidates using SAME logic as HairpinClassifier.scan_long_sequence().
    
    This ensures distribution analysis matches actual scanner behavior.
    """
    print(f"[gen] Generating scanner candidates from {genome_fa}...")
    print(f"[gen] Using scanner params: window_len={window_len}, step={step}")
    
    rng = random.Random(seed)
    
    # Create tier1 config matching scanner defaults
    tier1_cfg = TierConfig(
        min_len=40, max_len=120, min_pairs=18,
        min_mfe=-15.0, max_unpaired_frac=0.8
    ) if HAS_SCANNER else None
    
    candidates = []
    windows_seen = 0
    windows_passed = 0
    
    for rec in SeqIO.parse(genome_fa, "fasta"):
        seq = str(rec.seq).upper()
        n = len(seq)
        
        if n < window_len:
            continue
        
        # Generate windows like scanner
        starts = list(range(0, n - window_len + 1, step))
        rng.shuffle(starts)
        
        for start in starts:
            if len(candidates) >= max_candidates:
                break
            
            end = start + window_len
            win_seq = seq[start:end]
            
            if "N" in win_seq:
                continue
            
            windows_seen += 1
            
            # Fold (like scanner)
            struct, mfe = run_rnafold(win_seq)
            if not struct:
                continue
            
            # Tier1 filter (like scanner)
            if HAS_SCANNER and tier1_cfg:
                if not tier1_energy_filter(win_seq, struct, mfe, tier1_cfg):
                    continue
            else:
                pairs = struct.count("(")
                unpaired_frac = struct.count(".") / len(struct)
                if pairs < 18 or mfe > -15.0 or unpaired_frac > 0.8:
                    continue
            
            windows_passed += 1
            
            # Find hairpins (like scanner)
            if HAS_SCANNER:
                hairpins = find_hairpins(win_seq, struct)
                if not hairpins:
                    continue
                hp = hairpins[0]
                hp_seq = win_seq[hp.start:hp.end]
                if 50 <= len(hp_seq) <= 120:
                    candidates.append(hp_seq)
            else:
                if 50 <= len(win_seq) <= 120:
                    candidates.append(win_seq)
        
        if len(candidates) >= max_candidates:
            break
    
    print(f"[gen] Processed {windows_seen} windows, {windows_passed} passed tier1")
    print(f"[gen] Generated {len(candidates)} candidates")
    return candidates


def parallel_generate_scanner_candidates(
    genome_fa: str,
    pcfg: ParallelConfig,
    window_len: int = 100,
    step: int = 10,
    max_candidates: int = 500,
    seed: int = 42,
) -> List[str]:
    """Parallel version of generate_scanner_candidates using batch folding."""
    print(f"[gen] Generating scanner candidates (parallel, jobs={pcfg.jobs})...")
    rng = random.Random(seed)
    
    tier1_cfg = TierConfig(
        min_len=40, max_len=120, min_pairs=18,
        min_mfe=-15.0, max_unpaired_frac=0.8
    ) if HAS_SCANNER else None
    
    candidates = []
    
    # 1. Collect window metadata
    window_meta_buffer = []  # (seq)
    BATCH_SIZE = pcfg.chunksize * pcfg.jobs * 2
    
    for rec in SeqIO.parse(genome_fa, "fasta"):
        seq = str(rec.seq).upper()
        n = len(seq)
        if n < window_len: continue
        
        starts = list(range(0, n - window_len + 1, step))
        rng.shuffle(starts)
        
        for start in starts:
            if len(candidates) >= max_candidates: break
            
            end = start + window_len
            win_seq = seq[start:end]
            if "N" in win_seq: continue
            
            window_meta_buffer.append(win_seq)
            
            if len(window_meta_buffer) >= BATCH_SIZE:
                 candidates.extend(_process_candidate_batch(window_meta_buffer, pcfg, tier1_cfg))
                 window_meta_buffer = []
                 if len(candidates) >= max_candidates: break
        
        if len(candidates) >= max_candidates: break
            
    if window_meta_buffer and len(candidates) < max_candidates:
        candidates.extend(_process_candidate_batch(window_meta_buffer, pcfg, tier1_cfg))
        
    return candidates[:max_candidates]

def _process_candidate_batch(seqs: List[str], pcfg: ParallelConfig, tier1_cfg: TierConfig) -> List[str]:
    """Process batch for candidate generation."""
    found = []
    
    # Parallel fold
    if HAS_PARALLEL and pcfg.jobs > 1:
        from mirpv_ng.parallel import run_rnafold_parallel
        results = run_rnafold_parallel(seqs, pcfg)
    else:
        results = run_rnafold_batch(seqs) if HAS_PARALLEL else [run_rnafold(s) for s in seqs]
        
    for i, (struct, mfe) in enumerate(results):
        if not struct: continue
        win_seq = seqs[i]
        
        # Tier1 filter
        if HAS_SCANNER and tier1_cfg:
            if not tier1_energy_filter(win_seq, struct, mfe, tier1_cfg):
                continue
        else:
            pairs = struct.count("(")
            unpaired_frac = struct.count(".") / len(struct)
            if pairs < 18 or mfe > -15.0 or unpaired_frac > 0.8:
                continue
                
        # Find hairpins
        if HAS_SCANNER:
            hairpins = find_hairpins(win_seq, struct)
            if not hairpins: continue
            hp = hairpins[0]
            hp_seq = win_seq[hp.start:hp.end]
            if 50 <= len(hp_seq) <= 120:
                found.append(hp_seq)
        else:
            if 50 <= len(win_seq) <= 120:
                found.append(win_seq)
                
    return found


def compute_stats(seqs: List[str], name: str, sample_fold: int = 200) -> Dict:
    """Compute distribution statistics for a set of sequences."""
    if not seqs:
        return {"name": name, "count": 0}
    
    lengths = [len(s) for s in seqs]
    gcs = [compute_gc(s) for s in seqs]
    
    # Sample for folding (expensive)
    rng = random.Random(42)
    fold_sample = rng.sample(seqs, min(sample_fold, len(seqs)))
    
    mfes = []
    pairs = []
    for seq in fold_sample:
        struct, mfe = run_rnafold(seq)
        if struct:
            mfes.append(mfe)
            pairs.append(struct.count("("))
    
    stats = {
        "name": name,
        "count": len(seqs),
        "length_mean": np.mean(lengths),
        "length_std": np.std(lengths),
        "length_min": min(lengths),
        "length_max": max(lengths),
        "length_p10": np.percentile(lengths, 10),
        "length_p50": np.percentile(lengths, 50),
        "length_p90": np.percentile(lengths, 90),
        "gc_mean": np.mean(gcs),
        "gc_std": np.std(gcs),
    }
    
    if mfes:
        stats["mfe_mean"] = np.mean(mfes)
        stats["mfe_std"] = np.std(mfes)
        stats["mfe_p50"] = np.percentile(mfes, 50)
        stats["pairs_mean"] = np.mean(pairs)
        stats["pairs_std"] = np.std(pairs)
    
    return stats


def compare_distributions(stats1: Dict, stats2: Dict) -> List[str]:
    """Compare two distributions and flag significant differences."""
    flags = []
    name1, name2 = stats1["name"], stats2["name"]
    
    # Length comparison
    len_diff = abs(stats1.get("length_mean", 0) - stats2.get("length_mean", 0))
    if len_diff > 10:
        flags.append(f"⚠ Length differs by {len_diff:.1f}nt ({name1} vs {name2})")
    
    # GC comparison
    gc_diff = abs(stats1.get("gc_mean", 0) - stats2.get("gc_mean", 0))
    if gc_diff > 0.05:
        flags.append(f"⚠ GC differs by {gc_diff:.3f} ({name1} vs {name2})")
    
    # MFE comparison
    if "mfe_mean" in stats1 and "mfe_mean" in stats2:
        mfe_diff = abs(stats1["mfe_mean"] - stats2["mfe_mean"])
        if mfe_diff > 5.0:
            flags.append(f"⚠ MFE differs by {mfe_diff:.1f} kcal/mol ({name1} vs {name2})")
    
    # Pairs comparison  
    if "pairs_mean" in stats1 and "pairs_mean" in stats2:
        pairs_diff = abs(stats1["pairs_mean"] - stats2["pairs_mean"])
        if pairs_diff > 5:
            flags.append(f"⚠ Pairs differs by {pairs_diff:.1f} ({name1} vs {name2})")
    
    return flags


def format_stats(stats: Dict) -> str:
    """Format statistics for display."""
    if stats["count"] == 0:
        return f"{stats['name']}: No data"
    
    lines = [f"\n{stats['name']} (n={stats['count']})"]
    lines.append("-" * 50)
    lines.append(f"  Length: {stats['length_mean']:.1f} ± {stats['length_std']:.1f} "
                 f"[{stats['length_min']}, {stats['length_max']}]")
    lines.append(f"  GC: {stats['gc_mean']:.3f} ± {stats['gc_std']:.3f}")
    
    if "mfe_mean" in stats:
        lines.append(f"  MFE: {stats['mfe_mean']:.1f} ± {stats['mfe_std']:.1f} kcal/mol")
        lines.append(f"  Base pairs: {stats['pairs_mean']:.1f} ± {stats['pairs_std']:.1f}")
    
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Analyze candidate distribution vs training data (scanner-matched)"
    )
    ap.add_argument("--genome", required=True, help="Genome FASTA for scanning simulation")
    ap.add_argument("--positives", required=True, help="Training positives FASTA")
    ap.add_argument("--negatives", required=True, help="Training negatives FASTA")
    ap.add_argument("--out", default=None, help="Output report file")
    ap.add_argument("--max-candidates", type=int, default=500, help="Max scanning candidates")
    ap.add_argument("--fold-sample", type=int, default=200, help="Sample size for folding")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    if HAS_PARALLEL:
        add_parallel_args(ap)
    
    args = ap.parse_args()
    
    print("=" * 70)
    print("CANDIDATE DISTRIBUTION ANALYSIS (Scanner-Matched)")
    print("=" * 70)
    
    # Load positives
    print("\n[1] Loading positives...")
    positives = [get_clean_seq(r.seq) for r in SeqIO.parse(args.positives, "fasta")]
    print(f"    Loaded {len(positives)} positive sequences")
    
    # Load negatives
    print("\n[2] Loading negatives...")
    negatives = [get_clean_seq(r.seq) for r in SeqIO.parse(args.negatives, "fasta")]
    print(f"    Loaded {len(negatives)} negative sequences")
    
    # Generate scanner candidates using SAME logic as classifier
    print("\n[3] Generating scanner candidates (same logic as HairpinClassifier)...")
    
    pcfg = ParallelConfig.from_args(args) if HAS_PARALLEL else None
    print(f"[config] Threads: {pcfg.jobs if pcfg else 1}")
    
    if pcfg and pcfg.jobs > 1:
        candidates = parallel_generate_scanner_candidates(
            args.genome, pcfg, 
            max_candidates=args.max_candidates, seed=args.seed
        )
    else:
        candidates = generate_scanner_candidates(
            args.genome,
            max_candidates=args.max_candidates,
            seed=args.seed,
        )
    
    # Compute statistics
    print("\n[4] Computing statistics...")
    
    pos_stats = compute_stats(positives, "POSITIVES", args.fold_sample)
    neg_stats = compute_stats(negatives, "NEGATIVES", args.fold_sample)
    cand_stats = compute_stats(candidates, "SCANNER_CANDIDATES", args.fold_sample)
    
    # Format output
    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append("DISTRIBUTION ANALYSIS REPORT")
    output_lines.append("=" * 70)
    
    output_lines.append(format_stats(pos_stats))
    output_lines.append(format_stats(neg_stats))
    output_lines.append(format_stats(cand_stats))
    
    # Compare distributions
    output_lines.append("\n" + "=" * 70)
    output_lines.append("MISMATCH FLAGS")
    output_lines.append("=" * 70)
    
    all_flags = []
    all_flags.extend(compare_distributions(cand_stats, neg_stats))
    all_flags.extend(compare_distributions(cand_stats, pos_stats))
    
    if all_flags:
        for flag in all_flags:
            output_lines.append(flag)
    else:
        output_lines.append("✓ No significant distribution mismatches detected")
    
    # Key insight
    output_lines.append("\n" + "=" * 70)
    output_lines.append("KEY INSIGHT")
    output_lines.append("=" * 70)
    
    cand_neg_mismatch = False
    if "length_mean" in cand_stats and "length_mean" in neg_stats:
        if abs(cand_stats["length_mean"] - neg_stats["length_mean"]) > 10:
            cand_neg_mismatch = True
    
    if cand_neg_mismatch:
        output_lines.append("""
⚠ SCANNER-NEGATIVE MISMATCH DETECTED

The scanner generates candidates that differ significantly from training negatives.
This causes distribution shift: the model sees different patterns at inference time
than it saw during training, leading to crushed or overconfident scores.

SOLUTION: Use scanner-matched negatives (N2 bucket in build_negatives_v2.py):
  python training/build_negatives_v2.py --genome <genome.fa> ...

This ensures training negatives match the distribution of scanning candidates.
""")
    else:
        output_lines.append("""
✓ Scanner candidates and negatives appear well-matched.

If scores are still problematic, check:
  1. Calibration pipeline (run verify_calibration.py)
  2. Feature computation consistency
  3. Model threshold appropriateness for your task
""")
    
    output_lines.append("=" * 70)
    
    # Print and save
    report = "\n".join(output_lines)
    print(report)
    
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(report)
        print(f"\n[OUTPUT] Wrote report to {out_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
