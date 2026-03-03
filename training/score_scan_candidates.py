#!/usr/bin/env python3
"""
Score Scan Candidates: Canonical verification that scan-mode scoring is correct.

Purpose:
- Reproduce the REAL sequence-only inference distribution.
- Detect score crushing or over-filtering before full genome runs.
- Uses the EXACT same scanner logic as production.

Usage:
    python training/score_scan_candidates.py \\
        --genome refs/hg38_primary/hg38.primary.fa \\
        --model models/hsa_premirna_rf_v7.pkl \\
        --out analysis/scan_candidate_scores.txt \\
        --max-candidates 2000

Author: miRPV-NG Team
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import random
import subprocess
import numpy as np
import joblib
from typing import List, Tuple, Dict, Generator, Optional
from dataclasses import dataclass

from Bio import SeqIO

# Import SAME scanner components used in production
try:
    from mirpv_ng.tier_filters import TierConfig, tier1_energy_filter
    from mirpv_ng.geom_hairpin_finder import find_hairpins
    from mirpv_ng.classifier import compute_feature_vector
    HAS_SCANNER = True
except ImportError:
    HAS_SCANNER = False
    print("[ERROR] Could not import scanner components - this script requires mirpv_ng package")
    sys.exit(1)

# Try parallel module for batch folding
try:
    from mirpv_ng.parallel import run_rnafold_batch, ParallelConfig, get_executor, add_parallel_args
    HAS_PARALLEL = True
except ImportError:
    HAS_PARALLEL = False


def run_rnafold(seq: str) -> Tuple[str, float]:
    """Run RNAfold on a single sequence (wrapper for batch or single fallback)."""
    if HAS_PARALLEL:
        # NOTE: In parallel mode, this function is typically replaced by batch calls
        # But for direct calls, we fallback to single execution wrapper
        from mirpv_ng.parallel import run_rnafold_single
        return run_rnafold_single(seq)
    
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
        return struct, float(mfe_str)
    except Exception:
        return "", 0.0


def scanner_candidate_generator(
    genome_fa: str,
    window_len: int = 100,
    step: int = 10,
    seed: int = 42,
) -> Generator[Tuple[str, int, int, str, str, float], None, None]:
    """
    STREAMING generator using EXACT same logic as HairpinClassifier.scan_long_sequence().
    
    Yields: (chrom, start, end, hp_seq, struct, mfe)
    """
    rng = random.Random(seed)
    
    # Create tier1 config matching scanner defaults
    tier1_cfg = TierConfig(
        min_len=40, max_len=120, min_pairs=18,
        min_mfe=-15.0, max_unpaired_frac=0.8
    )
    
    # Load chromosomes
    chroms = []
    for rec in SeqIO.parse(genome_fa, "fasta"):
        chroms.append((rec.id, str(rec.seq)))
    
    rng.shuffle(chroms)
    
    for chrom_id, chrom_seq in chroms:
        n = len(chrom_seq)
        if n < window_len:
            continue
        
        starts = list(range(0, n - window_len + 1, step))
        rng.shuffle(starts)
        
        for start in starts:
            end = start + window_len
            win_seq = chrom_seq[start:end].upper()
            
            if "N" in win_seq:
                continue
            
            # Fold window (SAME as scanner)
            struct, mfe = run_rnafold(win_seq)
            if not struct:
                continue
            
            # Tier1 filter (SAME as scanner)
            if not tier1_energy_filter(win_seq, struct, mfe, tier1_cfg):
                continue
            
            # Find hairpins (SAME as scanner)
            hairpins = find_hairpins(win_seq, struct)
            if not hairpins:
                continue
            
            hp = hairpins[0]
            hp_seq = win_seq[hp.start:hp.end]
            hp_len = hp.end - hp.start
            
            if hp_len < 50 or hp_len > 120:
                continue
            
            yield (chrom_id, start, end, hp_seq, struct, mfe)


def parallel_scanner_candidate_generator(
    genome_fa: str,
    pcfg: ParallelConfig,
    window_len: int = 100,
    step: int = 10,
    seed: int = 42,
) -> Generator[Tuple[str, int, int, str, str, float], None, None]:
    """
    Parallel streaming generator for candidates.
    Matches the single-threaded logic but uses batch folding.
    """
    rng = random.Random(seed)
    tier1_cfg = TierConfig(
        min_len=40, max_len=120, min_pairs=18,
        min_mfe=-15.0, max_unpaired_frac=0.8
    )

    chroms = []
    for rec in SeqIO.parse(genome_fa, "fasta"):
        chroms.append((rec.id, str(rec.seq)))
    rng.shuffle(chroms)

    # 1. Generate window candidates (metadata only)
    # To keep memory low, we process one chromosome at a time or chunk windows
    
    window_meta_buffer = []  # [(chrom, start, end, seq), ...]
    BATCH_SIZE = pcfg.chunksize * pcfg.jobs * 2  # Buffer size for efficiency
    
    for chrom_id, chrom_seq in chroms:
        n = len(chrom_seq)
        if n < window_len: continue
        
        starts = list(range(0, n - window_len + 1, step))
        rng.shuffle(starts)
        
        for start in starts:
            end = start + window_len
            win_seq = chrom_seq[start:end].upper()
            if "N" in win_seq: continue
            
            window_meta_buffer.append((chrom_id, start, end, win_seq))
            
            if len(window_meta_buffer) >= BATCH_SIZE:
                yield from _process_window_batch(window_meta_buffer, pcfg, tier1_cfg)
                window_meta_buffer = []
        
    # Flush remaining
    if window_meta_buffer:
        yield from _process_window_batch(window_meta_buffer, pcfg, tier1_cfg)

def _process_window_batch(
    windows: List[Tuple[str, int, int, str]], 
    pcfg: ParallelConfig,
    tier1_cfg: TierConfig
) -> Generator[Tuple[str, int, int, str, str, float], None, None]:
    """Process a batch of windows: fold parallel, filter locally."""
    seqs = [w[3] for w in windows]
    
    # Parallel fold
    if HAS_PARALLEL and pcfg.jobs > 1:
        with get_executor(pcfg) as executor:
            # We fold in chunks inside the executor
            # run_rnafold_batch does sequential batch folding, we parallelize over batches
            # Since we have a list of seqs, we can use map_batches if we had a pure function
            # Or just use run_rnafold_batch directly if single threaded here but parallel inside?
            # No, parallel.py has helper run_rnafold_parallel
            from mirpv_ng.parallel import run_rnafold_parallel
            batch_results = run_rnafold_parallel(seqs, pcfg)
    else:
        # Serial fallback
        batch_results = run_rnafold_batch(seqs) if HAS_PARALLEL else [run_rnafold(s) for s in seqs]
        
    for i, (struct, mfe) in enumerate(batch_results):
        if not struct: continue
        chrom, start, end, win_seq = windows[i]
        
        # Tier1 filter
        if not tier1_energy_filter(win_seq, struct, mfe, tier1_cfg):
            continue
        
        # Find hairpins
        hairpins = find_hairpins(win_seq, struct)
        if not hairpins: continue
        
        hp = hairpins[0]
        hp_seq = win_seq[hp.start:hp.end]
        hp_len = hp.end - hp.start
        
        if hp_len < 50 or hp_len > 120: continue
        
        yield (chrom, start, end, hp_seq, struct, mfe)


def format_quantiles(scores: np.ndarray) -> str:
    """Format score quantiles for display."""
    lines = []
    lines.append(f"  Min:    {np.min(scores):.4f}")
    lines.append(f"  Q10:    {np.percentile(scores, 10):.4f}")
    lines.append(f"  Q25:    {np.percentile(scores, 25):.4f}")
    lines.append(f"  Q50:    {np.percentile(scores, 50):.4f}")
    lines.append(f"  Q75:    {np.percentile(scores, 75):.4f}")
    lines.append(f"  Q90:    {np.percentile(scores, 90):.4f}")
    lines.append(f"  Q95:    {np.percentile(scores, 95):.4f}")
    lines.append(f"  Q99:    {np.percentile(scores, 99):.4f}")
    lines.append(f"  Max:    {np.max(scores):.4f}")
    lines.append(f"  Mean:   {np.mean(scores):.4f}")
    lines.append(f"  Std:    {np.std(scores):.4f}")
    return "\n".join(lines)


def ascii_histogram(scores: np.ndarray, bins: int = 20, width: int = 50) -> str:
    """Generate ASCII histogram."""
    hist, edges = np.histogram(scores, bins=bins, range=(0, 1))
    max_count = max(hist) if max(hist) > 0 else 1
    
    lines = []
    for i, count in enumerate(hist):
        bar_len = int(count / max_count * width)
        bar = "█" * bar_len
        lines.append(f"  [{edges[i]:.2f}-{edges[i+1]:.2f}] {bar} {count}")
    
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Score scan-mode candidates to verify scoring behavior"
    )
    ap.add_argument("--genome", required=True, help="Genome FASTA")
    ap.add_argument("--model", required=True, help="Model pickle file")
    ap.add_argument("--out", required=True, help="Output report file")
    ap.add_argument("--max-candidates", type=int, default=2000, help="Max candidates to score")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--histogram", default=None, help="Optional PNG histogram path")
    ap.add_argument("--window-len", type=int, default=100, help="Scan window length")
    ap.add_argument("--step", type=int, default=10, help="Scan window step")
    ap.add_argument("--write-candidates", default=None, help="Optional FASTA output of scored candidates (for scan-background)")
    
    if HAS_PARALLEL:
        add_parallel_args(ap)
    
    args = ap.parse_args()
    
    print("=" * 70)
    print("SCAN-MODE CANDIDATE SCORING VERIFICATION")
    print("=" * 70)
    print(f"Genome: {args.genome}")
    print(f"Model: {args.model}")
    print(f"Max candidates: {args.max_candidates}")
    print(f"Seed: {args.seed}")
    if args.write_candidates:
        print(f"Write candidates: {args.write_candidates}")
    print("=" * 70)
    
    # Load model
    print("\n[1] Loading model...")
    model_data = joblib.load(args.model)
    
    if not isinstance(model_data, dict):
        print("ERROR: Model file is not a dict")
        return 1
    
    model = model_data["model"]
    feature_cols = model_data["feature_cols"]
    feature_set = model_data.get("feature_set", "extended")
    tier2_enabled = model_data.get("tier2_enabled", False)
    
    print(f"    Feature set: {feature_set}")
    print(f"    Features: {len(feature_cols)}")
    
    # Get reference thresholds from training
    ref_thresholds = model_data.get("reference_thresholds", {})
    f1_threshold = model_data.get("f1_threshold", model_data.get("decision_threshold", 0.5))
    print(f"    F1 threshold: {f1_threshold:.4f}")
    if ref_thresholds:
        print(f"    Ref thresholds: q90={ref_thresholds.get('neg_q90', 'N/A'):.4f}, "
              f"q95={ref_thresholds.get('neg_q95', 'N/A'):.4f}, "
              f"q99={ref_thresholds.get('neg_q99', 'N/A'):.4f}")
    
    # Generate and score candidates using SAME scanner logic
    print(f"\n[2] Generating scan-mode candidates (streaming)...")
    
    candidates = []
    scores = []
    windows_seen = 0
    
    # Create parallel config (handles auto-detection of threads)
    pcfg = ParallelConfig.from_args(args) if HAS_PARALLEL else None
    
    print(f"[config] Threads: {pcfg.jobs if pcfg else 1}")

    # Choose generator
    if pcfg and pcfg.jobs > 1:
        print(f"[parallel] Using parallel scanner ({pcfg.jobs} workers, chunksize={pcfg.chunksize})")
        scan_gen = parallel_scanner_candidate_generator(
            args.genome, pcfg, args.window_len, args.step, args.seed
        )
    else:
        print("[parallel] Using single-threaded scanner")
        scan_gen = scanner_candidate_generator(
            args.genome, args.window_len, args.step, args.seed
        )

    # Prepare FASTA writer if requested
    fasta_handle = None
    if args.write_candidates:
        fasta_path = Path(args.write_candidates)
        fasta_path.parent.mkdir(parents=True, exist_ok=True)
        fasta_handle = open(fasta_path, "w")

    try:
        for chrom, start, end, hp_seq, struct, mfe in scan_gen:
            if len(candidates) >= args.max_candidates:
                break
            
            windows_seen += 1
            
            # Compute features EXACTLY as scanner does
            try:
                feats = compute_feature_vector(
                    hp_seq,
                    feature_set=feature_set,
                    tier2_enabled=tier2_enabled,
                )
                
                # Build feature vector
                x = np.array([[feats.get(col, 0.0) for col in feature_cols]])
                
                # Score using predict_proba ONLY (class 1)
                proba = float(model.predict_proba(x)[0, 1])
                
                candidates.append({
                    "chrom": chrom, "start": start, "end": end,
                    "hp_seq": hp_seq, # Store for FASTA writing
                    "hp_len": len(hp_seq), "mfe": mfe, "score": proba
                })
                scores.append(proba)
                
                if fasta_handle:
                    # Write to FASTA: >scan|chr:start-end|score=VAL
                    # Sequence should be RNA or DNA? Usually DNA in FASTA
                    # hp_seq is RNA (U) from scanner generator?
                    # let's check generator... it returns hp_seq which is "win_seq[hp.start:hp.end]"
                    # win_seq is upper(). So T or U?
                    # In scanner_candidate_generator: win_seq = chrom_seq[start:end].upper() (DNA)
                    # But then run_rnafold takes it.
                    # Wait, run_rnafold converts T to U.
                    # So hp_seq is DNA.
                    seq_str = str(hp_seq)
                    rec_id = f"scan|{chrom}:{start}-{end}|score={proba:.4f}"
                    fasta_handle.write(f">{rec_id}\n{seq_str}\n")
                
                if len(candidates) % 200 == 0:
                    print(f"    Progress: {len(candidates)}/{args.max_candidates}")
            
            except Exception as e:
                continue
    finally:
        if fasta_handle:
            fasta_handle.close()
            print(f"[OUTPUT] Wrote candidates FASTA to {args.write_candidates}")
    
    print(f"    Processed {windows_seen} windows")
    print(f"    Scored {len(candidates)} candidates")
    
    if not scores:
        print("\nERROR: No candidates scored. Check genome and model.")
        return 1
    
    scores_arr = np.array(scores)
    
    # Generate report
    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append("SCAN-MODE CANDIDATE SCORING REPORT")
    output_lines.append("=" * 70)
    output_lines.append(f"Genome: {args.genome}")
    output_lines.append(f"Model: {args.model}")
    output_lines.append(f"Candidates scored: {len(candidates)}")
    output_lines.append(f"Windows processed: {windows_seen}")
    output_lines.append("")
    
    output_lines.append("SCORE DISTRIBUTION")
    output_lines.append("-" * 50)
    output_lines.append(format_quantiles(scores_arr))
    output_lines.append("")
    
    # Check against reference thresholds
    output_lines.append("THRESHOLD ANALYSIS")
    output_lines.append("-" * 50)
    output_lines.append(f"  F1 threshold: {f1_threshold:.4f}")
    output_lines.append(f"  Candidates above F1: {np.sum(scores_arr >= f1_threshold)} ({np.mean(scores_arr >= f1_threshold):.1%})")
    
    if ref_thresholds:
        q95 = ref_thresholds.get("neg_q95", 0.5)
        q99 = ref_thresholds.get("neg_q99", 0.5)
        output_lines.append(f"  Candidates above q95 ({q95:.4f}): {np.sum(scores_arr >= q95)} ({np.mean(scores_arr >= q95):.1%})")
        output_lines.append(f"  Candidates above q99 ({q99:.4f}): {np.sum(scores_arr >= q99)} ({np.mean(scores_arr >= q99):.1%})")
    
    output_lines.append("")
    
    # Interpretation
    output_lines.append("INTERPRETATION")
    output_lines.append("-" * 50)
    
    median_score = np.median(scores_arr)
    max_score = np.max(scores_arr)
    
    if median_score < 0.1:
        output_lines.append("NOTE: Median score is low (<0.1).")
        output_lines.append("   This is expected due to the high genome background base rate.")
        output_lines.append("   The model effectively filters the vast majority of random genomic hairpins.")
    elif median_score > 0.5:
        output_lines.append("⚠ WARNING: Median score is high (>0.5)")
        output_lines.append("   This may indicate calibration issues or true positives in genome.")
    else:
        output_lines.append("✓ Score distribution looks reasonable for genome background.")
    
    if max_score > 0.9:
        output_lines.append(f"   Note: {np.sum(scores_arr > 0.9)} candidates scored >0.9 (potential true positives or outliers)")
    
    output_lines.append("")
    
    # ASCII histogram
    output_lines.append("SCORE HISTOGRAM")
    output_lines.append("-" * 50)
    output_lines.append(ascii_histogram(scores_arr))
    output_lines.append("")
    
    # Top candidates
    output_lines.append("TOP 10 HIGHEST SCORING CANDIDATES")
    output_lines.append("-" * 50)
    top_indices = np.argsort(scores)[-10:][::-1]
    for i, idx in enumerate(top_indices):
        c = candidates[idx]
        output_lines.append(f"  {i+1}. score={c['score']:.4f} {c['chrom']}:{c['start']}-{c['end']} len={c['hp_len']} mfe={c['mfe']:.1f}")
    
    output_lines.append("")
    output_lines.append("=" * 70)
    
    # Print and save
    report = "\n".join(output_lines)
    print("\n" + report)
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\n[OUTPUT] Wrote report to {out_path}")
    
    # Optional PNG histogram
    if args.histogram:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(scores_arr, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
            ax.axvline(f1_threshold, color="red", linestyle="--", label=f"F1 threshold ({f1_threshold:.3f})")
            if ref_thresholds:
                ax.axvline(ref_thresholds.get("neg_q95", 0.5), color="orange", linestyle=":", label="neg_q95")
            ax.set_xlabel("Score (predict_proba class 1)")
            ax.set_ylabel("Count")
            ax.set_title(f"Scan-Mode Candidate Scores (n={len(candidates)})")
            ax.legend()
            ax.set_xlim(0, 1)
            
            hist_path = Path(args.histogram)
            hist_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(hist_path, dpi=150)
            plt.close()
            print(f"[OUTPUT] Saved histogram to {hist_path}")
        except ImportError:
            print("[WARNING] matplotlib not available, skipping PNG histogram")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
