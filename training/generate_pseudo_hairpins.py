#!/usr/bin/env python3
"""
Generate pseudo pre-miRNA hairpins from a genome FASTA.

- Slides fixed-length windows across the genome
- Folds each window with RNAfold (in parallel)
- Applies simple hairpin-like filters (length, MFE, base pairs, loop/stem constraints)
- Optionally excludes windows overlapping known miRNA loci (BED)
"""
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import subprocess
import multiprocessing as mp
import random

from Bio import SeqIO

try:
    from intervaltree import IntervalTree
    HAS_INTERVALTREE = True
except ImportError:
    HAS_INTERVALTREE = False


# ---------- helper functions for structure ----------

def run_rnafold(seq: str) -> Tuple[Optional[str], float]:
    """Run RNAfold and return (dot-bracket structure, MFE)."""
    p = subprocess.Popen(
        ["RNAfold", "--noPS"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, _ = p.communicate(seq + "\n")
    lines = out.strip().splitlines()
    if len(lines) < 2:
        return None, 0.0
    struct_line = lines[1]
    struct = struct_line.split()[0]
    try:
        mfe_str = struct_line.split("(")[-1].strip(" )")
        mfe = float(mfe_str)
    except Exception:
        mfe = 0.0
    return struct, mfe


def runs(s: str, char: str) -> List[int]:
    """Return run lengths of char in string s."""
    out = []
    cur = 0
    for c in s:
        if c == char:
            cur += 1
        else:
            if cur > 0:
                out.append(cur)
                cur = 0
    if cur > 0:
        out.append(cur)
    return out


def hairpin_passes_filters(
    seq: str,
    struct: str,
    mfe: float,
    min_pairs: int = 18,       # Increased from 16
    min_mfe: float = -18.0,    # Adjusted threshold
    max_len: int = 120,
    max_loops: int = 3,        # STRICTER: Limits complex branching
) -> bool:
    """
    Stricter filters to generate 'smooth' high-quality pseudo-hairpins.
    Rejects 'messy' structures to force the model to learn sequence motifs.
    """
    length = len(seq)
    if length > max_len:
        return False

    # 1. Energy threshold
    if mfe > min_mfe:
        return False

    # 2. Pair counting
    pairs = struct.count("(")
    if pairs < min_pairs:
        return False

    # 3. Loop counting (Structure complexity)
    loop_runs = runs(struct, ".")
    num_loops = len(loop_runs)
    if num_loops == 0 or num_loops > max_loops:
        return False

    # 4. Stem validation
    stem_runs = runs(struct, "(") + runs(struct, ")")
    if not stem_runs:
        return False
    max_stem = max(stem_runs)
    if max_stem < 10:
        return False

    # 5. NEW: Bulge Density Check
    # If more than 45% of the sequence is unpaired, it's too messy to be a good hairpin
    unpaired_count = struct.count(".")
    if unpaired_count / float(length) > 0.45: 
        return False

    # 6. NEW: "Big Bulge" Rejection
    # Real miRNAs usually have 1 big terminal loop. If there is a second huge loop >8nt, it's likely junk.
    loop_runs_sorted = sorted(loop_runs, reverse=True)
    if len(loop_runs_sorted) > 1:
        # loop_runs_sorted[0] is the terminal loop (usually).
        # loop_runs_sorted[1] is the largest internal bulge.
        if loop_runs_sorted[1] > 8: 
            return False

    return True

# ---------- known miRNA overlap ----------

def load_mirna_intervals(bed_path: Optional[str]):
    """
    Load known miRNA coords as an IntervalTree per chrom.
    BED format: chrom, start, end, name, ...
    """
    if bed_path is None:
        return None
    if not HAS_INTERVALTREE:
        print("[warn] intervaltree not installed; known miRNA filter will be skipped.")
        return None

    trees = {}
    with open(bed_path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            trees.setdefault(chrom, IntervalTree()).addi(start, end)
    return trees


def overlaps_known_mirna(
    chrom: str,
    start: int,
    end: int,
    trees,
) -> bool:
    if trees is None:
        return False
    tree = trees.get(chrom)
    if tree is None:
        return False
    return bool(tree.overlap(start, end))


# ---------- worker ----------

def window_worker(args):
    """
    Worker for multiprocessing. args is:
    (chrom, start, end, seq, params_dict)
    """
    chrom, start, end, seq, params = args
    seq_u = seq.upper().replace("T", "U")
    struct, mfe = run_rnafold(seq_u)
    if struct is None:
        return None

    if not hairpin_passes_filters(
        seq_u,
        struct,
        mfe,
        min_pairs=params["min_pairs"],
        min_mfe=params["min_mfe"],
        max_len=params["max_len"],
        max_loops=params["max_loops"],
    ):
        return None

    # Optionally skip overlap; we handle this in the main process (faster)
    return (chrom, start, end, seq_u, mfe)


# ---------- main ----------

def generate_windows(
    fasta_path: str,
    window_len: int,
    step: int,
) -> List[Tuple[str, int, int, str]]:
    """
    Generate (chrom, start, end, seq) windows from genome.
    """
    windows = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        chrom = rec.id
        seq = str(rec.seq)
        n = len(seq)
        for start in range(0, n - 80, step):
        # Randomly choose a length typical of pre-miRNAs
            current_len = random.randint(60, 100)
            end = start + current_len
            win_seq = seq[start:end]
            if "N" in win_seq.upper():
                continue
            windows.append((chrom, start, end, win_seq))
    return windows


def main():
    ap = argparse.ArgumentParser(
        description="Generate pseudo pre-miRNA hairpins from a genome."
    )
    ap.add_argument("--genome", required=True, help="Genome FASTA (e.g., GRCh38).")
    ap.add_argument("--out-fa", required=True, help="Output FASTA for negative hairpins.")
    ap.add_argument("--out-bed", default=None, help="Optional BED output.")
    ap.add_argument("--known-mirna-bed", default=None,
                    help="Optional BED of known miRNA loci to exclude.")
    ap.add_argument("--window-len", type=int, default=80,
                    help="Window length (nt) for scanning.")
    ap.add_argument("--step", type=int, default=10,
                    help="Step size (nt) between windows.")
    ap.add_argument("--min-pairs", type=int, default=16,
                    help="Minimum number of base pairs for hairpin-like structure.")
    ap.add_argument("--min-mfe", type=float, default=-20.0,
                    help="Maximum MFE (kcal/mol), e.g., -20 means mfe <= -20.")
    ap.add_argument("--max-len", type=int, default=120,
                    help="Maximum allowed hairpin length.")
    ap.add_argument("--max-loops", type=int, default=4,
                    help="Maximum number of loop segments.")
    ap.add_argument("--threads", type=int, default=4,
                    help="Number of worker processes.")
    args = ap.parse_args()

    params = {
        "min_pairs": args.min_pairs,
        "min_mfe": args.min_mfe,
        "max_len": args.max_len,
        "max_loops": args.max_loops,
    }

    print(f"[pseudo] Loading known miRNA intervals (if provided)...")
    mirna_trees = load_mirna_intervals(args.known_mirna_bed)

    print(f"[pseudo] Generating windows from {args.genome}...")
    windows = generate_windows(args.genome, args.window_len, args.step)
    print(f"[pseudo] Total windows to evaluate: {len(windows)}")

    worker_args = [
        (chrom, start, end, seq, params)
        for (chrom, start, end, seq) in windows
    ]

    # Parallel RNAfold + structure filtering
    print(f"[pseudo] Running RNAfold in parallel with {args.threads} processes...")
    pool = mp.Pool(processes=args.threads)
    results = pool.imap_unordered(window_worker, worker_args, chunksize=100)
    pool.close()

    out_fa = Path(args.out_fa)
    out_fa.parent.mkdir(parents=True, exist_ok=True)
    fa_handle = open(out_fa, "w")

    if args.out_bed:
        out_bed = Path(args.out_bed)
        out_bed.parent.mkdir(parents=True, exist_ok=True)
        bed_handle = open(out_bed, "w")
    else:
        bed_handle = None

    kept = 0
    skipped_overlap = 0

    for res in results:
        if res is None:
            continue
        chrom, start, end, seq_u, mfe = res

        # Exclude overlaps with known miRNAs here (cheaper than in worker)
        if overlaps_known_mirna(chrom, start, end, mirna_trees):
            skipped_overlap += 1
            continue

        kept += 1
        hid = f"{chrom}:{start}-{end};mfe={mfe:.2f}"
        fa_handle.write(f">{hid}\n{seq_u}\n")
        if bed_handle is not None:
            bed_handle.write(f"{chrom}\t{start}\t{end}\t{hid}\t0\t+\n")

    fa_handle.close()
    if bed_handle is not None:
        bed_handle.close()

    print(f"[pseudo] Kept {kept} pseudo hairpins.")
    if skipped_overlap:
        print(f"[pseudo] Discarded {skipped_overlap} overlapping known miRNAs.")


if __name__ == "__main__":
    main()
