#!/usr/bin/env python3
"""
Generate pseudo pre-miRNA hairpins from a genome FASTA.

- Slides fixed-length windows across the genome
- Folds each window with RNAfold (in parallel)
- Applies simple hairpin-like filters (length, MFE, base pairs, loop/stem constraints)
- Optionally excludes windows overlapping known miRNA loci (BED)
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import subprocess
import multiprocessing as mp

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
    min_pairs: int = 16,
    min_mfe: float = -20.0,
    max_len: int = 120,
    max_loops: int = 4,
) -> bool:
    """
    Rough hairpin-like criteria inspired by Triplet-SVM/microPred-style filters.

    You will likely tune these thresholds, but this is a reasonable starting point.
    """
    length = len(seq)
    if length > max_len:
        return False

    # energy threshold
    if mfe > min_mfe:  # e.g., mfe must be <= -20
        return False

    pairs = struct.count("(")
    if pairs < min_pairs:
        return False

    loop_runs = runs(struct, ".")
    num_loops = len(loop_runs)
    if num_loops == 0 or num_loops > max_loops:
        return False

    # reject if structure is totally messy (no clear stem)
    stem_runs = runs(struct, "(") + runs(struct, ")")
    if not stem_runs:
        return False
    max_stem = max(stem_runs)
    if max_stem < 10:
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
        for start in range(0, n - window_len + 1, step):
            end = start + window_len
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
