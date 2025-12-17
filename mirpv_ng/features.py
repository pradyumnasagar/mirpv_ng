# mirpv_ng/features.py

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .geom_stem_features import compute_pair_composition, compute_helix_stats
from .geom_energy import energy_features_global
from .geom_bulges import compute_bulge_features
from .pgs_features import compute_pgs_features


# ----------------- RNAfold wrapper ----------------- #

def run_rnafold(seq: str, rnafold_bin: str = "RNAfold") -> Tuple[str, float]:
    """
    Run RNAfold on a single sequence and return (structure, MFE).
    """
    seq = seq.strip().upper().replace("T", "U")
    if not seq:
        return "", 0.0

    proc = subprocess.Popen(
        [rnafold_bin, "--noPS"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = proc.communicate(seq + "\n")
    if proc.returncode != 0:
        raise RuntimeError(f"RNAfold failed: {err}")

    lines = [l for l in out.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"Unexpected RNAfold output:\n{out}")

    struct_line = lines[1]
    parts = struct_line.split()
    struct = parts[0]
    mfe_str = parts[-1].strip("()")
    mfe = float(mfe_str)
    return struct, mfe


# ----------------- core 36-ish features ----------------- #

def core36_features(seq: str, struct: str, mfe: float) -> Dict[str, float]:
    """
    Compact baseline feature set.
    """
    feats: Dict[str, float] = {}
    seq_u = seq.upper().replace("T", "U")
    L = max(len(seq_u), 1)

    A = seq_u.count("A")
    C = seq_u.count("C")
    G = seq_u.count("G")
    U = seq_u.count("U")

    feats["len"] = float(L)
    feats["gc_frac"] = (G + C) / L
    feats["au_frac"] = (A + U) / L
    feats["g_frac"] = G / L
    feats["c_frac"] = C / L
    feats["a_frac"] = A / L
    feats["u_frac"] = U / L

    n_pairs = struct.count("(")
    feats["num_pairs"] = float(n_pairs)
    feats["paired_frac"] = n_pairs * 2.0 / L if L > 0 else 0.0
    feats["unpaired_frac"] = 1.0 - feats["paired_frac"]

    num_loops, mean_loop, max_loop, loop_unpaired_fraction = classify_loops(struct)
    feats["num_loops"] = float(num_loops)
    feats["mean_loop_size"] = mean_loop
    feats["max_loop_size"] = float(max_loop)
    feats["loop_unpaired_frac"] = loop_unpaired_fraction

    num_bulges, bulge_density = bulge_stats(struct)
    feats["num_bulges"] = float(num_bulges)
    feats["bulge_density"] = bulge_density

    feats["mfe"] = float(mfe)
    feats["mfe_per_nt"] = float(mfe) / L

    return feats


# ----------------- extended feature set ----------------- #

def extended_features(seq: str, struct: str, mfe: float) -> Dict[str, float]:
    """
    Extended feature set: core + miRNAFold-like + PGS.
    """
    feats: Dict[str, float] = {}
    core = core36_features(seq, struct, mfe)
    feats.update(core)

    seq_u = seq.upper().replace("T", "U")
    L = max(len(seq_u), 1)

    # Global energy
    feats.update(energy_features_global(mfe, L))

    # Composition and imbalance
    A = seq_u.count("A")
    C = seq_u.count("C")
    G = seq_u.count("G")
    U = seq_u.count("U")

    feats["pct_A"] = A / L
    feats["pct_C"] = C / L
    feats["pct_G"] = G / L
    feats["pct_U"] = U / L

    ga = G + A
    cu = C + U
    feats["imbalance_GA_CU"] = abs(ga - cu) / L
    feats["imbalance_G_C"] = abs(G - C) / L

    # Dinucleotides
    dinucs = [
        "AA", "AC", "AG", "AU",
        "CA", "CC", "CG", "CU",
        "GA", "GC", "GG", "GU",
        "UA", "UC", "UG", "UU",
    ]
    di_den = max(L - 1, 1)
    for di in dinucs:
        feats[f"di_{di}"] = seq_u.count(di) / di_den

    # Simple tri-nucs
    tri_targets = ["UGU", "UGUG", "GUG", "GCG", "CGC", "AUG"]
    tri_den = max(L - 2, 1)
    for tri in tri_targets:
        feats[f"tri_{tri}"] = seq_u.count(tri) / tri_den

    # Segment-wise GC/paired
    seg_feats = segment_gc_and_pairing(seq_u, struct, n_segments=4)
    feats.update(seg_feats)

    # Pair composition + helix stats
    feats.update(compute_pair_composition(seq_u, struct))
    feats.update(compute_helix_stats(struct))

    # Bulge features
    feats.update(compute_bulge_features(seq_u, struct))

    # PGS features
    feats.update(compute_pgs_features(seq_u, struct))

    return feats


# ----------------- FASTA helpers ----------------- #

def read_fasta(path: str) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    current_id: str | None = None
    current_seq: List[str] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(current_seq)))
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_id is not None:
        records.append((current_id, "".join(current_seq)))
    return records


# ----------------- dot-bracket helpers ----------------- #

def classify_loops(struct: str) -> Tuple[int, float, int, float]:
    n = len(struct)
    loop_runs: List[int] = []
    loop_positions = set()

    i = 0
    while i < n:
        if struct[i] == ".":
            start = i
            while i < n and struct[i] == ".":
                i += 1
            end = i
            left = struct[start - 1] if start > 0 else None
            right = struct[end] if end < n else None
            if left in ("(", ")") and right in ("(", ")"):
                loop_runs.append(end - start)
                loop_positions.update(range(start, end))
        else:
            i += 1

    if not loop_runs:
        return 0, 0.0, 0, 0.0

    num_loops = len(loop_runs)
    mean_loop = float(np.mean(loop_runs))
    max_loop = int(np.max(loop_runs))
    loop_unpaired_fraction = len(loop_positions) / n if n > 0 else 0.0

    return num_loops, mean_loop, max_loop, loop_unpaired_fraction


def bulge_stats(struct: str) -> Tuple[int, float]:
    n = len(struct)
    bulge_positions = set()
    for i, c in enumerate(struct):
        if c != ".":
            continue
        left = struct[i - 1] if i > 0 else None
        right = struct[i + 1] if i < n - 1 else None
        if left in ("(", ")") or right in ("(", ")"):
            bulge_positions.add(i)
    num_bulges = len(bulge_positions)
    bulge_density = num_bulges / n if n > 0 else 0.0
    return num_bulges, bulge_density


def segment_gc_and_pairing(
    seq: str,
    struct: str,
    n_segments: int = 4,
) -> Dict[str, float]:
    L = len(seq)
    if L == 0 or n_segments <= 0:
        return {}

    seg_len = max(L // n_segments, 1)
    feats: Dict[str, float] = {}

    seg_gc: List[float] = []
    seg_paired: List[float] = []

    for i in range(n_segments):
        start = i * seg_len
        end = L if i == n_segments - 1 else min(L, (i + 1) * seg_len)
        if start >= L:
            seg_gc.append(0.0)
            seg_paired.append(0.0)
            continue

        seg_seq = seq[start:end]
        seg_struct = struct[start:end]

        seg_L = max(len(seg_seq), 1)
        G = seg_seq.count("G")
        C = seg_seq.count("C")
        gc_frac = (G + C) / seg_L
        paired_frac = seg_struct.count("(") * 2.0 / seg_L if seg_L > 0 else 0.0

        seg_gc.append(gc_frac)
        seg_paired.append(paired_frac)

        feats[f"seg{i+1}_gc"] = gc_frac
        feats[f"seg{i+1}_paired_frac"] = paired_frac

    feats["gc_grad_last_first"] = seg_gc[-1] - seg_gc[0]
    feats["paired_grad_last_first"] = seg_paired[-1] - seg_paired[0]

    return feats


# ----------------- batch feature computation ----------------- #

def compute_features_for_sequences(
    records: List[Tuple[str, str]],
    feature_set: str = "core36",
    rnafold_bin: str = "RNAfold",
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for rid, seq in records:
        struct, mfe = run_rnafold(seq, rnafold_bin=rnafold_bin)
        if feature_set == "core36":
            feats = core36_features(seq, struct, mfe)
        else:
            feats = extended_features(seq, struct, mfe)
        row: Dict[str, object] = {"id": rid, "seq": seq, "struct": struct, "mfe": mfe}
        row.update(feats)
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------- CLI ----------------- #

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute hairpin features from FASTA using RNAfold."
    )
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--feature-set", choices=["core36", "extended"], default="core36")
    ap.add_argument("--rnafold-bin", default="RNAfold")
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    records = read_fasta(args.fasta)
    print(f"[features] Loaded {len(records)} sequences from {args.fasta}")

    df = compute_features_for_sequences(
        records, feature_set=args.feature_set, rnafold_bin=args.rnafold_bin
    )
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[features] Wrote features to {out_path}")


if __name__ == "__main__":
    main()
