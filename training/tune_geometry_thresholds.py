#!/usr/bin/env python3
"""
Estimate Tier-2 geometry-aware thresholds from training FASTA sets.

Usage (from project root):

  python training/tune_geometry_thresholds.py \
      --pos-fasta mirpv_ng/training/data/hsa_mirgene_premirna.fa \
      --neg-fasta mirpv_ng/training/data/hsa_neg_hairpins.fa \
      --rnafold-bin RNAfold \
      --max-per-class 2000

This will:
  - fold up to N positives and N negatives with RNAfold
  - compute simple loop/bulge metrics from the dot-bracket structure
  - print per-class distributions
  - emit a suggested GeometryConfig block for Tier-2 filtering
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import sys

# ---------------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------------

# Ensure project root (the directory that contains 'mirpv_ng/' and 'training/')
# is on sys.path, so we can import the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mirpv_ng.features import read_fasta, run_rnafold  # type: ignore
from mirpv_ng.geom_bulges import compute_bulge_features  # type: ignore


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def find_internal_loops(struct: str) -> List[Tuple[int, int]]:
    """
    Find contiguous runs of '.' between the first and last paired base.

    Returns a list of (start, end) indices (end is exclusive).
    This approximates apical + internal loops for the purposes of
    geometry threshold tuning.
    """
    struct = struct.strip()
    if not struct:
        return []

    # identify the "core" hairpin span: from first paired to last paired
    paired_positions = [i for i, ch in enumerate(struct) if ch in ("(", ")")]
    if not paired_positions:
        return []

    start_core = min(paired_positions)
    end_core = max(paired_positions)

    loops: List[Tuple[int, int]] = []
    i = start_core
    L = len(struct)

    while i <= end_core:
        if struct[i] == ".":
            j = i
            while j <= end_core and struct[j] == ".":
                j += 1
            loops.append((i, j))
            i = j
        else:
            i += 1

    return loops


@dataclass
class GeometryMetrics:
    """
    Minimal geometry metrics per sequence, for tuning Tier-2 thresholds.
    """

    length: int
    num_loops: int
    max_loop_size: int
    bulge_count: float
    bulge_chain_count: float
    bulge_density: float


def compute_geometry_metrics(
    seq: str, struct: str
) -> GeometryMetrics:
    """
    Compute simple geometry metrics from sequence and dot-bracket structure.

    Uses:
      - find_internal_loops (loop count and max size)
      - compute_bulge_features from geom_bulges (bulge stats)
    """
    seq = seq.strip().upper()
    struct = struct.strip()
    L = len(seq)

    loops = find_internal_loops(struct)
    num_loops = len(loops)
    max_loop_size = max((end - start for start, end in loops), default=0)

    bulge_feats: Dict[str, float] = compute_bulge_features(seq, struct)
    bulge_count = float(bulge_feats.get("bulge_count", 0.0))
    bulge_chain_count = float(bulge_feats.get("bulge_chain_count", 0.0))

    length = max(L, 1)
    bulge_density = bulge_count / float(length)

    return GeometryMetrics(
        length=length,
        num_loops=num_loops,
        max_loop_size=max_loop_size,
        bulge_count=bulge_count,
        bulge_chain_count=bulge_chain_count,
        bulge_density=bulge_density,
    )


def subsample_records(
    records: List[Tuple[str, str]],
    max_n: int | None,
    seed: int = 1,
) -> List[Tuple[str, str]]:
    """
    Optionally subsample at most max_n records in a reproducible way.
    """
    if max_n is None or max_n <= 0 or len(records) <= max_n:
        return records

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(records), size=max_n, replace=False)
    return [records[i] for i in sorted(idx)]


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def featurize_fasta(
    fasta_path: Path,
    label: int,
    rnafold_bin: str,
    max_per_class: int | None,
) -> pd.DataFrame:
    """
    Fold sequences from FASTA, compute geometry metrics, and return a DataFrame.
    """
    records = read_fasta(str(fasta_path))
    records = subsample_records(records, max_per_class)

    rows: List[Dict[str, float]] = []

    for seq_id, seq in records:
        struct, mfe = run_rnafold(seq, rnafold_bin=rnafold_bin)
        if not struct:
            continue

        gm = compute_geometry_metrics(seq, struct)

        rows.append(
            {
                "id": seq_id,
                "label": label,
                "length": gm.length,
                "num_loops": gm.num_loops,
                "max_loop_size": gm.max_loop_size,
                "bulge_count": gm.bulge_count,
                "bulge_chain_count": gm.bulge_chain_count,
                "bulge_density": gm.bulge_density,
                "mfe": mfe,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["id", "label"])

    return pd.DataFrame(rows)


def summarize_metric(
    df: pd.DataFrame,
    metric: str,
    label: int,
) -> Dict[str, float]:
    """
    Compute summary stats for a given metric and class label.
    """
    subset = df[df["label"] == label][metric].values
    if subset.size == 0:
        return {}

    return {
        "mean": float(np.mean(subset)),
        "std": float(np.std(subset)),
        "median": float(np.median(subset)),
        "p95": float(np.percentile(subset, 95)),
        "p99": float(np.percentile(subset, 99)),
        "min": float(np.min(subset)),
        "max": float(np.max(subset)),
        "n": int(subset.size),
    }


def print_summary(df: pd.DataFrame) -> None:
    """
    Print per-metric summary for positives vs negatives.
    """
    metrics = ["num_loops", "max_loop_size", "bulge_count", "bulge_chain_count", "bulge_density"]

    print("# Geometry metric distributions")
    print()

    for metric in metrics:
        pos_stats = summarize_metric(df, metric, label=1)
        neg_stats = summarize_metric(df, metric, label=0)

        print(f"Metric: {metric}")
        if not pos_stats:
            print("  [no positive examples]")
            continue

        def fmt(stats: Dict[str, float]) -> str:
            return (
                f"n={stats['n']}, "
                f"mean={stats['mean']:.3f}, "
                f"median={stats['median']:.3f}, "
                f"p95={stats['p95']:.3f}, "
                f"p99={stats['p99']:.3f}, "
                f"min={stats['min']:.3f}, "
                f"max={stats['max']:.3f}"
            )

        print("  positives:", fmt(pos_stats))
        if neg_stats:
            print("  negatives:", fmt(neg_stats))
        print()


def suggest_geometry_config(df: pd.DataFrame) -> None:
    """
    Suggest GeometryConfig thresholds from positive distributions.

    We take the 99th percentile of positives for:
      - num_loops         -> max_num_loops
      - max_loop_size     -> max_loop_size
      - bulge_chain_count -> max_bulge_chain
      - bulge_density     -> max_bulge_density

    and print a ready-to-paste dataclass initializer snippet.
    """
    metrics_map = {
        "num_loops": "max_num_loops",
        "max_loop_size": "max_loop_size",
        "bulge_chain_count": "max_bulge_chain",
        "bulge_density": "max_bulge_density",
    }

    pos_df = df[df["label"] == 1]
    if pos_df.empty:
        print("# No positive examples; cannot suggest thresholds.")
        return

    print("# Suggested GeometryConfig (from positive 99th percentiles)")
    print("# You may want to adjust manually after inspection.")
    print()

    # Compute 99th percentile for each metric on positives
    suggestions: Dict[str, float] = {}
    for metric, cfg_name in metrics_map.items():
        vals = pos_df[metric].values
        if vals.size == 0:
            continue
        p99 = float(np.percentile(vals, 99))
        suggestions[cfg_name] = p99

    # Pretty-print as a dataclass initializer
    print("from dataclasses import dataclass")
    print()
    print("@dataclass")
    print("class GeometryConfig:")
    print("    max_num_loops: int | None = None")
    print("    max_loop_size: int | None = None")
    print("    max_bulge_chain: int | None = None")
    print("    max_bulge_density: float | None = None")
    print("    max_cactus_score: float | None = None")
    print()
    print("geom_cfg = GeometryConfig(")

    # Coerce loop counts etc to ints where appropriate
    for cfg_name, value in suggestions.items():
        if cfg_name in ("max_num_loops", "max_bulge_chain", "max_loop_size"):
            v = int(round(value))
        else:
            v = value
        print(f"    {cfg_name}={v!r},")

    print("    max_cactus_score=None,  # not used yet")
    print(")")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Tune Tier-2 geometry thresholds from positive/negative FASTA sets."
    )
    p.add_argument(
        "--pos-fasta",
        required=True,
        help="FASTA file with known pre-miRNAs (label=1).",
    )
    p.add_argument(
        "--neg-fasta",
        required=True,
        help="FASTA file with pseudo-hairpins / negatives (label=0).",
    )
    p.add_argument(
        "--rnafold-bin",
        default="RNAfold",
        help="Path to RNAfold executable (default: RNAfold in $PATH).",
    )
    p.add_argument(
        "--max-per-class",
        type=int,
        default=2000,
        help="Maximum number of sequences per class to use (default: 2000).",
    )
    p.add_argument(
        "--out-tsv",
        type=str,
        default=None,
        help="Optional path to write raw geometry metrics as TSV.",
    )

    args = p.parse_args()

    pos_path = Path(args.pos_fasta)
    neg_path = Path(args.neg_fasta)

    print(f"[tune] positives: {pos_path}")
    print(f"[tune] negatives: {neg_path}")
    print(f"[tune] RNAfold:   {args.rnafold_bin}")
    print(f"[tune] max per class: {args.max_per_class}")
    print()

    df_pos = featurize_fasta(pos_path, label=1, rnafold_bin=args.rnafold_bin, max_per_class=args.max_per_class)
    df_neg = featurize_fasta(neg_path, label=0, rnafold_bin=args.rnafold_bin, max_per_class=args.max_per_class)

    df = pd.concat([df_pos, df_neg], ignore_index=True)
    if df.empty:
        print("No data collected; check inputs and RNAfold.")
        raise SystemExit(1)

    if args.out_tsv is not None:
        out_path = Path(args.out_tsv)
        df.to_csv(out_path, sep="\t", index=False)
        print(f"[tune] wrote raw geometry metrics to {out_path}")

    print_summary(df)
    suggest_geometry_config(df)


if __name__ == "__main__":
    main()
