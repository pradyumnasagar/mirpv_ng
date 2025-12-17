# mirpv_ng/cli.py

"""
Command-line interface for miRPV-NG.

Currently provides:
- score-fasta: score sequences in FASTA (sequence-only mode)
"""

import argparse
import csv
import sys
from pathlib import Path

from .classifier import HairpinClassifier
from .features import read_fasta  # same helper used in training

from .structure3d_af3 import AF3Config, run_af3_for_rna


def cmd_score_fasta(args: argparse.Namespace) -> int:
    
    """
    Sequence-only mode scoring for a FASTA file.
    """
    fasta_path = Path(args.fasta)
    if not fasta_path.exists():
        print(f"[score-fasta] ERROR: FASTA not found: {fasta_path}", file=sys.stderr)
        return 1

    # Instantiate classifier
    clf = HairpinClassifier(
        model_path=args.model,
        species=args.species,
        feature_set=args.feature_set,
        max_hairpin_len=args.max_hairpin_len,
        max_seq_only_len=args.max_seq_only_len,
        window_len=args.window_len,
        step=args.step,
        tier1_min_pairs=args.tier1_min_pairs,
        tier1_min_mfe=args.tier1_min_mfe,
        tier2_enabled=args.tier2,
    )

    records = read_fasta(str(fasta_path))
    print(f"[score-fasta] Loaded {len(records)} sequences from {fasta_path}", file=sys.stderr)

    # Prepare output
    out_path = Path(args.out_tsv) if args.out_tsv else Path("-")
    out_fh = sys.stdout if str(out_path) == "-" else open(out_path, "w", newline="")

    fieldnames = [
        "input_id",
        "mode",
        "start",
        "end",
        "length",
        "rf_score",
        "pred_label",
    ]
    writer = csv.DictWriter(out_fh, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()

    total_candidates = 0
    for seq_id, seq in records:
        results = clf.score_sequence_record(seq_id, seq)
        for rec in results:
            writer.writerow(rec)
            total_candidates += 1

    if out_fh is not sys.stdout:
        out_fh.close()

    print(
        f"[score-fasta] Wrote {total_candidates} candidate records from "
        f"{len(records)} input sequences to {out_path}",
        file=sys.stderr,
    )
    return 0
def cmd_annotate_3d(args: argparse.Namespace) -> int:
    """
    Optional: run AlphaFold3 on top-N precursors/hairpins.
    """
    fasta_path = Path(args.fasta)
    if not fasta_path.exists():
        print(f"[annotate-3d] ERROR: FASTA not found: {fasta_path}", file=sys.stderr)
        return 1

    # Load sequences
    records = read_fasta(str(fasta_path))
    if not records:
        print("[annotate-3d] ERROR: FASTA has no sequences.", file=sys.stderr)
        return 1

    # Select top N in FASTA order
    selected = records[: args.top_n]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = AF3Config(
        docker_image=args.docker_image,
        model_dir=Path(args.model_dir),
        db_dir=Path(args.db_dir),
    )

    print(f"[annotate-3d] Running AF3 for {len(selected)} sequences", file=sys.stderr)

    for seq_id, seq in selected:
        try:
            pdb_path = run_af3_for_rna(
                seq_id=seq_id,
                seq=seq,
                out_dir=out_dir,
                cfg=cfg,
                max_len=args.max_len,
                dry_run=args.dry_run,
            )
            print(f"{seq_id}\t{pdb_path}")
        except ValueError as e:
            print(f"[annotate-3d] Skipping {seq_id}: {e}", file=sys.stderr)

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mirpv-ng", description="miRPV-NG CLI")
    sub = p.add_subparsers(dest="command", required=True)

    # score-fasta subcommand
    sp = sub.add_parser(
        "score-fasta",
        help="Score sequences from a FASTA file (sequence-only mode).",
    )
    sp.add_argument(
        "--tier2", 
        action="store_true", 
        help="Enable Tier-2 soft-gated features during scoring"
    )

    sp.add_argument(
        "--fasta",
        required=True,
        help="Input FASTA of sequences (hairpins or longer regions).",
    )
    sp.add_argument(
        "--model",
        required=True,
        help="Path to trained RF model (.pkl) from train_premirna_model.py",
    )
    sp.add_argument(
        "--species",
        default="hsa",
        help="Species code (currently just informative).",
    )
    sp.add_argument(
        "--feature-set",
        choices=["core36", "extended"],
        default="extended",
        help="Feature set expected by the model.",
    )
    sp.add_argument(
        "--max-hairpin-len",
        type=int,
        default=120,
        help="Max length treated as a single hairpin.",
    )
    sp.add_argument(
        "--max-seq-only-len",
        type=int,
        default=5000,
        help="Max length allowed in sequence-only mode; above this is 'too_long'.",
    )
    sp.add_argument(
        "--window-len",
        type=int,
        default=100,
        help="Window length for scanning longer sequences.",
    )
    sp.add_argument(
        "--step",
        type=int,
        default=20,
        help="Step size for sliding window in scanning mode.",
    )
    sp.add_argument(
        "--tier1-min-pairs",
        type=int,
        default=18,
        help="Tier-1 filter: minimum number of base pairs in window.",
    )
    sp.add_argument(
        "--tier1-min-mfe",
        type=float,
        default=-15.0,
        help="Tier-1 filter: maximum allowed MFE (e.g., -15.0 kcal/mol).",
    )
    sp.add_argument(
        "--out-tsv",
        default="-",
        help="Output TSV file (default: stdout).",
    )
    sp.set_defaults(func=cmd_score_fasta)
        # ---------------------------------------------------------------------
    # annotate-3d subcommand (optional AlphaFold3-RNA integration)
    # ---------------------------------------------------------------------
    sp3 = sub.add_parser(
        "annotate-3d",
        help="Run AlphaFold3 on top-N candidate sequences to produce RNA 3D structures (optional).",
    )

    sp3.add_argument(
        "--fasta",
        required=True,
        help="FASTA file containing precursor/hairpin sequences for 3D modelling.",
    )
    sp3.add_argument(
        "--out-dir",
        required=True,
        help="Output directory where AF3 PDB files will be written.",
    )
    sp3.add_argument(
        "--docker-image",
        required=True,
        help="Name of the AlphaFold3 Docker image to use.",
    )
    sp3.add_argument(
        "--model-dir",
        required=True,
        help="Host path to AlphaFold3 model directory.",
    )
    sp3.add_argument(
        "--db-dir",
        required=True,
        help="Host path to AlphaFold3 database directory.",
    )
    sp3.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Maximum number of sequences to process (default: 20).",
    )
    sp3.add_argument(
        "--max-len",
        type=int,
        default=120,
        help="Maximum RNA length allowed for AF3 modelling (default: 120).",
    )
    sp3.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the AF3 docker command but do NOT run it.",
    )

    sp3.set_defaults(func=cmd_annotate_3d)


    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
