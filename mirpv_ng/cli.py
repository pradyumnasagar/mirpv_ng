# mirpv_ng/cli.py
"""
Command-line interface for miRPV-NG.

Commands:
- score-fasta: sequence-only mode (hairpin scanning + RF)
- predict-mature: mature miRNA prediction (XGBRanker)
- annotate-3d: optional AlphaFold3 RNA structure prediction
- fastq-to-peaks: sRNA-seq Stage 1–8 (Peak+Pad): FASTQ -> peaks (+optional excision -> candidates.fa/tsv)
- candidates-to-scored: sRNA-seq Stage 9: candidates -> features -> RF score -> candidates.scored.tsv
- scored-to-peaks: sRNA-seq Stage 9.5: collapse pad-level candidates -> per-peak best choice
- peaks-to-known: sRNA-seq Stage 10: annotate peaks with MirGeneDB + miRBase known status
"""

import argparse
import csv
import sys
from pathlib import Path

from .classifier import HairpinClassifier
from .features import read_fasta
from .structure3d_af3 import AF3Config, run_af3_for_rna

from .fastq_to_peaks import run_fastq_to_peaks
from .candidates_to_scored import run_candidates_to_scored
from .scored_to_peaks import run_scored_to_peaks
from .peaks_to_known import run_peaks_to_known


def cmd_score_fasta(args: argparse.Namespace) -> int:
    fasta_path = Path(args.fasta)
    if not fasta_path.exists():
        print(f"[score-fasta] ERROR: FASTA not found: {fasta_path}", file=sys.stderr)
        return 1

    mature_ranker = None
    if args.predict_mature:
        if not args.mature_model:
            print("[score-fasta] ERROR: --predict-mature requires --mature-model", file=sys.stderr)
            return 1
        from .mature_model import MatureRanker
        mature_ranker = MatureRanker.load(args.mature_model)

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

    out_path = Path(args.out_tsv) if args.out_tsv else Path("-")
    out_fh = sys.stdout if str(out_path) == "-" else open(out_path, "w", newline="")

    base_fields = ["input_id", "mode", "start", "end", "length", "rf_score", "pred_label"]

    mature_fields = [
        "mature_arm",
        "mature_start",
        "mature_end",
        "mature_len",
        "mature_score",
        "mature_seq",
        "mature_start_global",
        "mature_end_global",
    ]

    extra_fields = ["hairpin_seq"]
    fieldnames = base_fields + (mature_fields if args.predict_mature else []) + extra_fields

    writer = csv.DictWriter(out_fh, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()

    for seq_id, full_seq in records:
        results = clf.score_sequence_record(seq_id, full_seq)
        for rec in results:
            try:
                s = int(rec["start"])
                e = int(rec["end"])
                rec["hairpin_seq"] = full_seq[s:e].upper().replace("T", "U")
            except Exception:
                rec["hairpin_seq"] = "NA"

            if args.predict_mature:
                rec.update({k: "NA" for k in mature_fields})

            if args.predict_mature and (mature_ranker is not None):
                try:
                    pred_label = int(rec.get("pred_label", 0))
                except Exception:
                    pred_label = 0

                if pred_label == 1 or args.predict_mature_all:
                    cand_seq = rec["hairpin_seq"]
                    if cand_seq != "NA" and len(cand_seq) >= 40:
                        pred = mature_ranker.predict_top1(
                            cand_seq,
                            rnafold_bin=args.mature_rnafold_bin,
                            lengths=tuple(args.mature_lengths),
                            max_per_arm=args.mature_max_per_arm,
                            min_paired_context=args.mature_min_paired_context,
                            loop_buffer=args.mature_loop_buffer,
                            fallback_loop_buffer=args.mature_fallback_loop_buffer,
                            fallback_max_per_arm=args.mature_fallback_max_per_arm,
                            fallback_min_paired_context=args.mature_fallback_min_paired_context,
                        )
                        rec.update(pred)

            writer.writerow(rec)

    if out_fh is not sys.stdout:
        out_fh.close()
    print(f"[score-fasta] Wrote results to {out_path}", file=sys.stderr)
    return 0


def cmd_predict_mature(args: argparse.Namespace) -> int:
    from .mature_model import MatureRanker
    ranker = MatureRanker.load(args.mature_model)

    records = read_fasta(args.fasta)
    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "input_id",
                "mature_arm",
                "mature_start",
                "mature_end",
                "mature_len",
                "mature_score",
                "mature_seq",
                "mature_start_global",
                "mature_end_global",
            ],
            delimiter="\t",
        )
        w.writeheader()
        for seq_id, seq in records:
            pred = ranker.predict_top1(
                seq,
                rnafold_bin=args.rnafold_bin,
                loop_buffer=args.loop_buffer,
                fallback_loop_buffer=args.fallback_loop_buffer,
            )
            pred["input_id"] = seq_id
            w.writerow(pred)

    print(f"[predict-mature] Wrote: {out_path}", file=sys.stderr)
    return 0


def cmd_annotate_3d(args: argparse.Namespace) -> int:
    cfg = AF3Config(
        docker_image=args.docker_image,
        model_dir=args.model_dir,
        db_dir=args.db_dir,
        out_dir=args.out_dir,
        dry_run=args.dry_run,
    )
    return run_af3_for_rna(
        fasta=args.fasta,
        cfg=cfg,
        top_n=args.top_n,
        max_len=args.max_len,
    )


def cmd_fastq_to_peaks(args: argparse.Namespace) -> int:
    return run_fastq_to_peaks(args)


def cmd_candidates_to_scored(args: argparse.Namespace) -> int:
    return run_candidates_to_scored(args)


def cmd_scored_to_peaks(args: argparse.Namespace) -> int:
    return run_scored_to_peaks(args)


def cmd_peaks_to_known(args: argparse.Namespace) -> int:
    return run_peaks_to_known(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mirpv-ng")
    sub = p.add_subparsers(dest="cmd", required=True)

    # -------- Sequence-only mode --------
    sp = sub.add_parser("score-fasta", help="Score sequences in FASTA using RF classifier (sequence-only mode).")
    sp.add_argument("--model", required=True)
    sp.add_argument("--fasta", required=True)
    sp.add_argument("--out-tsv", required=True)
    sp.add_argument("--species", default="hsa")
    sp.add_argument("--feature-set", default="extended")
    sp.add_argument("--max-hairpin-len", type=int, default=180)
    sp.add_argument("--max-seq-only-len", type=int, default=5000)
    sp.add_argument("--window-len", type=int, default=200)
    sp.add_argument("--step", type=int, default=20)
    sp.add_argument("--tier1-min-pairs", type=int, default=18)
    sp.add_argument("--tier1-min-mfe", type=float, default=-18.0)
    sp.add_argument("--tier2", action="store_true")
    sp.add_argument("--predict-mature", action="store_true")
    sp.add_argument("--predict-mature-all", action="store_true")
    sp.add_argument("--mature-model", default=None)
    sp.add_argument("--mature-rnafold-bin", default="RNAfold")
    sp.add_argument("--mature-lengths", nargs="+", type=int, default=[21, 22, 23, 24])
    sp.add_argument("--mature-max-per-arm", type=int, default=30)
    sp.add_argument("--mature-min-paired-context", type=int, default=6)
    sp.add_argument("--mature-loop-buffer", type=int, default=0)
    sp.add_argument("--mature-fallback-loop-buffer", type=int, default=10)
    sp.add_argument("--mature-fallback-max-per-arm", type=int, default=120)
    sp.add_argument("--mature-fallback-min-paired-context", type=int, default=0)
    sp.set_defaults(func=cmd_score_fasta)

    spm = sub.add_parser("predict-mature", help="Predict mature miRNA position on precursor FASTA using XGBRanker.")
    spm.add_argument("--mature-model", required=True)
    spm.add_argument("--fasta", required=True)
    spm.add_argument("--out", required=True)
    spm.add_argument("--rnafold-bin", default="RNAfold")
    spm.add_argument("--loop-buffer", type=int, default=0)
    spm.add_argument("--fallback-loop-buffer", type=int, default=10)
    spm.set_defaults(func=cmd_predict_mature)

    sp3 = sub.add_parser("annotate-3d", help="Run AlphaFold3 on top-N sequences (optional).")
    sp3.add_argument("--fasta", required=True)
    sp3.add_argument("--out-dir", required=True)
    sp3.add_argument("--docker-image", required=True)
    sp3.add_argument("--model-dir", required=True)
    sp3.add_argument("--db-dir", required=True)
    sp3.add_argument("--top-n", type=int, default=20)
    sp3.add_argument("--max-len", type=int, default=120)
    sp3.add_argument("--dry-run", action="store_true")
    sp3.set_defaults(func=cmd_annotate_3d)

    # -------- sRNA-seq mode --------
    sps = sub.add_parser("fastq-to-peaks", help="sRNA-seq Stage 1–8: FASTQ -> peaks (+optional excision -> candidates).")
    sps.add_argument("--fastq", required=True)
    sps.add_argument("--sample-id", required=True)
    sps.add_argument("--outdir", required=True)

    sps.add_argument("--cutadapt", default="cutadapt")
    sps.add_argument("--bowtie", default="bowtie")
    sps.add_argument("--bowtie-index", required=True)
    sps.add_argument("--threads", type=int, default=8)
    sps.add_argument("--adapter", default=None)
    sps.add_argument("--max-multimaps", type=int, default=50)

    # Blocklist (Stage 2)
    sps.add_argument("--blocklist-index", default=None, help="Bowtie index basename for blocklist (Rfam/tRNA etc).")
    sps.add_argument("--blocklist-name", default="rfam")
    sps.add_argument("--blocklist-mismatches", type=int, default=0)
    sps.add_argument("--blocklist-max-align", type=int, default=1)

    # Islands + gates
    sps.add_argument("--island-gap", type=int, default=50)
    sps.add_argument("--min-depth", type=int, default=5)
    sps.add_argument("--min-cpm", type=float, default=0.5)

    # Signal processing flags
    sps.add_argument("--smooth-w", type=int, default=3, dest="smooth_w")
    sps.add_argument("--peak-distance", type=int, default=35, dest="peak_distance")
    sps.add_argument("--peak-micromerge", type=int, default=8, dest="peak_micromerge")

    sps.add_argument("--use-scipy", action="store_true", dest="use_scipy")
    sps.add_argument("--scipy-prominence", type=float, default=None, dest="scipy_prominence")
    sps.add_argument("--scipy-width-min", type=float, default=None, dest="scipy_width_min")
    sps.add_argument("--scipy-width-max", type=float, default=None, dest="scipy_width_max")
    sps.add_argument("--fallback-prom-frac", type=float, default=0.30, dest="fallback_prom_frac")

    sps.add_argument("--support-window", type=int, default=15)
    sps.add_argument("--hard-frac-20-24", type=float, default=0.30)

    sps.add_argument("--anchor-unique-dominance", type=float, default=0.50)
    sps.add_argument("--anchor-unique-prec5p", type=float, default=0.70)
    sps.add_argument("--anchor-multi-dominance", type=float, default=0.60)
    sps.add_argument("--anchor-multi-prec5p", type=float, default=0.85)

    sps.add_argument("--repeat-bed", default=None)
    sps.add_argument("--bedtools", default="bedtools")
    sps.add_argument("--repeat-multi-prec5p", type=float, default=0.90)
    sps.add_argument("--repeat-multi-dominance", type=float, default=0.70)

    # Excision (optional)
    sps.add_argument("--genome-fasta", default=None)
    sps.add_argument("--pads", nargs="+", type=int, default=[70, 100])
    sps.add_argument("--samtools", default="samtools")

    sps.set_defaults(func=cmd_fastq_to_peaks)

    # Stage 9
    s9 = sub.add_parser("candidates-to-scored", help="sRNA-seq Stage 9: candidates -> RF scored candidates.scored.tsv")
    s9.add_argument("--candidates-tsv", required=True)
    s9.add_argument("--candidates-fa", required=True)
    s9.add_argument("--model", required=True)
    s9.add_argument("--outdir", required=True)
    s9.add_argument("--sample-id", required=True)
    s9.add_argument("--species", default="hsa")
    s9.add_argument("--feature-set", default="extended")
    s9.add_argument("--tier2", action="store_true")
    s9.set_defaults(func=cmd_candidates_to_scored)

    # Stage 9.5
    s95 = sub.add_parser("scored-to-peaks", help="sRNA-seq Stage 9.5: collapse pad-level scored candidates -> per-peak best")
    s95.add_argument("--scored-tsv", required=True, help="candidates.scored.tsv")
    s95.add_argument("--outdir", required=True)
    s95.set_defaults(func=cmd_scored_to_peaks)

    # Stage 10
    s10 = sub.add_parser("peaks-to-known", help="sRNA-seq Stage 10: annotate peaks as Known-Confirmed/Atypical/Unknown")
    s10.add_argument("--peaks-tsv", required=True, help="peaks.scored.tsv (from scored-to-peaks)")
    s10.add_argument("--outdir", required=True)
    s10.add_argument("--sample-id", default=None)

    s10.add_argument("--mirgenedb-bed", required=True, help="MirGeneDB precursor BED6")
    s10.add_argument("--mirbase-bed", required=True, help="miRBase precursor BED6")

    s10.add_argument("--min-any-overlap-bp", type=int, default=1)
    s10.add_argument("--min-confirm-overlap-bp", type=int, default=20)
    s10.add_argument("--min-confirm-frac-peak", type=float, default=0.20)
    s10.set_defaults(func=cmd_peaks_to_known)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
