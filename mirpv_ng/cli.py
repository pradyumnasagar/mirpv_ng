"""
miRPV-NG command-line interface.

Sequence-only mode:
  - score-fasta
  - predict-mature
  - annotate-3d

sRNA-seq mode (Peak + Pad, Evidence-First Hybrid):
  Stage 1–8:  fastq-to-peaks
  Stage 9:    candidates-to-scored
  Stage 9.5:  scored-to-peaks
  Stage 10:   peaks-to-known
  Stage 11:   peaks-to-finalists
  Stage 12:   finalists-to-struct
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .features import read_fasta

# Optional AF3 (sequence-only add-on)
# Optional: AlphaFold3-based 3D annotation.
# The 3D workflow is experimental and may not be shipped in all builds.
try:
    from .structure3d_af3 import AF3Config, run_af3_for_rna  # type: ignore
    _AF3_AVAILABLE = True
except Exception:  # pragma: no cover
    AF3Config = None  # type: ignore

    def run_af3_for_rna(*args, **kwargs):  # type: ignore
        raise RuntimeError(
            "3D annotation (AlphaFold3) is not available in this build. "
            "Remove 'annotate-3d' usage or add mirpv_ng/structure3d_af3.py."
        )

    _AF3_AVAILABLE = False

from .fastq_to_peaks import run_fastq_to_peaks
from .candidates_to_scored import run_candidates_to_scored
from .scored_to_peaks import run_scored_to_peaks
from .peaks_to_known import run_peaks_to_known
from .peaks_to_finalists import run_peaks_to_finalists
from .finalists_to_struct import run_finalists_to_struct
from .final_candidates import run_final_candidates
from .final_report import stage14_report

# Import parallel module
try:
    from .parallel import ParallelConfig, add_parallel_args
    HAS_PARALLEL = True
except ImportError:
    HAS_PARALLEL = False
    ParallelConfig = None


# -----------------------------
# Sequence-only commands
# -----------------------------
def _normalize_rna(seq: str) -> str:
    seq = (seq or "").strip().upper().replace("T", "U")
    return seq


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

    from .classifier import HairpinClassifier
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
    
    # Configure parallelism if available and requested
    if HAS_PARALLEL and hasattr(args, 'jobs') and args.jobs > 1:
        pcfg = ParallelConfig.from_args(args)
        clf.set_parallel_config(pcfg)
        print(f"[score-fasta] Parallel mode: jobs={pcfg.jobs}, backend={pcfg.backend}", file=sys.stderr)

    records = read_fasta(str(fasta_path))
    print(f"[score-fasta] Loaded {len(records)} sequences from {fasta_path}", file=sys.stderr)

    # Output path: --out and --out-tsv both write to dest="out_tsv"
    out_val = getattr(args, "out_tsv", None)
    if out_val is None:
        raise SystemExit("score-fasta: internal error: --out-tsv is missing from parsed args")

    out_path = Path(out_val)

    if str(out_path) == "-":
        out_fh = sys.stdout
    else:
       
        if out_path.suffix == "":
            out_path.mkdir(parents=True, exist_ok=True)
            out_path = out_path / "scored.tsv"
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)

        out_fh = open(out_path, "w", newline="")


    base_fields = ["input_id", "mode", "start", "end", "length", "rf_score", "pred_label"]
    mature_fields = [
        "mature_arm", "mature_start", "mature_end", "mature_len",
        "mature_score", "mature_seq", "mature_start_global", "mature_end_global"
    ]
    extra_fields = ["hairpin_seq"]

    fieldnames = base_fields + (mature_fields if args.predict_mature else []) + extra_fields
    writer = csv.DictWriter(out_fh, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()

    total_candidates = 0

    for seq_id, full_seq in records:
        # Normalize to RNA alphabet (T->U) and uppercase, to match training expectations
        full_seq_norm = _normalize_rna(full_seq)

        results = clf.score_sequence_record(seq_id, full_seq_norm)

        if not results:
            continue

        # Prefer precursor-like lengths; avoid returning tiny fragments for long inputs
        MIN_LEN = 60
        # If user set a small max-hairpin-len, still allow reasonable precursors
        MAX_LEN = max(int(getattr(args, "max_hairpin_len", 120)), 220)

        def _rec_len(r):
            try:
                return int(r.get("length", int(r["end"]) - int(r["start"])))
            except Exception:
                return 0

        def _rec_score(r):
            try:
                return float(r.get("rf_score", "nan"))
            except Exception:
                return float("-inf")

        filtered = [r for r in results if MIN_LEN <= _rec_len(r) <= MAX_LEN]
        use = filtered if filtered else results

        # Choose best by RF score, then by length (tie-break)
        best = max(use, key=lambda r: (_rec_score(r), _rec_len(r)))

        rec = best

        try:
            s = int(rec["start"])
            e = int(rec["end"])
            rec["hairpin_seq"] = full_seq_norm[s:e]
        except Exception:
            s, e = 0, len(full_seq_norm)
            rec["start"] = s
            rec["end"] = e
            rec["length"] = len(full_seq_norm)
            rec["hairpin_seq"] = full_seq_norm

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
                    if pred is not None:
                        rec["mature_arm"] = pred["arm"]
                        rec["mature_start"] = int(pred["start"])
                        rec["mature_end"] = int(pred["end"])
                        rec["mature_len"] = int(pred["length"])
                        rec["mature_score"] = f"{pred['score']:.6f}"
                        rec["mature_seq"] = pred["mature_seq"]
                        rec["mature_start_global"] = s + int(pred["start"])
                        rec["mature_end_global"] = s + int(pred["end"])

        writer.writerow(rec)
        total_candidates += 1

    if out_fh is not sys.stdout:
        out_fh.close()

    print(
        f"[score-fasta] Wrote {total_candidates} candidate records from "
        f"{len(records)} input sequences to {('-' if str(out_path) == '-' else out_path)}",
        file=sys.stderr,
    )
    return 0

def cmd_predict_mature(args: argparse.Namespace) -> int:
    from .mature_model import MatureRanker
    from Bio import SeqIO

    ranker = MatureRanker.load(args.mature_model)

    fasta_path = Path(args.fasta)
    if not fasta_path.exists():
        print(f"[predict-mature] ERROR: FASTA not found: {fasta_path}", file=sys.stderr)
        return 1

    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    if not records:
        print("[predict-mature] ERROR: FASTA has no sequences.", file=sys.stderr)
        return 1
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as out:
        out.write("id\tarm\tstart\tend\tlength\tscore\tmature_seq\n")
        for r in records:
            lengths = tuple(int(x) for x in args.lengths.split(",") if x.strip())
            pred = ranker.predict_top1(
                str(r.seq),
                rnafold_bin=args.rnafold_bin,
                lengths=lengths,
                max_per_arm=args.max_per_arm,
                min_paired_context=args.min_paired_context,
                loop_buffer=args.loop_buffer,
                fallback_loop_buffer=args.fallback_loop_buffer,
                fallback_max_per_arm=args.fallback_max_per_arm,
                fallback_min_paired_context=args.fallback_min_paired_context,
            )
            if pred is None:
                out.write(f"{r.id}\tNA\tNA\tNA\tNA\tNA\tNA\n")
            else:
                out.write(
                    f"{r.id}\t{pred['arm']}\t{pred['start']}\t{pred['end']}\t{pred['length']}\t{pred['score']:.6f}\t{pred['mature_seq']}\n"
                )

    print(f"[predict-mature] Wrote predictions to {out_path}", file=sys.stderr)
    return 0


def cmd_validate_seqonly(args: argparse.Namespace) -> int:
    """
    Validate sequence-only inputs with two-stage validation.
    
    Stage A: Mature-centric duplex evidence (tolerates boundary truncation)
    Stage B: Precursor-centric RF scoring on extracted hairpin
    """
    from .seqonly_validate import (
        run_stage_a, run_stage_b, compute_verdict,
        parse_coords_from_header, ValidationResult
    )
    from .features import run_rnafold
    from .classifier import load_rf_model
    from Bio import SeqIO
    
    # Validate inputs
    fasta_path = Path(args.fasta)
    if not fasta_path.exists():
        print(f"[validate-seqonly] ERROR: FASTA not found: {fasta_path}", file=sys.stderr)
        return 1
    
    model_path = Path(args.premirna_model)
    if not model_path.exists():
        print(f"[validate-seqonly] ERROR: Model not found: {model_path}", file=sys.stderr)
        return 1
    
    # Load models
    print(f"[validate-seqonly] Loading pre-miRNA model from {model_path}...", file=sys.stderr)
    model_info = load_rf_model(model_path)
    # Store model_path for later use in run_stage_b
    model_info.model_path = model_path
    tier2_enabled = model_info.tier2_enabled
    
    mature_model = None
    if args.mature_model:
        mature_model_path = Path(args.mature_model)
        if mature_model_path.exists():
            from .mature_model import MatureRanker
            print(f"[validate-seqonly] Loading mature model from {mature_model_path}...", file=sys.stderr)
            mature_model = MatureRanker.load(str(mature_model_path))
        else:
            print(f"[validate-seqonly] WARNING: Mature model not found: {mature_model_path}", file=sys.stderr)
            print(f"[validate-seqonly] Using heuristic mode for Stage A", file=sys.stderr)
    
    # Load genome if provided
    genome_seqs = {}
    if args.genome:
        genome_path = Path(args.genome)
        if genome_path.exists():
            print(f"[validate-seqonly] Loading genome from {genome_path}...", file=sys.stderr)
            for rec in SeqIO.parse(str(genome_path), "fasta"):
                genome_seqs[rec.id] = str(rec.seq).upper().replace("T", "U")
        else:
            print(f"[validate-seqonly] WARNING: Genome not found: {genome_path}", file=sys.stderr)
    
    # Load input sequences
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    if not records:
        print("[validate-seqonly] ERROR: FASTA has no sequences.", file=sys.stderr)
        return 1
    
    print(f"[validate-seqonly] Processing {len(records)} sequences...", file=sys.stderr)
    
    # Prepare output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = []
    extracted_seqs = []  # For --emit-extracted-fasta
    
    # Process each sequence
    for rec in records:
        seq_id = rec.id
        seq = str(rec.seq).upper().replace("T", "U")
        
        # Parse coordinates
        coords = parse_coords_from_header(rec.description)
        coords_parsed = coords is not None
        
        # Fold once
        struct, mfe = run_rnafold(seq)
        
        # Stage A
        stageA = run_stage_a(seq, struct, mature_model)
        
        # Stage B - pass model_path for threshold loading
        stageB = run_stage_b(seq, struct, mfe, model_info, model_path, stageA, tier2_enabled)
        
        # Verdict
        verdict, recommendation = compute_verdict(
            stageB.precursor_score,
            stageA.mature_score,
            stageB.threshold_q95,
            stageB.threshold_q99,
            mature_min=args.mature_min
        )
        
        # Re-extraction if coords + genome available
        reext_seq_len = None
        reext_stageB_score = None
        
        if coords and genome_seqs:
            chrom, start, end = coords
            if chrom in genome_seqs:
                new_start = max(0, start - args.flank)
                new_end = end + args.flank
                reext_seq = genome_seqs[chrom][new_start:new_end]
                
                # Re-run validation
                reext_struct, reext_mfe = run_rnafold(reext_seq)
                reext_stageA = run_stage_a(reext_seq, reext_struct, mature_model)
                reext_stageB = run_stage_b(reext_seq, reext_struct, reext_mfe, model_info, model_path, reext_stageA, tier2_enabled)
                
                reext_seq_len = len(reext_seq)
                reext_stageB_score = reext_stageB.precursor_score
                
                # Update verdict if re-extracted score is better
                if reext_stageB_score > stageB.precursor_score:
                    verdict, recommendation = compute_verdict(
                        reext_stageB_score,
                        reext_stageA.mature_score,
                        reext_stageB.threshold_q95,
                        reext_stageB.threshold_q99,
                        mature_min=args.mature_min
                    )
                    recommendation = f"[Re-extracted] {recommendation}"
        elif coords and not genome_seqs:
            recommendation += " Provide --genome for standardized re-extraction."
        
        results.append({
            "input_id": seq_id,
            "seq_len": len(seq),
            "stageA_mature_score": f"{stageA.mature_score:.4f}",
            "stageA_5p_start": stageA.best_5p_start,
            "stageA_5p_end": stageA.best_5p_end,
            "stageA_3p_start": stageA.best_3p_start,
            "stageA_3p_end": stageA.best_3p_end,
            "stageA_mode": stageA.mode,
            "stageA_notes": stageA.notes,
            "stageB_precursor_score": f"{stageB.precursor_score:.4f}",
            "stageB_threshold_q90": stageB.threshold_q90,
            "stageB_threshold_q95": stageB.threshold_q95,
            "stageB_threshold_q99": stageB.threshold_q99,
            "extracted_start": stageB.extracted_start,
            "extracted_end": stageB.extracted_end,
            "verdict": verdict,
            "recommendation": recommendation,
            "coords_parsed": coords_parsed,
            "reext_seq_len": reext_seq_len if reext_seq_len else "",
            "reext_stageB_score": f"{reext_stageB_score:.4f}" if reext_stageB_score else "",
        })
        
        # Store extracted hairpin for optional output
        if args.emit_extracted_fasta:
            hp_seq = seq[stageB.extracted_start:stageB.extracted_end]
            extracted_seqs.append((seq_id, hp_seq))
    
    # Write output TSV
    with open(out_path, "w", newline="") as f:
        fieldnames = [
            "input_id", "seq_len",
            "stageA_mature_score", "stageA_5p_start", "stageA_5p_end",
            "stageA_3p_start", "stageA_3p_end", "stageA_mode", "stageA_notes",
            "stageB_precursor_score", "stageB_threshold_q90", "stageB_threshold_q95",
            "stageB_threshold_q99", "extracted_start", "extracted_end",
            "verdict", "recommendation", "coords_parsed",
            "reext_seq_len", "reext_stageB_score"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(results)
    
    print(f"[validate-seqonly] Wrote {len(results)} results to {out_path}", file=sys.stderr)
    
    # Write extracted FASTAs if requested
    if args.emit_extracted_fasta:
        extract_path = Path(args.emit_extracted_fasta)
        extract_path.parent.mkdir(parents=True, exist_ok=True)
        with open(extract_path, "w") as f:
            for seq_id, hp_seq in extracted_seqs:
                f.write(f">{seq_id}\n{hp_seq}\n")
        print(f"[validate-seqonly] Wrote extracted hairpins to {extract_path}", file=sys.stderr)
    
    return 0



def cmd_annotate_3d(args: argparse.Namespace) -> int:
    fasta_path = Path(args.fasta)
    if not fasta_path.exists():
        print(f"[annotate-3d] ERROR: FASTA not found: {fasta_path}", file=sys.stderr)
        return 1

    records = read_fasta(str(fasta_path))
    if not records:
        print("[annotate-3d] ERROR: FASTA has no sequences.", file=sys.stderr)
        return 1

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

def cmd_final_report(args):
    return stage14_report(
        sample_id=args.sample_id,
        final_candidates_tsv=Path(args.final_candidates_tsv),
        outdir=Path(args.outdir),
        rejects_paths=[Path(x) for x in (args.rejects or [])] if args.rejects else None,
        qc_json_paths=[Path(x) for x in (args.qc_json or [])] if args.qc_json else None,
        known_early_tsv=Path(args.known_early_tsv) if getattr(args, "known_early_tsv", None) else None,
    )




# -----------------------------
# sRNA-seq wrappers
# -----------------------------

def cmd_fastq_to_peaks(args: argparse.Namespace) -> int:
    return run_fastq_to_peaks(args)

def cmd_candidates_to_scored(args: argparse.Namespace) -> int:
    return run_candidates_to_scored(args)

def cmd_scored_to_peaks(args: argparse.Namespace) -> int:
    return run_scored_to_peaks(args)

def cmd_peaks_to_known(args: argparse.Namespace) -> int:
    return run_peaks_to_known(args)

def cmd_peaks_to_known_early(args: argparse.Namespace) -> int:
    from .peaks_to_known_early import run_peaks_to_known_early
    run_peaks_to_known_early(
        peaks_tsv=args.peaks_tsv,
        outdir=args.outdir,
        sample_id=args.sample_id,
        mirgenedb_gff=args.mirgenedb_gff,
        mirbase_gff=args.mirbase_gff,
        max_pad=args.max_pad,
        bin_size=args.bin_size,
    )
    return 0


def cmd_candidates_filter(args: argparse.Namespace) -> int:
    from .candidates_filter import run_candidates_filter
    run_candidates_filter(
        candidates_tsv=args.candidates_tsv,
        candidates_fa=args.candidates_fa,
        peaks_known_early_tsv=args.peaks_known_early_tsv,
        outdir=args.outdir,
    )
    return 0


def cmd_peaks_to_finalists(args: argparse.Namespace) -> int:
    return run_peaks_to_finalists(args)

def cmd_finalists_to_struct(args: argparse.Namespace) -> int:
    return run_finalists_to_struct(args)


# -----------------------------
# Parser
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mirpv-ng", description="miRPV-NG CLI")
    sub = p.add_subparsers(dest="command", required=True)

    # ---- Sequence-only ----
    sp = sub.add_parser("score-fasta", help="Score sequences from a FASTA file (sequence-only mode).")
    sp.add_argument("--fasta", required=True)
    sp.add_argument("--model", required=True)
    sp.add_argument("--species", default="hsa")
    sp.add_argument("--feature-set", choices=["core36", "extended"], default="extended")

    sp.add_argument("--max-hairpin-len", type=int, default=220)
    sp.add_argument("--max-seq-only-len", type=int, default=5000)
    sp.add_argument("--window-len", type=int, default=100)
    sp.add_argument("--step", type=int, default=20)
    sp.add_argument("--tier1-min-pairs", type=int, default=18)
    sp.add_argument("--tier1-min-mfe", type=float, default=-15.0)
    # tier2: None = use model's setting, True = force on, False = force off
    tier2_group = sp.add_mutually_exclusive_group()
    tier2_group.add_argument("--tier2", action="store_true", dest="tier2", default=None,
                              help="Force enable tier2 features (override model setting)")
    tier2_group.add_argument("--no-tier2", action="store_false", dest="tier2",
                              help="Force disable tier2 features (override model setting)")
    sp.add_argument("--out-tsv", default="-")
    sp.add_argument("--out", dest="out_tsv", default=None, help="Alias for --out-tsv")

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
    # Add parallelism arguments
    if HAS_PARALLEL:
        add_parallel_args(sp)
    sp.set_defaults(func=cmd_score_fasta)

    spm = sub.add_parser("predict-mature", help="Predict mature miRNA position on precursor FASTA using XGBRanker.")
    spm.add_argument("--mature-model", required=True)
    spm.add_argument("--fasta", required=True)
    spm.add_argument("--out", required=True)
    spm.add_argument("--rnafold-bin", default="RNAfold")
    spm.add_argument("--loop-buffer", type=int, default=0)
    spm.add_argument("--fallback-loop-buffer", type=int, default=10)
    spm.set_defaults(func=cmd_predict_mature)
    spm.add_argument("--lengths", type=str, default="21,22,23,24",
                 help="Comma-separated mature lengths to consider, e.g. 18,19,20,21,22,23,24")
    spm.add_argument("--max-per-arm", type=int, default=30)
    spm.add_argument("--min-paired-context", type=int, default=6)
    spm.add_argument("--fallback-max-per-arm", type=int, default=120)
    spm.add_argument("--fallback-min-paired-context", type=int, default=0)

    # validate-seqonly: Two-stage validation for sequence-only inputs
    svs = sub.add_parser("validate-seqonly",
                         help="Validate sequence-only inputs with two-stage validation (mature + precursor evidence)")
    svs.add_argument("--fasta", required=True,
                     help="Input FASTA file")
    svs.add_argument("--premirna-model", required=True,
                     help="Pre-miRNA RF model (.pkl)")
    svs.add_argument("--mature-model", default=None,
                     help="Mature miRNA model (.pkl) - if absent, Stage A uses heuristic fallback")
    svs.add_argument("--genome", default=None,
                     help="Genome FASTA for coordinate-based re-extraction (optional)")
    svs.add_argument("--out", required=True,
                     help="Output TSV file")
    svs.add_argument("--flank", type=int, default=30,
                     help="Flanking bases for re-extraction (default: 30)")
    svs.add_argument("--mature-min", type=float, default=0.6,
                     help="Minimum mature score threshold for HIGH_CONFIDENCE verdict (default: 0.6)")
    svs.add_argument("--emit-extracted-fasta", default=None,
                     help="Optional: write extracted hairpin regions to FASTA")
    svs.set_defaults(func=cmd_validate_seqonly)

    if _AF3_AVAILABLE:
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

    # ---- sRNA Stage 1–8 ----
    sps = sub.add_parser("fastq-to-peaks", help="sRNA-seq Stage 1–8: FASTQ -> peaks (+optional excision -> candidates)")
    sps.add_argument("--fastq", required=True)
    sps.add_argument("--sample-id", required=True)
    sps.add_argument("--outdir", required=True)

    sps.add_argument("--cutadapt", default="cutadapt")
    sps.add_argument("--bowtie", default="bowtie")
    sps.add_argument("--bowtie-index", required=True)
    sps.add_argument("--threads", type=int, default=8)

    sps.add_argument("--adapter", default=None)
    sps.add_argument("--max-multimaps", type=int, default=50)

    # Blocklist
    sps.add_argument("--blocklist-index", default=None)
    sps.add_argument("--blocklist-name", default="rfam")
    sps.add_argument("--blocklist-mismatches", type=int, default=0)
    sps.add_argument("--blocklist-max-align", type=int, default=1)

    # Islands / depth
    sps.add_argument("--island-gap", type=int, default=50)
    sps.add_argument("--min-depth", type=int, default=5)
    sps.add_argument("--min-cpm", type=float, default=0.5)

    # Signal processing
    sps.add_argument("--use-scipy", action="store_true", dest="use_scipy")
    sps.add_argument("--smooth-w", type=int, default=3, dest="smooth_w")
    sps.add_argument("--peak-distance", type=int, default=35, dest="peak_distance")
    sps.add_argument("--peak-micromerge", type=int, default=8, dest="peak_micromerge")
    sps.add_argument("--scipy-prominence", type=float, default=None, dest="scipy_prominence")
    sps.add_argument("--scipy-width-min", type=float, default=None, dest="scipy_width_min")
    sps.add_argument("--scipy-width-max", type=float, default=None, dest="scipy_width_max")
    sps.add_argument("--fallback-prom-frac", type=float, default=0.30, dest="fallback_prom_frac")

    # Support window + hard length filter
    sps.add_argument("--support-window", type=int, default=15)
    sps.add_argument("--hard-frac-20-24", type=float, default=0.30)
    sps.add_argument(
    "--disable-hard-length-prefilter",
    action="store_true",
    help="Disable hard length-distribution prefilter (BadLengthDist). Recommended for IonTorrent / PCR-biased sRNA-seq."
)

    # Anchor gate thresholds
    sps.add_argument("--anchor-unique-dominance", type=float, default=0.50)
    sps.add_argument("--anchor-unique-prec5p", type=float, default=0.70)
    sps.add_argument("--anchor-multi-dominance", type=float, default=0.60)
    sps.add_argument("--anchor-multi-prec5p", type=float, default=0.85)

    # Repeat gate
    sps.add_argument("--repeat-bed", default=None)
    sps.add_argument("--bedtools", default="bedtools")
    sps.add_argument("--repeat-multi-prec5p", type=float, default=0.90)
    sps.add_argument("--repeat-multi-dominance", type=float, default=0.70)

    # Excision (enables candidates.tsv/fa)
    sps.add_argument("--genome-fasta", default=None)
    sps.add_argument("--pads", nargs="+", type=int, default=[70, 100])
    sps.add_argument("--samtools", default="samtools")

    sps.set_defaults(func=cmd_fastq_to_peaks)

    # ---- Filter candidates (between Stage 1 and Stage 9) ----
    sf = sub.add_parser(
        "candidates-filter",
        help="Filter candidates.tsv/fa based on early known peak labels (skip Known-Confirmed)."
    )
    sf.add_argument("--candidates-tsv", required=True)
    sf.add_argument("--candidates-fa", required=True)
    sf.add_argument("--peaks-known-early-tsv", required=True)
    sf.add_argument("--outdir", required=True)
    sf.set_defaults(func=cmd_candidates_filter)



    # ---- Stage 9 ----
    s9 = sub.add_parser("candidates-to-scored", help="sRNA-seq Stage 9: candidates.tsv/fa -> RF score")
    s9.add_argument("--candidates-tsv", required=True)
    s9.add_argument("--candidates-fa", required=True)
    s9.add_argument("--model", required=True)
    s9.add_argument("--outdir", required=True)
    s9.add_argument("--sample-id", required=True)
    s9.add_argument("--species", default="hsa")
    s9.add_argument("--feature-set", choices=["core36", "extended"], default="extended")
    s9.add_argument("--tier2", action="store_true")

    # Must exist because candidates_to_scored passes them through
    s9.add_argument("--max-hairpin-len", type=int, default=120)
    s9.add_argument("--max-seq-only-len", type=int, default=5000)
    s9.add_argument("--window-len", type=int, default=100)
    s9.add_argument("--step", type=int, default=10)
    s9.add_argument("--tier1-min-pairs", type=int, default=18)
    s9.add_argument("--tier1-min-mfe", type=float, default=-15.0)
    s9.add_argument("--threads", type=int, default=18)

    s9.set_defaults(func=cmd_candidates_to_scored)

    # ---- Stage 9.5 ----
    s95 = sub.add_parser("scored-to-peaks", help="sRNA-seq Stage 9.5: candidates.scored.tsv -> peaks.scored.tsv")
    s95.add_argument("--scored-tsv", required=True)
    s95.add_argument("--outdir", required=True)
    s95.set_defaults(func=cmd_scored_to_peaks)


    # ---- Stage 10 (EARLY) ----
    s10e = sub.add_parser(
        "peaks-to-known-early",
        help="Early known labeling for peaks (fast): label peaks as Known-Confirmed / Known-Region / Unknown."
    )
    s10e.add_argument("--peaks-tsv", required=True)
    s10e.add_argument("--outdir", required=True)
    s10e.add_argument("--sample-id", default=None)

    # Known annotations (GFF/GFF3)
    s10e.add_argument("--mirgenedb-gff", default=None)
    s10e.add_argument("--mirbase-gff", default=None)

    # Query window around peak center: peak_center +/- max_pad
    s10e.add_argument("--max-pad", type=int, default=100)

    # Interval index bin size (performance knob)
    s10e.add_argument("--bin-size", type=int, default=4096)

    s10e.set_defaults(func=cmd_peaks_to_known_early)



    # ---- Stage 10 ----
    
    s10 = sub.add_parser("peaks-to-known", help="sRNA-seq Stage 10: known labeling (MirGeneDB + miRBase GFF/GFF3)")
    s10.add_argument("--peaks-tsv", required=True)
    s10.add_argument("--outdir", required=True)
    s10.add_argument("--sample-id", default=None)
    
    
    # Preferred inputs: GFF/GFF3
    s10.add_argument("--mirgenedb-gff", default=None)
    s10.add_argument("--mirbase-gff", default=None)
    
    # Optional fast-path inputs: BED
    # (If provided, script can skip parsing GFF and just intersect intervals.)
    s10.add_argument("--mirgenedb-precursor-bed", default=None)
    s10.add_argument("--mirgenedb-mature-bed", default=None)
    s10.add_argument("--mirbase-precursor-bed", default=None)
    s10.add_argument("--mirbase-mature-bed", default=None)

    # GFF feature types to consider (important: must exist because peaks_to_known.py reads these)
    s10.add_argument(
        "--mirgenedb-precursor-types",
        nargs="+",
        default=["pre_miRNA", "pre-miRNA", "miRNA_primary_transcript"],
    )
    s10.add_argument(
        "--mirgenedb-mature-types",
        nargs="+",
        default=["miRNA"],
    )
    s10.add_argument(
        "--mirbase-precursor-types",
        nargs="+",
        default=["miRNA_primary_transcript", "pre_miRNA", "pre-miRNA"],
    )
    s10.add_argument(
        "--mirbase-mature-types",
        nargs="+",
        default=["miRNA"],
    )


    # Tolerances / overlap logic
    s10.add_argument("--min-mature-overlap-bp", type=int, default=10)
    s10.add_argument("--mature-query-pad", type=int, default=2)
    s10.add_argument("--center-tol", type=int, default=2)

    s10.set_defaults(func=cmd_peaks_to_known)

    # ---- Stage 11 ----
    s11 = sub.add_parser("peaks-to-finalists", help="sRNA-seq Stage 11: merge known+scored peaks -> candidates.tsv + strict finalists")
    s11.add_argument("--peaks-scored-tsv", required=True)
    s11.add_argument("--peaks-known-tsv", required=True)
    s11.add_argument("--outdir", required=True)
    s11.add_argument("--sample-id", default=None)

    s11.add_argument("--known-atyp-min-rf", type=float, default=0.55)
    s11.add_argument("--novel-high-min-rf", type=float, default=0.65)
    s11.add_argument("--known-atyp-min-rf-repeat", type=float, default=0.65)
    s11.add_argument("--novel-high-min-rf-repeat", type=float, default=0.75)

    s11.add_argument("--repeat-block", nargs="*", default=["LINE", "SINE", "LTR", "DNA", "Satellite", "Simple_repeat", "Low_complexity"])
    s11.add_argument("--repeat-allow", nargs="*", default=[])

    s11.set_defaults(func=cmd_peaks_to_finalists)

    # ---- Stage 12 ----
    s12 = sub.add_parser("finalists-to-struct", help="sRNA-seq Stage 12: strict finalists -> RNAfold structures + FASTA")
    s12.add_argument("--strict-finalists-tsv", required=True)
    s12.add_argument("--candidates-fa", required=True)
    s12.add_argument("--outdir", required=True)
    s12.add_argument("--sample-id", default=None)
    s12.add_argument("--rnafold", default="RNAfold")
    s12.add_argument("--threads", type=int, default=4)
    s12.set_defaults(func=cmd_finalists_to_struct)
    
    
    # ----- Stage 13 -----
    s13 = sub.add_parser("final-candidates", help="Stage 13: merge structure finalists with mature predictions"
    )
    s13.add_argument("--candidates-struct-tsv", required=True)
    s13.add_argument("--mature-tsv", required=True)
    s13.add_argument("--outdir", required=True)
    s13.add_argument("--sample-id", required=True)
    s13.set_defaults(func=run_final_candidates)
    
    # ------ stage14_report ------
    s14 = sub.add_parser("final-report", help="Stage 14: final summary + merged rejects (auditable)")
    s14.add_argument("--final-candidates-tsv", required=True)
    s14.add_argument("--outdir", required=True)
    s14.add_argument("--sample-id", required=True)
    s14.add_argument("--rejects", nargs="*", default=None, help="One or more rejects.tsv paths to merge")
    s14.add_argument("--qc-json", nargs="*", default=None, help="One or more qc.json paths to embed")
    s14.add_argument("--known-early-tsv", default=None, help="Optional peaks.known_early.tsv to output known-only tables.")
    s14.set_defaults(func=cmd_final_report)

    
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())