from __future__ import annotations

"""
mirpv_ng/candidates_to_scored.py

sRNA-seq mode — Stage after harvesting (Peak+Pad) and sequence excision.
Uses Multiprocessing for parallel scoring.

Inputs:
  - candidates.tsv : metadata for each excised window
  - candidates.fa  : sequences for each excised window
  - RF model (.pkl)

Outputs:
  - candidates.scored.tsv
  - qc_stage2.json
  - rejects.tsv
"""

import argparse
import csv
import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from .classifier import HairpinClassifier
from .features import read_fasta


REJECTS_HEADER = [
    "sample_id", "stage", "chrom", "start", "end", "strand", "island_id", "peak_center",
    "anchor_type", "reject_id", "reason", "metric", "value", "threshold", "action", "notes"
]

# --- Global worker variable ---
_WORKER_CLF = None


@dataclass
class CandidateMeta:
    candidate_id: str
    sample_id: str
    chrom: str
    start0: int
    end0: int
    strand: str
    peak_center0: int
    pad: int
    island_id: str
    anchor_type: str
    depth_raw: int
    cpm: float
    len_mode: int
    frac_20_24: float
    dominance: float
    prec5p: float
    start_entropy: float
    repeat_class: str
    row: Dict[str, str]


def _ensure_rejects_header(rejects_path: Path) -> None:
    if rejects_path.exists() and rejects_path.stat().st_size > 0:
        return
    rejects_path.parent.mkdir(parents=True, exist_ok=True)
    with rejects_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(REJECTS_HEADER)


def _log_reject_direct(writer, meta: CandidateMeta, reason: str, notes: str = ""):
    """Helper to write reject row dictionary for the main process."""
    return {
        "sample_id": meta.sample_id,
        "stage": "SCORE_CANDIDATE",
        "chrom": meta.chrom,
        "start": str(meta.start0),
        "end": str(meta.end0),
        "strand": meta.strand,
        "island_id": meta.island_id or "",
        "peak_center": str(meta.peak_center0) or "",
        "anchor_type": meta.anchor_type or "",
        "reject_id": meta.candidate_id or "",
        "reason": reason,
        "metric": "classifier_candidates",
        "value": "0",
        "threshold": ">=1",
        "action": "Reject",
        "notes": notes
    }


def _read_candidates_tsv(path: Path) -> List[CandidateMeta]:
    metas: List[CandidateMeta] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = [
            "candidate_id", "sample_id", "chrom", "start0", "end0", "strand",
            "peak_center0", "pad", "island_id", "anchor_type",
            "depth_raw", "cpm", "len_mode", "frac_20_24", "dominance", "prec5p", "start_entropy",
            "repeat_class",
        ]
        # Allow missing columns if files are from older versions, but warn?
        # For now, strict check as per original code
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"candidates.tsv missing columns: {missing}")

        for row in reader:
            metas.append(CandidateMeta(
                candidate_id=row["candidate_id"],
                sample_id=row["sample_id"],
                chrom=row["chrom"],
                start0=int(row["start0"]),
                end0=int(row["end0"]),
                strand=row["strand"],
                peak_center0=int(row["peak_center0"]),
                pad=int(row["pad"]),
                island_id=row["island_id"],
                anchor_type=row["anchor_type"],
                depth_raw=int(float(row["depth_raw"])),
                cpm=float(row["cpm"]),
                len_mode=int(float(row["len_mode"])),
                frac_20_24=float(row["frac_20_24"]),
                dominance=float(row["dominance"]),
                prec5p=float(row["prec5p"]),
                start_entropy=float(row["start_entropy"]),
                repeat_class=row.get("repeat_class", "None") or "None",
                row=row,
            ))
    return metas


def _pick_best_candidate(cands: List[Dict]) -> Optional[Dict]:
    best = None
    best_score = -1.0
    for c in cands:
        try:
            s = float(c.get("rf_score", float("nan")))
        except Exception:
            continue
        if math.isnan(s):
            continue
        if s > best_score:
            best_score = s
            best = c
    return best


def _to_genomic_coords(meta: CandidateMeta, local_start: int, local_end: int) -> Tuple[int, int]:
    if meta.strand == "+":
        return meta.start0 + local_start, meta.start0 + local_end
    hp_start0 = meta.end0 - local_end
    hp_end0 = meta.end0 - local_start
    return hp_start0, hp_end0


# --- Worker Functions ---

def _init_worker(
    model_path: str,
    species: str,
    feature_set: str,
    max_hairpin_len: int,
    max_seq_only_len: int,
    window_len: int,
    step: int,
    tier1_min_pairs: int,
    tier1_min_mfe: float,
    tier2_enabled: bool
):
    """Initialize the classifier once per worker process."""
    global _WORKER_CLF
    _WORKER_CLF = HairpinClassifier(
        model_path=model_path,
        species=species,
        feature_set=feature_set,
        max_hairpin_len=max_hairpin_len,
        max_seq_only_len=max_seq_only_len,
        window_len=window_len,
        step=step,
        tier1_min_pairs=tier1_min_pairs,
        tier1_min_mfe=tier1_min_mfe,
        tier2_enabled=tier2_enabled,
    )


def _process_batch(batch_data: List[Tuple[CandidateMeta, str]]) -> Tuple[List[Dict], List[Dict]]:
    """
    Process a batch of (metadata, sequence) tuples.
    Returns (list_of_valid_rows, list_of_reject_dicts).
    """
    valid_rows = []
    reject_rows = []
    
    global _WORKER_CLF
    if _WORKER_CLF is None:
        raise RuntimeError("Worker classifier not initialized")

    for meta, seq in batch_data:
        # Pre-processing sequence (upper, T->U) done here to save transfer size? 
        # Or usually passed as string.
        seq_norm = seq.upper().replace("T", "U")
        results = _WORKER_CLF.score_sequence_record(meta.candidate_id, seq_norm)
        
        best = _pick_best_candidate(results)
        row = dict(meta.row)

        if best is None:
            # Reject logic
            rej = _log_reject_direct(None, meta, "NoHairpinFound", "scan_long_sequence returned empty")
            reject_rows.append(rej)
            
            # Write row with NA
            row.update({
                "best_mode": "NA",
                "local_hp_start": "NA",
                "local_hp_end": "NA",
                "hp_start0": "NA",
                "hp_end0": "NA",
                "hp_len": "NA",
                "rf_score": "NA",
                "pred_label": "0",
            })
            valid_rows.append(row)
            continue

        try:
            local_s = int(best.get("start", 0))
            local_e = int(best.get("end", 0))
        except Exception:
            local_s, local_e = 0, 0

        hp_start0, hp_end0 = _to_genomic_coords(meta, local_s, local_e)
        hp_len = max(0, hp_end0 - hp_start0)
        rf = float(best.get("rf_score", float("nan")))
        pred = int(best.get("pred_label", 0))

        row.update({
            "best_mode": str(best.get("mode", "NA")),
            "local_hp_start": local_s,
            "local_hp_end": local_e,
            "hp_start0": hp_start0,
            "hp_end0": hp_end0,
            "hp_len": hp_len,
            "rf_score": f"{rf:.6f}" if not math.isnan(rf) else "NA",
            "pred_label": str(pred),
        })
        valid_rows.append(row)

    return valid_rows, reject_rows


def run_candidates_to_scored(args: argparse.Namespace) -> int:
    candidates_tsv = Path(args.candidates_tsv)
    candidates_fa = Path(args.candidates_fa)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rejects_path = outdir / "rejects.tsv"
    _ensure_rejects_header(rejects_path)

    # 1. Read Inputs
    print(f"[candidates-to-scored] Reading inputs...")
    metas = _read_candidates_tsv(candidates_tsv)
    seqs = dict(read_fasta(str(candidates_fa)))

    # 2. Validation
    missing_fa = [m.candidate_id for m in metas if m.candidate_id not in seqs]
    if missing_fa:
        # Log first 25 missing
        with rejects_path.open("a", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            for cid in missing_fa[:25]:
                # Minimal construction to reuse existing log function logic manualy
                w.writerow([
                    args.sample_id or "NA", "SCORE_CANDIDATE", ".", "-1", "-1", ".", 
                    "", "", "", cid, "MissingFASTARecord", "candidate_id", cid, 
                    "present", "Reject", f"candidates_fa={candidates_fa}"
                ])
        raise RuntimeError(f"{len(missing_fa)} candidate_ids missing from candidates.fa")

    # 3. Prepare Batches
    BATCH_SIZE = 100
    tasks = []
    current_batch = []
    
    for meta in metas:
        s = seqs.get(meta.candidate_id, "")
        current_batch.append((meta, s))
        if len(current_batch) >= BATCH_SIZE:
            tasks.append(current_batch)
            current_batch = []
    if current_batch:
        tasks.append(current_batch)

    out_path = outdir / "candidates.scored.tsv"
    base_cols = list(metas[0].row.keys()) if metas else []
    score_cols = [
        "best_mode", "local_hp_start", "local_hp_end", 
        "hp_start0", "hp_end0", "hp_len", "rf_score", "pred_label"
    ]
    fieldnames = base_cols + [c for c in score_cols if c not in base_cols]

    scored_count = 0
    rejected_count = 0
    
    # 4. Run Parallel
    print(f"[candidates-to-scored] Processing {len(metas)} candidates with {args.threads} threads...")
    
    with out_path.open("w", encoding="utf-8", newline="") as f_out, \
         rejects_path.open("a", encoding="utf-8", newline="") as f_rej:
        
        writer = csv.DictWriter(f_out, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        
        rej_writer = csv.writer(f_rej, delimiter="\t")


        with ProcessPoolExecutor(
            max_workers=args.threads,
            initializer=_init_worker,
            initargs=(
                args.model, args.species, args.feature_set,
                args.max_hairpin_len, args.max_seq_only_len,
                args.window_len, args.step,
                args.tier1_min_pairs, args.tier1_min_mfe,
                args.tier2
            )
        ) as executor:
            
            futures = {executor.submit(_process_batch, batch): batch for batch in tasks}
            
            for future in as_completed(futures):
                try:
                    valid_rows, reject_dicts = future.result()
                    
                    # Write Results
                    for row in valid_rows:
                        writer.writerow(row)
                        scored_count += 1
                        # Check if it was a 'soft' reject (NA score) to count as rejected
                        if row.get("rf_score") == "NA":
                            rejected_count += 1
                    
                    # Write Rejects
                    for rj in reject_dicts:
                        if not rj.get("sample_id"):
                            rj["sample_id"] = args.sample_id or "NA"
                        rej_writer.writerow([rj.get(col, "") for col in REJECTS_HEADER])
                        
                except Exception as e:
                    print(f"[candidates-to-scored] Error in worker batch: {e}")
                    raise e

    # 5. QC
    qc = {
        "stage": "candidates-to-scored",
        "sample_id": args.sample_id or (metas[0].sample_id if metas else "NA"),
        "candidates_total": len(metas),
        "scored_rows_written": scored_count,
        # Note: 'scored_count' includes rows with NA scores.
        # 'rejected_count' tracks how many were NA.
        "scored_ok": scored_count - rejected_count,
        "scored_rejected": rejected_count,
        "out_tsv": str(out_path),
        "inputs": {
            "candidates_tsv": str(candidates_tsv),
            "candidates_fa": str(candidates_fa),
            "model": str(args.model),
        },
        "params": {
            "feature_set": args.feature_set,
            "tier2": bool(args.tier2),
            "max_hairpin_len": args.max_hairpin_len,
            "threads": args.threads
        },
    }
    (outdir / "qc_stage2.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")

    print(f"[candidates-to-scored] Done. Scored: {scored_count}. Rejected: {rejected_count}.")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="miRPV-NG sRNA-seq: score excised Peak±pad candidates with RF model.")
    p.add_argument("--candidates-tsv", required=True, help="Input candidates.tsv from fastq-to-peaks --genome-fasta")
    p.add_argument("--candidates-fa", required=True, help="Input candidates.fa from fastq-to-peaks --genome-fasta")
    p.add_argument("--model", required=True, help="RF model pickle (.pkl)")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--sample-id", default=None, help="Optional override for sample_id (used only in qc/rejects)")

    p.add_argument("--species", default="hsa")
    p.add_argument("--feature-set", choices=["core36", "extended"], default="extended")
    p.add_argument("--max-hairpin-len", type=int, default=120)
    p.add_argument("--max-seq-only-len", type=int, default=5000)
    p.add_argument("--window-len", type=int, default=100)
    p.add_argument("--step", type=int, default=20)
    p.add_argument("--tier1-min-pairs", type=int, default=18)
    p.add_argument("--tier1-min-mfe", type=float, default=-15.0)
    p.add_argument("--tier2", action="store_true", help="Enable Tier-2 soft-gated features during scoring")
    
    # NEW ARGUMENT
    p.add_argument("--threads", type=int, default=max(1, multiprocessing.cpu_count() - 1), help="Number of parallel threads")
    
    return p


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)
    return run_candidates_to_scored(args)


if __name__ == "__main__":
    raise SystemExit(main())