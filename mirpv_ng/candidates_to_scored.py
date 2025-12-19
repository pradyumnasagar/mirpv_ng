from __future__ import annotations

"""
mirpv_ng/candidates_to_scored.py

sRNA-seq mode — Stage after harvesting (Peak+Pad) and sequence excision.

This stage scores each excised candidate window (Peak±pad) using the existing
geometry-aware hairpin classifier (RandomForest), and writes an augmented TSV.

Inputs:
  - candidates.tsv : metadata for each excised window
  - candidates.fa  : sequences for each excised window, already strand-oriented
  - RF model (.pkl)

Outputs (in outdir):
  - candidates.scored.tsv
  - qc_stage2.json
  - rejects.tsv (appends if exists; otherwise creates with header)

Auditing:
  - Any failure to score a candidate is recorded in rejects.tsv with stage=SCORE_CANDIDATE.

Coordinate conventions:
  - candidates.tsv stores genomic window [start0, end0) and strand.
  - sequences in candidates.fa are oriented 5'->3' relative to that strand:
      strand '+' => sequence == genome[start0:end0]
      strand '-' => sequence == revcomp(genome[start0:end0])
  - The classifier reports best hairpin coordinates in oriented-sequence space:
      local_hp_start, local_hp_end (0-based, end-exclusive)
    We convert those into genomic coordinates:
      '+' : hp_start0 = start0 + local_hp_start; hp_end0 = start0 + local_hp_end
      '-' : hp_start0 = end0   - local_hp_end;   hp_end0 = end0   - local_hp_start
"""

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .classifier import HairpinClassifier
from .features import read_fasta


REJECTS_HEADER = [
    "sample_id", "stage", "chrom", "start", "end", "strand", "island_id", "peak_center",
    "anchor_type", "reject_id", "reason", "metric", "value", "threshold", "action", "notes"
]


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


def _log_reject(
    rejects_path: Path,
    *,
    sample_id: str,
    stage: str,
    chrom: str,
    start: int,
    end: int,
    strand: str,
    island_id: str,
    peak_center: str,
    anchor_type: str,
    reject_id: str,
    reason: str,
    metric: str,
    value: str,
    threshold: str,
    action: str,
    notes: str = "",
) -> None:
    _ensure_rejects_header(rejects_path)
    with rejects_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            sample_id, stage, chrom, str(start), str(end), strand,
            island_id or "", peak_center or "", anchor_type or "",
            reject_id or "", reason, metric, value, threshold, action, notes
        ])


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


def run_candidates_to_scored(args: argparse.Namespace) -> int:
    candidates_tsv = Path(args.candidates_tsv)
    candidates_fa = Path(args.candidates_fa)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rejects_path = outdir / "rejects.tsv"
    _ensure_rejects_header(rejects_path)

    metas = _read_candidates_tsv(candidates_tsv)
    seqs = dict(read_fasta(str(candidates_fa)))

    missing_fa = [m.candidate_id for m in metas if m.candidate_id not in seqs]
    if missing_fa:
        for cid in missing_fa[:25]:
            _log_reject(
                rejects_path,
                sample_id=args.sample_id or "NA",
                stage="SCORE_CANDIDATE",
                chrom=".", start=-1, end=-1, strand=".",
                island_id="", peak_center="", anchor_type="",
                reject_id=cid,
                reason="MissingFASTARecord",
                metric="candidate_id",
                value=cid,
                threshold="present",
                action="Reject",
                notes=f"candidates_fa={candidates_fa}",
            )
        raise RuntimeError(f"{len(missing_fa)} candidate_ids missing from candidates.fa (logged first 25).")

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

    out_path = outdir / "candidates.scored.tsv"

    base_cols = list(metas[0].row.keys()) if metas else []
    score_cols = [
        "best_mode",
        "local_hp_start",
        "local_hp_end",
        "hp_start0",
        "hp_end0",
        "hp_len",
        "rf_score",
        "pred_label",
    ]
    fieldnames = base_cols + [c for c in score_cols if c not in base_cols]

    scored = 0
    rejected = 0

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for meta in metas:
            seq = seqs[meta.candidate_id].upper().replace("T", "U")
            results = clf.score_sequence_record(meta.candidate_id, seq)

            best = _pick_best_candidate(results)
            row = dict(meta.row)

            if best is None:
                rejected += 1
                _log_reject(
                    rejects_path,
                    sample_id=meta.sample_id,
                    stage="SCORE_CANDIDATE",
                    chrom=meta.chrom,
                    start=meta.start0,
                    end=meta.end0,
                    strand=meta.strand,
                    island_id=meta.island_id,
                    peak_center=str(meta.peak_center0),
                    anchor_type=meta.anchor_type,
                    reject_id=meta.candidate_id,
                    reason="NoHairpinFound",
                    metric="classifier_candidates",
                    value="0",
                    threshold=">=1",
                    action="Reject",
                    notes="scan_long_sequence returned empty",
                )
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
                writer.writerow(row)
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
            writer.writerow(row)
            scored += 1

    qc = {
        "stage": "candidates-to-scored",
        "sample_id": args.sample_id or (metas[0].sample_id if metas else "NA"),
        "candidates_total": len(metas),
        "scored_rows_written": scored + rejected,
        "scored_ok": scored,
        "scored_rejected": rejected,
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
            "window_len": args.window_len,
            "step": args.step,
            "tier1_min_pairs": args.tier1_min_pairs,
            "tier1_min_mfe": args.tier1_min_mfe,
        },
    }
    (outdir / "qc_stage2.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")

    print(f"[candidates-to-scored] Read candidates: {len(metas)}")
    print(f"[candidates-to-scored] Wrote: {out_path}")
    print(f"[candidates-to-scored] qc: {outdir / 'qc_stage2.json'}")
    print(f"[candidates-to-scored] rejects.tsv: {rejects_path}")
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
    return p


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)
    return run_candidates_to_scored(args)


if __name__ == "__main__":
    raise SystemExit(main())
