#!/usr/bin/env python3
"""
mirpv_ng.peaks_to_known

sRNA-seq mode — Stage 10: Known labeling (MirGeneDB + miRBase)

Input:
  - peaks.scored.tsv (from scored-to-peaks, Stage 9.5)
  - known precursor loci as BED6 (chrom, start0, end0, name, score, strand)
    for MirGeneDB + miRBase.

Output (in --outdir):
  - peaks.known.tsv        : peaks table + known annotations + known_status
  - known_hits.tsv         : best-hit summary per peak (auditable)
  - rejects.tsv            : stage10 decision audit (Unknown/Atypical/Confirmed reasons)
  - qc_stage10.json

Labels:
  - Known-Confirmed: strong same-strand overlap and peak_center is within known locus (configurable)
  - Known-Atypical : overlaps a known locus but strand mismatch and/or center outside, or weak overlap
  - Unknown        : no overlap with any known precursor

This stage does NOT filter peaks; it only labels them.
"""

from __future__ import annotations

import argparse
import csv
import json
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class KnownInterval:
    chrom: str
    start0: int
    end0: int
    name: str
    strand: str
    db: str  # "MirGeneDB" or "miRBase"


def parse_bed6(path: Path, db: str) -> List[KnownInterval]:
    intervals: List[KnownInterval] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            chrom = parts[0]
            try:
                start0 = int(parts[1])
                end0 = int(parts[2])
            except Exception:
                continue
            name = parts[3] if parts[3] else "NA"
            strand = parts[5] if parts[5] in {"+", "-"} else "."
            if end0 <= start0:
                continue
            intervals.append(KnownInterval(chrom=chrom, start0=start0, end0=end0, name=name, strand=strand, db=db))
    return intervals


def build_index(intervals: List[KnownInterval]) -> Dict[str, Tuple[List[int], List[KnownInterval]]]:
    """
    For each chrom: store (starts_sorted, intervals_sorted_by_start).
    Query via bisect and local scan. Known sets are small (few 1000s), this is fast.
    """
    by_chr: Dict[str, List[KnownInterval]] = {}
    for iv in intervals:
        by_chr.setdefault(iv.chrom, []).append(iv)

    idx: Dict[str, Tuple[List[int], List[KnownInterval]]] = {}
    for chrom, arr in by_chr.items():
        arr.sort(key=lambda x: x.start0)
        starts = [x.start0 for x in arr]
        idx[chrom] = (starts, arr)
    return idx


def overlap_bp(a0: int, a1: int, b0: int, b1: int) -> int:
    lo = max(a0, b0)
    hi = min(a1, b1)
    return max(0, hi - lo)


def best_hit_for_peak(
    chrom: str,
    q_start0: int,
    q_end0: int,
    q_strand: str,
    peak_center0: int,
    idx: Dict[str, Tuple[List[int], List[KnownInterval]]],
    min_any_overlap_bp: int,
    prefer_db_order: Tuple[str, str] = ("MirGeneDB", "miRBase"),
) -> Tuple[Optional[KnownInterval], Optional[KnownInterval], List[Tuple[KnownInterval, int]]]:
    """
    Returns:
      (best_same_strand, best_any_strand, overlaps_list)

    overlaps_list contains all intervals with overlap >= min_any_overlap_bp (with their overlap bp).
    """
    if chrom not in idx:
        return None, None, []

    starts, arr = idx[chrom]
    # find first interval with start > q_end0, scan neighbors around insertion point
    i = bisect_left(starts, q_start0)

    cand: List[Tuple[KnownInterval, int]] = []

    # scan left a bit (intervals might start before q_start0 but overlap)
    left = max(0, i - 200)
    right = min(len(arr), i + 200)

    # expand right until interval.start0 > q_end0 (safe break), but cap scans
    for j in range(left, right):
        iv = arr[j]
        if iv.start0 > q_end0 and j > i:
            break
        ov = overlap_bp(q_start0, q_end0, iv.start0, iv.end0)
        if ov >= min_any_overlap_bp:
            cand.append((iv, ov))

    if not cand:
        return None, None, []

    def db_rank(db: str) -> int:
        try:
            return prefer_db_order.index(db)
        except ValueError:
            return len(prefer_db_order)

    # choose best by overlap, then db preference, then center distance (smaller is better)
    def score_key(item: Tuple[KnownInterval, int]) -> Tuple[int, int, int]:
        iv, ov = item
        center_dist = 0 if (iv.start0 <= peak_center0 < iv.end0) else min(abs(peak_center0 - iv.start0), abs(peak_center0 - (iv.end0 - 1)))
        return (ov, -db_rank(iv.db), -center_dist)  # ov high, db earlier, center inside

    best_any = max(cand, key=score_key)[0]

    same = [x for x in cand if x[0].strand == q_strand and q_strand in {"+", "-"}]
    best_same = max(same, key=score_key)[0] if same else None

    return best_same, best_any, cand


def infer_query_interval(row: Dict[str, str]) -> Tuple[int, int, str]:
    """
    For overlap, prefer best_hp_start0/end0 if present; else best_start0/end0; else (center±best_pad).
    """
    strand = row.get("strand", row.get("peak_strand", row.get("best_strand", ".")))
    try:
        strand = strand if strand in {"+", "-"} else "."
    except Exception:
        strand = "."

    # 1) hairpin interval if available
    if "best_hp_start0" in row and "best_hp_end0" in row:
        try:
            s = int(row["best_hp_start0"])
            e = int(row["best_hp_end0"])
            if e > s:
                return s, e, strand
        except Exception:
            pass

    # 2) best_start0/end0 if available
    if "best_start0" in row and "best_end0" in row:
        try:
            s = int(row["best_start0"])
            e = int(row["best_end0"])
            if e > s:
                return s, e, strand
        except Exception:
            pass

    # 3) fallback from center + pad
    try:
        c = int(row["peak_center0"])
    except Exception:
        c = int(row.get("peak_center", "0"))
    try:
        pad = int(row.get("best_pad", "70"))
    except Exception:
        pad = 70

    return max(0, c - pad), c + pad + 1, strand


def write_reject(rejects_path: Path, sample_id: str, peak_id: str, status: str, reason: str, details: str) -> None:
    # Stage 10 is not a filter; we still log decisions as audit trail in rejects.tsv format.
    # Keep it compatible with your existing rejects schema: tab-separated, with "stage" and "reason".
    with rejects_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([sample_id, "KNOWN_GATE", ".", ".", ".", ".", peak_id, status, reason, details])


def run_peaks_to_known(args: argparse.Namespace) -> int:
    peaks_tsv = Path(args.peaks_tsv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_peaks = outdir / "peaks.known.tsv"
    out_hits = outdir / "known_hits.tsv"
    out_rejects = outdir / "rejects.tsv"
    out_qc = outdir / "qc_stage10.json"

    # Prepare rejects header (lightweight; stage10 has different fields than Stage1–8)
    if not out_rejects.exists():
        with out_rejects.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["sample_id", "stage", "chrom", "start", "end", "strand", "item_id", "status", "reason", "details"])

    # Load known annotations
    known: List[KnownInterval] = []
    if args.mirgenedb_bed:
        known.extend(parse_bed6(Path(args.mirgenedb_bed), db="MirGeneDB"))
    if args.mirbase_bed:
        known.extend(parse_bed6(Path(args.mirbase_bed), db="miRBase"))

    if not known:
        raise RuntimeError("No known annotations loaded. Provide --mirgenedb-bed and/or --mirbase-bed")

    idx = build_index(known)

    # Read peaks table
    with peaks_tsv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        fieldnames = list(r.fieldnames or [])
        if "peak_id" not in fieldnames:
            raise RuntimeError("Input peaks TSV must contain column: peak_id")

        # Output columns appended
        add_cols = [
            "known_status",
            "known_db",
            "known_id",
            "known_strand",
            "known_overlap_bp",
            "known_overlap_frac_peak",
            "known_overlap_frac_known",
            "known_center_in_known",
            "known_center_dist",
            "known_reason",
        ]
        out_fields = fieldnames + add_cols

        with out_peaks.open("w", encoding="utf-8", newline="") as fo:
            w_out = csv.DictWriter(fo, fieldnames=out_fields, delimiter="\t")
            w_out.writeheader()

            with out_hits.open("w", encoding="utf-8", newline="") as fh:
                w_hit = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "peak_id",
                        "sample_id",
                        "chrom",
                        "strand",
                        "peak_center0",
                        "query_start0",
                        "query_end0",
                        "known_db",
                        "known_id",
                        "known_start0",
                        "known_end0",
                        "known_strand",
                        "overlap_bp",
                        "overlap_frac_peak",
                        "overlap_frac_known",
                        "center_in_known",
                        "center_dist",
                        "status",
                        "reason",
                    ],
                    delimiter="\t",
                )
                w_hit.writeheader()

                n_total = 0
                n_confirmed = 0
                n_atypical = 0
                n_unknown = 0

                for row in r:
                    n_total += 1
                    peak_id = row["peak_id"]
                    sample_id = row.get("sample_id", args.sample_id or "NA")
                    chrom = row.get("chrom", "NA")
                    strand = row.get("strand", ".")
                    peak_center0 = int(row.get("peak_center0", "0"))

                    q_start0, q_end0, q_strand = infer_query_interval(row)
                    q_len = max(1, q_end0 - q_start0)

                    best_same, best_any, overlaps = best_hit_for_peak(
                        chrom=chrom,
                        q_start0=q_start0,
                        q_end0=q_end0,
                        q_strand=q_strand,
                        peak_center0=peak_center0,
                        idx=idx,
                        min_any_overlap_bp=args.min_any_overlap_bp,
                        prefer_db_order=("MirGeneDB", "miRBase"),
                    )

                    status = "Unknown"
                    reason = "NoOverlap"
                    chosen = None

                    # choose hit
                    if best_same is not None:
                        chosen = best_same
                        reason = "SameStrandOverlap"
                    elif best_any is not None:
                        chosen = best_any
                        reason = "AnyStrandOverlap"

                    if chosen is None:
                        n_unknown += 1
                        row.update({
                            "known_status": status,
                            "known_db": "NA",
                            "known_id": "NA",
                            "known_strand": "NA",
                            "known_overlap_bp": "0",
                            "known_overlap_frac_peak": "0.0",
                            "known_overlap_frac_known": "0.0",
                            "known_center_in_known": "0",
                            "known_center_dist": "NA",
                            "known_reason": reason,
                        })
                        w_out.writerow(row)
                        write_reject(out_rejects, sample_id, peak_id, status, reason, f"chrom={chrom};q={q_start0}-{q_end0};strand={strand}")
                        continue

                    ov = overlap_bp(q_start0, q_end0, chosen.start0, chosen.end0)
                    known_len = max(1, chosen.end0 - chosen.start0)
                    frac_peak = ov / q_len
                    frac_known = ov / known_len
                    center_in = 1 if (chosen.start0 <= peak_center0 < chosen.end0) else 0
                    center_dist = 0 if center_in else min(abs(peak_center0 - chosen.start0), abs(peak_center0 - (chosen.end0 - 1)))

                    # classify
                    if (chosen.strand == q_strand and q_strand in {"+", "-"}
                        and ov >= args.min_confirm_overlap_bp
                        and frac_peak >= args.min_confirm_frac_peak
                        and center_in == 1):
                        status = "Known-Confirmed"
                        reason = "StrongSameStrand+CenterIn"
                        n_confirmed += 1
                    else:
                        status = "Known-Atypical"
                        # encode why
                        bits = []
                        if chosen.strand != q_strand:
                            bits.append("StrandMismatch")
                        if ov < args.min_confirm_overlap_bp:
                            bits.append("LowOverlapBP")
                        if frac_peak < args.min_confirm_frac_peak:
                            bits.append("LowOverlapFracPeak")
                        if center_in == 0:
                            bits.append("CenterOutside")
                        reason = ";".join(bits) if bits else "Atypical"
                        n_atypical += 1

                    row.update({
                        "known_status": status,
                        "known_db": chosen.db,
                        "known_id": chosen.name,
                        "known_strand": chosen.strand,
                        "known_overlap_bp": str(ov),
                        "known_overlap_frac_peak": f"{frac_peak:.6f}",
                        "known_overlap_frac_known": f"{frac_known:.6f}",
                        "known_center_in_known": str(center_in),
                        "known_center_dist": str(center_dist),
                        "known_reason": reason,
                    })
                    w_out.writerow(row)

                    w_hit.writerow({
                        "peak_id": peak_id,
                        "sample_id": sample_id,
                        "chrom": chrom,
                        "strand": strand,
                        "peak_center0": peak_center0,
                        "query_start0": q_start0,
                        "query_end0": q_end0,
                        "known_db": chosen.db,
                        "known_id": chosen.name,
                        "known_start0": chosen.start0,
                        "known_end0": chosen.end0,
                        "known_strand": chosen.strand,
                        "overlap_bp": ov,
                        "overlap_frac_peak": f"{frac_peak:.6f}",
                        "overlap_frac_known": f"{frac_known:.6f}",
                        "center_in_known": center_in,
                        "center_dist": center_dist,
                        "status": status,
                        "reason": reason,
                    })

                    write_reject(out_rejects, sample_id, peak_id, status, reason,
                                 f"db={chosen.db};id={chosen.name};ov={ov};fracP={frac_peak:.3f};center_in={center_in}")

                qc = {
                    "sample_id": args.sample_id or "NA",
                    "peaks_total": n_total,
                    "known_confirmed": n_confirmed,
                    "known_atypical": n_atypical,
                    "unknown": n_unknown,
                    "params": {
                        "mirgenedb_bed": args.mirgenedb_bed,
                        "mirbase_bed": args.mirbase_bed,
                        "min_any_overlap_bp": args.min_any_overlap_bp,
                        "min_confirm_overlap_bp": args.min_confirm_overlap_bp,
                        "min_confirm_frac_peak": args.min_confirm_frac_peak,
                        "require_center_in": True,
                    },
                    "outputs": {
                        "peaks_known_tsv": str(out_peaks),
                        "known_hits_tsv": str(out_hits),
                        "rejects_tsv": str(out_rejects),
                    }
                }
                out_qc.write_text(json.dumps(qc, indent=2), encoding="utf-8")

    print(f"[peaks-to-known] wrote: {out_peaks}")
    print(f"[peaks-to-known] hits:  {out_hits}")
    print(f"[peaks-to-known] rejects.tsv: {out_rejects}")
    print(f"[peaks-to-known] qc: {out_qc}")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Stage 10: annotate peaks as Known-Confirmed/Known-Atypical/Unknown.")
    ap.add_argument("--peaks-tsv", required=True, help="peaks.scored.tsv from scored-to-peaks")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sample-id", default=None)

    ap.add_argument("--mirgenedb-bed", required=True, help="MirGeneDB precursor loci BED6")
    ap.add_argument("--mirbase-bed", required=True, help="miRBase precursor loci BED6")

    ap.add_argument("--min-any-overlap-bp", type=int, default=1,
                    help="Minimum overlap bp to consider a known locus as an overlap (for Atypical).")
    ap.add_argument("--min-confirm-overlap-bp", type=int, default=20,
                    help="Minimum overlap bp for Known-Confirmed (same strand).")
    ap.add_argument("--min-confirm-frac-peak", type=float, default=0.20,
                    help="Minimum overlap fraction of peak query interval for Known-Confirmed.")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)
    return run_peaks_to_known(args)


if __name__ == "__main__":
    raise SystemExit(main())
