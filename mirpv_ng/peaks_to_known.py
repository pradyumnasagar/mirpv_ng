#!/usr/bin/env python3
"""
Stage 10 — Known labeling (MirGeneDB + miRBase) using genome annotations (GFF3).


Input:
- peaks.scored.tsv (from scored-to-peaks; one row per peak)
- MirGeneDB GFF3 (or BED6) + miRBase GFF3 (or BED6)

Output:
- peaks.known.tsv     (peaks + known columns)
- known_hits.tsv      (best mature + best precursor hit per peak; auditable)
- rejects.tsv         (auditable stage10 labeling reasons)
- qc_stage10.json
"""

from __future__ import annotations

import argparse
import csv
import json
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Interval:
    chrom: str
    start0: int
    end0: int
    strand: str
    name: str
    db: str
    feature: str  # "mature" or "precursor"


def _overlap_bp(a0: int, a1: int, b0: int, b1: int) -> int:
    lo = max(a0, b0)
    hi = min(a1, b1)
    return max(0, hi - lo)


def _parse_attrs(attr_str: str) -> Dict[str, str]:
    # GFF3 attributes: key=value;key=value
    out: Dict[str, str] = {}
    for item in attr_str.split(";"):
        item = item.strip()
        if not item or "=" not in item:
            continue
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def load_bed6(path: Path, db: str, feature: str) -> List[Interval]:
    ivs: List[Interval] = []
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
            name = parts[3] or "NA"
            strand = parts[5] if parts[5] in {"+", "-"} else "."
            if end0 <= start0:
                continue
            ivs.append(Interval(chrom, start0, end0, strand, name, db, feature))
    return ivs


def load_gff3(
    path: Path,
    db: str,
    precursor_types: Tuple[str, ...],
    mature_types: Tuple[str, ...],
    name_keys: Tuple[str, ...] = ("Name", "ID", "Alias"),
) -> Tuple[List[Interval], List[Interval]]:
    prec: List[Interval] = []
    mat: List[Interval] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, source, ftype, start1, end1, score, strand, phase, attrs = parts
            if strand not in {"+", "-"}:
                strand = "."
            try:
                start0 = int(start1) - 1
                end0 = int(end1)
            except Exception:
                continue
            if end0 <= start0:
                continue

            ftype_l = ftype.strip()
            if (ftype_l not in precursor_types) and (ftype_l not in mature_types):
                continue

            ad = _parse_attrs(attrs)
            name = None
            for k in name_keys:
                if k in ad and ad[k]:
                    name = ad[k]
                    break
            if not name:
                # last resort: try Parent / Derives_from
                name = ad.get("Parent") or ad.get("Derives_from") or "NA"

            if ftype_l in precursor_types:
                prec.append(Interval(chrom, start0, end0, strand, name, db, "precursor"))
            elif ftype_l in mature_types:
                mat.append(Interval(chrom, start0, end0, strand, name, db, "mature"))

    return prec, mat


def build_index(intervals: List[Interval]) -> Dict[str, Tuple[List[int], List[Interval]]]:
    by_chr: Dict[str, List[Interval]] = {}
    for iv in intervals:
        by_chr.setdefault(iv.chrom, []).append(iv)
    idx: Dict[str, Tuple[List[int], List[Interval]]] = {}
    for chrom, arr in by_chr.items():
        arr.sort(key=lambda x: x.start0)
        idx[chrom] = ([x.start0 for x in arr], arr)
    return idx


def _query_candidates(idx: Dict[str, Tuple[List[int], List[Interval]]], chrom: str, q0: int, q1: int) -> List[Interval]:
    if chrom not in idx:
        return []
    starts, arr = idx[chrom]
    i = bisect_left(starts, q0)

    # local scan window; annotations are not huge
    left = max(0, i - 300)
    right = min(len(arr), i + 300)

    hits: List[Interval] = []
    for j in range(left, right):
        iv = arr[j]
        if iv.start0 > q1 and j > i:
            break
        if _overlap_bp(q0, q1, iv.start0, iv.end0) > 0:
            hits.append(iv)
    return hits


def infer_peak_query_interval(row: Dict[str, str]) -> Tuple[int, int, str]:
    """
    For overlap tests, prefer best_hp_start0/end0 if present, else best_start0/end0,
    else reconstruct from peak_center0 +/- best_pad.
    """
    strand = row.get("strand", ".")
    if strand not in {"+", "-"}:
        strand = "."

    for a, b in (("best_hp_start0", "best_hp_end0"), ("best_start0", "best_end0")):
        if a in row and b in row:
            try:
                s = int(row[a])
                e = int(row[b])
                if e > s:
                    return s, e, strand
            except Exception:
                pass

    c = int(row.get("peak_center0", "0"))
    pad = int(row.get("best_pad", "70"))
    return max(0, c - pad), c + pad + 1, strand


def best_hit(
    hits: List[Interval],
    q0: int,
    q1: int,
    peak_center0: int,
    q_strand: str,
    prefer_db: Tuple[str, ...],
) -> Tuple[Optional[Interval], int, float, int]:
    """
    Pick best by overlap bp, then db preference, then center-in.
    Returns (interval, overlap_bp, overlap_frac_of_query, center_in)
    """
    if not hits:
        return None, 0, 0.0, 0

    qlen = max(1, q1 - q0)

    def db_rank(db: str) -> int:
        try:
            return prefer_db.index(db)
        except ValueError:
            return len(prefer_db)

    best_iv = None
    best_key = None
    best_ov = 0
    best_center = 0

    for iv in hits:
        ov = _overlap_bp(q0, q1, iv.start0, iv.end0)
        center_in = 1 if (iv.start0 <= peak_center0 < iv.end0) else 0
        # penalize strand mismatch gently (still keep as info)
        strand_ok = 1 if (iv.strand == q_strand and q_strand in {"+", "-"}) else 0
        key = (ov, strand_ok, -db_rank(iv.db), center_in)
        if best_key is None or key > best_key:
            best_key = key
            best_iv = iv
            best_ov = ov
            best_center = center_in

    return best_iv, best_ov, best_ov / qlen, best_center


def ensure_rejects_header(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["sample_id", "stage", "chrom", "start", "end", "strand", "item_id", "status", "reason", "details"])


def append_reject(path: Path, sample_id: str, chrom: str, start: int, end: int, strand: str,
                  peak_id: str, status: str, reason: str, details: str) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([sample_id, "KNOWN_GATE", chrom, start, end, strand, peak_id, status, reason, details])


def run_peaks_to_known(args: argparse.Namespace) -> int:
    peaks_tsv = Path(args.peaks_tsv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_peaks = outdir / "peaks.known.tsv"
    out_hits = outdir / "known_hits.tsv"
    out_rejects = outdir / "rejects.tsv"
    out_qc = outdir / "qc_stage10.json"

    ensure_rejects_header(out_rejects)

    # Load annotations
    intervals_prec: List[Interval] = []
    intervals_mat: List[Interval] = []

    prefer_db = ("MirGeneDB", "miRBase")

    # MirGeneDB
    if args.mirgenedb_gff:
        p, m = load_gff3(
            Path(args.mirgenedb_gff),
            db="MirGeneDB",
            precursor_types=tuple(args.mirgenedb_precursor_types),
            mature_types=tuple(args.mirgenedb_mature_types),
        )
        intervals_prec.extend(p)
        intervals_mat.extend(m)
    if args.mirgenedb_precursor_bed:
        intervals_prec.extend(load_bed6(Path(args.mirgenedb_precursor_bed), db="MirGeneDB", feature="precursor"))
    if args.mirgenedb_mature_bed:
        intervals_mat.extend(load_bed6(Path(args.mirgenedb_mature_bed), db="MirGeneDB", feature="mature"))

    # miRBase
    if args.mirbase_gff:
        p, m = load_gff3(
            Path(args.mirbase_gff),
            db="miRBase",
            precursor_types=tuple(args.mirbase_precursor_types),
            mature_types=tuple(args.mirbase_mature_types),
        )
        intervals_prec.extend(p)
        intervals_mat.extend(m)
    if args.mirbase_precursor_bed:
        intervals_prec.extend(load_bed6(Path(args.mirbase_precursor_bed), db="miRBase", feature="precursor"))
    if args.mirbase_mature_bed:
        intervals_mat.extend(load_bed6(Path(args.mirbase_mature_bed), db="miRBase", feature="mature"))

    if not intervals_prec and not intervals_mat:
        raise RuntimeError("No known annotations loaded. Provide MirGeneDB/miRBase GFF and/or BED inputs.")

    idx_prec = build_index(intervals_prec) if intervals_prec else {}
    idx_mat = build_index(intervals_mat) if intervals_mat else {}

    # Read peaks and annotate
    with peaks_tsv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        if not r.fieldnames:
            raise RuntimeError("peaks TSV has no header")
        fieldnames = list(r.fieldnames)

        required = {"peak_id", "chrom", "strand", "peak_center0"}
        missing = [c for c in required if c not in fieldnames]
        if missing:
            raise RuntimeError(f"peaks TSV missing required columns: {missing}")

        add_cols = [
            "known_status",
            "known_basis",  # mature / precursor / none
            "known_db",
            "known_id",
            "known_feature",  # mature / precursor
            "known_start0",
            "known_end0",
            "known_strand",
            "known_overlap_bp",
            "known_overlap_frac_query",
            "known_center_in",
            "known_reason",
            # also keep best precursor separately for transparency
            "precursor_db",
            "precursor_id",
            "precursor_start0",
            "precursor_end0",
            "precursor_strand",
            "precursor_overlap_bp",
            "precursor_overlap_frac_query",
        ]
        out_fields = fieldnames + add_cols

        with out_peaks.open("w", encoding="utf-8", newline="") as fo, out_hits.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fo, fieldnames=out_fields, delimiter="\t")
            w.writeheader()

            wh = csv.DictWriter(
                fh,
                fieldnames=[
                    "peak_id", "sample_id", "chrom", "strand", "peak_center0",
                    "query_start0", "query_end0",
                    "best_mature_db", "best_mature_id", "best_mature_start0", "best_mature_end0", "best_mature_strand",
                    "mature_overlap_bp", "mature_overlap_frac_query", "mature_center_in",
                    "best_precursor_db", "best_precursor_id", "best_precursor_start0", "best_precursor_end0", "best_precursor_strand",
                    "precursor_overlap_bp", "precursor_overlap_frac_query",
                    "known_status", "known_basis", "known_reason",
                ],
                delimiter="\t",
            )
            wh.writeheader()

            n_total = 0
            n_confirm = 0
            n_atyp = 0
            n_unknown = 0

            for row in r:
                n_total += 1
                peak_id = row["peak_id"]
                sample_id = row.get("sample_id", args.sample_id or "NA")
                chrom = row["chrom"]
                strand = row["strand"] if row["strand"] in {"+", "-"} else "."
                peak_center0 = int(row["peak_center0"])

                q0, q1, q_strand = infer_peak_query_interval(row)

                # Expand query a bit for mature overlap (tolerance for isomiR shifts)
                q0m = max(0, q0 - args.mature_query_pad)
                q1m = q1 + args.mature_query_pad

                mat_hits = _query_candidates(idx_mat, chrom, q0m, q1m) if idx_mat else []
                prec_hits = _query_candidates(idx_prec, chrom, q0, q1) if idx_prec else []

                best_mat, mat_ov, mat_frac, mat_center = best_hit(mat_hits, q0m, q1m, peak_center0, q_strand, prefer_db)
                best_prec, prec_ov, prec_frac, _ = best_hit(prec_hits, q0, q1, peak_center0, q_strand, prefer_db)

                # Decide status
                status = "Unknown"
                basis = "none"
                reason = "NoKnownOverlap"
                chosen = None

                # Mature-first confirmation
                if best_mat is not None:
                    # require same strand and (center in mature interval +/- tolerance)
                    center_in_tol = 1 if (best_mat.start0 - args.center_tol <= peak_center0 < best_mat.end0 + args.center_tol) else 0
                    if best_mat.strand == q_strand and q_strand in {"+", "-"} and mat_ov >= args.min_mature_overlap_bp and center_in_tol == 1:
                        status = "Known-Confirmed"
                        basis = "mature"
                        reason = "MatureOverlap+CenterIn"
                        chosen = best_mat
                        n_confirm += 1
                    else:
                        status = "Known-Atypical"
                        basis = "mature"
                        bits = []
                        if best_mat.strand != q_strand:
                            bits.append("StrandMismatch")
                        if mat_ov < args.min_mature_overlap_bp:
                            bits.append("LowMatureOverlapBP")
                        if center_in_tol == 0:
                            bits.append("CenterOutsideMatureTol")
                        reason = ";".join(bits) if bits else "MatureAtypical"
                        chosen = best_mat
                        n_atyp += 1

                # If no mature hit, but precursor hit exists -> atypical (context)
                elif best_prec is not None:
                    status = "Known-Atypical"
                    basis = "precursor"
                    reason = "PrecursorOverlapOnly"
                    chosen = best_prec
                    n_atyp += 1
                else:
                    n_unknown += 1

                # Fill output columns
                if chosen is None:
                    row.update({
                        "known_status": status,
                        "known_basis": basis,
                        "known_db": "NA",
                        "known_id": "NA",
                        "known_feature": "NA",
                        "known_start0": "NA",
                        "known_end0": "NA",
                        "known_strand": "NA",
                        "known_overlap_bp": "0",
                        "known_overlap_frac_query": "0.0",
                        "known_center_in": "0",
                        "known_reason": reason,
                    })
                else:
                    # overlap shown depends on chosen feature
                    if chosen.feature == "mature":
                        ov = mat_ov
                        frac = mat_frac
                        center_in = 1 if (chosen.start0 <= peak_center0 < chosen.end0) else 0
                    else:
                        ov = prec_ov
                        frac = prec_frac
                        center_in = 1 if (chosen.start0 <= peak_center0 < chosen.end0) else 0

                    row.update({
                        "known_status": status,
                        "known_basis": basis,
                        "known_db": chosen.db,
                        "known_id": chosen.name,
                        "known_feature": chosen.feature,
                        "known_start0": str(chosen.start0),
                        "known_end0": str(chosen.end0),
                        "known_strand": chosen.strand,
                        "known_overlap_bp": str(ov),
                        "known_overlap_frac_query": f"{frac:.6f}",
                        "known_center_in": str(center_in),
                        "known_reason": reason,
                    })

                # Always emit best precursor context fields (even if Unknown)
                if best_prec is None:
                    row.update({
                        "precursor_db": "NA",
                        "precursor_id": "NA",
                        "precursor_start0": "NA",
                        "precursor_end0": "NA",
                        "precursor_strand": "NA",
                        "precursor_overlap_bp": "0",
                        "precursor_overlap_frac_query": "0.0",
                    })
                else:
                    row.update({
                        "precursor_db": best_prec.db,
                        "precursor_id": best_prec.name,
                        "precursor_start0": str(best_prec.start0),
                        "precursor_end0": str(best_prec.end0),
                        "precursor_strand": best_prec.strand,
                        "precursor_overlap_bp": str(prec_ov),
                        "precursor_overlap_frac_query": f"{prec_frac:.6f}",
                    })

                w.writerow(row)

                wh.writerow({
                    "peak_id": peak_id,
                    "sample_id": sample_id,
                    "chrom": chrom,
                    "strand": strand,
                    "peak_center0": peak_center0,
                    "query_start0": q0,
                    "query_end0": q1,
                    "best_mature_db": best_mat.db if best_mat else "NA",
                    "best_mature_id": best_mat.name if best_mat else "NA",
                    "best_mature_start0": best_mat.start0 if best_mat else "NA",
                    "best_mature_end0": best_mat.end0 if best_mat else "NA",
                    "best_mature_strand": best_mat.strand if best_mat else "NA",
                    "mature_overlap_bp": mat_ov,
                    "mature_overlap_frac_query": f"{mat_frac:.6f}",
                    "mature_center_in": mat_center,
                    "best_precursor_db": best_prec.db if best_prec else "NA",
                    "best_precursor_id": best_prec.name if best_prec else "NA",
                    "best_precursor_start0": best_prec.start0 if best_prec else "NA",
                    "best_precursor_end0": best_prec.end0 if best_prec else "NA",
                    "best_precursor_strand": best_prec.strand if best_prec else "NA",
                    "precursor_overlap_bp": prec_ov,
                    "precursor_overlap_frac_query": f"{prec_frac:.6f}",
                    "known_status": status,
                    "known_basis": basis,
                    "known_reason": reason,
                })

                # Audit trail (even Unknown gets a line — keeps everything explorable)
                append_reject(
                    out_rejects,
                    sample_id=sample_id,
                    chrom=chrom,
                    start=q0,
                    end=q1,
                    strand=strand,
                    peak_id=peak_id,
                    status=status,
                    reason=reason,
                    details=f"best_mature={best_mat.db+':'+best_mat.name if best_mat else 'NA'};best_prec={best_prec.db+':'+best_prec.name if best_prec else 'NA'}",
                )

            qc = {
                "sample_id": args.sample_id or "NA",
                "peaks_total": n_total,
                "known_confirmed": n_confirm,
                "known_atypical": n_atyp,
                "unknown": n_unknown,
                "inputs": {
                    "peaks_tsv": str(peaks_tsv),
                    "mirgenedb_gff": args.mirgenedb_gff,
                    "mirbase_gff": args.mirbase_gff,
                    "mirgenedb_precursor_bed": args.mirgenedb_precursor_bed,
                    "mirgenedb_mature_bed": args.mirgenedb_mature_bed,
                    "mirbase_precursor_bed": args.mirbase_precursor_bed,
                    "mirbase_mature_bed": args.mirbase_mature_bed,
                },
                "params": {
                    "min_mature_overlap_bp": args.min_mature_overlap_bp,
                    "center_tol": args.center_tol,
                    "mature_query_pad": args.mature_query_pad,
                    "mirgenedb_precursor_types": list(args.mirgenedb_precursor_types),
                    "mirgenedb_mature_types": list(args.mirgenedb_mature_types),
                    "mirbase_precursor_types": list(args.mirbase_precursor_types),
                    "mirbase_mature_types": list(args.mirbase_mature_types),
                },
                "outputs": {
                    "peaks_known_tsv": str(out_peaks),
                    "known_hits_tsv": str(out_hits),
                    "rejects_tsv": str(out_rejects),
                },
            }
            out_qc.write_text(json.dumps(qc, indent=2), encoding="utf-8")

    print(f"[peaks-to-known] peaks.known.tsv: {out_peaks}")
    print(f"[peaks-to-known] known_hits.tsv:  {out_hits}")
    print(f"[peaks-to-known] rejects.tsv:     {out_rejects}")
    print(f"[peaks-to-known] qc:              {out_qc}")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Stage 10: annotate peaks as known (mature-first).")
    ap.add_argument("--peaks-tsv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sample-id", default=None)

    # Prefer GFF inputs (native for miRBase); BED optional for power-users
    ap.add_argument("--mirgenedb-gff", default=None)
    ap.add_argument("--mirbase-gff", default=None)

    ap.add_argument("--mirgenedb-precursor-bed", default=None)
    ap.add_argument("--mirgenedb-mature-bed", default=None)
    ap.add_argument("--mirbase-precursor-bed", default=None)
    ap.add_argument("--mirbase-mature-bed", default=None)

    # Feature type names in GFF3 can differ across sources/versions.
    # Defaults are miRBase-typical. You can override on the command line if needed.
    ap.add_argument("--mirgenedb-precursor-types", nargs="+", default=["miRNA_primary_transcript", "pre_miRNA", "pre-miRNA"])
    ap.add_argument("--mirgenedb-mature-types", nargs="+", default=["miRNA", "mature_miRNA", "mature-miRNA"])
    ap.add_argument("--mirbase-precursor-types", nargs="+", default=["miRNA_primary_transcript"])
    ap.add_argument("--mirbase-mature-types", nargs="+", default=["miRNA"])

    # Mature confirmation knobs
    ap.add_argument("--min-mature-overlap-bp", type=int, default=10,
                    help="Minimum bp overlap with mature feature to consider for confirmation (after query padding).")
    ap.add_argument("--mature-query-pad", type=int, default=2,
                    help="Expand query interval by +/- this many nt when testing mature overlap (isomiR tolerance).")
    ap.add_argument("--center-tol", type=int, default=2,
                    help="Allow peak_center within mature interval +/- this tolerance for Known-Confirmed.")

    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    return run_peaks_to_known(args)


if __name__ == "__main__":
    raise SystemExit(main())
