# mirpv_ng/peaks_to_known_early.py
"""
Early known-miRNA labeling for peaks (pre-RF).

Goal:
- Label each peak as one of:
  - Known-Confirmed: peak center falls inside a known mature interval on the same strand (when known strand is defined)
  - Known-Region: overlaps a known mature/precursor region but not "confirmed"
  - Unknown: no overlap with known sets

Inputs:
- peaks TSV from fastq-to-peaks (must include at least: chrom, strand, peak_center0, sample_id or sample_id can be injected)
- miRNA annotations as GFF3 (MirGeneDB / miRBase) and/or BED6

Output:
- peaks.known_early.tsv

This module is self-contained (no dependency on other project modules).
"""

from __future__ import annotations

import csv
import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, DefaultDict
from collections import defaultdict


# ---------------------------
# interval model + indexing
# ---------------------------

@dataclass(frozen=True)
class Interval:
    chrom: str
    start0: int          # 0-based inclusive
    end0: int            # 0-based exclusive
    strand: str          # '+', '-', or '.'
    db: str              # 'MirGeneDB'/'miRBase'/'BED'
    feature: str         # 'mature'/'precursor'/'other'
    id: str              # stable ID or Name
    name: str            # human-readable (optional)


def _overlap_bp(a0: int, a1: int, b0: int, b1: int) -> int:
    x0 = max(a0, b0)
    x1 = min(a1, b1)
    return max(0, x1 - x0)


def _center_in_interval(center0: int, iv: Interval) -> bool:
    return iv.start0 <= center0 < iv.end0


def _strand_ok(query_strand: str, iv_strand: str) -> bool:
    # If known strand is '.', accept any. Else require match.
    if iv_strand in (".", "", None):
        return True
    return query_strand == iv_strand


def _bin_key(start0: int, bin_size: int) -> int:
    return start0 // bin_size


class IntervalIndex:
    """
    Simple binned interval index: chrom -> bin -> [intervals]
    """
    def __init__(self, bin_size: int = 4096):
        self.bin_size = int(bin_size)
        self._by_chrom: DefaultDict[str, DefaultDict[int, List[Interval]]] = defaultdict(lambda: defaultdict(list))

    def add(self, iv: Interval) -> None:
        b0 = _bin_key(iv.start0, self.bin_size)
        b1 = _bin_key(max(iv.end0 - 1, iv.start0), self.bin_size)
        for b in range(b0, b1 + 1):
            self._by_chrom[iv.chrom][b].append(iv)

    def query(self, chrom: str, start0: int, end0: int) -> List[Interval]:
        if chrom not in self._by_chrom:
            return []
        b0 = _bin_key(start0, self.bin_size)
        b1 = _bin_key(max(end0 - 1, start0), self.bin_size)
        out: List[Interval] = []
        bins = self._by_chrom[chrom]
        for b in range(b0, b1 + 1):
            out.extend(bins.get(b, []))
        return out


# ---------------------------
# parsing helpers
# ---------------------------

def _open_text(path: Path):
    p = str(path)
    if p.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _parse_gff3_attrs(attr_str: str) -> Dict[str, str]:
    d: Dict[str, str] = {}
    for part in attr_str.split(";"):
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            d[k.strip()] = v.strip()
    return d


def _feature_kind_from_gff3(ftype: str, attrs: Dict[str, str]) -> str:
    """
    Best-effort mapping of GFF3 types to "mature" vs "precursor".
    Different sources use different vocab. We stay conservative.
    """
    t = (ftype or "").lower()

    # direct
    if t in ("mirna", "mature_mirna", "mature-miRNA".lower(), "mature_miRNA".lower()):
        return "mature"
    if t in ("mirna_primary_transcript", "pri_mirna", "pre_mirna", "mirna_precursor", "stem_loop", "hairpin"):
        return "precursor"

    # attr hints
    so = (attrs.get("so_term") or attrs.get("Ontology_term") or "").lower()
    if "mature" in so:
        return "mature"
    if "precursor" in so or "stem" in so or "hairpin" in so:
        return "precursor"

    # heuristic: ID/Name often contains "-5p"/"-3p" for mature
    name = (attrs.get("Name") or attrs.get("ID") or "").lower()
    if name.endswith(("-5p", "-3p")):
        return "mature"

    return "other"


def load_gff3_intervals(gff_path: Path, db_name: str) -> List[Interval]:
    out: List[Interval] = []
    with _open_text(gff_path) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, source, ftype, start1, end1, score, strand, phase, attrs_s = parts
            try:
                s0 = int(start1) - 1
                e0 = int(end1)  # 1-based inclusive -> 0-based exclusive
            except Exception:
                continue
            strand = strand if strand in ("+", "-", ".") else "."
            attrs = _parse_gff3_attrs(attrs_s)
            feature = _feature_kind_from_gff3(ftype, attrs)
            _id = attrs.get("ID") or attrs.get("Name") or attrs.get("Alias") or f"{db_name}:{chrom}:{s0}-{e0}:{strand}"
            name = attrs.get("Name") or attrs.get("ID") or _id
            out.append(Interval(chrom=chrom, start0=s0, end0=e0, strand=strand,
                                db=db_name, feature=feature, id=_id, name=name))
    return out


def load_bed6_intervals(bed_path: Path, db_name: str, feature: str = "mature") -> List[Interval]:
    out: List[Interval] = []
    with _open_text(bed_path) as f:
        for line in f:
            if not line or line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                continue
            parts = re.split(r"\s+", line.strip())
            if len(parts) < 3:
                continue
            chrom = parts[0]
            try:
                s0 = int(parts[1])
                e0 = int(parts[2])
            except Exception:
                continue
            name = parts[3] if len(parts) >= 4 else f"{db_name}:{chrom}:{s0}-{e0}"
            strand = parts[5] if len(parts) >= 6 and parts[5] in ("+", "-") else "."
            out.append(Interval(chrom=chrom, start0=s0, end0=e0, strand=strand,
                                db=db_name, feature=feature, id=name, name=name))
    return out


# ---------------------------
# main logic
# ---------------------------

def build_indices(
    mirgenedb_gff: Optional[Path] = None,
    mirbase_gff: Optional[Path] = None,
    bed_files: Optional[List[Path]] = None,
    bin_size: int = 4096,
) -> Tuple[IntervalIndex, IntervalIndex]:
    """
    Returns: (mature_index, precursor_index)
    """
    mature = IntervalIndex(bin_size=bin_size)
    precursor = IntervalIndex(bin_size=bin_size)

    intervals: List[Interval] = []

    if mirgenedb_gff and mirgenedb_gff.exists():
        intervals.extend(load_gff3_intervals(mirgenedb_gff, "MirGeneDB"))
    if mirbase_gff and mirbase_gff.exists():
        intervals.extend(load_gff3_intervals(mirbase_gff, "miRBase"))
    if bed_files:
        for b in bed_files:
            if b and b.exists():
                intervals.extend(load_bed6_intervals(b, "BED", feature="mature"))

    for iv in intervals:
        if iv.feature == "mature":
            mature.add(iv)
        elif iv.feature == "precursor":
            precursor.add(iv)
        else:
            # unknown feature: treat as precursor-ish so we don't over-confirm
            precursor.add(iv)

    return mature, precursor


def _best_mature_hit(
    hits: List[Interval],
    q_chrom: str,
    q_strand: str,
    q_start0: int,
    q_end0: int,
    peak_center0: int,
) -> Optional[Tuple[Interval, int, bool, bool]]:
    """
    Choose the best mature overlap hit.
    Returns: (interval, overlap_bp, center_in, strand_ok)
    """
    best = None
    best_key = (-1, -1)  # (center_in, overlap)
    for iv in hits:
        if iv.chrom != q_chrom:
            continue
        ov = _overlap_bp(q_start0, q_end0, iv.start0, iv.end0)
        if ov <= 0:
            continue
        cin = _center_in_interval(peak_center0, iv)
        sok = _strand_ok(q_strand, iv.strand)
        # prioritize: center_in True > False, then overlap size
        key = (1 if cin else 0, ov)
        if key > best_key:
            best_key = key
            best = (iv, ov, cin, sok)
    return best


def label_peaks_early(
    peaks_tsv: Path,
    out_tsv: Path,
    mature_idx: IntervalIndex,
    precursor_idx: IntervalIndex,
    max_pad: int,
    sample_id_override: Optional[str] = None,
) -> None:
    """
    Read peaks.tsv and write peaks.known_early.tsv.
    """
    peaks_tsv = Path(peaks_tsv)
    out_tsv = Path(out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    with peaks_tsv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        hdr = r.fieldnames or []
        required = {"chrom", "strand", "peak_center0"}
        missing = required - set(hdr)
        if missing:
            raise RuntimeError(f"[peaks-to-known-early] peaks TSV missing columns: {sorted(missing)}")

        # optional columns we propagate if present
        pass_cols = [
            "sample_id", "chrom", "strand", "peak_center0",
            "island_id", "anchor_type", "peak_id",
            "depth_raw", "cpm", "len_mode", "dominance", "frac_20_24",
        ]
        pass_cols = [c for c in pass_cols if c in hdr]

        out_cols = pass_cols + [
            "known_early_status",  # Known-Confirmed / Known-Region / Unknown
            "known_db",
            "known_id",
            "known_feature",       # mature/precursor/other
            "overlap_bp",
            "center_in",
            "strand_ok",
        ]

        w = csv.DictWriter(out_tsv.open("w", encoding="utf-8", newline=""), delimiter="\t", fieldnames=out_cols)
        w.writeheader()

        for row in r:
            chrom = row["chrom"]
            strand = row["strand"] if row["strand"] in ("+", "-", ".") else "."
            try:
                peak_center0 = int(row["peak_center0"])
            except Exception:
                continue

            sample_id = sample_id_override or row.get("sample_id", "NA")

            q0 = max(0, peak_center0 - int(max_pad))
            q1 = peak_center0 + int(max_pad) + 1

            # gather candidate intervals
            mature_hits = [iv for iv in mature_idx.query(chrom, q0, q1) if _overlap_bp(q0, q1, iv.start0, iv.end0) > 0]
            precursor_hits = [iv for iv in precursor_idx.query(chrom, q0, q1) if _overlap_bp(q0, q1, iv.start0, iv.end0) > 0]

            # best mature
            best_m = _best_mature_hit(mature_hits, chrom, strand, q0, q1, peak_center0)

            status = "Unknown"
            known_db = ""
            known_id = ""
            known_feature = ""
            overlap_bp = "0"
            center_in = "0"
            strand_ok = "1"

            if best_m is not None:
                iv, ov, cin, sok = best_m
                # Confirm only if center-in AND strand OK (when strand known)
                if cin and sok:
                    status = "Known-Confirmed"
                else:
                    status = "Known-Region"
                known_db = iv.db
                known_id = iv.name or iv.id
                known_feature = "mature"
                overlap_bp = str(ov)
                center_in = "1" if cin else "0"
                strand_ok = "1" if sok else "0"
            elif precursor_hits:
                # overlaps precursor/other but no mature: still Known-Region
                # choose max-overlap precursor for bookkeeping
                best_iv = None
                best_ov = -1
                best_sok = True
                for iv in precursor_hits:
                    ov = _overlap_bp(q0, q1, iv.start0, iv.end0)
                    if ov > best_ov:
                        best_ov = ov
                        best_iv = iv
                        best_sok = _strand_ok(strand, iv.strand)
                status = "Known-Region"
                known_db = best_iv.db if best_iv else ""
                known_id = (best_iv.name or best_iv.id) if best_iv else ""
                known_feature = best_iv.feature if best_iv else "precursor"
                overlap_bp = str(best_ov if best_ov >= 0 else 0)
                center_in = "0"
                strand_ok = "1" if best_sok else "0"

            out_row: Dict[str, str] = {}
            for c in pass_cols:
                out_row[c] = row.get(c, "")
            out_row["sample_id"] = out_row.get("sample_id", sample_id)

            out_row.update({
                "known_early_status": status,
                "known_db": known_db,
                "known_id": known_id,
                "known_feature": known_feature,
                "overlap_bp": overlap_bp,
                "center_in": center_in,
                "strand_ok": strand_ok,
            })
            w.writerow(out_row)


def run_peaks_to_known_early(
    peaks_tsv: str,
    outdir: str,
    sample_id: Optional[str] = None,
    mirgenedb_gff: Optional[str] = None,
    mirbase_gff: Optional[str] = None,
    max_pad: int = 100,
    bin_size: int = 4096,
) -> Path:
    """
    Convenience function for CLI wrapper.
    Returns output TSV path.
    """
    peaks_tsv_p = Path(peaks_tsv)
    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    mature_idx, precursor_idx = build_indices(
        mirgenedb_gff=Path(mirgenedb_gff) if mirgenedb_gff else None,
        mirbase_gff=Path(mirbase_gff) if mirbase_gff else None,
        bed_files=None,
        bin_size=bin_size,
    )

    out_tsv = outdir_p / "peaks.known_early.tsv"
    label_peaks_early(
        peaks_tsv=peaks_tsv_p,
        out_tsv=out_tsv,
        mature_idx=mature_idx,
        precursor_idx=precursor_idx,
        max_pad=max_pad,
        sample_id_override=sample_id,
    )
    return out_tsv
