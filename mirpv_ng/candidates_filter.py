# mirpv_ng/candidates_filter.py
"""
Filter candidates.tsv / candidates.fa based on early-known peak labels.

Primary use-case:
- Skip RF/structure for "Known-Confirmed" peaks.
- Keep everything else (Known-Region + Unknown) for discovery / atypical processing.

Inputs:
- candidates.tsv (from fastq-to-peaks)
- candidates.fa  (from fastq-to-peaks)
- peaks.known_early.tsv (from peaks_to_known_early.py)

Outputs:
- candidates.filtered.tsv
- candidates.filtered.fa
- qc_filter.json

This module is self-contained.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


def _read_tsv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        hdr = r.fieldnames or []
        rows = [row for row in r]
    return hdr, rows


def _write_tsv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _read_fasta_ids(path: Path) -> Set[str]:
    ids: Set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        for line in f:
            if line.startswith(">"):
                ids.add(line[1:].strip().split()[0])
    return ids


def _filter_fasta(in_fa: Path, out_fa: Path, keep_ids: Set[str]) -> int:
    out_fa.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    write = False
    with in_fa.open("r", encoding="utf-8", newline="") as fin, out_fa.open("w", encoding="utf-8", newline="") as fout:
        for line in fin:
            if line.startswith(">"):
                seq_id = line[1:].strip().split()[0]
                write = seq_id in keep_ids
                if write:
                    kept += 1
                    fout.write(line)
            else:
                if write:
                    fout.write(line)
    return kept


def _peak_key(row: Dict[str, str]) -> str:
    """
    Construct a robust peak key that should match between peaks TSV and candidates TSV.
    We try multiple fallbacks. You can change this to your canonical identifiers once you confirm columns.

    Priority:
    1) peak_id if present
    2) sample_id|chrom|strand|peak_center0|anchor_type
    3) chrom|strand|peak_center0
    """
    if row.get("peak_id"):
        return f"peak_id:{row['peak_id']}"
    sample_id = row.get("sample_id", "NA")
    chrom = row.get("chrom", "NA")
    strand = row.get("strand", "NA")
    pc = row.get("peak_center0", row.get("peak_center", "NA"))
    anchor = row.get("anchor_type", "NA")
    if chrom != "NA" and strand != "NA" and pc != "NA":
        return f"{sample_id}|{chrom}|{strand}|{pc}|{anchor}"
    return f"{chrom}|{strand}|{pc}"


def build_skip_peaks(peaks_known_early_tsv: Path, skip_status: str = "Known-Confirmed") -> Set[str]:
    hdr, rows = _read_tsv(peaks_known_early_tsv)
    if "known_early_status" not in hdr:
        raise RuntimeError("[candidates-filter] peaks.known_early.tsv missing 'known_early_status' column")
    skip: Set[str] = set()
    for r in rows:
        if (r.get("known_early_status") or "").strip() == skip_status:
            skip.add(_peak_key(r))
    return skip


def filter_candidates(
    candidates_tsv: Path,
    candidates_fa: Path,
    peaks_known_early_tsv: Path,
    outdir: Path,
) -> Tuple[Path, Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)

    skip_peaks = build_skip_peaks(peaks_known_early_tsv)

    hdr, rows = _read_tsv(candidates_tsv)

    # candidates.tsv must have candidate_id; and should have info to map back to peak
    if "candidate_id" not in hdr:
        raise RuntimeError("[candidates-filter] candidates.tsv missing 'candidate_id' column")

    kept_rows: List[Dict[str, str]] = []
    dropped_rows: List[Dict[str, str]] = []

    for r in rows:
        pk = _peak_key(r)
        if pk in skip_peaks:
            dropped_rows.append(r)
        else:
            kept_rows.append(r)

    out_tsv = outdir / "candidates.filtered.tsv"
    _write_tsv(out_tsv, hdr, kept_rows)

    keep_ids = {r["candidate_id"] for r in kept_rows if r.get("candidate_id")}

    out_fa = outdir / "candidates.filtered.fa"
    kept_fa_n = _filter_fasta(candidates_fa, out_fa, keep_ids)

    qc = {
        "inputs": {
            "candidates_tsv": str(candidates_tsv),
            "candidates_fa": str(candidates_fa),
            "peaks_known_early_tsv": str(peaks_known_early_tsv),
        },
        "counts": {
            "candidates_in": len(rows),
            "candidates_kept": len(kept_rows),
            "candidates_dropped": len(dropped_rows),
            "fasta_records_kept": kept_fa_n,
            "skip_peaks_n": len(skip_peaks),
        },
        "notes": [
            "Dropped candidates whose peak key matched a Known-Confirmed peak in peaks.known_early.tsv.",
            "Peak key prefers peak_id when available; otherwise sample|chrom|strand|peak_center0|anchor_type.",
        ],
    }

    qc_path = outdir / "qc_filter.json"
    qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")

    return out_tsv, out_fa, qc_path


def run_candidates_filter(
    candidates_tsv: str,
    candidates_fa: str,
    peaks_known_early_tsv: str,
    outdir: str,
) -> Tuple[Path, Path, Path]:
    return filter_candidates(
        candidates_tsv=Path(candidates_tsv),
        candidates_fa=Path(candidates_fa),
        peaks_known_early_tsv=Path(peaks_known_early_tsv),
        outdir=Path(outdir),
    )
