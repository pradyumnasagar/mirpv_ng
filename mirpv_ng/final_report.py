#!/usr/bin/env python3
"""
miRPV-NG sRNA-seq Stage 14: Final report + merged rejects (auditable)

Inputs:
- final_candidates.tsv from Stage 13
- one or more rejects.tsv files from earlier stages (optional but recommended)
- optional qc.json files from earlier stages (optional)

Outputs (in outdir):
- final_report.json            (counts + breakdowns)
- final_report.tsv             (small human-readable summary)
- rejects.merged.tsv           (auditable merged trail; if rejects inputs provided)

Design notes:
- No redesign of ladder. This is only a reporting/packaging step.
- Keeps everything auditable.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def _read_tsv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        rows = list(r)
        return (r.fieldnames or [], rows)


def _safe_int(x: str, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def _safe_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def merge_rejects(reject_paths: List[Path], out_path: Path) -> Dict[str, int]:
    """
    Merge multiple rejects.tsv into one file.
    Keeps a superset header; adds 'source_rejects' column.
    """
    all_headers = set()
    parsed: List[Tuple[Path, List[str], List[Dict[str, str]]]] = []

    for p in reject_paths:
        if not p.exists():
            continue
        hdr, rows = _read_tsv(p)
        if not hdr:
            continue
        all_headers.update(hdr)
        parsed.append((p, hdr, rows))

    if not parsed:
        return {"reject_files_in": 0, "reject_rows_in": 0, "reject_rows_out": 0}

    # stable header ordering
    base = [
        "sample_id", "stage", "chrom", "start0", "end0", "strand",
        "record_id", "island_id", "peak_id", "candidate_id",
        "reason_code", "reason_detail", "value", "threshold", "action"
    ]
    header = []
    for c in base:
        if c in all_headers:
            header.append(c)
    for c in sorted(all_headers):
        if c not in header:
            header.append(c)
    if "source_rejects" not in header:
        header.append("source_rejects")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_in = 0
    rows_out = 0
    with out_path.open("w", encoding="utf-8", newline="") as out:
        w = csv.DictWriter(out, delimiter="\t", fieldnames=header)
        w.writeheader()
        for src, hdr, rows in parsed:
            for row in rows:
                rows_in += 1
                out_row = {k: row.get(k, "") for k in header}
                out_row["source_rejects"] = str(src)
                w.writerow(out_row)
                rows_out += 1

    return {"reject_files_in": len(parsed), "reject_rows_in": rows_in, "reject_rows_out": rows_out}


def read_qc_jsons(qc_paths: List[Path]) -> Dict[str, dict]:
    qc = {}
    for p in qc_paths:
        if not p.exists():
            continue
        try:
            qc[p.name] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            qc[p.name] = {"_error": f"failed_to_parse:{p}"}
    return qc


def stage14_report(
    sample_id: str,
    final_candidates_tsv: Path,
    outdir: Path,
    rejects_paths: Optional[List[Path]] = None,
    qc_json_paths: Optional[List[Path]] = None,
) -> int:
    outdir.mkdir(parents=True, exist_ok=True)

    hdr, rows = _read_tsv(final_candidates_tsv)
    n = len(rows)

    # Columns we *expect* but won’t hard-require
    # We’ll compute with whatever is present.
    def col(name: str) -> bool:
        return name in hdr

    # Common breakdowns
    label_col = None
    for c in ("final_label", "label", "status"):
        if c in hdr:
            label_col = c
            break

    known_db_col = None
    for c in ("known_db", "precursor_db"):
        if c in hdr:
            known_db_col = c
            break

    repeat_col = "repeat_class" if "repeat_class" in hdr else None

    rf_col = None
    for c in ("best_rf_score", "rf_score"):
        if c in hdr:
            rf_col = c
            break

    # Counters
    label_counts = Counter()
    known_db_counts = Counter()
    repeat_counts = Counter()
    by_label_repeat = defaultdict(Counter)

    top_rf: List[Tuple[float, str]] = []
    for row in rows:
        if label_col:
            label_counts[(row.get(label_col) or "NA").strip()] += 1
        if known_db_col:
            known_db_counts[(row.get(known_db_col) or "NA").strip()] += 1
        if repeat_col:
            repeat_counts[(row.get(repeat_col) or "NA").strip()] += 1
            if label_col:
                by_label_repeat[(row.get(label_col) or "NA").strip()][(row.get(repeat_col) or "NA").strip()] += 1

        if rf_col:
            score = _safe_float(row.get(rf_col, ""), default=-1.0)
            rid = row.get("candidate_id") or row.get("peak_id") or row.get("record_id") or ""
            if rid:
                top_rf.append((score, rid))

    top_rf.sort(reverse=True)
    top_rf_20 = [{"id": rid, "score": sc} for sc, rid in top_rf[:20]]

    # Merge rejects if provided
    rejects_stats = {}
    merged_rejects_path = None
    if rejects_paths:
        merged_rejects_path = outdir / "rejects.merged.tsv"
        rejects_stats = merge_rejects([Path(p) for p in rejects_paths], merged_rejects_path)

    qc_blob = {}
    if qc_json_paths:
        qc_blob = read_qc_jsons([Path(p) for p in qc_json_paths])

    report = {
        "sample_id": sample_id,
        "inputs": {
            "final_candidates_tsv": str(final_candidates_tsv),
            "rejects_inputs": [str(p) for p in (rejects_paths or [])],
            "qc_json_inputs": [str(p) for p in (qc_json_paths or [])],
        },
        "counts": {
            "final_candidates_rows": n,
        },
        "breakdowns": {
            "by_label": dict(label_counts),
            "by_known_db": dict(known_db_counts),
            "by_repeat_class": dict(repeat_counts),
            "by_label_repeat_class": {k: dict(v) for k, v in by_label_repeat.items()},
        },
        "top": {
            "top20_by_rf": top_rf_20,
        },
        "rejects_merge": {
            "enabled": bool(rejects_paths),
            "merged_path": str(merged_rejects_path) if merged_rejects_path else None,
            **rejects_stats,
        },
        "qc_jsons": qc_blob,
        "outputs": {
            "final_report_json": str(outdir / "final_report.json"),
            "final_report_tsv": str(outdir / "final_report.tsv"),
        },
    }

    (outdir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Small TSV summary
    with (outdir / "final_report.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["key", "value"])
        w.writerow(["sample_id", sample_id])
        w.writerow(["final_candidates_rows", n])

        if label_counts:
            for k, v in label_counts.most_common():
                w.writerow([f"label::{k}", v])

        if known_db_counts:
            for k, v in known_db_counts.most_common():
                w.writerow([f"known_db::{k}", v])

        if repeat_counts:
            for k, v in repeat_counts.most_common():
                w.writerow([f"repeat_class::{k}", v])

        if merged_rejects_path:
            w.writerow(["rejects_merged_path", str(merged_rejects_path)])
            for kk, vv in rejects_stats.items():
                w.writerow([f"rejects_merge::{kk}", vv])

    print(f"[final-report] final_report.json: {outdir / 'final_report.json'}")
    print(f"[final-report] final_report.tsv:  {outdir / 'final_report.tsv'}")
    if merged_rejects_path:
        print(f"[final-report] rejects.merged.tsv: {merged_rejects_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="final-report", description="miRPV-NG Stage 14: final report + merged rejects")
    ap.add_argument("--final-candidates-tsv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sample-id", required=True)
    ap.add_argument("--rejects", nargs="*", default=None, help="One or more rejects.tsv paths to merge.")
    ap.add_argument("--qc-json", nargs="*", default=None, help="One or more qc.json paths to include in report.")
    args = ap.parse_args()

    return stage14_report(
        sample_id=args.sample_id,
        final_candidates_tsv=Path(args.final_candidates_tsv),
        outdir=Path(args.outdir),
        rejects_paths=[Path(x) for x in args.rejects] if args.rejects else None,
        qc_json_paths=[Path(x) for x in args.qc_json] if args.qc_json else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
