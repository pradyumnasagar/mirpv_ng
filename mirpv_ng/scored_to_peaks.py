# mirpv_ng/scored_to_peaks.py
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PAD_RE = re.compile(r"\|pad(\d+)$")


REJECTS_HEADER = [
    "sample_id", "stage", "chrom", "start", "end", "strand", "island_id", "peak_center",
    "anchor_type", "reject_id", "reason", "metric", "value", "threshold", "action", "notes"
]


def _ensure_rejects_header(rejects_path: Path) -> None:
    if rejects_path.exists() and rejects_path.stat().st_size > 0:
        return
    rejects_path.parent.mkdir(parents=True, exist_ok=True)
    with rejects_path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t").writerow(REJECTS_HEADER)


def _log_reject(
    rejects_path: Path,
    *,
    sample_id: str,
    chrom: str,
    start0: int,
    end0: int,
    strand: str,
    island_id: str,
    peak_center0: str,
    anchor_type: str,
    reject_id: str,
    reason: str,
    metric: str,
    value: str,
    threshold: str,
    notes: str = "",
) -> None:
    _ensure_rejects_header(rejects_path)
    with rejects_path.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t").writerow([
            sample_id, "COLLAPSE_PADS",
            chrom, str(start0), str(end0), strand,
            island_id or "", peak_center0 or "", anchor_type or "",
            reject_id, reason, metric, value, threshold, "Reject", notes
        ])


def _parse_float(x: str) -> Optional[float]:
    try:
        if x is None:
            return None
        x = str(x).strip()
        if x == "" or x.upper() == "NA":
            return None
        return float(x)
    except Exception:
        return None


def _parse_int(x: str) -> Optional[int]:
    try:
        if x is None:
            return None
        x = str(x).strip()
        if x == "" or x.upper() == "NA":
            return None
        return int(float(x))
    except Exception:
        return None


def _base_id_and_pad(candidate_id: str) -> Tuple[str, Optional[int]]:
    m = PAD_RE.search(candidate_id)
    if not m:
        return candidate_id, None
    pad = int(m.group(1))
    base = candidate_id[: m.start()]
    return base, pad


def _sort_key(row: Dict[str, str]) -> Tuple:
    # deterministic ordering for output: chrom, peak_center0, strand, base_id
    chrom = row.get("chrom", "")
    strand = row.get("strand", "")
    pc = _parse_int(row.get("peak_center0", "0")) or 0
    base, _ = _base_id_and_pad(row.get("candidate_id", ""))
    return (chrom, pc, strand, base)


def run_scored_to_peaks(args: argparse.Namespace) -> int:
    in_tsv = Path(args.scored_tsv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rejects_path = outdir / "rejects.tsv"
    _ensure_rejects_header(rejects_path)

    # Read rows
    with in_tsv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError("Input scored TSV missing header.")
        fieldnames = list(reader.fieldnames)

        # Hard requirement per your statement
        if fieldnames[-1] != "pred_label":
            raise ValueError(f"Expected pred_label to be last column, found last={fieldnames[-1]}")

        required = ["candidate_id", "chrom", "start0", "end0", "strand", "peak_center0", "pad", "rf_score", "pred_label"]
        missing = [c for c in required if c not in fieldnames]
        if missing:
            raise ValueError(f"Input scored TSV missing required columns: {missing}")

        rows = [r for r in reader]

    if not rows:
        out_path = outdir / "peaks.scored.tsv"
        out_path.write_text("", encoding="utf-8")
        (outdir / "qc_stage9_5.json").write_text(json.dumps({"peaks_total": 0}, indent=2), encoding="utf-8")
        print("[scored-to-peaks] No rows found.")
        return 0

    # Group by base_id
    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        base, _pad = _base_id_and_pad(r["candidate_id"])
        groups[base].append(r)

    out_path = outdir / "peaks.scored.tsv"

    # Output schema: one record per base peak, with best window + audit of both pads
    out_cols = [
        "peak_id", "sample_id", "chrom", "strand", "peak_center0", "island_id", "anchor_type",
        "best_pad", "best_rf_score", "best_pred_label",
        "best_hp_start0", "best_hp_end0", "best_hp_len",
        "pad_scores", "pads_seen", "n_pads",
        "repeat_class",
        # keep these peak-evidence fields from the best row (they’re stage-1 biology, not pad-specific in spirit)
        "depth_raw", "cpm", "len_mode", "frac_20_24", "dominance", "prec5p", "start_entropy",
        # carry start/end window of the best pad (useful later)
        "best_start0", "best_end0",
        # tie-break info
        "tie_breaker",
    ]

    # Stats
    pads_per_peak = Counter()
    best_pad_counts = Counter()
    best_pred_counts = Counter()
    missing_rf_groups = 0
    short_groups = 0

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols, delimiter="\t")
        w.writeheader()

        # deterministic output
        for base_id in sorted(groups.keys()):
            rs = groups[base_id]
            rs.sort(key=_sort_key)

            pads_seen: List[int] = []
            pad_scores_map: Dict[int, float] = {}
            for r in rs:
                pad = _parse_int(r.get("pad"))
                if pad is not None:
                    pads_seen.append(pad)
                rf = _parse_float(r.get("rf_score"))
                if pad is not None and rf is not None:
                    pad_scores_map[pad] = rf

            pads_seen_sorted = sorted(set(pads_seen))
            pads_per_peak[len(pads_seen_sorted)] += 1
            if len(pads_seen_sorted) < 2:
                short_groups += 1
                # not rejecting, but logging for audit
                r0 = rs[0]
                _log_reject(
                    rejects_path,
                    sample_id=r0.get("sample_id", "NA"),
                    chrom=r0.get("chrom", "."),
                    start0=_parse_int(r0.get("start0")) or -1,
                    end0=_parse_int(r0.get("end0")) or -1,
                    strand=r0.get("strand", "."),
                    island_id=r0.get("island_id", ""),
                    peak_center0=r0.get("peak_center0", ""),
                    anchor_type=r0.get("anchor_type", ""),
                    reject_id=base_id,
                    reason="MissingPadVariant",
                    metric="n_pads",
                    value=str(len(pads_seen_sorted)),
                    threshold=">=2",
                    notes=f"pads_seen={pads_seen_sorted}",
                )

            # Select best by rf_score desc; tie-break: prefer smaller pad (usually 70) then earlier hp_len
            best_row = None
            best_rf = None
            tie_breaker = "rf_score"

            # Collect candidates with numeric rf
            candidates: List[Tuple[float, int, int, Dict[str, str]]] = []
            for r in rs:
                rf = _parse_float(r.get("rf_score"))
                pad = _parse_int(r.get("pad")) or 10**9
                hp_len = _parse_int(r.get("hp_len")) or 10**9
                if rf is None:
                    continue
                candidates.append((rf, pad, hp_len, r))

            if not candidates:
                missing_rf_groups += 1
                r0 = rs[0]
                _log_reject(
                    rejects_path,
                    sample_id=r0.get("sample_id", "NA"),
                    chrom=r0.get("chrom", "."),
                    start0=_parse_int(r0.get("start0")) or -1,
                    end0=_parse_int(r0.get("end0")) or -1,
                    strand=r0.get("strand", "."),
                    island_id=r0.get("island_id", ""),
                    peak_center0=r0.get("peak_center0", ""),
                    anchor_type=r0.get("anchor_type", ""),
                    reject_id=base_id,
                    reason="NoNumericRFScore",
                    metric="rf_score",
                    value="NA",
                    threshold="numeric",
                    notes="All pad variants missing/NA rf_score",
                )
                continue

            # sort: rf desc, pad asc, hp_len asc
            candidates.sort(key=lambda t: (-t[0], t[1], t[2]))
            best_rf, best_pad, _best_hp_len, best_row = candidates[0]
            if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
                tie_breaker = "rf_score,tie->pad,hp_len"

            assert best_row is not None

            best_pad_counts[str(best_pad)] += 1
            best_pred_counts[str(best_row.get("pred_label", "NA"))] += 1

            # audit string: "70:0.123,100:0.456"
            pad_scores_str = ",".join([f"{p}:{pad_scores_map[p]:.6f}" for p in sorted(pad_scores_map.keys())])

            out = {
                "peak_id": base_id,
                "sample_id": best_row.get("sample_id", "NA"),
                "chrom": best_row.get("chrom", "NA"),
                "strand": best_row.get("strand", "NA"),
                "peak_center0": best_row.get("peak_center0", "NA"),
                "island_id": best_row.get("island_id", "NA"),
                "anchor_type": best_row.get("anchor_type", "NA"),
                "best_pad": str(best_pad),
                "best_rf_score": f"{best_rf:.6f}",
                "best_pred_label": best_row.get("pred_label", "NA"),
                "best_hp_start0": best_row.get("hp_start0", "NA"),
                "best_hp_end0": best_row.get("hp_end0", "NA"),
                "best_hp_len": best_row.get("hp_len", "NA"),
                "pad_scores": pad_scores_str if pad_scores_str else "NA",
                "pads_seen": ",".join(map(str, pads_seen_sorted)) if pads_seen_sorted else "NA",
                "n_pads": str(len(pads_seen_sorted)),
                "repeat_class": best_row.get("repeat_class", "None") or "None",
                "depth_raw": best_row.get("depth_raw", "NA"),
                "cpm": best_row.get("cpm", "NA"),
                "len_mode": best_row.get("len_mode", "NA"),
                "frac_20_24": best_row.get("frac_20_24", "NA"),
                "dominance": best_row.get("dominance", "NA"),
                "prec5p": best_row.get("prec5p", "NA"),
                "start_entropy": best_row.get("start_entropy", "NA"),
                "best_start0": best_row.get("start0", "NA"),
                "best_end0": best_row.get("end0", "NA"),
                "tie_breaker": tie_breaker,
            }
            w.writerow(out)

    qc = {
        "stage": "scored-to-peaks",
        "input": str(in_tsv),
        "output": str(out_path),
        "peaks_total": len(groups),
        "pads_per_peak": dict(sorted(pads_per_peak.items())),
        "best_pad_counts": dict(sorted(best_pad_counts.items(), key=lambda x: int(x[0]))),
        "best_pred_counts": dict(sorted(best_pred_counts.items())),
        "peaks_missing_numeric_rf": missing_rf_groups,
        "peaks_with_missing_pad_variant": short_groups,
    }
    (outdir / "qc_stage9_5.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")

    print(f"[scored-to-peaks] peaks_total={len(groups)}")
    print(f"[scored-to-peaks] wrote: {out_path}")
    print(f"[scored-to-peaks] qc: {outdir / 'qc_stage9_5.json'}")
    print(f"[scored-to-peaks] rejects.tsv: {rejects_path}")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 9.5: collapse pad-variants into one best-scored peak record.")
    p.add_argument("--scored-tsv", required=True, help="Input candidates.scored.tsv")
    p.add_argument("--outdir", required=True, help="Output directory")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)
    return run_scored_to_peaks(args)


if __name__ == "__main__":
    raise SystemExit(main())
