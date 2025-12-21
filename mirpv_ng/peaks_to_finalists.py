#!/usr/bin/env python3
"""
sRNA-seq Stage 11 — Peaks -> Finalists (decision ladder output).

Inputs:
  - peaks.scored.tsv   (Stage 9.5 output, one row per peak with best_* fields)
  - peaks.known.tsv    (Stage 10 output, one row per peak with known_status etc.)

Outputs:
  - candidates.tsv         (ALL peaks, compact: metadata + best scores + known + final_decision)
  - strict_finalists.tsv   (ONLY strict finalists: Known-Confirmed, Known-Atypical, Novel-High)
  - rejects.tsv            (auditable, every non-strict peak gets a reason)
  - qc_stage11.json

Notes:
  - This stage does NOT fold RNA. It only labels and selects.
  - Structure generation remains separate (Stage 12), to obey your two-layer output rule.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


STAGE = "FINAL_SELECT"


def _as_int(x: str, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _as_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def ensure_rejects_header(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["sample_id", "stage", "chrom", "start", "end", "strand", "item_id", "status", "reason", "details"])


def append_reject(
    path: Path,
    sample_id: str,
    chrom: str,
    start: int,
    end: int,
    strand: str,
    item_id: str,
    status: str,
    reason: str,
    details: str,
) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([sample_id, STAGE, chrom, start, end, strand, item_id, status, reason, details])


def read_tsv_as_dict(path: Path, key_col: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        if not r.fieldnames:
            raise RuntimeError(f"{path} has no header")
        if key_col not in r.fieldnames:
            raise RuntimeError(f"{path} missing key column: {key_col}")
        for row in r:
            k = row.get(key_col)
            if not k:
                continue
            out[k] = row
    return out


def decision_ladder(
    known_status: str,
    best_rf_score: float,
    best_pred_label: int,
    repeat_class: str,
    args: argparse.Namespace,
) -> Tuple[str, str]:
    """
    Returns (final_label, reason).

    final_label:
      - Known-Confirmed
      - Known-Atypical
      - Novel-High
      - Reject
    """
    rc = repeat_class if repeat_class and repeat_class != "None" else "None"

    # Repeat gate (configurable)
    if rc != "None":
        if rc in args.repeat_block:
            return "Reject", f"RepeatBlocked:{rc}"
        # allowlist mode: if allowlist provided and rc not in it, reject
        if args.repeat_allow and rc not in args.repeat_allow:
            return "Reject", f"RepeatNotAllowed:{rc}"
        # otherwise, raise thresholds if configured
        rf_min_known_atyp = args.known_atyp_min_rf_repeat
        rf_min_novel = args.novel_high_min_rf_repeat
    else:
        rf_min_known_atyp = args.known_atyp_min_rf
        rf_min_novel = args.novel_high_min_rf

    ks = (known_status or "Unknown").strip()

    # Known confirmed always strict (it already had mature-first overlap)
    if ks == "Known-Confirmed":
        return "Known-Confirmed", "KnownConfirmed"

    # Known atypical: require RF support to avoid noise inside known loci
    if ks == "Known-Atypical":
        if best_rf_score >= rf_min_known_atyp:
            return "Known-Atypical", "KnownAtypical+RF"
        return "Reject", f"KnownAtypicalLowRF<{rf_min_known_atyp:.3f}"

    # Unknown: require model label + high score
    if ks == "Unknown":
        if best_pred_label == 1 and best_rf_score >= rf_min_novel:
            return "Novel-High", "Unknown+RFHigh"
        # Keep it explicitly rejected (auditable)
        if best_pred_label != 1:
            return "Reject", "UnknownPredLabel0"
        return "Reject", f"UnknownLowRF<{rf_min_novel:.3f}"

    # Anything else is treated conservatively
    return "Reject", f"UnhandledKnownStatus:{ks}"


def run_peaks_to_finalists(args: argparse.Namespace) -> int:
    peaks_scored = Path(args.peaks_scored_tsv)
    peaks_known = Path(args.peaks_known_tsv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_candidates = outdir / "candidates.tsv"
    out_strict = outdir / "strict_finalists.tsv"
    out_rejects = outdir / "rejects.tsv"
    out_qc = outdir / "qc_stage11.json"

    ensure_rejects_header(out_rejects)

    known_by_peak = read_tsv_as_dict(peaks_known, key_col="peak_id")

    with peaks_scored.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        if not r.fieldnames:
            raise RuntimeError("peaks.scored.tsv has no header")

        required = {"peak_id", "sample_id", "chrom", "strand", "peak_center0", "best_pad", "best_rf_score", "best_pred_label"}
        miss = [c for c in required if c not in r.fieldnames]
        if miss:
            raise RuntimeError(f"peaks.scored.tsv missing required columns: {miss}")

        # Compact schema for candidates.tsv
        cand_fields = [
            "peak_id",
            "sample_id",
            "chrom",
            "strand",
            "peak_center0",
            "island_id",
            "anchor_type",
            "repeat_class",
            "depth_raw",
            "cpm",
            "len_mode",
            "frac_20_24",
            "dominance",
            "prec5p",
            "start_entropy",
            "best_pad",
            "best_rf_score",
            "best_pred_label",
            "best_hp_start0",
            "best_hp_end0",
            "best_hp_len",
            "known_status",
            "known_basis",
            "known_db",
            "known_id",
            "known_feature",
            "final_label",
            "final_reason",
        ]

        strict_fields = cand_fields  # same columns, but subset rows

        n_total = 0
        n_strict = 0
        counts = {"Known-Confirmed": 0, "Known-Atypical": 0, "Novel-High": 0, "Reject": 0}

        with out_candidates.open("w", encoding="utf-8", newline="") as fo, out_strict.open("w", encoding="utf-8", newline="") as fs:
            wc = csv.DictWriter(fo, fieldnames=cand_fields, delimiter="\t")
            ws = csv.DictWriter(fs, fieldnames=strict_fields, delimiter="\t")
            wc.writeheader()
            ws.writeheader()

            for row in r:
                n_total += 1
                peak_id = row["peak_id"]
                sample_id = row.get("sample_id") or (args.sample_id or "NA")
                chrom = row["chrom"]
                strand = row["strand"] if row["strand"] in {"+", "-"} else "."
                peak_center0 = _as_int(row.get("peak_center0", "0"))

                # Merge known fields (if missing, treat as Unknown)
                krow = known_by_peak.get(peak_id, {})
                known_status = krow.get("known_status", "Unknown")
                known_basis = krow.get("known_basis", "none")
                known_db = krow.get("known_db", "NA")
                known_id = krow.get("known_id", "NA")
                known_feature = krow.get("known_feature", "NA")

                best_rf = _as_float(row.get("best_rf_score", "0"))
                best_pred = _as_int(row.get("best_pred_label", "0"))
                repeat_class = row.get("repeat_class", "None")

                final_label, final_reason = decision_ladder(
                    known_status=known_status,
                    best_rf_score=best_rf,
                    best_pred_label=best_pred,
                    repeat_class=repeat_class,
                    args=args,
                )
                counts[final_label] = counts.get(final_label, 0) + 1

                out_row = {
                    "peak_id": peak_id,
                    "sample_id": sample_id,
                    "chrom": chrom,
                    "strand": strand,
                    "peak_center0": row.get("peak_center0", "NA"),
                    "island_id": row.get("island_id", "NA"),
                    "anchor_type": row.get("anchor_type", "NA"),
                    "repeat_class": repeat_class or "None",
                    "depth_raw": row.get("depth_raw", "NA"),
                    "cpm": row.get("cpm", "NA"),
                    "len_mode": row.get("len_mode", "NA"),
                    "frac_20_24": row.get("frac_20_24", "NA"),
                    "dominance": row.get("dominance", "NA"),
                    "prec5p": row.get("prec5p", "NA"),
                    "start_entropy": row.get("start_entropy", "NA"),
                    "best_pad": row.get("best_pad", "NA"),
                    "best_rf_score": f"{best_rf:.6f}",
                    "best_pred_label": str(best_pred),
                    "best_hp_start0": row.get("best_hp_start0", "NA"),
                    "best_hp_end0": row.get("best_hp_end0", "NA"),
                    "best_hp_len": row.get("best_hp_len", "NA"),
                    "known_status": known_status,
                    "known_basis": known_basis,
                    "known_db": known_db,
                    "known_id": known_id,
                    "known_feature": known_feature,
                    "final_label": final_label,
                    "final_reason": final_reason,
                }

                wc.writerow(out_row)

                if final_label in {"Known-Confirmed", "Known-Atypical", "Novel-High"}:
                    ws.writerow(out_row)
                    n_strict += 1
                else:
                    # Log rejects (auditable)
                    q0 = _as_int(row.get("best_start0", row.get("best_hp_start0", "0")), 0)
                    q1 = _as_int(row.get("best_end0", row.get("best_hp_end0", "0")), 0)
                    append_reject(
                        out_rejects,
                        sample_id=sample_id,
                        chrom=chrom,
                        start=q0,
                        end=q1,
                        strand=strand,
                        item_id=peak_id,
                        status="Reject",
                        reason=final_reason,
                        details=f"known={known_status};rf={best_rf:.3f};pred={best_pred};repeat={repeat_class}",
                    )

        qc = {
            "sample_id": args.sample_id or "NA",
            "peaks_total": n_total,
            "strict_finalists": n_strict,
            "counts": counts,
            "inputs": {
                "peaks_scored_tsv": str(peaks_scored),
                "peaks_known_tsv": str(peaks_known),
            },
            "params": {
                "known_atyp_min_rf": args.known_atyp_min_rf,
                "novel_high_min_rf": args.novel_high_min_rf,
                "known_atyp_min_rf_repeat": args.known_atyp_min_rf_repeat,
                "novel_high_min_rf_repeat": args.novel_high_min_rf_repeat,
                "repeat_block": list(args.repeat_block),
                "repeat_allow": list(args.repeat_allow),
            },
            "outputs": {
                "candidates_tsv": str(out_candidates),
                "strict_finalists_tsv": str(out_strict),
                "rejects_tsv": str(out_rejects),
            },
        }
        out_qc.write_text(json.dumps(qc, indent=2), encoding="utf-8")

    print(f"[peaks-to-finalists] candidates.tsv:       {out_candidates}")
    print(f"[peaks-to-finalists] strict_finalists.tsv: {out_strict}")
    print(f"[peaks-to-finalists] rejects.tsv:          {out_rejects}")
    print(f"[peaks-to-finalists] qc:                   {out_qc}")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Stage 11: merge scored peaks + known labels and select strict finalists.")
    ap.add_argument("--peaks-scored-tsv", required=True)
    ap.add_argument("--peaks-known-tsv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sample-id", default=None)

    # Decision thresholds (defaults are conservative; tweak later without code edits)
    ap.add_argument("--known-atyp-min-rf", type=float, default=0.55)
    ap.add_argument("--novel-high-min-rf", type=float, default=0.65)

    # If repeat_class is not None, require stricter thresholds by default
    ap.add_argument("--known-atyp-min-rf-repeat", type=float, default=0.65)
    ap.add_argument("--novel-high-min-rf-repeat", type=float, default=0.75)

    # Repeat handling
    ap.add_argument("--repeat-block", nargs="*", default=["LINE", "SINE", "LTR", "DNA", "Satellite", "Simple_repeat", "Low_complexity"])
    ap.add_argument("--repeat-allow", nargs="*", default=[],
                    help="If provided, only these repeat classes are allowed; others are rejected.")

    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    return run_peaks_to_finalists(args)


if __name__ == "__main__":
    raise SystemExit(main())
