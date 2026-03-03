#!/usr/bin/env python3
"""
sRNA-seq Stage 12 — Strict finalists -> structures (RNAfold) + FASTA

Inputs:
  - strict_finalists.tsv (Stage 11 output)
  - candidates.fa        (from fastq-to-peaks excision; contains pad70/pad100 sequences)

Outputs (strict finalists only):
  - candidates_struct.tsv
  - candidates_struct.fa
  - rejects.tsv
  - qc_stage12.json

Key logic:
  - For each finalist peak row, choose candidate fasta id = peak_id + "|pad{best_pad}"
  - Fetch that exact sequence from candidates.fa
  - Run RNAfold to obtain dot-bracket + MFE
  - Write compact struct TSV + FASTA

Multithreading:
  - Uses ProcessPoolExecutor; each worker runs RNAfold for one sequence.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


STAGE = "STRUCT"


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


def load_fasta_to_dict(path: Path) -> Dict[str, str]:
    """Load FASTA into dict id->seq (expects unique IDs)."""
    seqs: Dict[str, List[str]] = {}
    cur_id: Optional[str] = None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                cur_id = line[1:].strip()
                if cur_id not in seqs:
                    seqs[cur_id] = []
                continue
            if cur_id is None:
                continue
            seqs[cur_id].append(line.strip())
    return {k: "".join(v).upper().replace("T", "U") for k, v in seqs.items()}


def parse_rnafold_output(stdout: str) -> Tuple[str, float]:
    """
    Returns (dotbracket, mfe).
    RNAfold typical output for fasta on stdin:
      >id
      SEQUENCE
      ....(((...))).... (-23.40)
    """
    lines = [x.strip() for x in stdout.splitlines() if x.strip()]
    if not lines:
        raise ValueError("RNAfold produced empty output")

    # last non-empty line should contain structure + mfe in parentheses
    last = lines[-1]
    m = re.search(r"^([().\[\]{}<>A-Za-z]+)\s+\(([-0-9.]+)\)", last)
    if not m:
        # Some versions format differently; try to pull mfe anywhere
        m2 = re.search(r"([().]+).*\(([-0-9.]+)\)", last)
        if not m2:
            raise ValueError(f"Could not parse RNAfold line: {last}")
        return m2.group(1), float(m2.group(2))

    dot = m.group(1)
    mfe = float(m.group(2))
    return dot, mfe


def rnafold_one(rnafold_bin: str, seq_id: str, seq: str) -> Tuple[str, str, str, float]:
    """
    Worker function. Returns (seq_id, seq, dotbracket, mfe).
    """
    inp = f">{seq_id}\n{seq}\n"
    cmd = [rnafold_bin, "--noPS"]
    proc = subprocess.run(
        cmd,
        input=inp.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace")[-500:])
    out = proc.stdout.decode("utf-8", errors="replace")
    dot, mfe = parse_rnafold_output(out)
    return seq_id, seq, dot, mfe


def run_finalists_to_struct(args: argparse.Namespace) -> int:
    strict_tsv = Path(args.strict_finalists_tsv)
    candidates_fa = Path(args.candidates_fa)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_tsv = outdir / "candidates_struct.tsv"
    out_fa = outdir / "candidates_struct.fa"
    out_rejects = outdir / "rejects.tsv"
    out_qc = outdir / "qc_stage12.json"
    ensure_rejects_header(out_rejects)

    fasta_map = load_fasta_to_dict(candidates_fa)

    # Read strict finalists
    with strict_tsv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        if not r.fieldnames:
            raise RuntimeError("strict_finalists.tsv has no header")

        needed = {"peak_id", "chrom", "strand", "best_pad", "final_label"}
        miss = [c for c in needed if c not in r.fieldnames]
        if miss:
            raise RuntimeError(f"strict_finalists.tsv missing required columns: {miss}")

        # We only fold strict labels (but strict file should already contain only them)
        strict_labels = {"Known-Confirmed", "Known-Atypical", "Novel-High"}

        jobs: List[Tuple[str, str, Dict[str, str]]] = []
        n_in = 0
        n_skip = 0

        for row in r:
            n_in += 1
            final_label = (row.get("final_label") or "").strip()
            if final_label not in strict_labels:
                n_skip += 1
                continue

            peak_id = row["peak_id"]
            pad = row.get("best_pad", "70")
            seq_id = f"{peak_id}|pad{pad}"

            seq = fasta_map.get(seq_id)
            if not seq:
                # Attempt small recovery: sometimes IDs differ by whitespace
                seq = fasta_map.get(seq_id.strip())
            if not seq:
                append_reject(
                    out_rejects,
                    sample_id=row.get("sample_id", args.sample_id or "NA"),
                    chrom=row.get("chrom", "NA"),
                    start=int(row.get("best_hp_start0", "0") or 0),
                    end=int(row.get("best_hp_end0", "0") or 0),
                    strand=row.get("strand", "."),
                    item_id=peak_id,
                    status="Reject",
                    reason="MissingCandidateSequence",
                    details=f"expected_fasta_id={seq_id}",
                )
                continue

            jobs.append((seq_id, seq, row))

    # Run RNAfold in parallel
    results: List[Tuple[str, str, str, float, Dict[str, str]]] = []
    n_ok = 0
    n_fail = 0

    with ProcessPoolExecutor(max_workers=max(1, int(args.threads))) as ex:
        futs = {}
        for seq_id, seq, row in jobs:
            futs[ex.submit(rnafold_one, args.rnafold, seq_id, seq)] = row

        for fut in as_completed(futs):
            row = futs[fut]
            peak_id = row.get("peak_id", "NA")
            try:
                seq_id, seq, dot, mfe = fut.result()
                results.append((seq_id, seq, dot, mfe, row))
                n_ok += 1
            except Exception as e:
                n_fail += 1
                append_reject(
                    out_rejects,
                    sample_id=row.get("sample_id", args.sample_id or "NA"),
                    chrom=row.get("chrom", "NA"),
                    start=int(row.get("best_hp_start0", "0") or 0),
                    end=int(row.get("best_hp_end0", "0") or 0),
                    strand=row.get("strand", "."),
                    item_id=peak_id,
                    status="Reject",
                    reason="RNAfoldFailed",
                    details=str(e)[:300],
                )

    # Write outputs
    struct_fields = [
        "candidate_id",
        "peak_id",
        "sample_id",
        "chrom",
        "strand",
        "peak_center0",
        "pad",
        "final_label",
        "known_status",
        "known_db",
        "known_id",
        "best_rf_score",
        "best_pred_label",
        "best_hp_start0",
        "best_hp_end0",
        "best_hp_len",
        "seq_len",
        "mfe",
        "dotbracket",
        "seq",
    ]

    with out_tsv.open("w", encoding="utf-8", newline="") as fo, out_fa.open("w", encoding="utf-8") as ff:
        w = csv.DictWriter(fo, fieldnames=struct_fields, delimiter="\t")
        w.writeheader()

        for seq_id, seq, dot, mfe, row in results:
            peak_id = row.get("peak_id", "NA")
            sample_id = row.get("sample_id", args.sample_id or "NA")
            chrom = row.get("chrom", "NA")
            strand = row.get("strand", ".")
            peak_center0 = row.get("peak_center0", row.get("peak_center", "NA"))
            pad = row.get("best_pad", "NA")

            w.writerow({
                "candidate_id": seq_id,
                "peak_id": peak_id,
                "sample_id": sample_id,
                "chrom": chrom,
                "strand": strand,
                "peak_center0": peak_center0,
                "pad": pad,
                "final_label": row.get("final_label", "NA"),
                "known_status": row.get("known_status", "NA"),
                "known_db": row.get("known_db", "NA"),
                "known_id": row.get("known_id", "NA"),
                "best_rf_score": row.get("best_rf_score", "NA"),
                "best_pred_label": row.get("best_pred_label", "NA"),
                "best_hp_start0": row.get("best_hp_start0", "NA"),
                "best_hp_end0": row.get("best_hp_end0", "NA"),
                "best_hp_len": row.get("best_hp_len", "NA"),
                "seq_len": str(len(seq)),
                "mfe": f"{mfe:.3f}",
                "dotbracket": dot,
                "seq": seq,
            })

            ff.write(f">{seq_id}\n{seq}\n")

    qc = {
        "sample_id": args.sample_id or "NA",
        "strict_in": len(jobs),
        "fold_ok": n_ok,
        "fold_failed": n_fail,
        "skipped_non_strict_rows": n_skip,
        "inputs": {
            "strict_finalists_tsv": str(strict_tsv),
            "candidates_fa": str(candidates_fa),
        },
        "params": {
            "rnafold": args.rnafold,
            "threads": int(args.threads),
        },
        "outputs": {
            "candidates_struct_tsv": str(out_tsv),
            "candidates_struct_fa": str(out_fa),
            "rejects_tsv": str(out_rejects),
        },
    }
    out_qc.write_text(json.dumps(qc, indent=2), encoding="utf-8")

    print(f"[finalists-to-struct] candidates_struct.tsv: {out_tsv}")
    print(f"[finalists-to-struct] candidates_struct.fa:  {out_fa}")
    print(f"[finalists-to-struct] rejects.tsv:          {out_rejects}")
    print(f"[finalists-to-struct] qc:                   {out_qc}")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Stage 12: strict finalists -> RNAfold structures (only strict).")
    ap.add_argument("--strict-finalists-tsv", required=True)
    ap.add_argument("--candidates-fa", required=True, help="FASTA from fastq-to-peaks excision (pad sequences).")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sample-id", default=None)
    ap.add_argument("--rnafold", default="RNAfold")
    ap.add_argument("--threads", type=int, default=4)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    return run_finalists_to_struct(args)


if __name__ == "__main__":
    raise SystemExit(main())
