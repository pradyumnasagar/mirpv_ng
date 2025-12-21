#!/usr/bin/env python3

import csv
import json
from pathlib import Path
from collections import defaultdict

def run_final_candidates(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    struct_tsv = Path(args.candidates_struct_tsv)
    mature_tsv = Path(args.mature_tsv)

    out_tsv = outdir / "final_candidates.tsv"
    qc_json = outdir / "qc_stage13.json"
    rejects_tsv = outdir / "rejects.tsv"
    
    
    
    
    # ---------------------------
    # Load mature predictions
    # ---------------------------
    mature_by_id = {}
    with mature_tsv.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            # FIX: Check for 'id' (from predict-mature) OR 'candidate_id' (generic)
            cid = row.get("candidate_id") or row.get("id")
            if not cid:
                continue
            mature_by_id[cid] = row
            

    # ---------------------------
    # Load mature predictions
    # ---------------------------
   # mature_by_id = {}
    #with mature_tsv.open() as f:
     #   r = csv.DictReader(f, delimiter="\t")
      #  for row in r:
       #     cid = row.get("candidate_id")
        #    if not cid:
         #       continue
          #  mature_by_id[cid] = row

    # ---------------------------
    # Merge with struct candidates
    # ---------------------------
    total_in = 0
    matched = 0
    missing = 0

    with struct_tsv.open() as f_in, \
         out_tsv.open("w", newline="") as f_out, \
         rejects_tsv.open("w", newline="") as f_rej:

        r = csv.DictReader(f_in, delimiter="\t")

        # Prepare output header
        mature_cols = []
        if mature_by_id:
            sample_row = next(iter(mature_by_id.values()))
            mature_cols = [c for c in sample_row.keys() if c != "candidate_id"]

        fieldnames = r.fieldnames + mature_cols
        w = csv.DictWriter(f_out, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()

        rej_fields = ["candidate_id", "reason"]
        wrej = csv.DictWriter(f_rej, fieldnames=rej_fields, delimiter="\t")
        wrej.writeheader()

        for row in r:
            total_in += 1
            cid = row["candidate_id"]

            if cid in mature_by_id:
                m = mature_by_id[cid]
                for c in mature_cols:
                    row[c] = m.get(c)
                w.writerow(row)
                matched += 1
            else:
                wrej.writerow({
                    "candidate_id": cid,
                    "reason": "MissingMaturePrediction"
                })
                missing += 1

    # ---------------------------
    # QC
    # ---------------------------
    qc = {
        "sample_id": args.sample_id,
        "inputs": {
            "candidates_struct_tsv": str(struct_tsv),
            "mature_tsv": str(mature_tsv)
        },
        "counts": {
            "struct_candidates": total_in,
            "mature_matched": matched,
            "missing_mature": missing
        },
        "outputs": {
            "final_candidates_tsv": str(out_tsv),
            "rejects_tsv": str(rejects_tsv)
        }
    }

    with qc_json.open("w") as f:
        json.dump(qc, f, indent=2)

    print(f"[final-candidates] written: {out_tsv}")
    print(f"[final-candidates] qc: {qc_json}")
    print(f"[final-candidates] rejects: {rejects_tsv}")

    return 0

