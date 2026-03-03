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
    
    # ---------------------------\
    # 1. Load Structure Info (Need SEQ and PAD for Read-Awareness)
    # ---------------------------\
    struct_data = {}
    with struct_tsv.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            cid = row["candidate_id"]
            struct_data[cid] = {
                "seq": row.get("seq", ""),
                "pad": int(row.get("pad", 0)) if row.get("pad") else 0
            }

    # ---------------------------\
    # 2. Load Mature Predictions
    # ---------------------------\
    mature_by_id = {}
    with mature_tsv.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            # Handle potential ID mismatch from upstream tools
            cid = row.get("candidate_id") or row.get("id")
            if not cid:
                continue
            mature_by_id[cid] = row

    # ---------------------------\
    # 3. Merge & Apply Read-Aware Logic
    # ---------------------------\
    total_in = 0
    matched = 0
    missing = 0
    overridden = 0

    with struct_tsv.open() as f_in, \
         out_tsv.open("w", newline="") as f_out, \
         rejects_tsv.open("w", newline="") as f_rej:
        
        r = csv.DictReader(f_in, delimiter="\t")
        
        # Define output columns (Struct cols + Mature cols)
        mature_cols = ["arm", "start", "end", "length", "score", "mature_seq"]
        fieldnames = r.fieldnames + mature_cols
        
        w = csv.DictWriter(f_out, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()

        rej_fields = ["candidate_id", "reason"]
        wrej = csv.DictWriter(f_rej, fieldnames=rej_fields, delimiter="\t")
        wrej.writeheader()

        for row in r:
            total_in += 1
            cid = row["candidate_id"]
            
            # Get Structural Context
            seq = row.get("seq", "")
            pad = int(row.get("pad", 0)) if row.get("pad") else 0

            if cid in mature_by_id:
                m = mature_by_id[cid]
                
                # --- READ-AWARENESS CHECK ---
                # The 'pad' index is exactly where the Read Peak center is.
                # The ML model predicted 'start' and 'end'.
                try:
                    ml_start = int(m.get("start", 0))
                    ml_end = int(m.get("end", 0))
                    ml_center = (ml_start + ml_end) / 2
                    
                    # Distance between Read Peak and ML Prediction Center
                    dist = abs(pad - ml_center)
                    
                    # THRESHOLD: 20nt. 
                    # If distance > 20, the model likely picked the wrong arm (arms are usually ~40nt apart centers).
                    if dist > 20 and seq and pad > 0:
                        # OVERRIDE: Force mature to be centered on the Read Peak
                        # Choose length in a read-aware way:
                        # 1) use len_mode if available (from peaks/candidates metadata)
                        # 2) else use ML-predicted length
                        # 3) else default 21
                        override_len = None
                        try:
                            lm = row.get("len_mode")
                            if lm not in (None, "", "NA"):
                                override_len = int(float(lm))
                        except Exception:
                            override_len = None

                        if override_len is None:
                            try:
                                override_len = int(m.get("length", 21))
                            except Exception:
                                override_len = 21

                        # clamp to plausible mature range (you can widen if you want)
                        new_len = max(18, min(24, int(override_len)))
                        half = new_len // 2

                        
                        # Clamp bounds to be safe
                        new_start = max(0, pad - half)
                        new_end = min(len(seq), pad + half + (new_len % 2))
                        
                        # Construct new Override Data
                        row["start"] = new_start
                        row["end"] = new_end
                        row["length"] = new_end - new_start
                        row["mature_seq"] = seq[new_start:new_end]
                        # Guess arm based on position in precursor
                        row["arm"] = "5p" if new_start < (len(seq)/2) else "3p"
                        row["score"] = "0.0" # Mark score as 0/Dummy to indicate non-ML
                        
                        overridden += 1
                    else:
                        # TRUST ML: The prediction is close enough to the reads
                        for c in mature_cols:
                            row[c] = m.get(c)
                            
                except ValueError:
                    # Fallback if parsing fails, just use ML data
                    for c in mature_cols:
                        row[c] = m.get(c)

                w.writerow(row)
                matched += 1
            else:
                # Missing prediction entirely
                wrej.writerow({
                    "candidate_id": cid,
                    "reason": "MissingMaturePrediction"
                })
                missing += 1

    # ---------------------------\
    # QC
    # ---------------------------\
    qc = {
        "sample_id": args.sample_id,
        "inputs": {
            "candidates_struct_tsv": str(struct_tsv),
            "mature_tsv": str(mature_tsv)
        },
        "counts": {
            "struct_candidates": total_in,
            "mature_matched": matched,
            "missing_mature": missing,
            "read_position_overrides": overridden
        },
        "outputs": {
            "final_candidates_tsv": str(out_tsv),
            "rejects_tsv": str(rejects_tsv)
        }
    }

    with qc_json.open("w") as f:
        json.dump(qc, f, indent=2)

    print(f"[final-candidates] Processed {total_in}. Matched {matched}. Overridden {overridden} (Read-Guided). Missing {missing}.")