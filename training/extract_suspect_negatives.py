
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
from Bio import SeqIO

def get_clean_seq(seq):
    """Normalize sequence: Upper case, U to T (or T to U) for comparison."""
    return str(seq).upper().replace("U", "T")

def main():
    parser = argparse.ArgumentParser(description="Extract 'Suspect' negatives by subtracting MirGeneDB from miRBase using substring matching")
    parser.add_argument("--mirbase", required=True, help="Path to miRBase hairpin.fa")
    parser.add_argument("--mirgenedb", required=True, help="Path to MirGeneDB hsa.fa (Positives)")
    parser.add_argument("--species", default="hsa", help="Species prefix to filter miRBase (e.g., hsa)")
    parser.add_argument("--out", required=True, help="Output file for Hard Negatives")
    args = parser.parse_args()

    # 1. Load MirGeneDB Sequences (The Truth)
    # Storing as a list to iterate for substring checking
    positive_seqs = []
    print(f"Loading Positives from {args.mirgenedb}...")
    for rec in SeqIO.parse(args.mirgenedb, "fasta"):
        positive_seqs.append(get_clean_seq(rec.seq))
    
    print(f"Loaded {len(positive_seqs)} verified positive sequences.")

    # 2. Scan miRBase for 'Suspects'
    suspects = []
    real_count = 0
    total_species = 0

    print(f"Scanning miRBase {args.mirbase} for {args.species} entries...")
    for rec in SeqIO.parse(args.mirbase, "fasta"):
        if not rec.id.startswith(args.species):
            continue

        total_species += 1
        mb_seq = get_clean_seq(rec.seq)
        is_real = False

        # SUBSTRING CHECK (The Fix)
        # Check if any verified positive is contained within this miRBase entry
        # OR if this miRBase entry is contained within a verified positive
        for pos_seq in positive_seqs:
            if (pos_seq in mb_seq) or (mb_seq in pos_seq):
                is_real = True
                break
        
        if is_real:
            real_count += 1
        else:
            suspects.append(rec)

    # 3. Report & Write
    print(f"Total {args.species} in miRBase: {total_species}")
    print(f"Identified {real_count} Real miRNAs (matches MirGeneDB) -> SKIPPED")
    print(f"Extracted {len(suspects)} Suspects (Type 3 Negatives) -> WRITING to {args.out}")

    with open(args.out, "w") as f:
        SeqIO.write(suspects, f, "fasta-2line")

if __name__ == "__main__":
    main()