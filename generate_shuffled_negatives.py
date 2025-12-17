import random
import argparse
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def shuffle_sequence(seq_str, n=5):
    """
    Generate n shuffled versions of the sequence.
    Simple shuffling preserves mononucleotide frequency (GC content).
    """
    shuffled_seqs = []
    chars = list(seq_str)
    seen = set()
    seen.add(seq_str) # Don't output the original

    attempts = 0
    while len(shuffled_seqs) < n and attempts < n * 10:
        random.shuffle(chars)
        new_seq = "".join(chars)
        if new_seq not in seen:
            shuffled_seqs.append(new_seq)
            seen.add(new_seq)
        attempts += 1
    
    return shuffled_seqs

def main():
    parser = argparse.ArgumentParser(description="Generate shuffled negatives from positives")
    parser.add_argument("--input", required=True, help="Input Positive FASTA")
    parser.add_argument("--output", required=True, help="Output Negative FASTA")
    parser.add_argument("--multiplier", type=int, default=10, help="How many shuffles per positive?")
    args = parser.parse_args()

    negatives = []
    print(f"Reading {args.input}...")
    
    count = 0
    for record in SeqIO.parse(args.input, "fasta"):
        seq_str = str(record.seq).upper().replace("U", "T")
        
        # Generate shuffles
        fakes = shuffle_sequence(seq_str, n=args.multiplier)
        
        for i, fake_seq in enumerate(fakes):
            neg_id = f"shuffle_{record.id}_{i}"
            neg_rec = SeqRecord(
                Seq(fake_seq),
                id=neg_id,
                description="shuffled_negative"
            )
            negatives.append(neg_rec)
        count += 1

    print(f"Generated {len(negatives)} shuffled negatives from {count} positives.")
    
    with open(args.output, "w") as f:
        SeqIO.write(negatives, f, "fasta")
    print("Done.")

if __name__ == "__main__":
    main()
