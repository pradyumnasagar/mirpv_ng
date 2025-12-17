import random
from Bio import SeqIO

def shuffle_seq(seq):
    s = list(seq)
    random.shuffle(s)
    return "".join(s)

positives = list(SeqIO.parse("training/data/hsa_mirgene_premirna.fa", "fasta"))
with open("training/data/hsa_shuffled_negatives.fa", "w") as f:
    for i, rec in enumerate(positives):
        # Generate 5 shuffled versions for every real miRNA
        for j in range(5):
            seq = str(rec.seq)
            shuffled = shuffle_seq(seq)
            # Give it a unique ID
            f.write(f">shuffled_{i}_{j}\n{shuffled}\n")
