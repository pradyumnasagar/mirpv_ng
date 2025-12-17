import random

# Real miRNAs (hsa-let-7a, hsa-mir-21, hsa-mir-155)
targets = [
    "UGGGAUGAGGUAGUAGGUUGUAUAGUUUUAGGGUCACACCCACCACUGGGAGAUAACUAUACAAUCUACUGUCUUUCCUA",
    "UGUCGGGUAGCUUAUCAGACUGAUGUUGACUGUUGAAUCUCAUGGCAACACCAGUCGAUGGGCUGUCUGACA",
    "CUGUUAAUGCUAAUCGUGAUAGGGGUUUUUGCCUCCAACUGACUCCUACAUAUUAGCAUUAACAG"
]

# Generate 5000nt of random junk (A/C/G/T)
seq = "".join(random.choices("ACGT", k=5000))

# Insert miRNAs at specific spots
seq += targets[0] # let-7a at ~5000
seq += "".join(random.choices("ACGT", k=2000))
seq += targets[1] # mir-21 at ~7000
seq += "".join(random.choices("ACGT", k=2000))
seq += targets[2] # mir-155 at ~9000
seq += "".join(random.choices("ACGT", k=1000))

# Save
with open("examples/synthetic_chr.fa", "w") as f:
    f.write(f">synthetic_test_chr\n{seq}\n")

print(f"Generated synthetic_chr.fa with {len(seq)} nt.")