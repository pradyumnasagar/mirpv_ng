from Bio import SeqIO
import pandas as pd

def _predict_mature_simple(seq: str, length: int = 22):
    if len(seq) <= length:
        start = 0
    else:
        start = (len(seq) - length) // 2
    end = start + length
    return start, end, seq[start:end]

def predict_from_fasta(fasta_path: str, scores_df: pd.DataFrame, score_cutoff: float = 0.5) -> pd.DataFrame:
    """
    For each sequence in fasta, if its premirna_score >= cutoff,
    produce a naive mature prediction.
    """
    score_map = dict(zip(scores_df["id"], scores_df["premirna_score"]))
    rows = []

    for rec in SeqIO.parse(fasta_path, "fasta"):
        hid = rec.id
        seq = str(rec.seq).upper()
        score = score_map.get(hid, 0.0)
        if score < score_cutoff:
            continue
        start, end, mseq = _predict_mature_simple(seq)
        rows.append({
            "id": hid,
            "premirna_score": score,
            "mature_start": start,
            "mature_end": end,
            "mature_seq": mseq
        })

    return pd.DataFrame(rows)
