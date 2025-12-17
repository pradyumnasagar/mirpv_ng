# mirpv_ng/pgs_features.py

from __future__ import annotations

from typing import Dict, Tuple, List


def _find_internal_loops(struct: str) -> List[Tuple[int, int]]:
    """
    Very simple loop finder: contiguous '.' regions flanked by paired bases.

    Returns list of (start, end) indices [0-based, end exclusive].
    """
    loops: List[Tuple[int, int]] = []
    n = len(struct)
    i = 0
    while i < n:
        if struct[i] == ".":
            start = i
            while i < n and struct[i] == ".":
                i += 1
            end = i
            left = struct[start - 1] if start > 0 else None
            right = struct[end] if end < n else None
            if left in ("(", ")") and right in ("(", ")"):
                loops.append((start, end))
        else:
            i += 1
    return loops


def _best_loop(loops: List[Tuple[int, int]]) -> Tuple[int, int] | None:
    """
    Pick the largest loop (by size). Heuristic proxy for apical loop.
    """
    if not loops:
        return None
    sizes = [(end - start, (start, end)) for (start, end) in loops]
    sizes.sort(reverse=True)
    return sizes[0][1]


def compute_pgs_features(seq: str, struct: str) -> Dict[str, float]:
    """
    pgs motif and geometry features.

    Uses a simplified scoring scheme:

    - UGUG / UGU in the (largest) apical loop
    - basal UG motif near the 5' end of the stem
    - CNNC motif near the loop
    - loop size and stem length relative to ~40 nt target
    """
    feats: Dict[str, float] = {}

    seq = seq.upper().replace("T", "U")
    L = max(len(seq), 1)
    if len(struct) != len(seq):
        struct = struct[: len(seq)]

    loops = _find_internal_loops(struct)
    best = _best_loop(loops)

    # Default: no loops
    loop_size = 0
    loop_seq = ""
    loop_start = 0
    loop_end = 0

    if best is not None:
        loop_start, loop_end = best
        loop_size = loop_end - loop_start
        loop_seq = seq[loop_start:loop_end]

    # ---- motif counts ---- #

    # UGUG / UGU in apical loop
    ugug_count = loop_seq.count("UGUG") if loop_seq else 0
    ugu_count = loop_seq.count("UGU") if loop_seq else 0

    # Basal UG: in first ~8 nt of sequence
    basal_region = seq[: min(8, L)]
    basal_ug_count = basal_region.count("UG")

    # CNNC motif anywhere in loop-proximal region
    loop_prox_start = max(loop_start - 10, 0)
    loop_prox_end = min(loop_end + 10, L)
    loop_prox_seq = seq[loop_prox_start:loop_prox_end] if loop_end > 0 else seq
    cnnc_count = loop_prox_seq.count("CNNC")  # 'N' not literal; rough proxy
    # simplistic: treat any 'C..C' as CNNC-like
    loose_cnnc = 0
    for i in range(len(loop_prox_seq) - 3):
        if loop_prox_seq[i] == "C" and loop_prox_seq[i + 3] == "C":
            loose_cnnc += 1

    # ---- stem length heuristic: number of paired positions / 2 ---- #
    num_pairs = struct.count("(")
    stem_len = num_pairs  # this is already paired-bases count (not /2) in typical hairpins

    # ---- scoring, pgs ---- #
    score = 0.0

    # UGUG/UGU in apical loop
    if ugug_count > 0 or ugu_count > 0:
        score += 1.0

    # stem length closeness to 40 nt
    stem_target = 40
    diff = abs(stem_target - stem_len)
    if diff <= 3:
        score += 1.0
    elif diff <= 8:
        score += 0.75
    elif diff <= 13:
        score += 0.5
    elif diff <= 18:
        score += 0.25

    # basal UG
    if basal_ug_count > 0:
        score += 1.0

    # CNNC-like motifs
    if loose_cnnc > 0:
        score += 1.0

    # ---- store features ---- #
    feats["pgs_score"] = score
    feats["pgs_loop_size"] = float(loop_size)
    feats["pgs_stem_len"] = float(stem_len)
    feats["pgs_ugug_count"] = float(ugug_count)
    feats["pgs_ugu_count"] = float(ugu_count)
    feats["pgs_basal_ug_count"] = float(basal_ug_count)
    feats["pgs_cnnc_loose_count"] = float(loose_cnnc)

    # Normalised versions (to make RF training easier)
    feats["pgs_loop_size_norm"] = loop_size / L
    feats["pgs_stem_len_norm"] = stem_len / L

    return feats

