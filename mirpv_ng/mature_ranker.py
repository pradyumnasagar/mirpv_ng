# mirpv_ng/mature_ranker.py
# Traverse, Pair, Rank — mature duplex candidate generation with bulge-robust traversal

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class DuplexCandidate:
    arm: str                 # "5p" or "3p"
    start: int               # 0-based start
    length: int              # candidate length
    end: int                 # start + length (exclusive)
    partner_start: int       # inferred partner start (0-based)
    partner_end: int         # inferred partner end (exclusive)
    overhang: int            # 3' overhang proxy / true depending on logic
    paired_frac: float
    feasible: float          # 1.0 if feasible, else 0.0
    features: Dict[str, float]


# ----------------------------- Utilities ----------------------------- #

def dotbracket_pairs(struct: str) -> List[int]:
    """Map index i -> index j. -1 if unpaired."""
    stack = []
    pair = [-1] * len(struct)
    for i, ch in enumerate(struct):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if not stack:
                continue
            j = stack.pop()
            pair[i] = j
            pair[j] = i
    return pair


def identify_loop_region(struct: str) -> Tuple[int, int]:
    """Find the apical loop (longest run of dots). Returns [start, end)"""
    best_len = 0
    best_start = 0
    run = 0
    cur_start = 0
    for i, ch in enumerate(struct):
        if ch == ".":
            if run == 0:
                cur_start = i
            run += 1
        else:
            if run > best_len:
                best_len = run
                best_start = cur_start
            run = 0
    if run > best_len:
        best_len = run
        best_start = cur_start
    return (best_start, best_start + best_len)


def score_end_stability(seq: str, start: int, end: int) -> float:
    """Simple end-stability proxy (GC=3, AU=1, other=2) on first 4 nts."""
    if start < 0 or end > len(seq):
        return 0.0
    sub = seq[start : min(end, start + 4)]
    score = 0.0
    for c in sub:
        if c in "GC":
            score += 3.0
        elif c in "AU":
            score += 1.0
        else:
            score += 2.0
    return score


# ----------------------------- Core Logic ----------------------------- #

def build_virtual_duplex(
    seq: str,
    struct: str,
    pair: List[int],
    loop_start: int,
    loop_end: int,
    start: int,
    L: int,
    arm: str
) -> DuplexCandidate:
    """
    Build duplex candidate; uses paired-space walk and returns feature dict.
    This assumes 'start' is on the chosen arm in global coordinates.
    """

    n = len(seq)
    end = start + L
    if end > n:
        return DuplexCandidate(arm, start, L, end, -1, -1, -99, 0.0, 0.0, {})

    # Reject if start is inside loop
    if loop_start <= start < loop_end:
        return DuplexCandidate(arm, start, L, end, -1, -1, -99, 0.0, 0.0, {})

    used = list(range(start, end))

    # Count pairing within candidate window
    paired_count = 0
    partner_pos = []
    for i in used:
        p = pair[i]
        if p != -1:
            paired_count += 1
            partner_pos.append(p)

    paired_frac = paired_count / float(L) if L > 0 else 0.0

    # Infer partner bounds (simple)
    if partner_pos:
        partner_start = min(partner_pos)
        partner_end = max(partner_pos) + 1
    else:
        partner_start, partner_end = -1, -1

    # Overhang proxy (keep whatever your existing logic is)
    overhang = 0
    if partner_pos:
        # a proxy: distance between inferred partner end and the pair of first paired base
        first_paired = None
        for i in used:
            if pair[i] != -1:
                first_paired = i
                break
        if first_paired is not None:
            anchor_partner = pair[first_paired]
            overhang = (partner_end - 1) - anchor_partner

    # Distances
    if arm == "5p":
        dist_loop = float(loop_start - end)
        dist_base = float(start)
    else:
        dist_loop = float(start - loop_end)
        dist_base = float(n - end)

    feat_dict: Dict[str, float] = {
        "feat_arm_5p": 1.0 if arm == "5p" else 0.0,
        "feat_len": float(L),
        "feat_dist_loop": float(dist_loop),
        "feat_dist_base": float(dist_base),
        "feat_paired_frac": float(paired_frac),
        "feat_starts_U": 1.0 if seq[start] == "U" else 0.0,
        "feat_stability": score_end_stability(seq, start, end),
        "feat_cut_unpaired": 1.0 if struct[start] == "." else 0.0,
        "feat_overhang_proxy": float(overhang),
    }

    # --- End-aware features to resolve length ambiguity (your fixed version) ---
    last_idx = end - 1
    if 0 <= last_idx < len(pair):
        feat_dict["feat_end_is_paired"] = 1.0 if pair[last_idx] != -1 else 0.0
    else:
        feat_dict["feat_end_is_paired"] = 0.0

    if arm == "3p":
        tail = 0
        i = last_idx
        while i >= start and 0 <= i < len(pair) and pair[i] == -1:
            tail += 1
            i -= 1
        feat_dict["feat_3p_tail_unpaired"] = float(tail)
    else:
        if partner_pos and partner_end > 0:
            last_paired_partner = max(partner_pos)
            feat_dict["feat_3p_tail_unpaired"] = float((partner_end - 1) - last_paired_partner)
        else:
            feat_dict["feat_3p_tail_unpaired"] = 0.0

    feat_dict["feat_3p_tail_abs2"] = abs(feat_dict.get("feat_3p_tail_unpaired", 0.0) - 2.0)

    return DuplexCandidate(
        arm=arm,
        start=start,
        length=L,
        end=end,
        partner_start=partner_start,
        partner_end=partner_end,
        overhang=overhang,
        paired_frac=paired_frac,
        feasible=1.0,
        features=feat_dict,
    )


def propose_candidate_starts(
    struct: str,
    pair: List[int],
    max_per_arm: int = 30,
    min_paired_context: int = 6,
    loop_buffer: int = 0,
) -> List[Tuple[int, str]]:
    """
    Scan plausible start sites on both arms.
    loop_buffer: allow starts closer to loop boundary by this many nts.
    Still does NOT allow starts inside the apical loop.
    """
    starts: List[Tuple[int, str]] = []
    L = len(struct)
    loop_start, loop_end = identify_loop_region(struct)

    def paired_context_ok(i: int) -> bool:
        if min_paired_context <= 0:
            return True
        w_end = min(L, i + min_paired_context)
        cnt = 0
        for k in range(i, w_end):
            if pair[k] != -1:
                cnt += 1
        return cnt >= max(1, min_paired_context // 2)

    # 5p range: up to loop_start-1, but allow approaching by loop_buffer
    max_5p = max(0, (loop_start - 1) + loop_buffer)
    max_5p = min(max_5p, L - 1)

    # 3p range: from loop_end, but allow approaching by loop_buffer
    min_3p = max(0, loop_end - loop_buffer)
    min_3p = min(min_3p, L - 1)

    # 5p starts
    starts_5p: List[Tuple[int, str]] = []
    for s in range(0, max_5p + 1):
        if loop_start <= s < loop_end:
            continue
        if paired_context_ok(s):
            starts_5p.append((s, "5p"))

    # 3p starts
    starts_3p: List[Tuple[int, str]] = []
    for s in range(min_3p, L):
        if loop_start <= s < loop_end:
            continue
        if paired_context_ok(s):
            starts_3p.append((s, "3p"))

    # downsample
    starts_5p = starts_5p[:max_per_arm]
    starts_3p = starts_3p[:max_per_arm]
    return starts_5p + starts_3p


def generate_duplex_candidates(
    seq: str,
    struct: str,
    lengths: Tuple[int, ...] = (21, 22, 23, 24),
    max_per_arm: int = 30,
    min_paired_context: int = 6,
    loop_buffer: int = 0,
) -> List[DuplexCandidate]:
    pair = dotbracket_pairs(struct)
    loop_start, loop_end = identify_loop_region(struct)
    starts = propose_candidate_starts(
        struct, pair,
        max_per_arm=max_per_arm,
        min_paired_context=min_paired_context,
        loop_buffer=loop_buffer,
    )

    cands: List[DuplexCandidate] = []
    for s, arm in starts:
        for L in lengths:
            c = build_virtual_duplex(seq, struct, pair, loop_start, loop_end, s, L, arm)
            if c.overhang > -90 and c.feasible > 0:
                cands.append(c)
    return cands
