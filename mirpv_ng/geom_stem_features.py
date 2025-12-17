# mirpv_ng/geom_stem_features.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


@dataclass
class StemRun:
    start_i: int
    start_j: int
    length: int


def get_basepairs(struct: str) -> List[Tuple[int, int]]:
    """
    Convert dot-bracket structure into a sorted list of basepairs (i, j),
    0-based, i < j.
    """
    stack: List[int] = []
    pairs: List[Tuple[int, int]] = []
    for i, c in enumerate(struct):
        if c == "(":
            stack.append(i)
        elif c == ")":
            if not stack:
                continue
            i0 = stack.pop()
            pairs.append((i0, i))
    pairs.sort()
    return pairs


def get_hairpin_span(pairs: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """Return (min_i, max_j) over all basepairs, or None if no pairs."""
    if not pairs:
        return None
    min_i = min(i for i, _ in pairs)
    max_j = max(j for _, j in pairs)
    return min_i, max_j


def find_exact_stem_runs(pairs: List[Tuple[int, int]]) -> List[StemRun]:
    """
    Group consecutive basepairs (i, j) such that:
      i increases by 1 and j decreases by 1 → exact helix.

    Returns a list of StemRun objects.
    """
    if not pairs:
        return []

    runs: List[StemRun] = []
    cur_i, cur_j = pairs[0]
    cur_len = 1

    for i, j in pairs[1:]:
        if i == cur_i + 1 and j == cur_j - 1:
            cur_i, cur_j = i, j
            cur_len += 1
        else:
            runs.append(
                StemRun(
                    start_i=cur_i - cur_len + 1,
                    start_j=cur_j + cur_len - 1,
                    length=cur_len,
                )
            )
            cur_i, cur_j = i, j
            cur_len = 1

    runs.append(
        StemRun(
            start_i=cur_i - cur_len + 1,
            start_j=cur_j + cur_len - 1,
            length=cur_len,
        )
    )
    return runs


def compute_pair_composition(seq: str, struct: str) -> Dict[str, float]:
    """
    Fraction of GC / AU / GU / other basepairs in the hairpin.

    Returns:
      {
        "pct_GC_pairs",
        "pct_AU_pairs",
        "pct_GU_pairs",
        "pct_other_pairs",
      }
    """
    seq_u = seq.upper().replace("T", "U")
    pairs = get_basepairs(struct)
    if not pairs:
        return {
            "pct_GC_pairs": 0.0,
            "pct_AU_pairs": 0.0,
            "pct_GU_pairs": 0.0,
            "pct_other_pairs": 0.0,
        }

    gc = au = gu = other = 0
    for i, j in pairs:
        if i < 0 or j >= len(seq_u):
            continue
        a, b = seq_u[i], seq_u[j]
        pair = a + b
        if pair in ("GC", "CG"):
            gc += 1
        elif pair in ("AU", "UA"):
            au += 1
        elif pair in ("GU", "UG"):
            gu += 1
        else:
            other += 1

    total = max(gc + au + gu + other, 1)
    return {
        "pct_GC_pairs": gc / total,
        "pct_AU_pairs": au / total,
        "pct_GU_pairs": gu / total,
        "pct_other_pairs": other / total,
    }


def compute_helix_stats(struct: str) -> Dict[str, float]:
    """
    Helix-level features from exact stem runs:
      - helix_count
      - helix_max_len
      - helix_mean_len
      - anchor_stem_len (longest helix)
    """
    pairs = get_basepairs(struct)
    runs = find_exact_stem_runs(pairs)

    if not runs:
        return {
            "helix_count": 0.0,
            "helix_max_len": 0.0,
            "helix_mean_len": 0.0,
            "anchor_stem_len": 0.0,
        }

    lengths = [r.length for r in runs]
    helix_count = len(lengths)
    helix_max = max(lengths)
    helix_mean = sum(lengths) / helix_count
    anchor_stem_len = helix_max

    return {
        "helix_count": float(helix_count),
        "helix_max_len": float(helix_max),
        "helix_mean_len": float(helix_mean),
        "anchor_stem_len": float(anchor_stem_len),
    }
