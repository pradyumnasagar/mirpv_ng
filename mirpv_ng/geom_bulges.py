# mirpv_ng/geom_bulges.py

"""
Bulge-related features derived from dot-bracket structure.

Implements:
- biggest bulge on left/right side
- number of bulges
- number of *consecutive* bulges
- number of consecutive bulges on same side

This is a simplified Python analogue of miRNAFold/HairpIndex bulge logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

from .geom_stem_features import get_basepairs, get_hairpin_span


@dataclass
class Bulge:
    start: int
    end: int
    length: int
    side: str   # "left" or "right"
    index: int  # order along stem


def _classify_bulges(seq: str, struct: str) -> List[Bulge]:
    """
    Crude but effective bulge classifier:

    - Work in the hairpin span (from min(i) to max(j) of basepairs).
    - Any contiguous '.' inside span with paired neighbors on at least one side
      is treated as a bulge/loop.
    - side:
        - if positions closer to left arm => "left"
        - else => "right"

    We also assign an ordering index along 5'→3' to detect consecutive bulges.
    """
    pairs = get_basepairs(struct)
    span = get_hairpin_span(pairs)
    if span is None:
        return []
    span_start, span_end = span
    n = len(struct)
    span_start = max(0, span_start)
    span_end = min(n - 1, span_end)

    mid = (span_start + span_end) / 2.0

    bulges: List[Bulge] = []
    i = span_start
    bulge_idx = 0

    while i <= span_end:
        if struct[i] == ".":
            j = i
            while j <= span_end and struct[j] == ".":
                j += 1
            length = j - i

            # classify side based on center position
            center = i + (length - 1) / 2.0
            side = "left" if center <= mid else "right"

            bulges.append(
                Bulge(
                    start=i,
                    end=j - 1,
                    length=length,
                    side=side,
                    index=bulge_idx,
                )
            )
            bulge_idx += 1
            i = j
        else:
            i += 1

    return bulges


def compute_bulge_features(seq: str, struct: str) -> Dict[str, float]:
    """
    Return bulge-related features:
      - bulge_count
      - biggest bulge size left/right
      - bulge_chain_count
      - bulge_chain_same_side_count
    """
    bulges = _classify_bulges(seq, struct)
    if not bulges:
        return {
            "bulge_count": 0.0,
            "bulge_max_left": 0.0,
            "bulge_max_right": 0.0,
            "bulge_chain_count": 0.0,
            "bulge_chain_same_side_count": 0.0,
        }

    max_left = max((b.length for b in bulges if b.side == "left"), default=0)
    max_right = max((b.length for b in bulges if b.side == "right"), default=0)
    bulge_count = len(bulges)

    # Consecutive bulge chains (indices differ by 1)
    bulges_sorted = sorted(bulges, key=lambda b: b.index)
    chain_count = 0
    chain_same_side = 0

    for i in range(len(bulges_sorted) - 1):
        b1 = bulges_sorted[i]
        b2 = bulges_sorted[i + 1]
        # "consecutive" in ordering; you can add spatial checks if needed
        if b2.index == b1.index + 1:
            chain_count += 1
            if b1.side == b2.side:
                chain_same_side += 1

    return {
        "bulge_count": float(bulge_count),
        "bulge_max_left": float(max_left),
        "bulge_max_right": float(max_right),
        "bulge_chain_count": float(chain_count),
        "bulge_chain_same_side_count": float(chain_same_side),
    }

