# mirpv_ng/geom_hairpin_finder.py
"""
Hairpin geometry detection for miRPV-NG.

This module identifies and characterizes RNA hairpin structures from
secondary structure predictions. It extracts geometric features used
for pre-miRNA classification.

Classes:
    HairpinGeometry: Dataclass containing all hairpin geometric properties.

Functions:
    find_primary_hairpin: Find the dominant hairpin in a structure.
    find_hairpins: Find all hairpins (currently returns primary only).

Example:
    >>> from mirpv_ng.geom_hairpin_finder import find_primary_hairpin
    >>> hp = find_primary_hairpin("GGCCAUUAGGCC", "((((....))))")
    >>> print(hp.num_pairs, hp.anchor_stem_len)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from .geom_stem_features import get_basepairs, StemRun, find_exact_stem_runs
from .geom_bulges import compute_bulge_features

@dataclass
class HairpinGeometry:
    start: int
    end: int
    loop_start: int
    loop_end: int
    num_pairs: int
    stem_runs: List[StemRun]
    helix_count: int
    helix_max_len: int
    helix_mean_len: float
    anchor_stem_len: int
    num_loops: int
    max_loop_size: int
    mean_loop_size: float
    bulge_features: Dict[str, float]
    unpaired_frac: float
    extra: Dict[str, float]

def _loop_stats(struct: str, start: int, end: int) -> Tuple[int, int, float]:
    segments = []
    in_loop = False
    seg_start = -1
    for idx in range(start, end):
        if struct[idx] == ".":
            if not in_loop: in_loop = True; seg_start = idx
        else:
            if in_loop: segments.append(idx - seg_start); in_loop = False
    if in_loop: segments.append(end - seg_start)
    if not segments: return 0, 0, 0.0
    return len(segments), max(segments), sum(segments) / float(len(segments))

def find_primary_hairpin(seq: str, struct: str) -> Optional[HairpinGeometry]:
    L = len(seq)
    if L == 0: return None
    basepairs = get_basepairs(struct)
    if not basepairs: return None
    runs = find_exact_stem_runs(basepairs)
    if not runs: return None

    anchor = max(runs, key=lambda r: r.length)
    hp_start = anchor.start_i
    hp_end = anchor.start_j + 1
    loop_start = anchor.start_i + anchor.length
    loop_end = anchor.start_j - anchor.length + 1
    if loop_start >= loop_end: loop_start = loop_end

    stem_runs = [r for r in runs if (r.start_i >= hp_start and (r.start_i + r.length - 1) < hp_end)]
    if not stem_runs: stem_runs = [anchor]
    
    lengths = [r.length for r in stem_runs]
    num_loops, max_loop_size, mean_loop_size = _loop_stats(struct, hp_start, hp_end)
    bulge_feats = compute_bulge_features(seq, struct)
    unpaired_frac = struct.count(".") / float(L)

    return HairpinGeometry(
        start=hp_start, end=hp_end, loop_start=loop_start, loop_end=loop_end,
        num_pairs=len(basepairs), stem_runs=stem_runs,
        helix_count=len(lengths), helix_max_len=max(lengths), helix_mean_len=sum(lengths)/len(lengths),
        anchor_stem_len=anchor.length,
        num_loops=num_loops, max_loop_size=max_loop_size, mean_loop_size=mean_loop_size,
        bulge_features=bulge_feats, unpaired_frac=unpaired_frac, extra={}
    )

def find_hairpins(seq: str, struct: str) -> List[HairpinGeometry]:
    hp = find_primary_hairpin(seq, struct)
    return [hp] if hp is not None else []