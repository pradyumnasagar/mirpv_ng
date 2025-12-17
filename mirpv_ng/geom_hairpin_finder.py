# mirpv_ng/geom_hairpin_finder.py

"""
Geometry-based hairpin finder for miRPV-NG.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from .geom_stem_features import get_basepairs, StemRun
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

def _find_stem_runs(basepairs: List[Tuple[int, int]]) -> List[StemRun]:
    if not basepairs: return []
    bp_sorted = sorted(basepairs, key=lambda x: x[0])
    runs: List[StemRun] = []
    cur_i, cur_j = bp_sorted[0]
    cur_len = 1
    for (i, j) in bp_sorted[1:]:
        if i == cur_i + 1 and j == cur_j - 1:
            cur_len += 1; cur_i, cur_j = i, j
        else:
            runs.append(StemRun(start_i=cur_i - cur_len + 1, start_j=cur_j + cur_len - 1, length=cur_len))
            cur_i, cur_j = i, j; cur_len = 1
    runs.append(StemRun(start_i=cur_i - cur_len + 1, start_j=cur_j + cur_len - 1, length=cur_len))
    return runs

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
    runs = _find_stem_runs(basepairs)
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