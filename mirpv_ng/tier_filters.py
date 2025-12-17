# mirpv_ng/tier_filters.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from .geom_hairpin_finder import HairpinGeometry

@dataclass
class TierConfig:
    min_len: int = 40
    max_len: int = 120
    min_pairs: int = 16
    min_mfe: float = -10.0
    max_unpaired_frac: float = 0.7
    max_loop_count: Optional[int] = None
    max_loop_size: Optional[int] = None

def tier1_energy_filter(seq: str, struct: str, mfe: float, cfg: TierConfig) -> bool:
    L = len(seq)
    if L < cfg.min_len or L > cfg.max_len: return False
    if struct.count("(") < cfg.min_pairs: return False
    if mfe > cfg.min_mfe: return False
    if struct.count(".") / float(L) > cfg.max_unpaired_frac: return False
    return True

@dataclass
class GeometryConfig:
    max_num_loops: Optional[int] = None
    max_loop_size: Optional[int] = None
    max_bulge_chain: Optional[int] = None
    max_bulge_density: Optional[float] = None
    max_cactus_score: Optional[float] = None

def tier2_geometry_filter(hp_geom: HairpinGeometry, cfg: GeometryConfig) -> bool:
    if cfg.max_num_loops is not None and hp_geom.num_loops > cfg.max_num_loops: return False
    if cfg.max_loop_size is not None and hp_geom.max_loop_size > cfg.max_loop_size: return False
    
    bulge_feats = hp_geom.bulge_features or {}
    if cfg.max_bulge_chain is not None:
        if float(bulge_feats.get("bulge_chain_count", 0.0)) > cfg.max_bulge_chain: return False
    
    if cfg.max_bulge_density is not None:
        bulge_count = float(bulge_feats.get("bulge_count", 0.0))
        density = bulge_count / max(1, hp_geom.end - hp_geom.start)
        if density > cfg.max_bulge_density: return False
        
    return True