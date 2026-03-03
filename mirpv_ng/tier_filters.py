# mirpv_ng/tier_filters.py
"""
Tier filtering module for miRPV-NG pre-miRNA classification.

This module provides configuration classes and filter functions for the
multi-tier filtering system used to reduce false positives:

- **Tier 1**: Hard energy/pairs filter based on MFE and base pair count
- **Tier 2**: Soft geometry filter based on loop/bulge structure (optional)

Classes:
    TierConfig: Configuration for tier-1 energy filter thresholds.
    GeometryConfig: Configuration for tier-2 geometry filter thresholds.
    Tier2Config: Configuration for tier-2 soft-gating penalty system.

Functions:
    tier1_energy_filter: Apply tier-1 hard filter.
    tier2_geometry_filter: Apply tier-2 geometry constraints.
    tier2_soft_features: Compute soft penalty features (never drops candidates).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

# --- Tier-1 (energy/pairs) config + filter -----------------------------------

@dataclass
class TierConfig:
    min_len: int = 40
    max_len: int = 120
    min_pairs: int = 18
    min_mfe: float = -15.0
    max_unpaired_frac: float = 0.8


def tier1_energy_filter(seq: str, struct: str, mfe: float, cfg: TierConfig) -> bool:
    L = len(seq)
    if L < cfg.min_len or L > cfg.max_len:
        return False

    # paired count = number of '(' (or ')')
    pairs = struct.count("(")
    if pairs < cfg.min_pairs:
        return False

    if mfe > cfg.min_mfe:
        return False

    unpaired_frac = struct.count(".") / float(len(struct) or 1)
    if unpaired_frac > cfg.max_unpaired_frac:
        return False

    return True


# --- Tier-2 geometry config + filter (kept permissive by default) ------------

@dataclass
class GeometryConfig:
    max_num_loops: Optional[int] = None
    max_loop_size: Optional[int] = None
    max_bulge_chain: Optional[int] = None
    max_bulge_density: Optional[float] = None
    max_cactus_score: Optional[float] = None


def tier2_geometry_filter(hp: Any, cfg: GeometryConfig) -> bool:
    """
    Returns True if passes geometry constraints.
    If all cfg fields are None => geometry filtering is effectively disabled.
    Works even if hp doesn't have specific attributes (then it won't filter).
    """
    if (
        cfg.max_num_loops is None and
        cfg.max_loop_size is None and
        cfg.max_bulge_chain is None and
        cfg.max_bulge_density is None and
        cfg.max_cactus_score is None
    ):
        return True

    # Only apply checks if the attribute exists on hp
    num_loops = getattr(hp, "num_loops", None)
    if cfg.max_num_loops is not None and num_loops is not None:
        if num_loops > cfg.max_num_loops:
            return False

    loop_size = getattr(hp, "loop_size", None)
    if cfg.max_loop_size is not None and loop_size is not None:
        if loop_size > cfg.max_loop_size:
            return False

    bulge_chain = getattr(hp, "bulge_chain", None)
    if cfg.max_bulge_chain is not None and bulge_chain is not None:
        if bulge_chain > cfg.max_bulge_chain:
            return False

    bulge_density = getattr(hp, "bulge_density", None)
    if cfg.max_bulge_density is not None and bulge_density is not None:
        if bulge_density > cfg.max_bulge_density:
            return False

    cactus_score = getattr(hp, "cactus_score", None)
    if cfg.max_cactus_score is not None and cactus_score is not None:
        if cactus_score > cfg.max_cactus_score:
            return False

    return True

# --- Tier-2 soft gating (do NOT drop candidates) -----------------------------


@dataclass
class Tier2Config:
    enabled: bool = False

    # “reasonable” defaults; we’ll tune later on your train/neg sets
    mfe_per_nt_max: float = -0.20     # e.g. -0.20 kcal/mol/nt (more negative is better)
    loop_len_max: int = 22
    bulge_frac_max: float = 0.30      # bulged positions / paired positions-ish
    unpaired_frac_max: float = 0.55   # unpaired fraction in whole structure

    # penalty scales (controls smoothness)
    mfe_scale: float = 0.05
    loop_scale: float = 5.0
    bulge_scale: float = 0.10
    unpaired_scale: float = 0.10

    # weights
    w_mfe: float = 1.0
    w_loop: float = 0.7
    w_bulge: float = 1.0
    w_unpaired: float = 0.8


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0:
        return default
    return a / b


def _ramp_excess(x: float, limit: float, scale: float) -> float:
    """0 if within limit, else smooth penalty increasing with excess."""
    if scale <= 0:
        scale = 1.0
    excess = x - limit
    if excess <= 0:
        return 0.0
    return excess / scale


def tier2_soft_features(seq: str, struct: str, mfe: float, cfg: Optional[Tier2Config] = None) -> Dict[str, float]:
    """
    Soft-gated Tier-2: returns penalty + pass flag + diagnostic metrics.
    Never filters out candidates.
    """
    if cfg is None:
        cfg = Tier2Config(enabled=False)

    # Always return keys (schema stability); if disabled => zeros.
    out = {
        "tier2_enabled": 1.0 if cfg.enabled else 0.0,
        "tier2_pass": 1.0,
        "tier2_penalty": 0.0,
        "tier2_mfe_per_nt": 0.0,
        "tier2_loop_len": 0.0,
        "tier2_unpaired_frac": 0.0,
        "tier2_bulge_frac": 0.0,
    }
    if not cfg.enabled:
        return out

    L = max(1, len(seq))
    mfe_per_nt = mfe / float(L)

    # loop length = longest run of '.' in the middle-ish; simple approximation.
    # (Good enough for gating; we can refine using your geometry modules later.)
    max_dot_run = 0
    run = 0
    for ch in struct:
        if ch == ".":
            run += 1
            max_dot_run = max(max_dot_run, run)
        else:
            run = 0
    loop_len = max_dot_run

    unpaired = struct.count(".")
    unpaired_frac = _safe_div(unpaired, len(struct), 0.0)

    paired = struct.count("(") + struct.count(")")
    # bulge-ish proxy: unpaired relative to paired (clipped)
    bulge_frac = _safe_div(unpaired, max(1, paired), 0.0)
    bulge_frac = min(1.0, bulge_frac)

    # penalties: note sign logic for mfe_per_nt
    # We want mfe_per_nt to be <= cfg.mfe_per_nt_max (more negative is better).
    # So penalize if mfe_per_nt is higher (less negative) than allowed.
    p_mfe = _ramp_excess(mfe_per_nt, cfg.mfe_per_nt_max, cfg.mfe_scale)
    p_loop = _ramp_excess(loop_len, float(cfg.loop_len_max), cfg.loop_scale)
    p_bulge = _ramp_excess(bulge_frac, cfg.bulge_frac_max, cfg.bulge_scale)
    p_unp = _ramp_excess(unpaired_frac, cfg.unpaired_frac_max, cfg.unpaired_scale)

    penalty = cfg.w_mfe * p_mfe + cfg.w_loop * p_loop + cfg.w_bulge * p_bulge + cfg.w_unpaired * p_unp

    out.update({
        "tier2_pass": 1.0 if penalty <= 0 else 0.0,
        "tier2_penalty": float(penalty),
        "tier2_mfe_per_nt": float(mfe_per_nt),
        "tier2_loop_len": float(loop_len),
        "tier2_unpaired_frac": float(unpaired_frac),
        "tier2_bulge_frac": float(bulge_frac),
    })
    return out
