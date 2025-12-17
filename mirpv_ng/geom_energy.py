# mirpv_ng/geom_energy.py

"""
Energy-related features for hairpins.

Implements:
- adjusted MFE (MFE / length)
- optional helper to compute MFE for a subsequence if needed
"""

from __future__ import annotations

from typing import Tuple, Dict
import subprocess


def compute_adjusted_mfe(mfe: float, length: int) -> float:
    """
    Adjusted MFE = MFE / length.
    Protect against division by zero.
    """
    L = max(length, 1)
    return float(mfe) / float(L)


def fold_subsequence(seq: str, rnafold_bin: str = "RNAfold") -> Tuple[str, float]:
    """
    Optional helper: compute structure + MFE for a given subsequence.

    This is not strictly needed for basic adjusted MFE, but if you want
    region-wise energies (e.g., exact stem energy), you can call this.
    """
    proc = subprocess.Popen(
        [rnafold_bin, "--noPS"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = proc.communicate(seq.strip() + "\n")
    if proc.returncode != 0:
        raise RuntimeError(f"{rnafold_bin} failed: {err}")
    lines = out.strip().splitlines()
    if len(lines) < 2:
        raise RuntimeError("Unexpected RNAfold output")
    struct_line = lines[1]
    parts = struct_line.split()
    struct = parts[0]
    mfe_str = parts[-1].strip("()")
    mfe_val = float(mfe_str)
    return struct, mfe_val


def energy_features_global(mfe: float, seq_len: int) -> Dict[str, float]:
    """
    Convenience wrapper: returns global energy features.
    """
    return {
        "mfe": float(mfe),
        "adj_mfe": compute_adjusted_mfe(mfe, seq_len),
    }
