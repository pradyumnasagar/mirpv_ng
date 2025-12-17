# mirpv_ng/structure3d_af3.py

"""
Optional AlphaFold3-RNA integration for miRPV-NG.

This module does NOT bundle AlphaFold3. It assumes the user
has installed the official AF3 stack (e.g. Docker image +
weights + databases) and exposes a thin Python wrapper to
run AF3 on short RNA sequences (pre-miRNA precursors).

Intended use:
  - run AF3 on top-N candidates from miRPV-NG
  - collect PDBs for visualisation / downstream analysis

All paths/commands are user-configurable and can be turned
off completely in normal runs.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class AF3Config:
    """
    Configuration needed to run AlphaFold3 for RNA.

    We keep this intentionally generic:
      - docker_image: name of AF3 docker image
      - model_dir: where AF3 weights/models live (on host)
      - db_dir: where AF3 databases live (on host)
      - extra_args: list of extra CLI flags for AF3 runner
    """
    docker_image: str
    model_dir: Path
    db_dir: Path
    extra_args: Optional[List[str]] = None


def write_af3_input_json(seq_id: str, seq: str, out_dir: Path) -> Path:
    """
    Write a minimal AF3 input JSON for a single RNA chain.

    This is a placeholder; you will need to adapt it to the
    exact JSON schema expected by the official AF3 code.

    Returns:
        Path to the JSON file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict = {
        "name": seq_id,
        "sequence": seq.replace("T", "U"),  # AF3 expects RNA
        "description": "miRPV-NG pre-miRNA candidate",
    }
    json_path = out_dir / f"{seq_id}.json"
    with json_path.open("w") as f:
        json.dump(payload, f)
    return json_path


def run_af3_for_rna(
    seq_id: str,
    seq: str,
    out_dir: Path,
    cfg: AF3Config,
    max_len: int = 120,
    dry_run: bool = False,
) -> Path:
    """
    Run AlphaFold3 on a short RNA sequence via Docker.

    Args:
        seq_id: identifier for this sequence (used for naming)
        seq: RNA/DNA sequence (A/C/G/U or A/C/G/T)
        out_dir: directory where AF3 outputs will be written
        cfg: AF3Config with docker_image, model_dir, db_dir
        max_len: safety guard on sequence length
        dry_run: if True, print command and return expected PDB path
                 without actually running AF3.

    Returns:
        Path to predicted PDB file (expected location). The caller
        should check existence and handle failures.

    Notes:
        - Assumes AF3 docker image exposes a CLI like:
          `run_alphafold3_rna --input_json ... --output_dir ...`
        - You must adapt the command template to your local AF3 setup.
    """
    if len(seq) > max_len:
        raise ValueError(f"Sequence length {len(seq)} exceeds max_len={max_len}")

    out_dir = out_dir.resolve()
    json_path = write_af3_input_json(seq_id, seq, out_dir)

    # Mount model + db directories into the container
    model_dir = cfg.model_dir.resolve()
    db_dir = cfg.db_dir.resolve()

    pdb_out = out_dir / f"{seq_id}.pdb"

    base_cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{json_path.parent}:/inputs",
        "-v",
        f"{out_dir}:/outputs",
        "-v",
        f"{model_dir}:/models",
        "-v",
        f"{db_dir}:/db",
        cfg.docker_image,
        # below is pseudo-CLI for AF3; adapt to real script
        "bash",
        "-lc",
        (
            "run_alphafold3_rna "
            "--input_json /inputs/{json} "
            "--output_dir /outputs "
            "--model_dir /models "
            "--db_dir /db "
        ).format(json=json_path.name),
    ]

    if cfg.extra_args:
        base_cmd.extend(cfg.extra_args)

    if dry_run:
        print("[af3] DRY RUN:", " ".join(str(x) for x in base_cmd))
        return pdb_out

    print("[af3] Running AlphaFold3 for", seq_id)
    print("[af3] Command:", " ".join(str(x) for x in base_cmd))

    try:
        subprocess.check_call(base_cmd)
    except subprocess.CalledProcessError as e:
        print(f"[af3] AlphaFold3 failed for {seq_id}: {e}")
        # caller can decide what to do; we still return the expected path
        return pdb_out

    return pdb_out
