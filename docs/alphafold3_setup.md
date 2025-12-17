# AlphaFold3-RNA Setup for miRPV-NG (Optional)

miRPV-NG can optionally interface with **AlphaFold3** to predict **3D structures of top pre-miRNA candidates**.  
This is **not required** for core miRNA discovery: all main results (hairpin detection, scoring, sRNA-seq mode) rely only on 2D structure (RNAfold).

Because of AlphaFold3’s license and distribution model:

- miRPV-NG **does not** bundle AlphaFold3 code, weights, or databases.
- Users must **install AlphaFold3 separately** (non-commercial research use only).
- miRPV-NG only provides a **thin wrapper** (`annotate-3d` CLI) around an existing AF3 setup.

This document explains a recommended setup for **Linux + Docker**.

---

## 1. Requirements

You will need:

- A Linux machine with:
  - Recent NVIDIA GPU (e.g. ≥ 16 GB VRAM recommended)
  - CUDA-compatible drivers installed
- Docker with NVIDIA support:
  - [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Sufficient disk space (100+ GB) for AF3 models + databases
- A valid research use case that complies with the AlphaFold3 license

miRPV-NG itself does **not** require a GPU; only the optional AF3 integration does.

---

## 2. Obtain AlphaFold3 code, weights, and databases

Follow the **official AlphaFold3 documentation** (DeepMind / Google) to:

1. Request access to the AlphaFold3 weights / models.
2. Download the **AlphaFold3 Docker image** (or build from source).
3. Download required **databases** (RNA-related, PDB, etc.).

We cannot redistribute links or scripts that bypass the official process.  
Always refer to the official AlphaFold3 repository and documentation for:

- `models/` directory content  
- `databases/` directory layout  
- recommended Docker images and tags

---

## 3. Directory layout

We recommend the following layout on your host:

```bash
$HOME/alphafold3/
  models/       # AF3 models / weights
  databases/    # AF3 databases
  run/          # working directory (inputs/outputs/logs)
