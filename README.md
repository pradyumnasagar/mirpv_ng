# miRPV-NG: Next-Generation pre-miRNA Prediction Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**miRPV-NG** (microRNA Prediction Verification - Next Gen) is a comprehensive pipeline for discovering and validating pre-miRNA candidates from small RNA-seq data or genomic sequences. It integrates tiered feature sets, Random Forest classification with probability calibration, and rigorous filtering steps to provide high-precision miRNA annotations.

## Key Features

*   **Calibrated Classification**: Uses a calibrated Random Forest model (isotonic calibration by default on held-out data, with sigmoid also supported) to provide meaningful probability scores.
*   **Tiered Feature Extraction**: Combines thermodynamic features (Tier 1) with soft-gated structural geometry checks (Tier 2).
*   **End-to-End Pipeline**: Handles raw FASTQ trimming, mapping (Bowtie), peak calling, candidate excision, and final reporting.
*   **Performance**: Multiprocessing support for high-throughput analysis.

## Model Specificity

> **Important**: miRBase hairpins are **never** used as negatives for published models. A deprecated legacy option exists in the training utilities for backward compatibility and must not be used for published training.

### Training Philosophy

- **Positives (Label 1)**: MirGeneDB gold-standard pre-miRNAs
- **N1 Negatives**: Dinucleotide-shuffled positives (composition controls)
- **N2 Negatives**: Scanner-matched genomic background (with miRBase loci excluded)
- **N3 Negatives**: Confusable structured RNAs (tRNA/snoRNA/snRNA/rRNA) — NOT miRBase!

miRBase may contain true miRNAs not yet validated in MirGeneDB. Using miRBase as negatives would cause systematic false negatives. Instead, N3 uses non-miRNA structured RNAs that look hairpin-like but are definitively not miRNAs.

### N3 Decoy Sources

To build the negative set, provide a FASTA of structured RNAs:

```bash
python training/build_negatives_v2.py \
    --positives data/train/hsa_mirgene_premirna.fa \
    --genome refs/hg38_primary/hg38.primary.fa \
    --n3-decoys-fasta refs/ncRNA/hsa_structured_rna.fa \
    --exclude-bed refs/known/hsa_known_mirnas.bed \
    --out data/train/negatives_v3.fa \
    --n1-count 2000 --n2-count 1500 --n3-count 500
```

### Sanity Check for Low-Scoring Hairpins

If a hairpin scores unexpectedly low, check if it's in the training data:

```bash
python training/check_n3_overlap.py \
    --seq "AUGC...your_hairpin_sequence..." \
    --negatives data/train/negatives_v3.fa \
    --positives data/train/hsa_mirgene_premirna.fa
```

### Sequence-Only Validation

The `validate-seqonly` command provides two-stage validation for sequence-only inputs, handling miRBase-style truncated hairpins:

- **Stage A**: Mature-centric duplex evidence (normalized 0-1 using percentile ranking)
- **Stage B**: Precursor-centric RF scoring on best hairpin region

miRBase hairpins may have incorrect or truncated precursor boundaries. This command detects mature-like duplex regions (Stage A) and assesses precursor plausibility (Stage B), providing actionable recommendations including:
- **HIGH_CONFIDENCE_PREMIRNA**: Strong precursor + mature-like evidence
- **MATURE_LIKE_BUT_TRUNCATED_PRECURSOR**: Good mature evidence, but precursor context weak (suggests re-extraction)
- **WEAK_EVIDENCE**: Some evidence, not definitive
- **NO_SUPPORT**: Unlikely under model assumptions

```bash
# Basic usage with miRBase hairpins (without genome)
python -m mirpv_ng.cli validate-seqonly \
  --fasta refs/known/hsa/hairpin.hsa_only.fa \
  --premirna-model models/hsa_premirna_rf_v10_sigmoid.pkl \
  --out results/validate_mirbase.tsv

# With mature model for better Stage A scoring
python -m mirpv_ng.cli validate-seqonly \
  --fasta refs/known/hsa/hairpin.hsa_only.fa \
  --premirna-model models/hsa_premirna_rf_v10_sigmoid.pkl \
  --mature-model models/hsa_mature_xgbrank_v3.pkl \
  --out results/validate_mirbase.tsv

# With genome for coordinate-based re-extraction
# (FASTA headers must contain chr:start-end pattern)
python -m mirpv_ng.cli validate-seqonly \
  --fasta examples/candidates_with_coords.fa \
  --genome refs/hg38_primary/hg38.primary.fa \
  --premirna-model models/hsa_premirna_rf_v10_sigmoid.pkl \
  --out results/validate_reextract.tsv \
  --flank 30 \
  --emit-extracted-fasta results/extracted.fa
```


## Installation

### Prerequisites
*   Python 3.9+
*   `RNAfold` (from ViennaRNA package) must be in your PATH.
*   `bowtie`, `samtools`, `bedtools`, `cutadapt` (for sRNA-seq pipeline).

### Install via pip

```bash
git clone https://github.com/pradyumnasagar/mirpv_ng.git
cd mirpv_ng
pip install .
```

## Quick Start

### 1. Acceptance Test (Verify Installation)
Run the end-to-end verification script to ensure everything is working:

```bash
bash scripts/acceptance_test.sh
```

### 2. Training a New Model (Gold Standard)
To train a model entirely from scratch with the publication-grade pipeline:

**A. Build Negative Set (N1/N2/N3)**
Constructs a balanced negative set mixing composition controls, scanner-matched background, and hard decoys.

```bash
python training/build_negatives_v2.py \
    --positives data/train/hsa_mirgene_premirna.fa \
    --genome refs/hg38_primary/hg38.primary.fa \
    --mirbase-hairpins refs/known/hsa/hairpin.fa \
    --exclude-bed refs/known/hsa_known_mirnas.bed \
    --out data/train/negatives_v2.fa \
    --n1-count 2000 --n2-count 1500 --n3-count 500 \
    --threads 8
```

**B. Train & Calibrate Model**
Trains a Random Forest classifier with probability calibration and scan-background injection.
By default, `training/train_premirna_model.py` uses isotonic calibration on a held-out
validation set (`--calibration isotonic`). When the validation set is small (e.g. fewer than
~1000 validation examples), sigmoid calibration (`--calibration sigmoid`) is recommended
to avoid potential isotonic overfitting or quantization artifacts.

```bash
python training/train_premirna_model.py \
    --pos-fasta data/train/hsa_mirgene_premirna.fa \
    --neg-fasta data/train/negatives_v2.fa \
    --model-out models/hsa_premirna_rf_v8.pkl \
    --metrics-out models/hsa_premirna_rf_v8_metrics.txt \
    --calibration isotonic \
    --scan-background-fasta data/train/negatives_v2.fa \
    --scan-background-limit 500 \
    --threads 8
```

### 3. Sequence-Only Mode (Score FASTA)
Score candidate hairpins provided in a FASTA file without sequencing data.

```bash
python training/score_scan_candidates.py \
    --genome refs/hg38_primary/hg38.primary.fa \
    --model models/hsa_premirna_rf_v8.pkl \
    --out analysis/scan_candidate_scores.txt \
    --max-candidates 2000 \
    --threads 8
```

### 4. sRNA-Seq Pipeline (FASTQ to Final Report)
Run the full discovery pipeline on a small RNA-seq sample using the provided helper scripts.

```bash
# Example: run full pipeline on normal samples
bash scripts/run_normals_full.sh

# Example: run full pipeline on premalignant samples
bash scripts/run_pre_full.sh
```

## Documentation
Full documentation is available in the `docs/` directory.

## Citation
If you use miRPV-NG in your research, please cite:
*Available soon*
