# Performance and Parallelism Guide

miRPV-NG supports production-scale parallel execution across all heavy computational steps.

## Quick Start

```bash
# Workstation: use 4 parallel workers
python -m mirpv_ng.cli score-fasta \
  --fasta candidates.fa --model models/hsa_rf_v7.pkl \
  --out scored.tsv --jobs 4

# HPC: use SLURM-allocated CPUs
python -m mirpv_ng.cli score-fasta \
  --fasta candidates.fa --model models/hsa_rf_v7.pkl \
  --out scored.tsv --jobs $SLURM_CPUS_PER_TASK
```

## Parallel CLI Flags

All parallelizable commands support these flags:

| Flag | Default | Description |
|------|---------|-------------|
| `-j, --jobs` | 1 | Number of parallel workers |
| `--backend` | process | Parallelization backend (process/thread) |
| `--chunksize` | 50 | Batch size for parallel processing |
| `--tmpdir` | system | Temporary directory for intermediate files |
| `--stable-order` | true | Preserve output ordering |

### Environment Variables & Thread Logic

Centralized logic determines the default thread count with the following precedence:

1.  **User Argument**: `--threads` / `--jobs` / `-j` flag (if > 0)
2.  **SLURM**: `SLURM_CPUS_PER_TASK` environment variable
3.  **PBS**: `PBS_NP` environment variable
4.  **SGE/GE**: `NSLOTS` environment variable
5.  **OMP**: `OMP_NUM_THREADS` environment variable
6.  **OS Detect**: `os.cpu_count()` (all available cores)
7.  **Fallback**: 1

You can override defaults globally:
```bash
export MIRPV_JOBS=4
export MIRPV_BACKEND=process
export MIRPV_TMPDIR=/scratch/$USER/mirpv_tmp
```

## Commands with Parallel Support

### score_scan_candidates.py (Sequence-Only Scoring)
```bash
python training/score_scan_candidates.py \
  --genome genome.fa \
  --model model.pkl \
  --out scores.txt \
  --threads 8
```

### score-fasta (CLI / Legacy)
```bash
python -m mirpv_ng.cli score-fasta \
  --fasta input.fa --model model.pkl --out out.tsv \
  -j 8
```

### build_negatives_v2.py (Training Data Generation)
```bash
python training/build_negatives_v2.py \
  --positives pos.fa --mirbase-hairpins hairpin.fa \
  --genome genome.fa --out negatives.fa \
  --jobs 8
```

### analyze_candidate_distribution.py
```bash
python training/analyze_candidate_distribution.py \
  --genome genome.fa \
  --positives pos.fa \
  --negatives neg.fa \
  --threads 4
```

### tail_risk_report.py
```bash
python training/tail_risk_report.py \
  --model model.pkl \
  --negatives negatives.fa \
  --out report.tsv \
  --threads 4
```

## SLURM Job Scripts

### Single Sample Job
```bash
#!/bin/bash
#SBATCH --job-name=mirpv_scan
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=mirpv_%j.log

# Set environment
export MIRPV_JOBS=$SLURM_CPUS_PER_TASK
export MIRPV_TMPDIR=$TMPDIR

# Run scoring
python -m mirpv_ng.cli score-fasta \
  --fasta $INPUT_FASTA \
  --model $MODEL_PATH \
  --out $OUTPUT_TSV \
  -j $SLURM_CPUS_PER_TASK
```

### Multi-Sample Job Array
For processing multiple samples in parallel:

```bash
#!/bin/bash
#SBATCH --job-name=mirpv_array
#SBATCH --array=1-10%4      # 10 samples, 4 concurrent
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=mirpv_%A_%a.log

# Read sample from list
SAMPLE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" samples.txt)
INPUT_FASTA="data/${SAMPLE}/${SAMPLE}.fa"
OUTPUT_TSV="results/${SAMPLE}/scored.tsv"

# Set environment
export MIRPV_JOBS=$SLURM_CPUS_PER_TASK
export MIRPV_TMPDIR=$TMPDIR

# Create output directory
mkdir -p "results/${SAMPLE}"

# Run
python -m mirpv_ng.cli score-fasta \
  --fasta $INPUT_FASTA \
  --model models/hsa_rf_v7.pkl \
  --out $OUTPUT_TSV \
  -j $SLURM_CPUS_PER_TASK
```

### Training Job (Negative Generation + Model Training)
```bash
#!/bin/bash
#SBATCH --job-name=mirpv_train
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=training_%j.log

export MIRPV_JOBS=$SLURM_CPUS_PER_TASK
export MIRPV_TMPDIR=$TMPDIR

# Generate negatives with parallel folding
python training/build_negatives_v2.py \
  --positives data/train/positives.fa \
  --mirbase-hairpins data/train/hairpin.fa \
  --genome refs/hg38.primary.fa \
  --exclude-bed refs/known_mirnas.bed \
  --out data/train/negatives_v2.fa \
  --report data/train/negatives_v2_report.tsv \
  -j $SLURM_CPUS_PER_TASK

# Train model
python training/train_premirna_model.py \
  --pos-fasta data/train/positives.fa \
  --neg-fasta data/train/negatives_v2.fa \
  --model-out models/hsa_rf_v7.pkl \
  --metrics-out models/hsa_rf_v7.metrics.txt \
  --tier2
```

## Performance Tips

### Batch Size Tuning
- **Large batches** (100-200): Better for HPC with fast local storage
- **Small batches** (25-50): Better for workstations or network storage

### Memory Considerations
- Each worker uses ~100-200MB for RNAfold batching
- Total memory: ~200MB × n_jobs + base process
- For 8 jobs, allocate at least 4GB

### Avoiding Oversubscription
When external tools (bowtie, RNAfold) already use threads:
```bash
# If RNAfold uses 2 threads internally, divide jobs
EFFECTIVE_JOBS=$(( SLURM_CPUS_PER_TASK / 2 ))
python -m mirpv_ng.cli score-fasta -j $EFFECTIVE_JOBS
```

### Determinism with Parallelism
When `--seed` is set and `--stable-order` is true (default):
- Results are deterministic regardless of job count
- `jobs=1` and `jobs=4` produce identical outputs
- Essential for reproducible research

## Verification

Test that parallelism preserves correctness:
```bash
# Run with 1 job
python -m mirpv_ng.cli score-fasta \
  --fasta test.fa --model model.pkl --out out_j1.tsv -j 1

# Run with 4 jobs
python -m mirpv_ng.cli score-fasta \
  --fasta test.fa --model model.pkl --out out_j4.tsv -j 4

# Verify identical output
diff out_j1.tsv out_j4.tsv
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  mirpv_ng/parallel.py                   │
├─────────────────────────────────────────────────────────┤
│  ParallelConfig     - Unified configuration dataclass   │
│  add_parallel_args  - CLI argument parser helper        │
│  run_rnafold_batch  - Single-process batch folding      │
│  run_rnafold_parallel - Multi-worker batch folding      │
│  ExecutorContext    - Worker pool manager               │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│  CLI Commands          Training Scripts                 │
├────────────────────────┬────────────────────────────────┤
│  score-fasta           │  build_negatives_v2.py         │
│  predict-mature        │  analyze_candidate_dist.py     │
│  fastq-to-peaks        │  score_distribution_check.py   │
└────────────────────────┴────────────────────────────────┘
```

## Limitations

1. **Process backend recommended**: Thread backend limited by Python GIL
2. **Temporary files**: Each batch creates temp files; ensure tmpdir has space
3. **Memory bounded**: Streaming design avoids loading entire genome into memory
4. **No GPU**: RNAfold and scikit-learn RF are CPU-only
