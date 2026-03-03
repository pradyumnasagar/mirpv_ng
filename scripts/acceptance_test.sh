#!/bin/bash
# acceptance_test.sh
# miRPV-NG Gold Standard Acceptance Test
# Verifies end-to-end functionality using REAL repository paths and flags.

set -e

# --- Configuration (Real Repo Paths) ---
# Inputs
POSITIVES="data/train/hsa_mirgene_premirna.fa"
MIRBASE_HAIRPINS="refs/known/hsa/hairpin.fa"
GENOME="refs/hg38_primary/hg38.primary.fa"
EXCLUDE_BED="refs/known/hsa_known_mirnas.bed"

# Outputs (Tmp Dir)
OUT_DIR="/tmp/mirpv_acceptance"
NEGATIVES_N2="${OUT_DIR}/negatives_v2.fa"
NEGATIVES_REPORT="${OUT_DIR}/negatives_v2_report.tsv"
MODEL_OUT="${OUT_DIR}/model.pkl"
METRICS_OUT="${OUT_DIR}/train.metrics.txt"
SCAN_CANDIDATES="${OUT_DIR}/scan_candidates.txt"
SCAN_HIST="${OUT_DIR}/scan_hist.png"
SCAN_FASTA="${OUT_DIR}/scan_candidates.fa"

# --- Cleanup & Setup ---
echo "[SETUP] Cleaning and creating ${OUT_DIR}..."
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

# --- Check Prerequisites ---
echo "[CHECK] Verifying input files..."
for f in "$POSITIVES" "$MIRBASE_HAIRPINS" "$GENOME" "$EXCLUDE_BED"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing required file: $f"
        exit 1
    fi
done

if ! command -v RNAfold &> /dev/null; then
    echo "ERROR: RNAfold not found in PATH"
    exit 1
fi

# --- Step 1: Build Negatives (V2) ---
echo "[STEP 1] Building Negatives (N1=200, N2=200, N3=50)..."
python training/build_negatives_v2.py \
    --positives "$POSITIVES" \
    --mirbase-hairpins "$MIRBASE_HAIRPINS" \
    --genome "$GENOME" \
    --exclude-bed "$EXCLUDE_BED" \
    --out "$NEGATIVES_N2" \
    --report "$NEGATIVES_REPORT" \
    --n1-count 200 --n2-count 200 --n3-count 50 \
    --seed 42 \
    --jobs 4

# Check outputs
if [ ! -s "$NEGATIVES_N2" ]; then echo "ERROR: Negatives FASTA missing or empty"; exit 1; fi
if [ ! -s "$NEGATIVES_REPORT" ]; then echo "ERROR: Negatives report missing or empty"; exit 1; fi

# --- Step 2: Train Model ---
echo "[STEP 2] Training Model (Isotonic)..."
python training/train_premirna_model.py \
    --pos-fasta "$POSITIVES" \
    --neg-fasta "$NEGATIVES_N2" \
    --model-out "$MODEL_OUT" \
    --metrics-out "$METRICS_OUT" \
    --model-version "acceptance_v1" \
    --tier2 \
    --calibration isotonic \
    --seed 42 \
    --threads 4

# Check outputs
if [ ! -s "$MODEL_OUT" ]; then echo "ERROR: Model PKL missing or empty"; exit 1; fi
if [ ! -s "$METRICS_OUT" ]; then echo "ERROR: Metrics file missing or empty"; exit 1; fi

# --- Step 3: Scan Candidates (Verification + Background Gen) ---
echo "[STEP 3] Scoring Scan Candidates (threads=auto)..."
python training/score_scan_candidates.py \
    --genome "$GENOME" \
    --model "$MODEL_OUT" \
    --out "$SCAN_CANDIDATES" \
    --histogram "$SCAN_HIST" \
    --write-candidates "$SCAN_FASTA" \
    --max-candidates 200 \
    --threads auto

# Check outputs
if [ ! -s "$SCAN_CANDIDATES" ]; then echo "ERROR: Scan candidates report missing or empty"; exit 1; fi
if [ ! -s "$SCAN_HIST" ]; then echo "ERROR: Scan histogram missing or empty"; exit 1; fi
if [ ! -s "$SCAN_FASTA" ]; then echo "ERROR: Scan candidates FASTA missing or empty"; exit 1; fi

# --- Step 4: Validate-Seqonly (without genome) ---
echo "[STEP 4] Running validate-seqonly on miRBase hairpins (no genome)..."
VALIDATE_OUT="${OUT_DIR}/validate_mirbase.tsv"

# Create hsa-only hairpins if not exists
HSA_HAIRPINS="refs/known/hsa/hairpin.hsa_only.fa"
if [ ! -f "$HSA_HAIRPINS" ]; then
    awk 'BEGIN{keep=0} /^>/{keep=($0 ~ /^>hsa-/)} {if(keep) print}' "$MIRBASE_HAIRPINS" > "$HSA_HAIRPINS"
fi

python -m mirpv_ng.cli validate-seqonly \
    --fasta "$HSA_HAIRPINS" \
    --premirna-model "$MODEL_OUT" \
    --out "$VALIDATE_OUT"

if [ ! -s "$VALIDATE_OUT" ]; then echo "ERROR: validate-seqonly output missing or empty"; exit 1; fi
echo "[STEP 4] validate-seqonly (no genome) completed: $VALIDATE_OUT"

# --- Step 5: Validate-Seqonly with coordinate re-extraction ---
echo "[STEP 5] Running validate-seqonly with genome re-extraction..."
VALIDATE_COORDS_OUT="${OUT_DIR}/validate_coords.tsv"
EXTRACTED_FASTA="${OUT_DIR}/extracted_precursors.fa"
COORDS_FASTA="examples/candidates_with_coords.fa"

if [ -f "$COORDS_FASTA" ] && [ -f "$GENOME" ]; then
    python -m mirpv_ng.cli validate-seqonly \
        --fasta "$COORDS_FASTA" \
        --genome "$GENOME" \
        --premirna-model "$MODEL_OUT" \
        --out "$VALIDATE_COORDS_OUT" \
        --flank 30 \
        --emit-extracted-fasta "$EXTRACTED_FASTA"
    
    if [ ! -s "$VALIDATE_COORDS_OUT" ]; then echo "ERROR: validate-seqonly coords output missing"; exit 1; fi
    if [ ! -s "$EXTRACTED_FASTA" ]; then echo "ERROR: extracted fasta missing or empty"; exit 1; fi
    echo "[STEP 5] validate-seqonly (with genome) completed: $VALIDATE_COORDS_OUT"
else
    echo "[STEP 5] SKIPPED: Missing $COORDS_FASTA or $GENOME"
fi

echo "[SUCCESS] All acceptance tests passed!"
echo "Outputs are in ${OUT_DIR}"
