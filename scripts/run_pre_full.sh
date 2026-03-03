#!/usr/bin/env bash
set -euo pipefail


# Resolve repo root even if script is launched from elsewhere
MIRPV_HOME="${MIRPV_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$MIRPV_HOME"
FASTQ_DIR="examples/vai/fastqs"
OUT_ROOT="results/vai_premalignant_run"
THREADS=18

# Use PRIMARY genome (you already have these)
GENOME_INDEX="refs/bowtie1_hg38_primary/hg38.primary"
GENOME_FASTA="refs/hg38_primary/hg38.primary.fa"

# Filters you already have
BLOCKLIST_INDEX="refs/indexes/blocklist/rfam_trna"   # "" to disable
REPEAT_BED="refs/repeats/hg38/hg38.rmsk.repClass.bed.sorted.bed"  # "" to disable

# Known miRNA annotations (adjust if your files differ)
MIRGENEDB_GFF="refs/known/hsa/hsa_mirgene.gff"
MIRBASE_GFF="refs/known/hsa/hsa_mirbase.gff3"

RF_MODEL="models/hsa_premirna_rf_extended_tier2_v6.pkl"
MATURE_MODEL="models/hsa_mature_xgbrank_len18_26_v4.pkl"
MATURE_LENS="18,19,20,21,22,23,24,25"

SPECIES="hsa"

# If already trimmed, keep empty
ADAPTER=""

# Your final repeat policy
REPEAT_ALLOW=(LINE SINE LTR DNA NONE)
REPEAT_BLOCK=(SATELLITE SIMPLE_REPEAT LOW_COMPLEXITY)
NOVEL_HIGH_MIN_RF_REPEAT="0.50"

mkdir -p "$OUT_ROOT"

need_file() { [[ -s "$1" ]] || { echo "[ERROR] Missing/empty: $1" >&2; exit 1; }; }

sample_id_from_fastq() {
  local fq="$1"
  local base
  base="$(basename "$fq")"
  base="${base%.gz}"
  base="${base%.fastq}"
  base="${base%.fq}"
  echo "$base"
}

shopt -s nullglob
FASTQS=( "$FASTQ_DIR"/premalignant*.fastq "$FASTQ_DIR"/premalignant*.fastq.gz "$FASTQ_DIR"/premalignant*.fq "$FASTQ_DIR"/premalignant*.fq.gz )
shopt -u nullglob
(( ${#FASTQS[@]} > 0 )) || { echo "[ERROR] No premalignant*.fastq found in $FASTQ_DIR" >&2; exit 1; }

echo "[INFO] Found ${#FASTQS[@]} premalignant FASTQ(s) in $FASTQ_DIR"

for FQ in "${FASTQS[@]}"; do
  S="$(sample_id_from_fastq "$FQ")"
  OD="$OUT_ROOT/$S"
  mkdir -p "$OD"

  echo
  echo "============================================================"
  echo "[SAMPLE] $S"
  echo "[FASTQ ] $FQ"
  echo "[OUT   ] $OD"
  echo "============================================================"

  # -----------------------
  echo 'RUNNING # Stage 01: fastq-to-peaks'
  # -----------------------
  ST01="$OD/01_fastq_to_peaks"; mkdir -p "$ST01"

  CMD01=(python -m mirpv_ng.cli fastq-to-peaks
    --fastq "$FQ"
    --sample-id "$S"
    --outdir "$ST01"
    --bowtie-index "$GENOME_INDEX"
    --genome-fasta "$GENOME_FASTA"
    --threads "$THREADS"
    --max-multimaps 50
    --island-gap 50
    --min-depth 5
    --min-cpm 0.5
    --smooth-w 1
    --peak-distance 5
    --support-window 15
    --hard-frac-20-24 0.0
  )
  [[ -n "$ADAPTER" ]] && CMD01+=(--adapter "$ADAPTER")
  [[ -n "$BLOCKLIST_INDEX" ]] && CMD01+=(--blocklist-index "$BLOCKLIST_INDEX")
  [[ -n "$REPEAT_BED" ]] && CMD01+=(--repeat-bed "$REPEAT_BED")
  "${CMD01[@]}"

  need_file "$ST01/candidates.tsv"
  need_file "$ST01/candidates.fa"
  PEAKS_TSV="$ST01/${S}.peaks.tsv"
  need_file "$PEAKS_TSV"

  # -----------------------
  echo 'RUNNING # Stage 02: peaks-to-known-early'
  # -----------------------
  ST02="$OD/02_peaks_to_known_early"; mkdir -p "$ST02"
  CMD02=(python -m mirpv_ng.cli peaks-to-known-early
    --peaks-tsv "$PEAKS_TSV"
    --outdir "$ST02"
    --max-pad 100
  )
  [[ -n "$MIRGENEDB_GFF" ]] && CMD02+=(--mirgenedb-gff "$MIRGENEDB_GFF")
  [[ -n "$MIRBASE_GFF" ]] && CMD02+=(--mirbase-gff "$MIRBASE_GFF")
  "${CMD02[@]}"
  KNOWN_EARLY="$ST02/peaks.known_early.tsv"
  need_file "$KNOWN_EARLY"

  # -----------------------
  echo 'RUNNING # Stage 03: candidates-filter'
  # -----------------------
  ST03="$OD/03_candidates_filter"; mkdir -p "$ST03"
  python -m mirpv_ng.cli candidates-filter \
    --candidates-tsv "$ST01/candidates.tsv" \
    --candidates-fa  "$ST01/candidates.fa" \
    --peaks-known-early-tsv "$KNOWN_EARLY" \
    --outdir "$ST03" \
 
  CAND_TSV="$ST03/candidates.filtered.tsv"
  CAND_FA="$ST03/candidates.filtered.fa"
  need_file "$CAND_TSV"
  need_file "$CAND_FA"

  # -----------------------
  echo 'RUNNING # Stage 09: candidates-to-scored (RF)'
  # -----------------------
  ST09="$OD/09_candidates_to_scored"; mkdir -p "$ST09"
  python -m mirpv_ng.cli candidates-to-scored \
    --candidates-tsv "$CAND_TSV" \
    --candidates-fa  "$CAND_FA" \
    --model "$RF_MODEL" \
    --outdir "$ST09" \
    --sample-id "$S" \
    --species "$SPECIES" \
    --feature-set extended \
    --tier2 \
    --window-len 120 \
    --step 3 \
    --threads "$THREADS"
  SCORED_TSV="$ST09/candidates.scored.tsv"
  need_file "$SCORED_TSV"

  # -----------------------
  echo 'RUNNING # Stage 095: scored-to-peaks'
  # -----------------------
  ST095="$OD/095_scored_to_peaks"; mkdir -p "$ST095"
  python -m mirpv_ng.cli scored-to-peaks \
    --scored-tsv "$SCORED_TSV" \
    --outdir "$ST095" 
    
  PEAKS_SCORED="$ST095/peaks.scored.tsv"
  need_file "$PEAKS_SCORED"

  # -----------------------
  echo 'RUNNING # Stage 10: peaks-to-known'
  # -----------------------
  ST10="$OD/10_peaks_to_known"; mkdir -p "$ST10"
  CMD10=(python -m mirpv_ng.cli peaks-to-known
    --peaks-tsv "$PEAKS_SCORED"
	--mirgenedb-gff "$MIRGENEDB_GFF"
	--mirbase-gff "$MIRBASE_GFF"
	--outdir "$ST10"
    --sample-id "$S"
  )
  [[ -n "$MIRGENEDB_GFF" ]] && CMD10+=(--mirgenedb-gff "$MIRGENEDB_GFF")
  [[ -n "$MIRBASE_GFF" ]] && CMD10+=(--mirbase-gff "$MIRBASE_GFF")
  "${CMD10[@]}"
  PEAKS_KNOWN="$ST10/peaks.known.tsv"
  need_file "$PEAKS_KNOWN"

  # -----------------------
  echo 'RUNNING # Stage 11: peaks-to-finalists (your repeat policy)'
  # -----------------------
  ST11="$OD/11_peaks_to_finalists"; mkdir -p "$ST11"
  python -m mirpv_ng.cli peaks-to-finalists \
    --peaks-scored-tsv "$PEAKS_SCORED" \
    --peaks-known-tsv  "$PEAKS_KNOWN" \
    --outdir "$ST11" \
    --sample-id "$S" \
    --repeat-allow "${REPEAT_ALLOW[@]}" \
    --repeat-block "${REPEAT_BLOCK[@]}" \
    --novel-high-min-rf-repeat "$NOVEL_HIGH_MIN_RF_REPEAT"
  STRICT="$ST11/strict_finalists.tsv"
  need_file "$STRICT"

  # -----------------------
  echo 'RUNNING # Stage 12: finalists-to-struct'
  # -----------------------
  ST12="$OD/12_finalists_to_struct"; mkdir -p "$ST12"
  python -m mirpv_ng.cli finalists-to-struct \
    --strict-finalists-tsv "$STRICT" \
    --candidates-fa "$ST01/candidates.fa" \
    --outdir "$ST12" \
    --sample-id "$S" \
    --sample-id "$S" \
    --threads "$THREADS"
  STRUCT_TSV="$ST12/candidates_struct.tsv"
  STRUCT_FA="$ST12/candidates_struct.fa"
  need_file "$STRUCT_TSV"
  need_file "$STRUCT_FA"

  # -----------------------
 echo 'RUNNING # Stage 12m: predict-mature (18–25 enabled)'
  # -----------------------
  ST12M="$OD/12m_predict_mature"; mkdir -p "$ST12M"
  python -m mirpv_ng.cli predict-mature \
    --mature-model "$MATURE_MODEL" \
    --fasta "$STRUCT_FA" \
    --out "$ST12M/predict_mature.tsv" \
    --lengths "$MATURE_LENS"
  MATURE_TSV="$ST12M/predict_mature.tsv"
  need_file "$MATURE_TSV"

  # -----------------------
  echo 'RUNNING # Stage 13: final-candidates'
  # -----------------------
  ST13="$OD/13_final_candidates"; mkdir -p "$ST13"
  python -m mirpv_ng.cli final-candidates \
    --candidates-struct-tsv "$STRUCT_TSV" \
    --mature-tsv "$MATURE_TSV" \
    --outdir "$ST13" \
    --sample-id "$S"
  FINAL_CAND="$ST13/final_candidates.tsv"
  need_file "$FINAL_CAND"

  # -----------------------
 echo 'RUNNING # Stage 14: final-report'
  # -----------------------
  ST14="$OD/14_final_report"; mkdir -p "$ST14"
  python -m mirpv_ng.cli final-report \
    --final-candidates-tsv "$FINAL_CAND" \
    --outdir "$ST14" \
    --sample-id "$S"
  need_file "$ST14/final_report.tsv"

  echo "[DONE] $S"
done

echo
echo "[ALL DONE] premalignants complete: $OUT_ROOT"
