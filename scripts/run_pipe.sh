#!/usr/bin/env bash
set -euo pipefail


# Resolve repo root even if script is launched from elsewhere
MIRPV_HOME="${MIRPV_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$MIRPV_HOME"
# ============================================================
# miRPV-NG sRNA-seq full pipeline (Tumor1–Tumor7) — v2 flow
# Stages:
#   1–8  fastq-to-peaks
#   2    peaks-to-known-early         (NEW; fast pre-RF labeling)
#   3    candidates-filter            (NEW; skip Known-Confirmed)
#   9    candidates-to-scored         (RF scoring)
#   9.5  scored-to-peaks
#   10   peaks-to-known               (full known labeling)
#   11   peaks-to-finalists
#   12   finalists-to-struct
#   12m  predict-mature               (18–24 enabled)
#   13   final-candidates
#   14   final-report                 (auditable + known tables)
# ============================================================

# ----------------------------
# Inputs you MUST set once
# ----------------------------
FASTQ_DIR="examples/vai/fastqs"
OUT_ROOT="results/vai_normal_run"
THREADS=18

GENOME_INDEX="refs/bowtie1_hg38/hg38"
GENOME_FASTA="refs/hg38/hg38.fa"

BLOCKLIST_INDEX="refs/indexes/blocklist/rfam_trna"   # set "" to disable
REPEAT_BED="refs/repeats/hg38/hg38.rmsk.repClass.bed.sorted.bed"    # set "" to disable

MIRGENEDB_GFF="refs/known/hsa/hsa_mirgene.gff"       # set "" to disable
MIRBASE_GFF="refs/known/hsa/hsa_mirbase.gff3"        # set "" to disable

RF_MODEL="models/hsa_premirna_rf_extended_tier2_v6.pkl"
MATURE_MODEL="models/hsa_mature_xgbrank_len18_26_v4.pkl"

SPECIES="hsa"

# If already trimmed, keep empty. Else set adapter sequence.
ADAPTER=""

mkdir -p "$OUT_ROOT"

# Tumor fastqs (explicit)
TUMORS=(tumor2 tumor3 tumor4 tumor5 tumor6 tumor7)

for S in "${TUMORS[@]}"; do
  FQ="${FASTQ_DIR}/${S}.fastq"
  [[ -s "$FQ" ]] || { echo "Missing FASTQ: $FQ" >&2; exit 1; }

  echo
  echo "=============================="
  echo "Running sample: $S"
  echo "FASTQ: $FQ"
  echo "OUT:   $OUT_ROOT/$S"
  echo "=============================="

  OD="$OUT_ROOT/$S"
  mkdir -p "$OD"

  # ----------------------------
  # Stage 1–8: FASTQ -> peaks (+ candidates.tsv/fa because genome-fasta provided)
  # ----------------------------
  ST1="$OD/01_fastq_to_peaks"
  mkdir -p "$ST1"

  CMD1=(python -m mirpv_ng.cli fastq-to-peaks
    --fastq "$FQ"
    --sample-id "$S"
    --outdir "$ST1"
    --bowtie-index "$GENOME_INDEX"
    --genome-fasta "$GENOME_FASTA"
    --threads "$THREADS"
    --max-multimaps 50
    --island-gap 50
    --min-depth 5
    --min-cpm 0.5
    --smooth-w 1
    --peak-distance 5
    --fallback-prom-frac 0.00
    --support-window 15
    --hard-frac-20-24 0.0
  )

  [[ -n "$ADAPTER" ]] && CMD1+=(--adapter "$ADAPTER")
  [[ -n "$BLOCKLIST_INDEX" ]] && CMD1+=(--blocklist-index "$BLOCKLIST_INDEX")
  [[ -n "$REPEAT_BED" ]] && CMD1+=(--repeat-bed "$REPEAT_BED")

  "${CMD1[@]}"

  PEAKS_TSV="$ST1/${S}.peaks.tsv"
  CAND_TSV="$ST1/candidates.tsv"
  CAND_FA="$ST1/candidates.fa"
  QC1="$ST1/qc.json"
  REJ1="$ST1/rejects.tsv"

  # ----------------------------
  # Stage 2 (NEW): Early known labeling (fast, pre-RF)
  # ----------------------------
  ST2="$OD/02_peaks_known_early"
  mkdir -p "$ST2"

  CMD2=(python -m mirpv_ng.cli peaks-to-known-early
    --peaks-tsv "$PEAKS_TSV"
    --outdir "$ST2"
    --max-pad 100
  )
  [[ -n "$MIRGENEDB_GFF" ]] && CMD2+=(--mirgenedb-gff "$MIRGENEDB_GFF")
  [[ -n "$MIRBASE_GFF" ]] && CMD2+=(--mirbase-gff "$MIRBASE_GFF")

  "${CMD2[@]}"

  KNOWN_EARLY="$ST2/peaks.known_early.tsv"

  # ----------------------------
  # Stage 3 (NEW): Filter candidates (skip Known-Confirmed peaks)
  # ----------------------------
  ST3="$OD/03_candidates_filtered"
  mkdir -p "$ST3"

  python -m mirpv_ng.cli candidates-filter \
    --candidates-tsv "$CAND_TSV" \
    --candidates-fa  "$CAND_FA" \
    --peaks-known-early-tsv "$KNOWN_EARLY" \
    --outdir "$ST3"

  # Redirect downstream to filtered candidates
  CAND_TSV="$ST3/candidates.filtered.tsv"
  CAND_FA="$ST3/candidates.filtered.fa"
  QC3="$ST3/qc_filter.json"

  # ----------------------------
  # Stage 9: candidates -> RF score
  # ----------------------------
  ST9="$OD/09_candidates_to_scored"
  mkdir -p "$ST9"

  python -m mirpv_ng.cli candidates-to-scored \
    --candidates-tsv "$CAND_TSV" \
    --candidates-fa  "$CAND_FA" \
    --model "$RF_MODEL" \
    --outdir "$ST9" \
    --sample-id "$S" \
    --species "$SPECIES" \
    --feature-set extended \
    --tier2 \
    --window-len 120 \
    --step 3 \
    --threads "$THREADS"

  SCORED_TSV="$ST9/candidates.scored.tsv"
  QC9="$ST9/qc_stage2.json"
  REJ9="$ST9/rejects.tsv"

  # ----------------------------
  # Stage 9.5: scored -> peaks.scored.tsv
  # ----------------------------
  ST95="$OD/095_scored_to_peaks"
  mkdir -p "$ST95"

  python -m mirpv_ng.cli scored-to-peaks \
    --scored-tsv "$SCORED_TSV" \
    --outdir "$ST95"

  PEAKS_SCORED="$ST95/peaks.scored.tsv"
  QC95="$ST95/qc_stage2p5.json"
  REJ95="$ST95/rejects.tsv"

  # ----------------------------
  # Stage 10: peaks -> known labeling (full)
  # ----------------------------
  ST10="$OD/10_peaks_to_known"
  mkdir -p "$ST10"

  CMD10=(python -m mirpv_ng.cli peaks-to-known
    --peaks-tsv "$PEAKS_TSV"
    --outdir "$ST10"
    --sample-id "$S"
  )
  [[ -n "$MIRGENEDB_GFF" ]] && CMD10+=(--mirgenedb-gff "$MIRGENEDB_GFF")
  [[ -n "$MIRBASE_GFF" ]] && CMD10+=(--mirbase-gff "$MIRBASE_GFF")

  "${CMD10[@]}"

  PEAKS_KNOWN="$ST10/peaks.known.tsv"
  QC10="$ST10/qc_stage10.json"
  REJ10="$ST10/rejects.tsv"

  # ----------------------------
  # Stage 11: merge known + scored -> strict finalists
  # ----------------------------
  ST11="$OD/11_peaks_to_finalists"
  mkdir -p "$ST11"

  python -m mirpv_ng.cli peaks-to-finalists \
    --peaks-scored-tsv "$PEAKS_SCORED" \
    --peaks-known-tsv  "$PEAKS_KNOWN" \
    --outdir "$ST11" \
    --sample-id "$S"

  STRICT_TSV="$ST11/strict_finalists.tsv"
  QC11="$ST11/qc_stage11.json"
  REJ11="$ST11/rejects.tsv"

  # ----------------------------
  # Stage 12: strict finalists -> RNAfold structures
  # ----------------------------
  ST12="$OD/12_finalists_to_struct"
  mkdir -p "$ST12"

  python -m mirpv_ng.cli finalists-to-struct \
    --strict-finalists-tsv "$STRICT_TSV" \
    --candidates-fa "$CAND_FA" \
    --outdir "$ST12" \
    --sample-id "$S" \
    --threads 4

  STRUCT_TSV="$ST12/candidates_struct.tsv"
  STRUCT_FA="$ST12/candidates_struct.fa"
  QC12="$ST12/qc_stage12.json"
  REJ12="$ST12/rejects.tsv"

  # ----------------------------
  # Stage 12m: Mature prediction (18–24)
  # ----------------------------
  ST12M="$OD/12m_predict_mature"
  mkdir -p "$ST12M"

  python -m mirpv_ng.cli predict-mature \
    --mature-model "$MATURE_MODEL" \
    --fasta "$STRUCT_FA" \
    --out "$ST12M/mature.tsv" \
    --rnafold-bin RNAfold \
    --lengths "18,19,20,21,22,23,24" \
    --max-per-arm 40 \
    --min-paired-context 4 \
    --loop-buffer 0 \
    --fallback-loop-buffer 10 \
    --fallback-max-per-arm 160 \
    --fallback-min-paired-context 0

  MATURE_TSV="$ST12M/mature.tsv"

  # ----------------------------
  # Stage 13: merge struct + mature -> final_candidates.tsv
  # ----------------------------
  ST13="$OD/13_final_candidates"
  mkdir -p "$ST13"

  python -m mirpv_ng.cli final-candidates \
    --candidates-struct-tsv "$STRUCT_TSV" \
    --mature-tsv "$MATURE_TSV" \
    --outdir "$ST13" \
    --sample-id "$S"

  FINAL_CAND="$ST13/final_candidates.tsv"
  QC13="$ST13/qc_stage13.json"
  REJ13="$ST13/rejects.tsv"

  # ----------------------------
  # Stage 14: final report (auditable + known tables)
  # ----------------------------
  ST14="$OD/14_final_report"
  mkdir -p "$ST14"

  python -m mirpv_ng.cli final-report \
    --final-candidates-tsv "$FINAL_CAND" \
    --outdir "$ST14" \
    --sample-id "$S" \
    --known-early-tsv "$KNOWN_EARLY" \
    --rejects \
      "$REJ1" "$REJ9" "$REJ95" "$REJ10" "$REJ11" "$REJ12" "$REJ13" \
    --qc-json \
      "$QC1" "$QC3" "$QC9" "$QC95" "$QC10" "$QC11" "$QC12" "$QC13"

  echo "[DONE] $S -> $ST14"
done

echo
echo "All tumors done. Outputs under: $OUT_ROOT"
