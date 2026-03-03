Pipeline Overview
=================

miRPV-NG implements a multi-stage pipeline for miRNA discovery from sRNA-seq data,
plus a standalone sequence-only mode.

.. contents:: Stages
   :local:
   :depth: 1

Sequence-Only Mode
------------------

Score FASTA sequences for pre-miRNA probability without sRNA-seq data.

.. code-block:: bash

    mirpv-ng score-fasta \
        --model models/hsa_premirna_rf_extended_tier2_v6.pkl \
        --fasta input.fa \
        --out-tsv scores.tsv

Two-Stage Validation
^^^^^^^^^^^^^^^^^^^^

Validate sequence-only inputs with mature + precursor evidence:

.. code-block:: bash

    mirpv-ng validate-seqonly \
        --fasta input.fa \
        --premirna-model models/hsa_premirna_rf_extended_tier2_v6.pkl \
        --mature-model models/hsa_mature_xgbrank_len18_26_v4.pkl \
        --out validation_results.tsv

Stage 1-8: FASTQ to Peaks
--------------------------

Processes raw FASTQ reads through alignment and peak calling.

**Input:** FASTQ file (raw or adapter-trimmed)

**Output:**

- ``peaks.tsv`` — Called peaks with read support statistics
- ``candidates.tsv`` — Candidate regions for classification
- ``candidates.fa`` — Sequences for candidates
- ``qc.json`` — Quality control metrics

**Key Steps:**

1. Adapter trimming (optional, via cutadapt)
2. Bowtie1 alignment to reference genome
3. Blocklist filtering (rRNA, tRNA removal)
4. Peak calling with island detection
5. Candidate sequence extraction

.. code-block:: bash

    mirpv-ng fastq-to-peaks \
        --fastq sample.fastq.gz \
        --sample-id sample1 \
        --outdir results/01_peaks \
        --bowtie-index refs/bowtie1_hg38/hg38 \
        --genome-fasta refs/hg38/hg38.fa \
        --threads 8

Stage 9: RF Classification
--------------------------

Scores candidate sequences using the trained Random Forest model.

**Input:** ``candidates.tsv``, ``candidates.fa``

**Output:** ``candidates.scored.tsv`` with ``rf_score`` column

.. code-block:: bash

    mirpv-ng candidates-to-scored \
        --candidates-tsv results/01_peaks/candidates.tsv \
        --candidates-fa results/01_peaks/candidates.fa \
        --model models/hsa_premirna_rf_extended_tier2_v6.pkl \
        --outdir results/02_scored \
        --sample-id sample1

Stage 9.5: Scored to Peaks
---------------------------

Aggregates per-candidate RF scores back to the peak level.

**Input:** ``candidates.scored.tsv``

**Output:** ``peaks.scored.tsv``

.. code-block:: bash

    mirpv-ng scored-to-peaks \
        --scored-tsv results/02_scored/candidates.scored.tsv \
        --outdir results/03_scored_peaks

Candidates Filter (optional)
-----------------------------

Filters candidates based on early known peak labels — skips Known-Confirmed
peaks to avoid redundant scoring.

**Input:** ``candidates.tsv``, ``candidates.fa``, ``peaks.known_early.tsv``

**Output:** Filtered ``candidates.tsv`` and ``candidates.fa``

.. code-block:: bash

    mirpv-ng candidates-filter \
        --candidates-tsv results/01_peaks/candidates.tsv \
        --candidates-fa results/01_peaks/candidates.fa \
        --peaks-known-early-tsv results/01_peaks/peaks.known_early.tsv \
        --outdir results/01_peaks

Stage 10e: Early Known Labeling
--------------------------------

Quick labeling of peaks as Known-Confirmed / Known-Region / Unknown using
GFF annotations only.

**Input:** ``peaks.tsv``

**Output:** ``peaks.known_early.tsv``

.. code-block:: bash

    mirpv-ng peaks-to-known-early \
        --peaks-tsv results/03_scored_peaks/peaks.scored.tsv \
        --outdir results/04_known \
        --mirgenedb-gff refs/known/hsa/hsa_mirgene.gff \
        --mirbase-gff refs/known/hsa/hsa_mirbase.gff3

Stage 10: Known miRNA Annotation
---------------------------------

Full known labeling using MirGeneDB and mirBase with precursor/mature BED
overlap checks.

**Input:** ``peaks.tsv``

**Output:** ``peaks.known.tsv`` with ``known_status`` column

.. code-block:: bash

    mirpv-ng peaks-to-known \
        --peaks-tsv results/03_scored_peaks/peaks.scored.tsv \
        --outdir results/04_known \
        --mirgenedb-gff refs/known/hsa/hsa_mirgene.gff \
        --mirbase-gff refs/known/hsa/hsa_mirbase.gff3

Stage 11: Finalist Selection
-----------------------------

Merges RF scores with known annotations to select strict finalists.

**Input:** ``peaks.scored.tsv``, ``peaks.known.tsv``

**Output:** ``strict_finalists.tsv``

.. code-block:: bash

    mirpv-ng peaks-to-finalists \
        --peaks-scored-tsv results/03_scored_peaks/peaks.scored.tsv \
        --peaks-known-tsv results/04_known/peaks.known.tsv \
        --outdir results/05_finalists \
        --sample-id sample1

Stage 12: Structure Prediction
-------------------------------

Runs ViennaRNA RNAfold on finalist sequences.

**Input:** ``strict_finalists.tsv``, ``candidates.fa``

**Output:** ``candidates_struct.tsv``, ``candidates_struct.fa``

.. code-block:: bash

    mirpv-ng finalists-to-struct \
        --strict-finalists-tsv results/05_finalists/strict_finalists.tsv \
        --candidates-fa results/01_peaks/candidates.fa \
        --outdir results/06_struct \
        --sample-id sample1

Stage 12m: Mature Prediction
-----------------------------

Predicts mature miRNA positions using XGBoost ranker.

**Input:** ``candidates_struct.fa``

**Output:** ``mature.tsv`` with ranked 5p/3p candidates

.. code-block:: bash

    mirpv-ng predict-mature \
        --mature-model models/hsa_mature_xgbrank_len18_26_v4.pkl \
        --fasta results/06_struct/candidates_struct.fa \
        --out results/07_mature/mature.tsv

Stage 13: Final Candidates
---------------------------

Merges structure and mature predictions.

**Input:** ``candidates_struct.tsv``, ``mature.tsv``

**Output:** ``final_candidates.tsv``

.. code-block:: bash

    mirpv-ng final-candidates \
        --candidates-struct-tsv results/06_struct/candidates_struct.tsv \
        --mature-tsv results/07_mature/mature.tsv \
        --outdir results/08_final \
        --sample-id sample1

Stage 14: Final Report
-----------------------

Generates auditable reports with all QC metrics.

**Input:** All previous stage outputs

**Output:** Final summary TSV + merged rejects

.. code-block:: bash

    mirpv-ng final-report \
        --final-candidates-tsv results/08_final/final_candidates.tsv \
        --outdir results/09_report \
        --sample-id sample1
