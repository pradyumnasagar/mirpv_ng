Quick Start Guide
=================

This guide walks you through the most common miRPV-NG workflows.

Sequence-Only Mode
------------------

For scoring pre-miRNA candidates from a FASTA file without sRNA-seq data.

Score FASTA for Pre-miRNA Probability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python -m mirpv_ng.cli score-fasta \
        --model models/hsa_premirna_rf_extended_tier2_v6.pkl \
        --fasta your_candidates.fa \
        --out scored_candidates.tsv

Output columns:

- ``input_id``: Sequence identifier
- ``rf_score``: Random Forest probability score (0-1, higher = more likely pre-miRNA)
- ``pred_label``: Binary prediction (1 = pre-miRNA, 0 = not)

Predict Mature miRNA Positions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python -m mirpv_ng.cli predict-mature \
        --mature-model models/hsa_mature_xgbrank_len18_26_v4.pkl \
        --fasta your_candidates.fa \
        --out mature_predictions.tsv

sRNA-Seq Mode
-------------

For the full pipeline from FASTQ to final predictions.

Validate Sequence-Only Inputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two-stage validation with mature + precursor evidence:

.. code-block:: bash

    python -m mirpv_ng.cli validate-seqonly \
        --fasta your_candidates.fa \
        --premirna-model models/hsa_premirna_rf_extended_tier2_v6.pkl \
        --mature-model models/hsa_mature_xgbrank_len18_26_v4.pkl \
        --out validation.tsv

Full Pipeline
^^^^^^^^^^^^^

Use the provided shell script::

    bash scripts/run_pipe.sh

Or run stages individually:

**Stage 1-8: FASTQ to Peaks**

.. code-block:: bash

    python -m mirpv_ng.cli fastq-to-peaks \
        --fastq sample.fastq \
        --sample-id sample1 \
        --outdir results/01_peaks \
        --bowtie-index refs/bowtie1_hg38/hg38 \
        --genome-fasta refs/hg38/hg38.fa \
        --threads 8

**Stage 9: Score Candidates**

.. code-block:: bash

    python -m mirpv_ng.cli candidates-to-scored \
        --candidates-tsv results/01_peaks/candidates.tsv \
        --candidates-fa results/01_peaks/candidates.fa \
        --model models/hsa_premirna_rf_extended_tier2_v6.pkl \
        --outdir results/09_scored

Python API
----------

Using the Classifier in Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from mirpv_ng.classifier import HairpinClassifier
    
    # Load classifier
    clf = HairpinClassifier(
        model_path="models/hsa_premirna_rf_extended_tier2_v6.pkl",
        species="hsa"
    )
    
    # Score a sequence
    result = clf.score_sequence_record("my_seq", "GGCCAUUAGGCC...")
    
    for r in result:
        print(f"{r['input_id']}: score={r['rf_score']:.3f}")

Computing Features Directly
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from mirpv_ng.features import run_rnafold, extended_features
    
    seq = "GGCCAUUAGGCC"
    struct, mfe = run_rnafold(seq)
    feats = extended_features(seq, struct, mfe)
    
    print(f"GC content: {feats['gc_frac']:.2f}")
    print(f"MFE: {feats['mfe']:.1f} kcal/mol")
