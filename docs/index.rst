miRPV-NG Documentation
=======================

**miRPV-NG** (miRNA Prediction and Validation - Next Generation) is a comprehensive 
bioinformatics pipeline for discovering and validating microRNAs from small RNA 
sequencing (sRNA-seq) data.

.. note::
   This documentation covers both the Python API and command-line interface.

Features
--------

- 14-stage sRNA-seq processing pipeline
- Machine learning classification (Random Forest + XGBoost)
- ViennaRNA integration for structure prediction
- Known miRNA annotation (miRGeneDB, mirBase)
- Mature miRNA position prediction
- PySide6 graphical user interface

Quick Start
-----------

Installation
^^^^^^^^^^^^

Create the conda environment::

    conda env create -f env.yml
    conda activate mirpv-ng

Score a FASTA file
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python -m mirpv_ng.cli score-fasta \
        --model models/hsa_premirna_rf_extended_tier2_v6.pkl \
        --fasta input.fa \
        --out scores.tsv

Python API
^^^^^^^^^^

.. code-block:: python

    from mirpv_ng.classifier import HairpinClassifier
    
    clf = HairpinClassifier("models/hsa_premirna_rf_extended_tier2_v6.pkl")
    results = clf.score_sequence_record("seq1", "GGCCAUUAGGCC...")
    print(results[0]["rf_score"])  # 0.85

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   pipeline

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/cli
   api/features
   api/classifier
   api/tier_filters
   api/pgs_features
   api/mature_ranker
   api/mature_model
   api/geom_hairpin_finder
   api/geom_stem_features
   api/geom_bulges
   api/geom_energy
   api/parallel

.. toctree::
   :maxdepth: 1
   :caption: Additional

   changelog
   contributing


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
