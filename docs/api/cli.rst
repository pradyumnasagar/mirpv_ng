CLI Module
==========

.. automodule:: mirpv_ng.cli
   :members:
   :undoc-members:
   :show-inheritance:

Subcommands
-----------

The ``mirpv-ng`` command-line tool provides the following subcommands:

Sequence-Only Mode
^^^^^^^^^^^^^^^^^^

- ``score-fasta`` — Score sequences from a FASTA file
- ``predict-mature`` — Predict mature miRNA position using XGBRanker
- ``validate-seqonly`` — Two-stage validation (mature + precursor evidence)

sRNA-Seq Mode
^^^^^^^^^^^^^

- ``fastq-to-peaks`` — Stage 1–8: FASTQ → peaks + candidate excision
- ``candidates-filter`` — Filter candidates based on early known labeling
- ``candidates-to-scored`` — Stage 9: RF scoring of candidates
- ``scored-to-peaks`` — Stage 9.5: Aggregate candidate scores to peaks
- ``peaks-to-known-early`` — Stage 10e: Early known peak labeling
- ``peaks-to-known`` — Stage 10: Full known miRNA annotation
- ``peaks-to-finalists`` — Stage 11: Merge known + scored → finalists
- ``finalists-to-struct`` — Stage 12: RNAfold structure prediction
- ``final-candidates`` — Stage 13: Merge structure + mature predictions
- ``final-report`` — Stage 14: Final auditable report
