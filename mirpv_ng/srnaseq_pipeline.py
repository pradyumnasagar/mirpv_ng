# mirpv_ng/srnaseq_pipeline.py

"""
sRNA-seq mode for miRPV-NG.

High-level workflow:

  FASTQ --> trim / size-select --> map to genome --> BAM
        --> sRNA loci --> precursor candidates
        --> RF hairpin scoring --> mature / isomiR calls

This module defines data structures and orchestration functions.
Implementation of trimming/mapping can initially shell out to
external tools (cutadapt/fastp + bowtie/bowtie2), or you can
provide BAM directly and skip those steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pysam  # we can later make this optional

from .classifier import HairpinClassifier
from .features import read_fasta
from .mature import infer_mature_from_profile  # we will adapt this later


# ------------ basic data types ------------ #


@dataclass
class ReadStats:
    total_reads: int
    kept_after_trim: int
    kept_after_size: int


@dataclass
class SRNALocus:
    """
    A compact representation of a sRNA locus derived from BAM coverage.

    Coordinates are 0-based, half-open [start, end) on a given reference.
    """
    chrom: str
    start: int
    end: int
    strand: str  # "+", "-", or "."
    depth: int   # max depth or total reads, depending on strategy
    read_count: int


@dataclass
class PrecursorCandidate:
    """
    A candidate precursor sequence extracted around a sRNA locus.
    """
    chrom: str
    start: int
    end: int
    strand: str
    locus: SRNALocus
    seq: str


@dataclass
class ScoredPrecursor:
    precursor: PrecursorCandidate
    rf_score: float
    pred_label: int
    # coordinates of predicted hairpin within precursor, relative
    hp_start: int
    hp_end: int


# ------------ trimming & size-selection ------------ #


def trim_and_filter_fastq(
    fastq_in: Path,
    fastq_out: Path,
    adapter_seq: Optional[str] = None,
    min_len: int = 18,
    max_len: int = 30,
    quality_trim: bool = True,
) -> ReadStats:
    """
    Trim adapters and perform size selection on input sRNA FASTQ.

    For now this is a stub; implementation options:
      - shell out to cutadapt / fastp
      - or use a pure-Python approach for simple experiments

    Returns:
        ReadStats with total and retained reads.
    """
    # TODO: implement trimming + size-selection
    raise NotImplementedError("trim_and_filter_fastq is not yet implemented")


# ------------ mapping & locus construction ------------ #


def map_srna_reads(
    fastq: Path,
    genome_index: Path,
    bam_out: Path,
    mapper: str = "bowtie",
    threads: int = 4,
) -> None:
    """
    Map sRNA reads to genome using an external aligner.

    For now this is a stub; we expect:
      - single-end small RNA reads
      - short alignments (no splicing)

    You can start by running mapping outside miRPV-NG and
    feeding a BAM to downstream functions.
    """
    # TODO: shell out to bowtie/bowtie2/STAR/etc.
    raise NotImplementedError("map_srna_reads is not yet implemented")


def loci_from_bam(
    bam_path: Path,
    min_depth: int = 5,
    max_gap: int = 2,
) -> List[SRNALocus]:
    """
    Build sRNA loci from a mapped BAM.

    Strategy (first version):
      - traverse genome per chromosome & strand
      - build coverage array for positions with any reads
      - cluster contiguous positions with coverage >= min_depth,
        merging gaps <= max_gap into one locus

    Returns:
        list of SRNALocus objects
    """
    loci: List[SRNALocus] = []
    bam = pysam.AlignmentFile(bam_path, "rb")

    # TODO: implement per-chromosome coverage + clustering
    # For now, leave stub.
    bam.close()
    raise NotImplementedError("loci_from_bam is not yet implemented")


# ------------ precursor extraction ------------ #


def extract_precursors_from_loci(
    genome_fasta: Path,
    loci: List[SRNALocus],
    flank_up: int = 70,
    flank_down: int = 70,
) -> List[PrecursorCandidate]:
    """
    For each sRNA locus, extract a putative precursor sequence
    from the genome with given flanking lengths.

    Returns:
        list of PrecursorCandidate objects
    """
    seqs = dict(read_fasta(str(genome_fasta)))  # chrom -> seq
    precursors: List[PrecursorCandidate] = []

    for loc in loci:
        chrom_seq = seqs.get(loc.chrom)
        if chrom_seq is None:
            continue

        # Expand locus with flanks (clip to chromosome bounds)
        start = max(0, loc.start - flank_up)
        end = min(len(chrom_seq), loc.end + flank_down)
        if end <= start:
            continue

        seq = chrom_seq[start:end]
        precursors.append(
            PrecursorCandidate(
                chrom=loc.chrom,
                start=start,
                end=end,
                strand=loc.strand,
                locus=loc,
                seq=seq,
            )
        )

    return precursors


# ------------ RF scoring of precursors ------------ #


def score_precursors_with_rf(
    clf: HairpinClassifier,
    precursors: List[PrecursorCandidate],
) -> List[ScoredPrecursor]:
    """
    Use the pre-trained HairpinClassifier to score each precursor
    sequence in sequence-only mode.

    We reuse score_sequence_record, which returns either:
      - one hairpin record (mode=hairpin) for short sequences
      - multiple (mode=scan) records for longer precursors
      - or a single too_long marker (mode=too_long)
    """
    results: List[ScoredPrecursor] = []

    for pc in precursors:
        hits = clf.score_sequence_record(pc.chrom, pc.seq)
        for h in hits:
            if h["mode"] == "too_long":
                continue

            hp_start = h["start"]
            hp_end = h["end"]
            sp = ScoredPrecursor(
                precursor=pc,
                rf_score=float(h["rf_score"]),
                pred_label=int(h["pred_label"]),
                hp_start=hp_start,
                hp_end=hp_end,
            )
            results.append(sp)

    return results


# ------------ overall orchestration ------------ #


def run_srnaseq_pipeline(
    fastq_in: Path,
    genome_fasta: Path,
    model_path: Path,
    out_tsv: Path,
    adapter_seq: Optional[str] = None,
    min_len: int = 18,
    max_len: int = 30,
    min_depth: int = 5,
    flank_up: int = 70,
    flank_down: int = 70,
    window_len: int = 80,
    step: int = 20,
) -> None:
    """
    High-level sRNA-seq pipeline from FASTQ to scored precursors.

    First version outline:
      - (1) trim + size-select FASTQ -> filtered FASTQ
      - (2) map to genome externally (or via map_srna_reads)
      - (3) derive loci from BAM
      - (4) extract precursor sequences with flanks
      - (5) score with HairpinClassifier
      - (6) write TSV with per-locus scores
    """
    # TODO: wire all steps once trimming/mapping/loci are implemented
    raise NotImplementedError("run_srnaseq_pipeline is not yet implemented")
