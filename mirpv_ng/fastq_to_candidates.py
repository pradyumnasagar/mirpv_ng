#!/usr/bin/env python3
"""
fastq_to_candidates.py
Production Upstream Pipeline for miRPV-NG.

WORKFLOW:
1. Preprocess: Cutadapt trim + Size selection (18-30nt).
2. Blocklist: Strict removal of rRNA/tRNA/snoRNA.
3. Mapping: Bowtie1 (v=1, m=50) to keep multi-mappers.
4. Quantification: Count Known miRNAs (MirGeneDB > miRBase).
5. The Harvester:
   - Signal-Based Peak Splitting (Separates polycistronic/adjacent loci).
   - Unique-First Anchoring (Prevents repeat explosion).
   - Biogenesis Gating (Sharpness, 5p-Precision, Complexity).
   - Dual-Window Excision (Peak+/-70, Peak+/-100).

Output: candidates.tsv (Input for miRPV-NG scoring), rejects.tsv, quantification.tsv
"""

import argparse
import subprocess
import sys
import os
import logging
import numpy as np
import pysam
import pybedtools
from Bio import SeqIO
from collections import Counter
from scipy.signal import find_peaks, convolve

# --- Configuration Constants ---
MIN_LEN_DISCOVERY = 18
MAX_LEN_DISCOVERY = 30
CLUSTER_GAP = 50          # Max distance to merge reads into an island
MIN_ISLAND_DEPTH = 5      # Minimum reads to consider an island
PEAK_MIN_DIST = 35        # Initial distance between peaks
PEAK_PROMINENCE_REL = 0.30 # Peak must be 30% higher than local valley
ISOMIR_MERGE_DIST = 8     # Merge peaks closer than this (same biological arm)
WINDOW_SIZES = [70, 100]  # Dual window strategy

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def run_cmd(cmd, shell=False):
    """Run a shell command and handle errors."""
    cmd_str = cmd if isinstance(cmd, str) else ' '.join(cmd)
    logger.info(f"EXEC: {cmd_str}")
    try:
        if shell:
            subprocess.check_call(cmd, shell=True, executable="/bin/bash")
        else:
            subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)

# -------------------------------------------------------------------------
# STAGE 1 & 2: Preprocess & Blocklist
# -------------------------------------------------------------------------

def preprocess_and_blocklist(fastq_in, out_dir, blocklist_index, adapter, threads):
    """
    Trim adapters, size select, and scrub rRNA/tRNA.
    """
    clean_fastq = os.path.join(out_dir, "clean.fastq")
    final_fastq = os.path.join(out_dir, "discovery_input.fastq")
    blocklist_sam = os.path.join(out_dir, "blocklist_hits.sam")
    
    # 1. Cutadapt
    cmd_cut = [
        "cutadapt", "-j", str(threads),
        "-a", adapter,
        "-m", str(MIN_LEN_DISCOVERY), "-M", str(MAX_LEN_DISCOVERY),
        "--discard-untrimmed",
        "-o", clean_fastq,
        fastq_in
    ]
    run_cmd(cmd_cut)
    
    # 2. Blocklist Scrub (Bowtie -v 0 -k 1)
    # Reads that DO NOT map to blocklist (--un) go to discovery.
    cmd_block = (
        f"bowtie -v 0 -k 1 -p {threads} --un {final_fastq} {blocklist_index} {clean_fastq} > {blocklist_sam}"
    )
    run_cmd(cmd_block, shell=True)
    
    return final_fastq

# -------------------------------------------------------------------------
# STAGE 3: Genome Mapping & Quantification
# -------------------------------------------------------------------------

def map_genome(fastq_in, out_dir, genome_index, threads):
    """
    Map to genome allowing multi-mappers (m=50).
    """
    out_bam = os.path.join(out_dir, "genome_mapped.bam")
    if os.path.exists(out_bam):
        logger.info("BAM exists, skipping mapping.")
        return out_bam

    cmd = (
        f"bowtie -v 1 -m 50 --best --strata -p {threads} -S {genome_index} {fastq_in} "
        f"| samtools view -bS - | samtools sort -o {out_bam}"
    )
    run_cmd(cmd, shell=True)
    run_cmd(["samtools", "index", out_bam])
    return out_bam

def quantify_knowns(bam_path, mirgenedb_gff, mirbase_gff, out_tsv):
    """
    Stage 4: Quantify known miRNAs (Parallel Track).
    """
    logger.info("Quantifying known miRNAs...")
    reads = pybedtools.BedTool(bam_path).bam_to_bed()
    results = {}

    def process_db(gff, db_name):
        if not gff or gff == "None": return
        # Filter GFF for miRNA features
        annot = pybedtools.BedTool(gff).filter(lambda x: x[2] in ["miRNA", "pre_miRNA", "miRNA_primary_transcript"])
        cov = annot.coverage(reads, s=True, counts=True)
        for feat in cov:
            key = (feat.chrom, feat.start, feat.end, feat.strand)
            count = int(feat[9])
            name = feat.attrs.get("Name", feat.name)
            if count > 0:
                # Prioritize MirGeneDB overwrites
                if key not in results or db_name == "MirGeneDB":
                    results[key] = {'db': db_name, 'name': name, 'count': count}

    process_db(mirbase_gff, "miRBase")
    process_db(mirgenedb_gff, "MirGeneDB")

    with open(out_tsv, "w") as f:
        f.write("chrom\tstart\tend\tstrand\tdatabase\tname\tcount\n")
        for (chrom, start, end, strand), data in results.items():
            f.write(f"{chrom}\t{start}\t{end}\t{strand}\t{data['db']}\t{data['name']}\t{data['count']}\n")

# -------------------------------------------------------------------------
# STAGE 6: Peak Calling Logic
# -------------------------------------------------------------------------

def smooth_signal(signal, window_size=3):
    """Rolling mean smoothing to reduce isomiR jitter."""
    if len(signal) < window_size: return signal
    window = np.ones(window_size) / window_size
    return convolve(signal, window, mode='same')

def call_peaks_in_island(bam, chrom, start, end, strand, min_depth):
    """
    Advanced Peak Calling:
    1. Build Coverage (Total & Unique).
    2. Unique-First Pass.
    3. Multi-Rescue Pass.
    4. IsomiR Merging.
    """
    length = end - start
    if length < 20: return []

    cov_total = np.zeros(length, dtype=int)
    cov_unique = np.zeros(length, dtype=int)

    reads = list(bam.fetch(chrom, start, end))
    for r in reads:
        if (("-" if r.is_reverse else "+") != strand): continue
        
        # Check uniqueness (Bowtie specific: NH:i:1 or MAPQ check)
        is_unique = False
        try:
            if r.get_tag("NH") == 1: is_unique = True
        except KeyError:
            if r.mapping_quality > 0: is_unique = True 

        r_s = max(0, r.reference_start - start)
        r_e = min(length, r.reference_end - start)
        cov_total[r_s:r_e] += 1
        if is_unique:
            cov_unique[r_s:r_e] += 1

    peaks_found = [] # (pos, type)

    # --- Pass A: Unique Anchors ---
    smooth_unique = smooth_signal(cov_unique)
    u_peaks, _ = find_peaks(smooth_unique, distance=PEAK_MIN_DIST, prominence=min_depth*PEAK_PROMINENCE_REL)
    
    # Merge IsomiR Micropeaks (Greedy)
    u_peaks_merged = []
    if len(u_peaks) > 0:
        # Sort by height to keep dominant
        u_peaks_sorted = sorted(u_peaks, key=lambda x: smooth_unique[x], reverse=True)
        kept = []
        for p in u_peaks_sorted:
            if not any(abs(p - k) < ISOMIR_MERGE_DIST for k in kept):
                kept.append(p)
        u_peaks_merged = sorted(kept)
    
    for p in u_peaks_merged:
        if smooth_unique[p] >= min_depth:
            peaks_found.append((start + p, "Unique"))

    # --- Pass B: Multi Rescue ---
    smooth_total = smooth_signal(cov_total)
    m_peaks, _ = find_peaks(smooth_total, distance=PEAK_MIN_DIST, prominence=min_depth*PEAK_PROMINENCE_REL)
    
    for p in m_peaks:
        # Rescue Rule: Must be >25nt away from any Unique Anchor
        dist_to_unique = min([abs(p - (up - start)) for up in u_peaks_merged]) if u_peaks_merged else 999
        
        if dist_to_unique > 25:
            if smooth_total[p] >= min_depth:
                peaks_found.append((start + p, "Multi"))

    return peaks_found

# -------------------------------------------------------------------------
# STAGE 7-9: Metrics & Gating
# -------------------------------------------------------------------------

def compute_dicer_metrics(bam, chrom, peak_pos, strand, window=15):
    """Compute Biogenesis Evidence Vector."""
    start = peak_pos - window
    end = peak_pos + window
    
    reads = [r for r in bam.fetch(chrom, start, end) if (("-" if r.is_reverse else "+") == strand)]
    if not reads: return None

    total = len(reads)
    
    # 1. Length Stats
    lens = [r.query_length for r in reads]
    c_len = Counter(lens)
    len_mode = c_len.most_common(1)[0][0]
    frac_20_24 = sum(c_len[l] for l in range(20, 25)) / total
    
    # 2. Start Stats (Precision/Sharpness)
    starts = [r.reference_start for r in reads]
    c_start = Counter(starts)
    top_starts = c_start.most_common(2)
    
    top1_n = top_starts[0][1]
    top2_n = top_starts[1][1] if len(top_starts) > 1 else 0
    dominance = (top1_n + top2_n) / total  # Sharpness
    
    # Precision 5p: Fraction at Dominant Start +/- 1nt
    dom_start = top_starts[0][0]
    prec_5p_n = sum(1 for s in starts if abs(s - dom_start) <= 1)
    precision_5p = prec_5p_n / total
    
    # 3. Complexity
    seqs = set(r.query_sequence for r in reads)
    distinct_seq_count = len(seqs)
    
    return {
        "len_mode": len_mode,
        "frac_20_24": frac_20_24,
        "dominance": dominance,
        "precision_5p": precision_5p,
        "distinct_seqs": distinct_seq_count,
        "depth": total
    }

def check_repeat_masker(chrom, start, end, repeat_bed):
    """Check overlap with RepeatMasker."""
    if not repeat_bed or repeat_bed == "None": return (False, False, "None")
    
    region = f"{chrom}\t{start}\t{end}"
    a = pybedtools.BedTool(region, from_string=True)
    hits = a.intersect(repeat_bed, wo=True)
    
    is_simple = False
    family = "None"
    
    for h in hits:
        # Adapt column index based on your RM BED format. usually col 3 is name.
        rep_name = h[3]
        family = rep_name
        lower = rep_name.lower()
        if "simple" in lower or "low" in lower or "satellite" in lower or ")n" in lower:
            is_simple = True
            
    return (len(hits) > 0, is_simple, family)

# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="miRPV-NG Upstream Harvester")
    parser.add_argument("--fastq", required=True)
    parser.add_argument("--genome-index", required=True)
    parser.add_argument("--genome-fa", required=True)
    parser.add_argument("--blocklist-index", required=True)
    parser.add_argument("--repeat-bed", default=None)
    parser.add_argument("--mirgenedb-gff", default=None)
    parser.add_argument("--mirbase-gff", default=None)
    parser.add_argument("--adapter", default="TGGAATTCTCGGGTGCCAAGG")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--threads", type=int, default=8)
    
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Logs
    reject_file = open(os.path.join(args.out_dir, "rejects.tsv"), "w")
    reject_file.write("locus\tstage\treason\tvalue\tthreshold\n")
    
    # --- Stages 1-3 ---
    logger.info("Starting Preprocessing & Mapping...")
    disc_fastq = preprocess_and_blocklist(args.fastq, args.out_dir, args.blocklist_index, args.adapter, args.threads)
    bam_path = map_genome(disc_fastq, args.out_dir, args.genome_index, args.threads)
    
    # --- Stage 4: Known Quantification ---
    if args.mirgenedb_gff:
        quantify_knowns(bam_path, args.mirgenedb_gff, args.mirbase_gff, os.path.join(args.out_dir, "quantification.tsv"))
    
    # --- Stage 5: Island Formation ---
    logger.info("Building Islands...")
    reads_bed = pybedtools.BedTool(bam_path).bam_to_bed()
    # Post-map length filter
    reads_bed = reads_bed.filter(lambda x: MIN_LEN_DISCOVERY <= int(x.end)-int(x.start) <= MAX_LEN_DISCOVERY)
    islands = reads_bed.merge(s=True, d=CLUSTER_GAP, c=1, o="count")
    
    # --- Stage 6-10: Harvest ---
    logger.info("Harvesting Candidates...")
    
    bam = pysam.AlignmentFile(bam_path, "rb")
    genome = pysam.FastaFile(args.genome_fa)
    repeat_bed = pybedtools.BedTool(args.repeat_bed) if args.repeat_bed else None
    
    out_tsv = open(os.path.join(args.out_dir, "candidates.tsv"), "w")
    cols = ["id", "chrom", "start", "end", "strand", "anchor_type", "window_size", 
            "depth", "sharpness", "precision_5p", "mirna_frac", "complexity", 
            "repeat_flag", "repeat_family", "sequence"]
    out_tsv.write("\t".join(cols) + "\n")
    
    cand_count = 0
    
    for island in islands:
        if int(island[3]) < MIN_ISLAND_DEPTH: continue
            
        peaks = call_peaks_in_island(bam, island.chrom, island.start, island.end, island.strand, MIN_ISLAND_DEPTH)
        
        if not peaks:
            reject_file.write(f"{island.chrom}:{island.start}\tPEAK_CALL\tNoPeak\t0\t0\n")
            continue
            
        for p_idx, (peak_pos, anchor) in enumerate(peaks):
            locus_id = f"{island.chrom}:{peak_pos}:{island.strand}"
            
            # Metric Calculation
            m = compute_dicer_metrics(bam, island.chrom, peak_pos, island.strand)
            if not m: continue
            
            # --- PREFILTERS ---
            if m['frac_20_24'] < 0.30 and m['len_mode'] not in range(20, 25):
                reject_file.write(f"{locus_id}\tPREFILTER\tBadLengthDist\t{m['frac_20_24']:.2f}\t0.30\n")
                continue
                
            if m['distinct_seqs'] < 2 and m['depth'] > 20:
                reject_file.write(f"{locus_id}\tPREFILTER\tMonoSeqPeak\t{m['distinct_seqs']}\t2\n")
                continue
            
            # --- ANCHOR GATES ---
            pass_gate = False
            if anchor == "Unique":
                if m['dominance'] >= 0.50 or m['precision_5p'] >= 0.70: pass_gate = True
            else: # Multi - Strict
                if m['precision_5p'] >= 0.85 and m['dominance'] >= 0.60: pass_gate = True
                    
            if not pass_gate:
                reject_file.write(f"{locus_id}\tANCHOR_GATE\tLowEvidence_{anchor}\t{m['precision_5p']:.2f}\tReq\n")
                continue
            
            # --- REPEAT GATE ---
            has_rep, is_simple, rep_fam = check_repeat_masker(island.chrom, peak_pos-10, peak_pos+10, repeat_bed)
            repeat_flag = False
            if has_rep:
                if is_simple:
                    reject_file.write(f"{locus_id}\tREPEAT_GATE\tSimpleRepeat\t{rep_fam}\tReject\n")
                    continue
                repeat_flag = True
                if anchor == "Multi": # Extra strict for Multi+Repeat
                    if not (m['precision_5p'] >= 0.90 and m['dominance'] >= 0.70):
                        reject_file.write(f"{locus_id}\tREPEAT_GATE\tRepeatStrictFail\t{m['precision_5p']:.2f}\t0.90\n")
                        continue

            # --- EXCISION ---
            for radius in WINDOW_SIZES:
                w_start = max(0, peak_pos - radius)
                w_end = peak_pos + radius
                try:
                    seq = genome.fetch(island.chrom, w_start, w_end)
                    if island.strand == "-":
                        seq = SeqIO.Seq(seq).reverse_complement()
                        seq = str(seq)
                except: continue

                cid = f"Cand_{cand_count}_P{p_idx}_W{radius*2}"
                row = [
                    cid, island.chrom, str(w_start), str(w_end), island.strand,
                    anchor, str(radius*2),
                    str(m['depth']), f"{m['dominance']:.3f}", f"{m['precision_5p']:.3f}",
                    f"{m['frac_20_24']:.2f}", str(m['distinct_seqs']),
                    str(repeat_flag), rep_fam, seq
                ]
                out_tsv.write("\t".join(row) + "\n")
                
        cand_count += 1

    out_tsv.close()
    reject_file.close()
    logger.info(f"Done. Generated {cand_count} candidate windows.")

if __name__ == "__main__":
    main()
