#!/usr/bin/env python3
"""
fastq_to_peaks.py

miRPV-NG sRNA-seq mode — Stage 1–8 (Peak+Pad Evidence-First Hybrid)
FASTQ -> candidate peaks + (optional) dual-window sequence excision -> candidates.tsv + candidates.fa

Key commitments (DO NOT redesign):
  - unique-first anchoring, multi-rescue
  - smoothing + SciPy signal processing optional
  - auditable rejects.tsv
  - two-layer outputs: coordinates/metadata always; structures later for strict finalists

External tools:
  - cutadapt
  - bowtie (Bowtie1)
Optional tools:
  - bedtools (only if repeat-bed enabled)
  - samtools (only if --genome-fasta used AND pysam not available)

Stage 2 (Blocklist) IMPORTANT:
  - Some Bowtie1 conda wrappers fail silently on gz FASTQ input (exit code 1).
  - Therefore, if blocklist is enabled and input is .gz, we transparently gunzip to
    a temp plain FASTQ for the blocklist bowtie step only, then delete temp.
  - Downstream continues with gz to keep I/O consistent.

Outputs:
  - <sample>.peaks.bed
  - <sample>.peaks.tsv (peak metrics)
  - candidates.tsv (dual windows) [only if --genome-fasta]
  - candidates.fa  (dual windows) [only if --genome-fasta]
  - rejects.tsv (auditable)
  - qc.json
  - run.log
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ----------------------------
# Utilities
# ----------------------------

_RC = str.maketrans("ACGTUNacgtun", "TGCAANtgcaan")


def revcomp(seq: str) -> str:
    return seq.translate(_RC)[::-1]


def have_exe(name: str) -> bool:
    return shutil.which(name) is not None


def run_cmd(cmd: List[str], log_path: Path, quiet_stdout: bool = False) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        if quiet_stdout:
            proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=log)
        else:
            proc = subprocess.run(cmd, stdout=log, stderr=log)
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def count_fastq_reads(path: Path) -> int:
    opener = gzip.open if str(path).endswith(".gz") else open
    n = 0
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            n += 1
    return n // 4


def gunzip_to_plain(in_gz: Path, out_plain: Path) -> None:
    with gzip.open(in_gz, "rb") as fin, open(out_plain, "wb") as fout:
        shutil.copyfileobj(fin, fout)


def gzip_plain_fastq(in_plain: Path, out_gz: Path) -> None:
    with open(in_plain, "rb") as fin, gzip.open(out_gz, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    in_plain.unlink(missing_ok=True)


def is_low_complexity(seq: str) -> bool:
    s = seq.upper().replace("U", "T")
    if not s:
        return True
    uniq = set(s)
    if len(uniq) <= 2:
        return True
    top = Counter(s).most_common(1)[0][1]
    return (top / len(s)) >= 0.8


# ----------------------------
# Data models
# ----------------------------

@dataclass
class ReadAln:
    chrom: str
    start0: int
    end0: int
    strand: str
    seq: str
    nm: int
    is_unique: bool


@dataclass
class Peak:
    chrom: str
    peak_center0: int
    strand: str
    island_id: str
    anchor_type: str  # Unique or Multi


# ----------------------------
# Bowtie1 text parsing (auditable uniqueness)
# ----------------------------

def parse_bowtie_line(line: str) -> Optional[Tuple[str, ReadAln]]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 6:
        return None
    rid, strand, ref, off_s, seq, _ = parts[:6]
    try:
        off = int(off_s)
    except ValueError:
        return None

    nm = 0
    if len(parts) >= 8:
        mm = parts[7]
        if mm and mm != "0":
            nm = mm.count(",") + 1

    aln = ReadAln(chrom=ref, start0=off, end0=off + len(seq), strand=strand, seq=seq, nm=nm, is_unique=False)
    return rid, aln


# ----------------------------
# Signal processing helpers
# ----------------------------

def smooth_rolling_mean(cov: List[int], w: int) -> List[float]:
    if w <= 1:
        return [float(x) for x in cov]
    half = w // 2
    out: List[float] = []
    n = len(cov)
    prefix = [0]
    for x in cov:
        prefix.append(prefix[-1] + x)
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        out.append((prefix[b] - prefix[a]) / (b - a))
    return out


def find_peaks_scipy(
    y: List[float],
    distance: int,
    prominence: Optional[float],
    width: Optional[Tuple[float, float]]
) -> List[int]:
    try:
        from scipy.signal import find_peaks  # type: ignore
    except Exception:
        return []

    kwargs = {"distance": distance}
    if prominence is not None:
        kwargs["prominence"] = prominence
    if width is not None:
        kwargs["width"] = width

    peaks, _props = find_peaks(y, **kwargs)
    return [int(p) for p in peaks]


def find_peaks_fallback(
    y: List[float],
    min_distance: int,
    min_prominence_frac: float
) -> List[int]:
    n = len(y)
    if n < 3:
        return []
    cands = [i for i in range(1, n - 1) if y[i] >= y[i - 1] and y[i] >= y[i + 1] and y[i] > 0]
    if not cands:
        return []

    prom_ok: List[int] = []
    for i in cands:
        a = max(0, i - min_distance)
        b = min(n, i + min_distance + 1)
        local_min = min(y[a:b]) if b > a else 0.0
        peak = y[i]
        if peak <= 0:
            continue
        prom = (peak - local_min) / peak
        if prom >= min_prominence_frac:
            prom_ok.append(i)

    if not prom_ok:
        return []

    prom_ok.sort(key=lambda i: y[i], reverse=True)
    selected: List[int] = []
    occupied = [False] * n
    for i in prom_ok:
        if occupied[i]:
            continue
        selected.append(i)
        a = max(0, i - min_distance)
        b = min(n, i + min_distance + 1)
        for j in range(a, b):
            occupied[j] = True
    selected.sort()
    return selected


def merge_micropeaks(peaks: List[int], max_sep: int) -> List[int]:
    if not peaks:
        return []
    peaks = sorted(peaks)
    merged = [peaks[0]]
    for p in peaks[1:]:
        if p - merged[-1] <= max_sep:
            continue
        merged.append(p)
    return merged


# ----------------------------
# Islands
# ----------------------------

def build_islands(reads: List[ReadAln], gap: int) -> Dict[str, List[ReadAln]]:
    by_chr_strand: Dict[Tuple[str, str], List[ReadAln]] = defaultdict(list)
    for r in reads:
        by_chr_strand[(r.chrom, r.strand)].append(r)

    islands: Dict[str, List[ReadAln]] = {}
    island_idx = 0

    for (chrom, strand), rs in by_chr_strand.items():
        rs.sort(key=lambda x: (x.start0, x.end0))
        cur: List[ReadAln] = []
        cur_end: Optional[int] = None

        for r in rs:
            if not cur:
                cur = [r]
                cur_end = r.end0
                continue
            assert cur_end is not None
            if r.start0 <= cur_end + gap:
                cur.append(r)
                cur_end = max(cur_end, r.end0)
            else:
                island_idx += 1
                iid = f"{chrom}:{cur[0].start0}-{cur_end}:{strand}:island{island_idx}"
                islands[iid] = cur
                cur = [r]
                cur_end = r.end0

        if cur:
            island_idx += 1
            assert cur_end is not None
            iid = f"{chrom}:{cur[0].start0}-{cur_end}:{strand}:island{island_idx}"
            islands[iid] = cur

    return islands


# ----------------------------
# Support stats
# ----------------------------

def compute_stack_stats(reads: List[ReadAln], peak_center0: int, window: int) -> Dict[str, float]:
    a = peak_center0 - window
    b = peak_center0 + window + 1
    win_reads = [r for r in reads if not (r.end0 <= a or r.start0 >= b)]
    depth = len(win_reads)
    if depth == 0:
        return {
            "depth_raw": 0.0,
            "len_mode": -1.0,
            "frac_len_20_24": 0.0,
            "distinct_seq_count": 0.0,
            "dominance_top1_top2": 0.0,
            "precision_5p": 0.0,
            "start_entropy": 0.0,
        }

    lengths = [len(r.seq) for r in win_reads]
    len_counts = Counter(lengths)
    len_mode = len_counts.most_common(1)[0][0]
    frac_20_24 = sum(c for L, c in len_counts.items() if 20 <= L <= 24) / depth

    starts = [r.start0 for r in win_reads]
    start_counts = Counter(starts)
    top = start_counts.most_common(2)
    top1 = top[0][1]
    top2 = top[1][1] if len(top) > 1 else 0
    dominance = (top1 + top2) / depth

    dom_pos = top[0][0]
    within = sum(c for s, c in start_counts.items() if abs(s - dom_pos) <= 1)
    precision_5p = within / depth

    ent = 0.0
    for c in start_counts.values():
        p = c / depth
        ent -= p * math.log(p + 1e-12, 2)

    distinct_seq_count = len(set(r.seq for r in win_reads))

    return {
        "depth_raw": float(depth),
        "len_mode": float(len_mode),
        "frac_len_20_24": float(frac_20_24),
        "distinct_seq_count": float(distinct_seq_count),
        "dominance_top1_top2": float(dominance),
        "precision_5p": float(precision_5p),
        "start_entropy": float(ent),
    }


# ----------------------------
# Genome fetch
# ----------------------------

class GenomeFetcher:
    def __init__(self, fasta_path: Path, samtools_bin: str = "samtools"):
        self.fasta_path = fasta_path
        self.samtools_bin = samtools_bin
        self._pysam = None
        try:
            import pysam  # type: ignore
            self._pysam = pysam
            self._fa = pysam.FastaFile(str(fasta_path))
        except Exception:
            self._pysam = None
            self._fa = None

    def has_fai(self) -> bool:
        return (Path(str(self.fasta_path) + ".fai")).exists() or self.fasta_path.with_suffix(self.fasta_path.suffix + ".fai").exists()

    def ensure_index(self) -> None:
        if self._pysam is not None and self.has_fai():
            return
        if not have_exe(self.samtools_bin):
            raise RuntimeError("Genome fetch requires pysam or samtools. Neither is available.")
        if not self.has_fai():
            subprocess.run([self.samtools_bin, "faidx", str(self.fasta_path)], check=True)

    def fetch(self, chrom: str, start0: int, end0: int) -> str:
        if start0 < 0:
            start0 = 0
        if end0 <= start0:
            return ""
        if self._pysam is not None:
            try:
                seq = self._fa.fetch(chrom, start0, end0)
            except Exception:
                return ""
            return seq.upper()

        if not have_exe(self.samtools_bin):
            raise RuntimeError("samtools not found for genome fetch fallback")
        region = f"{chrom}:{start0+1}-{end0}"
        proc = subprocess.run([self.samtools_bin, "faidx", str(self.fasta_path), region],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            return ""
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln and not ln.startswith(">")]
        return "".join(lines).upper()


# ----------------------------
# Repeat intersect (batch, sorted)
# ----------------------------

def batch_repeat_lookup(
    peaks_for_windows: List[Tuple[str, int, int, str, str]],
    repeat_bed: Path,
    bedtools_bin: str,
    outdir: Path,
) -> Dict[str, str]:
    if not have_exe(bedtools_bin):
        raise RuntimeError("bedtools not found but --repeat-bed enabled")

    a_bed = outdir / "._tmp_all_windows.bed"
    a_sorted = outdir / "._tmp_all_windows.sorted.bed"

    with a_bed.open("w", encoding="utf-8") as f:
        for chrom, s, e, key, strand in peaks_for_windows:
            s2 = 0 if s < 0 else s
            if e <= s2:
                continue
            f.write(f"{chrom}\t{s2}\t{e}\t{key}\t0\t{strand}\n")

    subprocess.run(
        ["sort", "-k1,1", "-k2,2n", str(a_bed)],
        check=True,
        stdout=a_sorted.open("w", encoding="utf-8")
    )

    repeat_bed = Path(repeat_bed)
    candidate_sorted = repeat_bed.with_suffix(repeat_bed.suffix + ".sorted.bed")
    if candidate_sorted.exists():
        b_sorted_path = candidate_sorted
    else:
        b_sorted_path = outdir / "._tmp_repeat.sorted.bed"
        subprocess.run(
            ["sort", "-k1,1", "-k2,2n", str(repeat_bed)],
            check=True,
            stdout=b_sorted_path.open("w", encoding="utf-8")
        )

    proc = subprocess.run(
        [bedtools_bin, "intersect", "-sorted", "-wa", "-wb", "-a", str(a_sorted), "-b", str(b_sorted_path)],
        capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"bedtools intersect failed: {proc.stderr}")

    rep: Dict[str, str] = {}
    for ln in proc.stdout.splitlines():
        if not ln.strip():
            continue
        fields = ln.split("\t")
        if len(fields) < 10:
            continue
        key = fields[3]
        rep_class = fields[9]
        if key not in rep:
            rep[key] = rep_class

    return rep


# ----------------------------
# Args (must match CLI)
# ----------------------------

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="miRPV-NG sRNA-seq Stage1-8: FASTQ -> peaks (+optional excision).")
    ap.add_argument("--fastq", required=True)
    ap.add_argument("--sample-id", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--cutadapt", default="cutadapt")
    ap.add_argument("--bowtie", default="bowtie")
    ap.add_argument("--bowtie-index", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--max-multimaps", type=int, default=50)

    # Stage 2: blocklist
    ap.add_argument("--blocklist-index", default=None)
    ap.add_argument("--blocklist-name", default="rfam")
    ap.add_argument("--blocklist-mismatches", type=int, default=0)
    ap.add_argument("--blocklist-max-align", type=int, default=1)

    ap.add_argument("--island-gap", type=int, default=50)
    ap.add_argument("--min-depth", type=int, default=5)
    ap.add_argument("--min-cpm", type=float, default=0.5)

    ap.add_argument("--smooth-w", type=int, default=3, dest="smooth_w")
    ap.add_argument("--peak-distance", type=int, default=35, dest="peak_distance")
    ap.add_argument("--peak-micromerge", type=int, default=8, dest="peak_micromerge")

    ap.add_argument("--use-scipy", action="store_true", dest="use_scipy")
    ap.add_argument("--scipy-prominence", type=float, default=None, dest="scipy_prominence")
    ap.add_argument("--scipy-width-min", type=float, default=None, dest="scipy_width_min")
    ap.add_argument("--scipy-width-max", type=float, default=None, dest="scipy_width_max")

    ap.add_argument("--fallback-prom-frac", type=float, default=0.30, dest="fallback_prom_frac")

    ap.add_argument("--support-window", type=int, default=15)
    ap.add_argument("--hard-frac-20-24", type=float, default=0.30)

    ap.add_argument("--anchor-unique-dominance", type=float, default=0.50)
    ap.add_argument("--anchor-unique-prec5p", type=float, default=0.70)
    ap.add_argument("--anchor-multi-dominance", type=float, default=0.60)
    ap.add_argument("--anchor-multi-prec5p", type=float, default=0.85)

    ap.add_argument("--repeat-bed", default=None)
    ap.add_argument("--bedtools", default="bedtools")
    ap.add_argument("--repeat-multi-prec5p", type=float, default=0.90)
    ap.add_argument("--repeat-multi-dominance", type=float, default=0.70)

    ap.add_argument("--genome-fasta", default=None)
    ap.add_argument("--pads", nargs="+", type=int, default=[70, 100])
    ap.add_argument("--samtools", default="samtools")

    return ap


# ----------------------------
# Entrypoint
# ----------------------------

def run_fastq_to_peaks(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log_path = outdir / "run.log"
    rejects_path = outdir / "rejects.tsv"
    peaks_bed = outdir / f"{args.sample_id}.peaks.bed"
    peaks_tsv = outdir / f"{args.sample_id}.peaks.tsv"
    qc_path = outdir / "qc.json"

    candidates_tsv = outdir / "candidates.tsv"
    candidates_fa = outdir / "candidates.fa"

    with rejects_path.open("w", encoding="utf-8") as rej:
        rej.write("\t".join([
            "sample_id", "stage", "chrom", "start", "end", "strand", "island_id", "peak_center",
            "anchor_type", "reject_id", "reason", "metric", "value", "threshold", "action", "notes"
        ]) + "\n")

    def log_reject(stage: str, chrom: str, start: int, end: int, strand: str,
                   island_id: str, peak_center: str, anchor_type: str,
                   reject_id: str, reason: str, metric: str, value: str,
                   threshold: str, action: str, notes: str = "") -> None:
        with rejects_path.open("a", encoding="utf-8") as rej:
            rej.write("\t".join([
                args.sample_id, stage, chrom, str(start), str(end), strand,
                island_id or "", peak_center or "", anchor_type or "",
                reject_id or "", reason, metric, value, threshold, action, notes
            ]) + "\n")

    if not have_exe(args.cutadapt):
        raise RuntimeError(f"cutadapt not found: {args.cutadapt}")
    if not have_exe(args.bowtie):
        raise RuntimeError(f"bowtie (Bowtie1) not found: {args.bowtie}")

    # ----------------------------
    # Stage 1: trim + size select
    # ----------------------------
    trimmed_fq = outdir / f"{args.sample_id}.trimmed.18_30.fastq.gz"
    adapter = args.adapter or "TGGAATTCTCGGGTGCCAAGG"

    run_cmd([
        args.cutadapt,
        "-a", adapter,
        "-m", "18", "-M", "30",
        "--cores", str(args.threads),
        "-o", str(trimmed_fq),
        str(args.fastq),
    ], log_path)

    # ----------------------------
    # Stage 2: blocklist (optional)
    # ----------------------------
    fq_for_genome = trimmed_fq

    blocklist_enabled = bool(args.blocklist_index)
    block_reads_in = None
    block_reads_pass = None
    block_reads_removed = None
    block_index = str(args.blocklist_index)

    tmp_plain_in: Optional[Path] = None

    if blocklist_enabled:
        block_reads_in = count_fastq_reads(trimmed_fq)

        # IMPORTANT: bowtie conda wrapper may fail on gz; use a temp plain FASTQ for blocklist mapping
        block_in = trimmed_fq
        if str(trimmed_fq).endswith(".gz"):
            tmp_plain_in = outdir / f"{args.sample_id}.trimmed.18_30.block_in.fastq"
            gunzip_to_plain(trimmed_fq, tmp_plain_in)
            block_in = tmp_plain_in

        un_plain = outdir / f"{args.sample_id}.blocklist.pass.fastq"
        pass_gz = outdir / f"{args.sample_id}.blocklist.pass.fastq.gz"
        block_aln = outdir / f"{args.sample_id}.blocklist.bowtie.txt"

        # Any hit to blocklist => removed (kept out of --un)
        cmd = [
            args.bowtie,
            "-q",
            "-v", str(int(args.blocklist_mismatches)),
            "-k", str(int(args.blocklist_max_align)),
            "--best",
            "-p", str(args.threads),
            "--un", str(un_plain),
            str(args.blocklist_index),
            str(block_in),
            str(block_aln),
        ]

        # DO NOT quiet stdout: wrappers sometimes emit relevant errors on stdout
        run_cmd(cmd, log_path, quiet_stdout=False)

        # Keep downstream I/O consistent (gz)
        gzip_plain_fastq(un_plain, pass_gz)
        fq_for_genome = pass_gz

        block_reads_pass = count_fastq_reads(pass_gz)
        block_reads_removed = int(block_reads_in - block_reads_pass)

        if tmp_plain_in is not None:
            tmp_plain_in.unlink(missing_ok=True)

    # ----------------------------
    # Stage 3: genome mapping (Bowtie1)
    # ----------------------------
    bowtie_out = outdir / f"{args.sample_id}.bowtie.txt"

    # IMPORTANT: bowtie conda wrapper may fail on gz; use a temp plain FASTQ
    tmp_plain_genome: Optional[Path] = None
    genome_in = fq_for_genome
    if str(fq_for_genome).endswith(".gz"):
        tmp_plain_genome = outdir / f"{args.sample_id}.genome_in.fastq"
        gunzip_to_plain(fq_for_genome, tmp_plain_genome)
        genome_in = tmp_plain_genome

    cmd = [
        args.bowtie,
        "-q",                     # <-- FIX: tell bowtie input is FASTQ
        "-v", "1",
        "-m", str(args.max_multimaps),
        "--best", "--strata",
        "-p", str(args.threads),
        args.bowtie_index,
        str(genome_in),
        str(bowtie_out),
    ]
    run_cmd(cmd, log_path, quiet_stdout=False)

    if tmp_plain_genome is not None:
        tmp_plain_genome.unlink(missing_ok=True)

    # Determine uniqueness by counting mappings per read id
    read_hits_by_id: Dict[str, List[ReadAln]] = defaultdict(list)
    with bowtie_out.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            parsed = parse_bowtie_line(line)
            if parsed is None:
                continue
            rid, aln = parsed
            read_hits_by_id[rid].append(aln)

    reads: List[ReadAln] = []
    too_many_maps = 0
    for rid, alns in read_hits_by_id.items():
        if len(alns) > args.max_multimaps:
            too_many_maps += 1
            log_reject("MAP", ".", -1, -1, ".", "", "", "", rid,
                       "TooManyMaps", "nhits", str(len(alns)),
                       f"<= {args.max_multimaps}", "Reject", "")
            continue
        is_unique = (len(alns) == 1)
        for a in alns:
            a.is_unique = is_unique
            reads.append(a)

    if not reads:
        peaks_bed.write_text("", encoding="utf-8")
        peaks_tsv.write_text("", encoding="utf-8")
        qc_path.write_text(json.dumps({
            "sample_id": args.sample_id,
            "mapped_alignments": 0,
            "unique_alignments": 0,
            "multimap_alignments": 0,
            "lib_size_for_cpm": 0,
            "islands_total": 0,
            "peaks_written": 0,
            "candidates_written": 0,
            "too_many_maps_reads": too_many_maps,
            "blocklist": {
                "enabled": blocklist_enabled,
                "name": args.blocklist_name,
                "index": args.blocklist_index,
                "reads_in": block_reads_in,
                "reads_pass": block_reads_pass,
                "reads_removed": block_reads_removed,
                "mismatches": args.blocklist_mismatches,
                "max_align": args.blocklist_max_align,
            },
        }, indent=2), encoding="utf-8")
        print(f"[fastq-to-peaks] No mapped reads after filters. outdir={outdir}")
        return 0

    unique_alns = sum(1 for r in reads if r.is_unique)
    mult_alns = len(reads) - unique_alns
    lib_size = len(reads)

    # ----------------------------
    # Stage 4: islands
    # ----------------------------
    islands = build_islands(reads, gap=args.island_gap)

    # Prepare outputs
    peaks_bed.write_text("", encoding="utf-8")
    with peaks_tsv.open("w", encoding="utf-8") as pt:
        pt.write("\t".join([
            "sample_id", "chrom", "peak_center0", "strand", "island_id", "anchor_type",
            "depth_raw", "cpm", "len_mode", "frac_20_24", "dominance", "prec5p", "start_entropy",
            "repeat_class"
        ]) + "\n")

    width = None
    if args.scipy_width_min is not None or args.scipy_width_max is not None:
        width = (
            args.scipy_width_min if args.scipy_width_min is not None else 0.0,
            args.scipy_width_max if args.scipy_width_max is not None else float("inf"),
        )

    def adaptive_prom(y: List[float]) -> float:
        m = max(y) if y else 0.0
        return max(1.0, 0.05 * m)

    pending_peaks: List[Tuple[Peak, Dict[str, float]]] = []

    # ----------------------------
    # Stage 5–7: peak calling + gates
    # ----------------------------
    for island_id, island_reads in islands.items():
        chrom = island_reads[0].chrom
        strand = island_reads[0].strand
        isl_start = min(r.start0 for r in island_reads)
        isl_end = max(r.end0 for r in island_reads)

        depth_island = len(island_reads)
        cpm_island = (depth_island / lib_size) * 1e6

        if depth_island < args.min_depth or cpm_island < args.min_cpm:
            log_reject("ISLAND", chrom, isl_start, isl_end, strand, island_id, "", "", island_id,
                       "LowDepth", "depth|cpm", f"{depth_island}|{cpm_island:.3f}",
                       f">={args.min_depth} and >={args.min_cpm}", "Reject", "")
            continue

        span_len = isl_end - isl_start
        if span_len <= 0:
            continue

        cov_unique = [0] * span_len
        cov_all = [0] * span_len
        for r in island_reads:
            a = max(isl_start, r.start0)
            b = min(isl_end, r.end0)
            if b <= a:
                continue
            for pos in range(a, b):
                idx = pos - isl_start
                cov_all[idx] += 1
                if r.is_unique:
                    cov_unique[idx] += 1

        y_unique = smooth_rolling_mean(cov_unique, w=args.smooth_w)
        y_all = smooth_rolling_mean(cov_all, w=args.smooth_w)

        if args.use_scipy:
            pu = find_peaks_scipy(
                y_unique,
                distance=args.peak_distance,
                prominence=args.scipy_prominence if args.scipy_prominence is not None else adaptive_prom(y_unique),
                width=width
            )
        else:
            pu = find_peaks_fallback(y_unique, min_distance=args.peak_distance, min_prominence_frac=args.fallback_prom_frac)
        pu = merge_micropeaks(pu, max_sep=args.peak_micromerge)

        peaks: List[Peak] = []
        unique_centers = set()
        for idx in pu:
            center = isl_start + idx
            peaks.append(Peak(chrom=chrom, peak_center0=center, strand=strand, island_id=island_id, anchor_type="Unique"))
            unique_centers.add(center)

        if args.use_scipy:
            pa = find_peaks_scipy(
                y_all,
                distance=args.peak_distance,
                prominence=args.scipy_prominence if args.scipy_prominence is not None else adaptive_prom(y_all),
                width=width
            )
        else:
            pa = find_peaks_fallback(y_all, min_distance=args.peak_distance, min_prominence_frac=args.fallback_prom_frac)
        pa = merge_micropeaks(pa, max_sep=args.peak_micromerge)

        for idx in pa:
            center = isl_start + idx
            # Rescue: keep multi peak only if no unique support near it
            if any(abs(center - c) <= 25 for c in unique_centers):
                local_a = max(0, idx - 5)
                local_b = min(span_len, idx + 6)
                if sum(cov_unique[local_a:local_b]) > 0:
                    continue
            peaks.append(Peak(chrom=chrom, peak_center0=center, strand=strand, island_id=island_id, anchor_type="Multi"))


        # ---- Forced peak rescue for tiny / edge-case islands ----
        if not peaks and max(y_all) > 0:
            idx = max(range(len(y_all)), key=lambda k: y_all[k])
            center = isl_start + idx
            peaks = [Peak(
                chrom=chrom,
                peak_center0=center,
                strand=strand,
                island_id=island_id,
                anchor_type="ForcedMax"
            )]

        if not peaks:
            log_reject("PEAK_CALL", chrom, isl_start, isl_end, strand,
                       island_id, "", "", island_id,
                       "NoPeak", "peaks", "0", ">=1", "Reject", "")
            continue


        for pk in peaks:
            stats = compute_stack_stats(island_reads, pk.peak_center0, window=args.support_window)
            depth_raw = int(stats["depth_raw"])
            len_mode = int(stats["len_mode"])
            frac_20_24 = float(stats["frac_len_20_24"])
            distinct_seq_count = int(stats["distinct_seq_count"])
            dominance = float(stats["dominance_top1_top2"])
            prec5p = float(stats["precision_5p"])

            #if frac_20_24 < args.hard_frac_20_24 and (len_mode < 20 or len_mode > 24):
             #   log_reject("PREFILTER", chrom,
              #             pk.peak_center0 - args.support_window, pk.peak_center0 + args.support_window + 1,
               #            strand, island_id, str(pk.peak_center0), pk.anchor_type,
                #           f"{island_id}|{pk.peak_center0}",
                 #          "BadLengthDist", "len_mode|frac_20_24", f"{len_mode}|{frac_20_24:.3f}",
                  #         f"mode in [20..24] or frac_20_24>={args.hard_frac_20_24}", "Reject", "")
                #continue
            
            
         
                



            
            
            
            
            
            
            
            
            

            if depth_raw >= 10 and distinct_seq_count < 2:
                log_reject("PREFILTER", chrom,
                           pk.peak_center0 - args.support_window, pk.peak_center0 + args.support_window + 1,
                           strand, island_id, str(pk.peak_center0), pk.anchor_type,
                           f"{island_id}|{pk.peak_center0}",
                           "MonoSeqPeak", "distinct_seq_count", str(distinct_seq_count),
                           ">=2", "Reject", "")
                continue

            if pk.anchor_type == "Unique":
                if not (dominance >= args.anchor_unique_dominance or prec5p >= args.anchor_unique_prec5p):
                    log_reject("ANCHOR_GATE", chrom,
                               pk.peak_center0 - args.support_window, pk.peak_center0 + args.support_window + 1,
                               strand, island_id, str(pk.peak_center0), pk.anchor_type,
                               f"{island_id}|{pk.peak_center0}",
                               "LowAnchorEvidenceUnique", "dominance|prec5p", f"{dominance:.3f}|{prec5p:.3f}",
                               f"dominance>={args.anchor_unique_dominance} OR prec5p>={args.anchor_unique_prec5p}",
                               "Reject", "")
                    continue
            else:
                if not (dominance >= args.anchor_multi_dominance and prec5p >= args.anchor_multi_prec5p):
                    log_reject("ANCHOR_GATE", chrom,
                               pk.peak_center0 - args.support_window, pk.peak_center0 + args.support_window + 1,
                               strand, island_id, str(pk.peak_center0), pk.anchor_type,
                               f"{island_id}|{pk.peak_center0}",
                               "LowAnchorEvidenceMulti", "dominance|prec5p", f"{dominance:.3f}|{prec5p:.3f}",
                               f"dominance>={args.anchor_multi_dominance} AND prec5p>={args.anchor_multi_prec5p}",
                               "Reject", "")
                    continue

            pending_peaks.append((pk, stats))

    # ----------------------------
    # Stage 8: repeat lookup (optional)
    # ----------------------------
    repeat_lookup: Dict[str, str] = {}
    if args.repeat_bed:
        win_list: List[Tuple[str, int, int, str, str]] = []
        for pk, _stats in pending_peaks:
            a = pk.peak_center0 - args.support_window
            b = pk.peak_center0 + args.support_window + 1
            key = f"{args.sample_id}|{pk.island_id}|{pk.peak_center0}|{pk.anchor_type}"
            win_list.append((pk.chrom, a, b, key, pk.strand))
        repeat_lookup = batch_repeat_lookup(
            peaks_for_windows=win_list,
            repeat_bed=Path(args.repeat_bed),
            bedtools_bin=args.bedtools,
            outdir=outdir
        )

    # ----------------------------
    # Optional excision
    # ----------------------------
    do_excision = bool(args.genome_fasta)
    gf = None
    if do_excision:
        gf = GenomeFetcher(Path(args.genome_fasta), samtools_bin=args.samtools)
        gf.ensure_index()
        with open(candidates_tsv, "w", encoding="utf-8") as ct:
            ct.write("\t".join([
                "candidate_id", "sample_id",
                "chrom", "start0", "end0", "strand",
                "peak_center0", "pad",
                "island_id", "anchor_type",
                "depth_raw", "cpm",
                "len_mode", "frac_20_24", "dominance", "prec5p", "start_entropy",
                "repeat_class"
            ]) + "\n")
        candidates_fa.write_text("", encoding="utf-8")

    peaks_written = 0
    candidates_written = 0

    for pk, stats in pending_peaks:
        depth_raw = int(stats["depth_raw"])
        len_mode = int(stats["len_mode"])
        frac_20_24 = float(stats["frac_len_20_24"])
        dominance = float(stats["dominance_top1_top2"])
        prec5p = float(stats["precision_5p"])
        start_entropy = float(stats["start_entropy"])

        peak_key = f"{args.sample_id}|{pk.island_id}|{pk.peak_center0}|{pk.anchor_type}"
        repeat_class = repeat_lookup.get(peak_key, "None")

        # Write BED + TSV
        with peaks_bed.open("a", encoding="utf-8") as out_bed:
            out_bed.write("\t".join([
                pk.chrom,
                str(pk.peak_center0 - 1),
                str(pk.peak_center0),
                peak_key,
                str(int(min(1000, round(1000.0 * dominance)))),
                pk.strand
            ]) + "\n")

        with peaks_tsv.open("a", encoding="utf-8") as pt:
            pt.write("\t".join(map(str, [
                args.sample_id, pk.chrom, pk.peak_center0, pk.strand, pk.island_id, pk.anchor_type,
                depth_raw, f"{(depth_raw/lib_size)*1e6:.6f}", len_mode, f"{frac_20_24:.6f}",
                f"{dominance:.6f}", f"{prec5p:.6f}", f"{start_entropy:.6f}",
                repeat_class
            ])) + "\n")

        peaks_written += 1

        # Excision into dual pads
        if do_excision and gf is not None:
            for pad in args.pads:
                start0 = pk.peak_center0 - pad
                end0 = pk.peak_center0 + pad + 1
                seq = gf.fetch(pk.chrom, start0, end0)
                if not seq:
                    log_reject("EXCISE", pk.chrom, start0, end0, pk.strand, pk.island_id, str(pk.peak_center0), pk.anchor_type,
                               f"{peak_key}|pad{pad}",
                               "FetchFailed", "genome_fetch", "NA", "non-empty", "Reject",
                               f"genome_fasta={args.genome_fasta}")
                    continue

                if pk.strand == "-":
                    seq = revcomp(seq)

                if is_low_complexity(seq):
                    log_reject("EXCISE", pk.chrom, max(0, start0), end0, pk.strand, pk.island_id, str(pk.peak_center0), pk.anchor_type,
                               f"{peak_key}|pad{pad}",
                               "LowComplexity", "low_complexity", "1", "0", "Reject", "")
                    continue

                cand_id = f"{peak_key}|pad{pad}"
                with open(candidates_tsv, "a", encoding="utf-8") as ct:
                    ct.write("\t".join(map(str, [
                        cand_id, args.sample_id,
                        pk.chrom, max(0, start0), end0, pk.strand,
                        pk.peak_center0, pad,
                        pk.island_id, pk.anchor_type,
                        depth_raw, f"{(depth_raw/lib_size)*1e6:.6f}",
                        len_mode, f"{frac_20_24:.6f}", f"{dominance:.6f}", f"{prec5p:.6f}", f"{start_entropy:.6f}",
                        repeat_class
                    ])) + "\n")

                with candidates_fa.open("a", encoding="utf-8") as fa:
                    fa.write(f">{cand_id}\n{seq}\n")

                candidates_written += 1

    qc = {
        "sample_id": args.sample_id,
        "mapped_alignments": len(reads),
        "unique_alignments": unique_alns,
        "multimap_alignments": mult_alns,
        "lib_size_for_cpm": lib_size,
        "islands_total": len(islands),
        "pending_peaks_pre_repeat": len(pending_peaks),
        "peaks_written": peaks_written,
        "candidates_written": candidates_written if do_excision else 0,
        "too_many_maps_reads": too_many_maps,
        "blocklist": {
            "enabled": blocklist_enabled,
            "name": args.blocklist_name,
            "index": args.blocklist_index,
            "reads_in": block_reads_in,
            "reads_pass": block_reads_pass,
            "reads_removed": block_reads_removed,
            "mismatches": args.blocklist_mismatches,
            "max_align": args.blocklist_max_align,
        },
        "params": {
            "adapter": adapter,
            "island_gap": args.island_gap,
            "min_depth": args.min_depth,
            "min_cpm": args.min_cpm,
            "smooth_w": args.smooth_w,
            "peak_distance": args.peak_distance,
            "peak_micromerge": args.peak_micromerge,
            "use_scipy": bool(args.use_scipy),
            "scipy_prominence": args.scipy_prominence,
            "scipy_width": width,
            "support_window": args.support_window,
            "repeat_bed": args.repeat_bed,
            "genome_fasta": args.genome_fasta,
            "pads": list(args.pads),
        },
    }
    qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")

    print(f"[fastq-to-peaks] Wrote peaks: {peaks_written}")
    print(f"[fastq-to-peaks] peaks.bed: {peaks_bed}")
    print(f"[fastq-to-peaks] peaks.tsv: {peaks_tsv}")
    if do_excision:
        print(f"[fastq-to-peaks] candidates.tsv: {candidates_tsv}")
        print(f"[fastq-to-peaks] candidates.fa: {candidates_fa}")
    print(f"[fastq-to-peaks] rejects.tsv: {rejects_path}")
    print(f"[fastq-to-peaks] qc.json: {qc_path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)
    return run_fastq_to_peaks(args)


if __name__ == "__main__":
    raise SystemExit(main())
