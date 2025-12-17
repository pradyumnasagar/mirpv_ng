# Pre-miRNA Geometry Signature (PGS) – Geometry Block

PGS (Pre-miRNA Geometry Signature) is the geometry-only feature block used inside the extended feature set of **miRPV-NG**.  
It describes how a candidate precursor folds as a hairpin: the size and position of the main loop, the effective stem length, and the presence of known regulatory motifs in loop-proximal regions.

PGS is not a full biogenesis model; it is a compact geometric summary that the RandomForest can combine with energy, composition, and other structural features.

---

## 1. Inputs and scope

The PGS geometry block is computed from:

- The **primary sequence** `seq` (5'→3') of a candidate hairpin.
- The corresponding **secondary structure** in dot–bracket format `struct`, typically obtained from RNAfold.

PGS assumes that `struct` represents a reasonably hairpin-like object: a dominant stem with one major loop and possibly a few smaller internal loops or bulges. It is tolerant of noise, but it is not a full general RNA structure descriptor.

---

## 2. Loop detection and “dominant loop”

Internal loops are defined as contiguous `'.'` segments in the dot–bracket string that are flanked on both sides by paired bases (`'('` and `')'`):

- The helper `_find_internal_loops(struct)` returns a list of `(start, end)` indices (0-based, end exclusive) for such contiguous unpaired regions.
- From this list, the PGS block selects a single **dominant loop**. The current implementation favors biologically plausible candidates (central, sufficiently long, and with a clear flanking stem).

For the dominant loop, PGS extracts:

- Loop start and end indices.
- Loop length (`loop_size`).
- Loop sequence (`loop_seq = seq[loop_start:loop_end]`).

If no internal loops are detected, PGS falls back to a degenerate configuration with `loop_size = 0` and an empty loop sequence, and most loop-derived features become zero.

Everything outside the dominant loop is treated as part of the stem or flanking regions for the purpose of PGS features.

---

## 3. Geometry features

The table below summarizes the current PGS geometry features and their intended meaning.  
Names are given in the `pgs_*` style; if you refactor the code, this table should be kept in sync with the actual keys emitted by `compute_pgs_features`.

| Feature name              | Type      | Description                                                                                             |
|---------------------------|-----------|---------------------------------------------------------------------------------------------------------|
| `pgs_geom_score`          | float     | Aggregate geometry score combining stem length, loop size, motif presence, and penalties for poor folds. |
| `pgs_loop_size`           | float     | Length (in nt) of the dominant loop (`loop_end - loop_start`).                                         |
| `pgs_stem_len`            | float     | Effective stem length in nt, derived from the number and span of paired positions.                      |
| `pgs_ugug_count`          | float     | Count of the `UGUG` motif in the dominant loop sequence.                                               |
| `pgs_ugu_count`           | float     | Count of the `UGU` motif in the dominant loop sequence.                                                |
| `pgs_basal_ug_count`      | float     | Count of `UG` dinucleotides in the basal region near the 5' end, typically the first ~8 nt of `seq`.   |
| `pgs_cnnc_loose_count`    | float     | Approximate count of CNNC-like motifs in a region around the loop (loop ± 10 nt) using a loose `C..C` proxy. |
| `pgs_loop_size_norm`      | float     | Loop size normalized by total sequence length (`loop_size / L`).                                       |
| `pgs_stem_len_norm`       | float     | Stem length normalized by total sequence length (`stem_len / L`).                                      |

The `pgs_geom_score` is a heuristic scalar that rewards hairpins with:

- A reasonably long and continuous stem.
- A non-pathological loop size (neither extremely tiny nor extremely large).
- Presence of favorable motifs (UGUG/UGU/CNNC-like) in appropriate regions.

and penalizes configurations with:

- Very short stems.
- Extremely large or fragmented loops.
- Excessive or poorly placed internal loops.

The precise weighting is implemented in the code and may evolve; the intention is to provide a single geometry-informed score that the classifier can use alongside the individual PGS components.

---

## 4. Motif and region definitions

PGS currently uses simple, explicit regions:

- **Basal region**  
  The first `min(8, L)` nucleotides of the sequence (`seq[:min(8, L)]`).  
  `pgs_basal_ug_count` is the number of `UG` dinucleotides in this window.

- **Loop-proximal region**  
  For a dominant loop covering `[loop_start, loop_end)`, the loop-proximal region is defined as:
  loop_prox_start = max(loop_start - 10, 0)
  loop_prox_end = min(loop_end + 10, L)
  loop_prox_seq = seq[loop_prox_start:loop_prox_end]
 
`pgs_cnnc_loose_count` is computed over `loop_prox_seq`, using a simple pattern where any `C..C` fragment is treated as CNNC-like. This is intentionally permissive and is meant as a proxy rather than an exact motif match.

- **Loop motifs**  
`pgs_ugug_count` and `pgs_ugu_count` are derived directly from the dominant loop sequence `loop_seq` using standard substring counting.

These definitions are intentionally simple and fast; they are not meant to capture all variants of experimentally reported regulatory motifs but to provide robust, geometry-aware proxies.

---

## 5. Relationship to other feature groups

PGS is only one component of the extended feature set used in miRPV-NG:

- Core features (`core36`) include base composition, simple structure descriptors, and energy-related measures.
- PGS adds a focused set of geometry-heavy descriptors derived from the dominant loop and stem configuration.
- Additional extended features can include entropy, dinucleotide/tri-nucleotide patterns, and other miRNAFold-like descriptors.

In training, the classifier sees:

- Core features (always present).
- PGS features (geometry block).
- Optional extended features (energy, entropy, composition) depending on the selected feature configuration.

PGS is designed so that it can, if required, be ablated or analyzed separately in feature importance studies, making it possible to quantify how much purely geometric information contributes to precursor discrimination.

---

## 6. Limitations and future refinements

The current PGS implementation is deliberately simple:

- It assumes a single dominant loop and does not explicitly model complex multi-loop architectures.
- Loop detection is based solely on dot–bracket strings with contiguous `'.'` regions flanked by paired bases.
- Motif detection uses straightforward substring counting and a loose CNNC proxy.

Possible future refinements (if needed for later versions) include:

- More nuanced selection of the dominant loop (e.g. weighting by position, asymmetry, or conservation).
- Explicit modeling of multiple internal loops and bulges rather than a single loop.
- Improved CNNC-like motif detection that respects actual nucleotide identity at the “N” positions.

For **PGS v1**, however, the emphasis is robustness and speed over maximal biophysical detail.

---


