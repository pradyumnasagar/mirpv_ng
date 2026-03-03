"""
Unified Concurrency Layer for miRPV-NG.

Provides consistent parallelization across all heavy computational steps:
- Batch RNAfold execution (biggest performance win)
- Parallel feature computation
- Worker pool management

Environment variables:
    MIRPV_JOBS      - Default number of workers
    MIRPV_BACKEND   - Default backend (process|thread)
    MIRPV_TMPDIR    - Default temporary directory
    MIRPV_CHUNKSIZE - Default chunk size for batching

Usage:
    from mirpv_ng.parallel import (
        ParallelConfig, get_executor, run_rnafold_batch,
        add_parallel_args, parse_parallel_args
    )

Author: miRPV-NG Team
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import subprocess
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Any, Iterator, TypeVar
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future, as_completed
import random

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


# =============================================================================
# CONFIGURATION
# =============================================================================

def get_default_threads(user_requested: Optional[int] = None) -> int:
    """
    Determine default thread count with strict precedence:
    1. user_requested (if valid > 0)
    2. SLURM_CPUS_PER_TASK
    3. PBS_NP
    4. NSLOTS
    5. OMP_NUM_THREADS
    6. os.cpu_count()
    7. Fallback: 1
    
    Returns:
        int: Valid thread count >= 1
    """
    def _parse_int(val: Any) -> Optional[int]:
        try:
            i = int(float(val))
            return i if i > 0 else None
        except (ValueError, TypeError):
            return None

    # 1. User requested
    if user_requested is not None:
        val = _parse_int(user_requested)
        if val: return val

    # 2. SLURM
    val = _parse_int(os.environ.get("SLURM_CPUS_PER_TASK"))
    if val: return val

    # 3. PBS
    val = _parse_int(os.environ.get("PBS_NP"))
    if val: return val

    # 4. SGE/GE
    val = _parse_int(os.environ.get("NSLOTS"))
    if val: return val

    # 5. OMP
    val = _parse_int(os.environ.get("OMP_NUM_THREADS"))
    if val: return val

    # 6. OS CPU count
    val = os.cpu_count()
    if val and val > 0:
        return val

    return 1

@dataclass
class ParallelConfig:
    """Configuration for parallel execution."""
    jobs: int = 1
    backend: str = "process"  # process | thread | external
    chunksize: int = 50
    tmpdir: Optional[str] = None
    seed: Optional[int] = None
    stable_order: bool = True
    
    # Optional existing executor to reuse (prevents nested pools)
    executor: Optional[Any] = None
    
    def __post_init__(self):
        # Validate
        if self.jobs < 1:
            self.jobs = 1
        if self.backend not in ("process", "thread", "external"):
            self.backend = "process"
            
        if self.tmpdir is None:
            self.tmpdir = os.environ.get("MIRPV_TMPDIR", tempfile.gettempdir())
            
        if self.chunksize == 50:
            env_chunk = os.environ.get("MIRPV_CHUNKSIZE")
            if env_chunk:
                try:
                    self.chunksize = int(env_chunk)
                except ValueError:
                    pass

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ParallelConfig":
        """Create config from parsed arguments."""
        # Resolve threads using centralized logic
        requested = getattr(args, "threads", getattr(args, "jobs", None))
        jobs = get_default_threads(requested)
        
        return cls(
            jobs=jobs,
            backend=getattr(args, "backend", "process"),
            chunksize=getattr(args, "chunksize", 50),
            tmpdir=getattr(args, "tmpdir", None),
            seed=getattr(args, "seed", None),
            stable_order=getattr(args, "stable_order", True),
        )


def add_parallel_args(parser: argparse.ArgumentParser) -> None:
    """Add standard parallel execution arguments to an argument parser."""
    group = parser.add_argument_group("Parallelism")
    group.add_argument(
        "--threads", "-j", "--jobs",
        dest="threads",
        type=str, default=None,
        help="Number of workers (int or 'auto'; default: auto-detect SLURM/PBS/OS)"
    )
    group.add_argument(
        "--backend",
        choices=["process", "thread", "external"],
        default="process",
        help="Parallelization backend (default: process)"
    )
    group.add_argument(
        "--chunksize",
        type=int, default=50,
        help="Batch size for parallel processing (default: 50)"
    )
    group.add_argument(
        "--tmpdir",
        type=str, default=None,
        help="Temporary directory for intermediate files"
    )
    group.add_argument(
        "--stable-order",
        action="store_true", default=True,
        help="Preserve output ordering (default: true)"
    )
    group.add_argument(
        "--no-stable-order",
        action="store_false", dest="stable_order",
        help="Allow out-of-order results for speed"
    )


# =============================================================================
# EXECUTOR MANAGEMENT
# =============================================================================

class ExecutorContext:
    """
    Context manager for worker pool.
    
    Handles nested execution by reusing existing executor if provided.
    """
    
    def __init__(self, cfg: ParallelConfig):
        self.cfg = cfg
        self.executor = cfg.executor  # Could be None
        self._owns_executor = False   # True if we created it
    
    def __enter__(self):
        if self.executor is None and self.cfg.jobs > 1:
            # Create new executor
            self._owns_executor = True
            if self.cfg.backend == "thread":
                self.executor = ThreadPoolExecutor(max_workers=self.cfg.jobs)
            else:
                self.executor = ProcessPoolExecutor(max_workers=self.cfg.jobs)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._owns_executor and self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        return False
    
    def map(
        self,
        fn: Callable[[T], R],
        items: List[T],
        chunksize: Optional[int] = None,
    ) -> List[R]:
        """
        Map function over items with optional parallelization.
        """
        if not items:
            return []
        
        cs = chunksize or self.cfg.chunksize
        
        if self.cfg.jobs == 1 or self.executor is None:
            # Sequential execution
            return [fn(item) for item in items]
        
        if self.cfg.stable_order:
            # Preserve order using executor.map()
            results = list(self.executor.map(fn, items, chunksize=cs))
        else:
            # Unordered for speed (use as_completed)
            futures = {self.executor.submit(fn, item): i for i, item in enumerate(items)}
            results = [None] * len(items)
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        
        return results
    
    def map_batches(
        self,
        fn: Callable[[List[T]], List[R]],
        items: List[T],
        batch_size: Optional[int] = None,
    ) -> List[R]:
        """
        Map function over batches of items.
        
        The function receives a batch and returns a list of results.
        Results are flattened and returned in order.
        """
        if not items:
            return []
        
        bs = batch_size or self.cfg.chunksize
        batches = [items[i:i+bs] for i in range(0, len(items), bs)]
        
        if self.cfg.jobs == 1 or self.executor is None:
            results = []
            for batch in batches:
                results.extend(fn(batch))
            return results
        
        # Parallel over batches
        if self.cfg.stable_order:
            batch_results = list(self.executor.map(fn, batches))
        else:
            futures = {self.executor.submit(fn, batch): i for i, batch in enumerate(batches)}
            batch_results = [None] * len(batches)
            for future in as_completed(futures):
                idx = futures[future]
                batch_results[idx] = future.result()
        
        # Flatten
        results = []
        for br in batch_results:
            results.extend(br)
        return results


def get_executor(cfg: ParallelConfig) -> ExecutorContext:
    """Get an executor context manager."""
    return ExecutorContext(cfg)


# =============================================================================
# BATCH RNAFOLD (BIGGEST WIN)
# =============================================================================

def run_rnafold_batch(
    sequences: List[str],
    tmpdir: Optional[str] = None,
    rnafold_bin: str = "RNAfold",
    timeout: int = 300,
    executor: Optional[Any] = None,
) -> List[Tuple[str, float]]:
    """
    Run RNAfold on a batch of sequences using a SINGLE process.
    
    This is the core worker function. It does NOT spawn its own pool.
    It runs one RNAfold process for the whole batch.
    
    Args:
        sequences: List of RNA sequences (uppercase, U alphabet)
        tmpdir: Temporary directory for files
        rnafold_bin: Path to RNAfold binary
        timeout: Timeout in seconds
        executor: IGNORED here (kept for signature compatibility), as this IS the worker task.
    
    Returns:
        List of (structure, mfe) tuples, one per input sequence
    """
    if not sequences:
        return []
    
    # Prepare input
    # Ensure T -> U conversion just in case
    input_text = "\n".join(seq.upper().replace("T", "U") for seq in sequences)
    
    try:
        proc = subprocess.Popen(
            [rnafold_bin, "--noPS"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=tmpdir,
        )
        stdout, stderr = proc.communicate(input_text + "\n", timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        logger.warning(
            "run_rnafold_batch: timed out after %ds for %d sequences; "
            "returning empty structures — downstream features will be zeroed (false-negative risk)",
            timeout, len(sequences),
        )
        return [("", 0.0)] * len(sequences)
    except FileNotFoundError:
        raise RuntimeError(
            f"RNAfold binary not found: {rnafold_bin!r}. "
            "Install ViennaRNA and ensure it is on PATH or pass --rnafold-bin."
        ) from None
    except Exception as exc:
        logger.error(
            "run_rnafold_batch: unexpected error for %d sequences: %s; "
            "returning empty structures — downstream features will be zeroed (false-negative risk)",
            len(sequences), exc,
        )
        return [("", 0.0)] * len(sequences)
    
    # Parse output
    results = []
    lines = stdout.strip().split("\n")
    
    i = 0
    while i < len(lines) and len(results) < len(sequences):
        # Each sequence produces 2 lines: sequence and structure+mfe
        if i + 1 >= len(lines):
            results.append(("", 0.0))
            i += 1
            continue
        
        struct_line = lines[i + 1]
        parts = struct_line.split()
        if parts:
            struct = parts[0]
            try:
                # Extract MFE from parentheses
                mfe_str = struct_line.split("(")[-1].strip(" )")
                mfe = float(mfe_str)
            except (ValueError, IndexError):
                mfe = 0.0
            results.append((struct, mfe))
        else:
            results.append(("", 0.0))
        
        i += 2
    
    # Pad if we didn't get enough results
    while len(results) < len(sequences):
        results.append(("", 0.0))
    
    return results


def run_rnafold_single(seq: str, rnafold_bin: str = "RNAfold") -> Tuple[str, float]:
    """
    Run RNAfold on a single sequence.
    """
    results = run_rnafold_batch([seq], rnafold_bin=rnafold_bin)
    return results[0] if results else ("", 0.0)


def run_rnafold_parallel(
    sequences: List[str],
    cfg: ParallelConfig,
    rnafold_bin: str = "RNAfold",
) -> List[Tuple[str, float]]:
    """
    Run RNAfold on many sequences with parallelism.
    
    Distributes batches to workers.
    Each worker runs a single RNAfold process for its batch.
    
    Uses cfg.executor if present to avoid nested pools.
    """
    if not sequences:
        return []
    
    # If jobs=1 or executor is unavailable and we can't make one (shouldn't happen with get_executor logic)
    # Just run direct batch
    if cfg.jobs == 1 and not cfg.executor:
        # Serial processing in chunks to keep memory/IO reasonable
        results = []
        for i in range(0, len(sequences), cfg.chunksize):
            batch = sequences[i:i + cfg.chunksize]
            results.extend(run_rnafold_batch(batch, cfg.tmpdir, rnafold_bin))
        return results
    
    # Parallel execution
    # We use ExecutorContext which handles whether to use existing or new executor
    with get_executor(cfg) as executor:
        # Lambda to capture arguments
        def _worker(batch):
            return run_rnafold_batch(batch, cfg.tmpdir, rnafold_bin)
            
        results = executor.map_batches(
            _worker,
            sequences,
            batch_size=cfg.chunksize,
        )
    
    return results


# =============================================================================
# PARALLEL ITERATORS FOR STREAMING
# =============================================================================

def chunked_iterator(iterable: Iterator[T], chunk_size: int) -> Iterator[List[T]]:
    """
    Yield chunks of items from an iterator.
    
    Useful for processing large streams in batches.
    """
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def parallel_map_stream(
    cfg: ParallelConfig,
    producer: Iterator[T],
    processor: Callable[[List[T]], List[R]],
    max_items: Optional[int] = None,
) -> Iterator[R]:
    """
    Process a stream with parallelism over batches.
    
    Args:
        cfg: Parallel configuration
        producer: Iterator yielding items
        processor: Function that processes a batch and returns results
        max_items: Optional maximum items to process
    
    Yields:
        Results from processor, preserving order if stable_order is True
    """
    items_yielded = 0
    
    for chunk in chunked_iterator(producer, cfg.chunksize):
        if max_items and items_yielded >= max_items:
            break
        
        results = processor(chunk)
        
        for result in results:
            if max_items and items_yielded >= max_items:
                break
            yield result
            items_yielded += 1


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_random_state(cfg: ParallelConfig) -> random.Random:
    """Get a seeded random state if seed is configured."""
    if cfg.seed is not None:
        return random.Random(cfg.seed)
    return random.Random()


def print_parallel_info(cfg: ParallelConfig, prefix: str = "") -> None:
    """Print parallelism configuration info."""
    print(f"{prefix}[parallel] jobs={cfg.jobs}, backend={cfg.backend}, "
          f"chunksize={cfg.chunksize}, stable_order={cfg.stable_order}")
    if cfg.tmpdir:
        print(f"{prefix}[parallel] tmpdir={cfg.tmpdir}")
