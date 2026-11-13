"""Process pools for self-play and benchmarks."""
from __future__ import annotations

import multiprocessing as mp
import os


def worker_pool(workers: int):
    """A pool whose workers each use a single maths thread. Without this every worker starts
    one BLAS thread per core as soon as the matrices get large enough, and the workers then
    fight each other for the processor (games became hundreds of times slower)."""
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"            # inherited by the workers; this process is already running
    return mp.Pool(workers)
