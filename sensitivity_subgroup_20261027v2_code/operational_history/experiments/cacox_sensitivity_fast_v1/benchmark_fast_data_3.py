"""Benchmark-only wrapper: same six jobs and votes, at most three active per branch."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import fast_data


def limited_executor(*, max_workers: int):
    return ThreadPoolExecutor(max_workers=min(max_workers, 3))


fast_data.ThreadPoolExecutor = limited_executor
fast_data.main()
