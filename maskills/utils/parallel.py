"""Thread pool executor wrappers."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable


def parallel_map(fn: Callable, items: Iterable, max_workers: int = 5) -> list:
    """Apply fn to each item in parallel, return results in order."""
    items_list = list(items)
    results = [None] * len(items_list)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(fn, item): i for i, item in enumerate(items_list)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()

    return results
