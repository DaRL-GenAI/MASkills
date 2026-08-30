"""GAIA task loader: read a JSONL of GAIA items into ``test_tasks``.

GAIA is eval-only here; we don't split into train/val/test. ``test_tasks``
exposes every item from the input JSONL deterministically (file order).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional


class GaiaTaskLoader:
    def __init__(
        self,
        benchmark_path: str,
        data_limit: Optional[int] = None,
    ):
        path = Path(benchmark_path)
        if not path.exists():
            raise FileNotFoundError(f"GAIA benchmark not found: {path}")
        items: List[dict] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
        if data_limit:
            items = items[:data_limit]
        for item in items:
            # Normalise keys that the eval harness reads.
            item.setdefault("question_id", item.get("task_id"))
            item.setdefault("question", item.get("Question", ""))
            item.setdefault("ground_truth", item.get("Final answer", ""))
            item.setdefault("category", item.get("Level"))
        self._items = items
        self.test_tasks = list(items)
        # No real train/val split; expose empty lists so callers don't crash.
        self.train_tasks: List[dict] = []
        self.val_tasks: List[dict] = []

    def sample_tasks(self, num_samples: int, seed=None, split: str = "test"):
        pool = {
            "train": self.train_tasks, "val": self.val_tasks,
            "test": self.test_tasks,
        }.get(split, self.test_tasks)
        return list(pool[: num_samples or len(pool)])
