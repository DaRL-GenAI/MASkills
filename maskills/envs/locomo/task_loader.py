"""LOCOMO task loader.

Loads ``locomo10.json`` (10 long multi-session conversations with QA
annotations) and explodes it into one task per (conversation, qa)
pair.  Each task carries the full conversation (so the env can build the
retriever context however it wants) and the QA item being asked.

The data file ships in ``env/locomo/data/locomo10.json``.  When
``benchmark_path`` points to that file directly the loader reads it; when
it points to the directory, the loader auto-resolves the filename.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

_DEFAULT_FILENAME = "locomo10.json"


class LocomoTaskLoader:
    """Load and split LOCOMO QA tasks.

    Mirrors the API of ``maskills/envs/language/task_loader.py``:
    ``tasks``, ``train_tasks``, ``test_tasks``, ``sample_tasks``,
    ``get_test_tasks``.
    """

    def __init__(
        self,
        benchmark_path: str,
        data_limit: Optional[int] = None,
        train_test_split: float = 1.0,
        split_seed: int = 42,
        category_filter: Optional[List[int]] = None,
    ):
        self.benchmark_path = Path(benchmark_path)
        self.data_limit = data_limit
        self.category_filter = (
            set(int(c) for c in category_filter) if category_filter else None
        )

        self.tasks: List[Dict] = []
        self._load_tasks()
        if self.data_limit is not None and self.data_limit > 0:
            self.tasks = self.tasks[: self.data_limit]

        self.train_tasks: List[Dict] = []
        self.val_tasks: List[Dict] = []
        self.test_tasks: List[Dict] = []
        self._split_tasks(train_test_split, split_seed)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _resolve_file(self) -> Path:
        if self.benchmark_path.is_file():
            return self.benchmark_path
        candidate = self.benchmark_path / _DEFAULT_FILENAME
        if candidate.exists():
            return candidate
        # Also accept benchmark_path pointing at env/locomo/ root.
        candidate = self.benchmark_path / "data" / _DEFAULT_FILENAME
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"No LOCOMO data file under {self.benchmark_path} "
            f"(looked for {_DEFAULT_FILENAME})."
        )

    def _load_tasks(self):
        path = self._resolve_file()
        with open(path, "r") as f:
            raw = json.load(f)

        for conv_entry in raw:
            sample_id = conv_entry.get("sample_id", "unknown")
            conv = conv_entry.get("conversation", {})
            session_meta = self._extract_session_meta(conv)
            speakers = self._extract_speakers(session_meta)

            for qa_idx, qa in enumerate(conv_entry.get("qa", [])):
                category = qa.get("category")
                if self.category_filter and category not in self.category_filter:
                    continue
                # cat 5 has ``adversarial_answer`` instead of ``answer``;
                # the ground-truth response is "not mentioned".
                if category == 5:
                    ground_truth = qa.get("adversarial_answer", "")
                else:
                    ground_truth = qa.get("answer", "")

                self.tasks.append({
                    "task_id": f"{sample_id}__qa_{qa_idx:04d}",
                    "type": "locomo_qa",
                    "sample_id": sample_id,
                    "question": qa.get("question", ""),
                    "ground_truth": str(ground_truth),
                    "category": category,
                    "evidence": list(qa.get("evidence", []) or []),
                    "speakers": speakers,
                    "session_meta": session_meta,
                    "metadata": {
                        "source": "locomo",
                        "conv_sample_id": sample_id,
                        "qa_index": qa_idx,
                        "category": category,
                        "is_adversarial": category == 5,
                    },
                })

        if not self.tasks:
            raise ValueError(f"No LOCOMO tasks loaded from {path}")

    # ------------------------------------------------------------------
    # Conversation helpers (kept here so env.py is light)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_session_meta(conv: Dict) -> List[Dict]:
        """Return ordered session metadata.

        Each element: ``{"name": "session_3", "index": 3, "date": "...",
        "turns": [{speaker, dia_id, text, blip_caption?}, ...]}``.
        Sessions are sorted by numeric index so old → new order is stable
        regardless of dict insertion order in the source JSON.
        """
        sessions = []
        for key, val in conv.items():
            if not key.startswith("session_"):
                continue
            if key.endswith("_date_time"):
                continue
            try:
                idx = int(key.split("_")[1])
            except (IndexError, ValueError):
                continue
            sessions.append((idx, key, val))
        sessions.sort(key=lambda t: t[0])

        meta = []
        for idx, name, turns in sessions:
            date = conv.get(f"{name}_date_time", "")
            meta.append({
                "name": name,
                "index": idx,
                "date": date,
                "turns": list(turns or []),
            })
        return meta

    @staticmethod
    def _extract_speakers(session_meta: List[Dict]) -> List[str]:
        seen = []
        for s in session_meta:
            for turn in s["turns"]:
                spk = turn.get("speaker")
                if spk and spk not in seen:
                    seen.append(spk)
                if len(seen) >= 2:
                    return seen
        return seen

    # ------------------------------------------------------------------
    # Splitting / sampling
    # ------------------------------------------------------------------

    def _split_tasks(self, train_ratio: float, seed: int):
        if train_ratio >= 1.0 or not self.tasks:
            self.train_tasks = self.tasks.copy()
            self.test_tasks = []
            return
        indices = list(range(len(self.tasks)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        n_train = max(1, int(len(self.tasks) * train_ratio))
        self.train_tasks = [self.tasks[i] for i in indices[:n_train]]
        self.test_tasks = [self.tasks[i] for i in indices[n_train:]]

    def split_by_conversation(
        self,
        n_train: int = 2,
        n_val: int = 2,
        n_test: int = 6,
        seed: int = 42,
    ) -> Dict[str, List[str]]:
        """Split tasks by long conversation (``sample_id``).

        The three splits hold *disjoint* conversations, so no long
        conversation leaks across train/val/test.  Returns the chosen
        ``sample_id`` lists per split (sorted) for logging.
        """
        sample_ids: List[str] = []
        for t in self.tasks:  # de-dup, keep first-seen order
            if t["sample_id"] not in sample_ids:
                sample_ids.append(t["sample_id"])
        random.Random(seed).shuffle(sample_ids)

        train_ids = set(sample_ids[:n_train])
        val_ids = set(sample_ids[n_train:n_train + n_val])
        test_ids = set(sample_ids[n_train + n_val:n_train + n_val + n_test])

        self.train_tasks = [t for t in self.tasks if t["sample_id"] in train_ids]
        self.val_tasks = [t for t in self.tasks if t["sample_id"] in val_ids]
        self.test_tasks = [t for t in self.tasks if t["sample_id"] in test_ids]
        return {
            "train": sorted(train_ids),
            "val": sorted(val_ids),
            "test": sorted(test_ids),
        }

    def sample_tasks(
        self,
        num_samples: int,
        seed: Optional[int] = None,
        split: str = "train",
    ) -> List[Dict]:
        pool = {
            "train": self.train_tasks,
            "val": self.val_tasks,
            "test": self.test_tasks,
        }.get(split, self.train_tasks)
        if not pool:
            pool = self.tasks
        rng = random.Random(seed) if seed is not None else random.Random()
        if num_samples >= len(pool):
            return pool.copy()
        return rng.sample(pool, num_samples)

    def get_test_tasks(self) -> List[Dict]:
        return self.test_tasks.copy()

    def __len__(self):
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)
