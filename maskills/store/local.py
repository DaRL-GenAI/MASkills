"""Filesystem-based experiment storage."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ..core.policy import AgentPolicy
from ..core.skills import SkillLibrary
from .base import BaseStore


class LocalStore(BaseStore):
    """Filesystem-backed storage for experiments."""

    def __init__(self, root_dir: str = "./experiments"):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def create_run(self, run_id: str, config) -> str:
        run_dir = self.root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "status": "running",
        }
        self._write_json(run_dir / "run_meta.json", meta)

        # Save frozen config
        if hasattr(config, 'to_json'):
            config.to_json(str(run_dir / "config.json"))
        return run_id

    def get_run_meta(self, run_id: str) -> dict:
        meta_path = self.root / "runs" / run_id / "run_meta.json"
        if meta_path.exists():
            return self._read_json(meta_path)
        return {}

    def list_runs(self, **filters) -> list[dict]:
        runs_dir = self.root / "runs"
        if not runs_dir.exists():
            return []
        result = []
        for d in sorted(runs_dir.iterdir()):
            if d.is_dir():
                meta = self.get_run_meta(d.name)
                if meta:
                    if all(meta.get(k) == v for k, v in filters.items()):
                        result.append(meta)
        return result

    # --- Trajectories ---
    def save_trajectory(self, run_id: str, iteration: int, episode_id: int, data: dict,
                        split: str = "train"):
        path = self._traj_dir(run_id, iteration, split)
        path.mkdir(parents=True, exist_ok=True)
        self._write_json(path / f"episode_{episode_id:03d}.json", data)

    def load_trajectories(self, run_id: str, iteration: int, limit: int = None,
                          split: str = "train") -> list[dict]:
        path = self._traj_dir(run_id, iteration, split)
        if not path.exists():
            return []
        files = sorted(path.glob("episode_*.json"))
        if limit:
            files = files[:limit]
        return [self._read_json(f) for f in files]

    def count_trajectories(self, run_id: str, iteration: int, split: str = "train") -> int:
        path = self._traj_dir(run_id, iteration, split)
        if not path.exists():
            return 0
        return len(list(path.glob("episode_*.json")))

    # --- Checkpoints ---
    def save_checkpoint(self, run_id: str, iteration: int, policies: dict, meta: dict):
        ckpt_dir = self._ckpt_dir(run_id, iteration)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for agent_id, policy in policies.items():
            if isinstance(policy, AgentPolicy):
                agent_dir = ckpt_dir / agent_id
                agent_dir.mkdir(parents=True, exist_ok=True)
                (agent_dir / "role.md").write_text(policy.role)
                # ``skills.md`` is the legacy flat view (kept for human
                # reading + back-compat); ``skill_library.json`` is the
                # authoritative discrete library that round-trips exactly.
                (agent_dir / "skills.md").write_text(policy.skills)
                self._write_json(
                    agent_dir / "skill_library.json", policy.skill_library.to_dict()
                )
            else:
                # Legacy fallback: plain string
                (ckpt_dir / f"{agent_id}.txt").write_text(policy)
        self._write_json(ckpt_dir / "meta.json", meta)

    def load_checkpoint(self, run_id: str, iteration: int) -> tuple[dict, dict]:
        ckpt_dir = self._ckpt_dir(run_id, iteration)
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"No checkpoint at iteration {iteration}")
        policies = {}
        # New format: agent_N/role.md + skills.md
        for agent_dir in sorted(ckpt_dir.iterdir()):
            if agent_dir.is_dir() and (agent_dir / "role.md").exists():
                role = (agent_dir / "role.md").read_text()
                lib_path = agent_dir / "skill_library.json"
                if lib_path.exists():
                    # Authoritative discrete library.
                    library = SkillLibrary.from_dict(self._read_json(lib_path))
                    policies[agent_dir.name] = AgentPolicy(role=role, skill_library=library)
                else:
                    # Pre-MASkills checkpoint: flat skills.md only.
                    skills_path = agent_dir / "skills.md"
                    skills = skills_path.read_text() if skills_path.exists() else ""
                    policies[agent_dir.name] = AgentPolicy(role=role, skills=skills)
        # Fallback: old format (agent_N.txt)
        if not policies:
            for txt_file in sorted(ckpt_dir.glob("*.txt")):
                agent_id = txt_file.stem
                policies[agent_id] = AgentPolicy.from_legacy(txt_file.read_text())
        meta_path = ckpt_dir / "meta.json"
        meta = self._read_json(meta_path) if meta_path.exists() else {}
        return policies, meta

    def latest_checkpoint(self, run_id: str) -> int | None:
        ckpt_root = self.root / "runs" / run_id / "checkpoints"
        if not ckpt_root.exists():
            return None
        iters = []
        for d in ckpt_root.iterdir():
            if d.is_dir() and d.name.startswith("iter_"):
                try:
                    iters.append(int(d.name.split("_")[1]))
                except ValueError:
                    pass
        return max(iters) if iters else None

    # --- Evaluations ---
    def save_evaluation(self, run_id: str, iteration: int, episode_id: int, eval_data: dict):
        eval_dir = self.root / "runs" / run_id / "evaluations" / f"iter_{iteration}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(eval_dir / f"eval_episode_{episode_id:03d}.json", eval_data)

    def load_evaluations(self, run_id: str, iteration: int) -> list[dict]:
        eval_dir = self.root / "runs" / run_id / "evaluations" / f"iter_{iteration}"
        if not eval_dir.exists():
            return []
        return [self._read_json(f) for f in sorted(eval_dir.glob("eval_episode_*.json"))]

    # --- Gradients ---
    def save_gradients(self, run_id: str, iteration: int, agent_id: str,
                       per_episode: list[str], aggregated: str):
        grad_dir = self.root / "runs" / run_id / "gradients" / f"iter_{iteration}"
        grad_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(grad_dir / f"{agent_id}_gradients.json", per_episode)
        (grad_dir / f"{agent_id}_aggregated.txt").write_text(aggregated)

    # --- Metrics ---
    def append_metrics(self, run_id: str, entry: dict):
        metrics_dir = self.root / "runs" / run_id / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_dir / "training_curve.jsonl"
        with open(metrics_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def load_metrics(self, run_id: str) -> list[dict]:
        metrics_file = self.root / "runs" / run_id / "metrics" / "training_curve.jsonl"
        if not metrics_file.exists():
            return []
        entries = []
        with open(metrics_file) as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries

    def get_log_path(self, run_id: str) -> str:
        log_dir = self.root / "runs" / run_id / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return str(log_dir / "train.log")

    def get_run_dir(self, run_id: str) -> str:
        """Return the filesystem directory for a run (creates if needed)."""
        run_dir = self.root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return str(run_dir)

    # --- Helpers ---
    def _traj_dir(self, run_id: str, iteration: int, split: str = "train") -> Path:
        sub = "trajectories" if split == "train" else f"trajectories_{split}"
        return self.root / "runs" / run_id / sub / f"iter_{iteration}"

    def _ckpt_dir(self, run_id: str, iteration: int) -> Path:
        return self.root / "runs" / run_id / "checkpoints" / f"iter_{iteration}"

    @staticmethod
    def _write_json(path: Path, data):
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def _read_json(path: Path) -> dict:
        with open(path) as f:
            return json.load(f)
