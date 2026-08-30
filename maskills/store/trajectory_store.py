"""Trajectory read/write/query, backed by BaseStore."""

from __future__ import annotations

from ..core.base import Trajectory
from .base import BaseStore


class TrajectoryStore:
    """Read/write/query episode trajectories."""

    def __init__(self, store: BaseStore, run_id: str):
        self.store = store
        self.run_id = run_id

    def save(self, iteration: int, episode_id: int, trajectory: Trajectory, split: str = "train"):
        data = {
            "episode_id": episode_id,
            "task": trajectory.task,
            "transitions": trajectory.steps,
            "reward": trajectory.reward,
            "metadata": trajectory.metadata,
            "skill_trace": trajectory.skill_trace,
        }
        self.store.save_trajectory(self.run_id, iteration, episode_id, data, split=split)

    def load(self, iteration: int, limit: int = None, split: str = "train") -> list[Trajectory]:
        raw = self.store.load_trajectories(self.run_id, iteration, limit, split=split)
        return [
            Trajectory(
                task=r["task"],
                steps=r["transitions"],
                reward=r["reward"],
                metadata=r.get("metadata", {}),
                skill_trace=r.get("skill_trace", {}),
            )
            for r in raw
        ]

    def count(self, iteration: int, split: str = "train") -> int:
        return self.store.count_trajectories(self.run_id, iteration, split=split)

    def get_stats(self, iteration: int) -> dict:
        trajectories = self.load(iteration)
        if not trajectories:
            return {"num_episodes": 0, "avg_reward": 0, "min_reward": 0, "max_reward": 0}
        rewards = [t.reward for t in trajectories]
        lengths = [len(t.steps) for t in trajectories]
        return {
            "num_episodes": len(trajectories),
            "avg_reward": sum(rewards) / len(rewards),
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "avg_length": sum(lengths) / len(lengths),
        }
