"""Abstract base class for experiment storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseStore(ABC):
    """Pluggable storage backend for experiment data."""

    @abstractmethod
    def create_run(self, run_id: str, config) -> str:
        """Initialize a new training run. Returns run_id."""
        ...

    @abstractmethod
    def get_run_meta(self, run_id: str) -> dict:
        ...

    @abstractmethod
    def list_runs(self, **filters) -> list[dict]:
        ...

    # Trajectories
    @abstractmethod
    def save_trajectory(self, run_id: str, iteration: int, episode_id: int, data: dict,
                        split: str = "train"):
        ...

    @abstractmethod
    def load_trajectories(self, run_id: str, iteration: int, limit: int = None,
                          split: str = "train") -> list[dict]:
        ...

    @abstractmethod
    def count_trajectories(self, run_id: str, iteration: int, split: str = "train") -> int:
        ...

    # Checkpoints
    @abstractmethod
    def save_checkpoint(self, run_id: str, iteration: int, policies: dict[str, str], meta: dict):
        ...

    @abstractmethod
    def load_checkpoint(self, run_id: str, iteration: int) -> tuple[dict[str, str], dict]:
        ...

    @abstractmethod
    def latest_checkpoint(self, run_id: str) -> int | None:
        ...

    # Evaluations
    @abstractmethod
    def save_evaluation(self, run_id: str, iteration: int, episode_id: int, eval_data: dict):
        ...

    @abstractmethod
    def load_evaluations(self, run_id: str, iteration: int) -> list[dict]:
        ...

    # Gradients
    @abstractmethod
    def save_gradients(self, run_id: str, iteration: int, agent_id: str,
                       per_episode: list[str], aggregated: str):
        ...

    # Metrics
    @abstractmethod
    def append_metrics(self, run_id: str, entry: dict):
        ...

    @abstractmethod
    def load_metrics(self, run_id: str) -> list[dict]:
        ...

    def get_log_path(self, run_id: str) -> str:
        """Return the path for the training log file."""
        ...
