"""Versioned policy checkpoint manager."""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional

from ..core.policy import AgentPolicy, default_agent_policy
from .base import BaseStore


class PolicyCheckpoint:
    """Versioned policy snapshot manager backed by BaseStore."""

    def __init__(
        self,
        store: BaseStore,
        run_id: str,
        num_agents: int,
        default_policy_fn: Callable[..., AgentPolicy] = None,
        task_type: str | None = None,
    ):
        self.store = store
        self.run_id = run_id
        self.num_agents = num_agents
        self.default_policy_fn = default_policy_fn or default_agent_policy
        self.task_type = task_type

    def get_policies(self, iteration: Optional[int] = None) -> dict[str, AgentPolicy]:
        """Load policies. If iteration=None, load latest. If none exists, generate defaults."""
        if iteration is None:
            iteration = self.store.latest_checkpoint(self.run_id)
        if iteration is None:
            return self._generate_defaults()
        policies, _ = self.store.load_checkpoint(self.run_id, iteration)
        # Ensure all values are AgentPolicy
        result = {}
        for k, v in policies.items():
            if isinstance(v, AgentPolicy):
                result[k] = v
            else:
                result[k] = AgentPolicy.from_legacy(v)
        return result

    def save_policies(self, iteration: int, policies: dict[str, AgentPolicy], stats: dict = None):
        """Save policies for a new iteration."""
        meta = {
            "timestamp": datetime.now().isoformat(),
            "num_agents": len(policies),
            "training_stats": stats,
        }
        self.store.save_checkpoint(self.run_id, iteration, policies, meta)

    def get_training_curve(self) -> list[dict]:
        return self.store.load_metrics(self.run_id)

    def diff_policies(self, iter_a: int, iter_b: int) -> dict[str, tuple[str, str]]:
        """Compare policies between two iterations."""
        policies_a, _ = self.store.load_checkpoint(self.run_id, iter_a)
        policies_b, _ = self.store.load_checkpoint(self.run_id, iter_b)
        return {
            agent: (policies_a.get(agent, ""), policies_b.get(agent, ""))
            for agent in set(policies_a) | set(policies_b)
        }

    def _generate_defaults(self) -> dict[str, AgentPolicy]:
        # Pass ``task_type`` so language-task agents (e.g. QA) get the
        # task-specific answer-format instructions in their initial role.
        return {
            f"agent_{i + 1}": self.default_policy_fn(
                i, self.num_agents, task_type=self.task_type
            )
            for i in range(self.num_agents)
        }
