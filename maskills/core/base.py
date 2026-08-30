"""Core abstract base classes for MASkills."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Trajectory:
    """A single episode trajectory."""

    task: dict
    steps: list[dict]  # [{agent_id, observation, action, ...}, ...]
    reward: float
    metadata: dict = field(default_factory=dict)
    # MASkills ξ: per-agent skill-invocation trace,
    # {agent_name: [{"step": idx, "skills": [skill_id, ...]}, ...]}.
    skill_trace: dict = field(default_factory=dict)


class BaseEnvironment(ABC):
    """Thin adapter over a third-party environment."""

    @abstractmethod
    def reset(self, task: dict) -> dict:
        """Reset environment, return initial observations."""
        ...

    @abstractmethod
    def step(self, agent_id: str, action: str) -> tuple[dict, float, bool, dict]:
        """Execute agent action, return (obs, reward, done, info)."""
        ...

    @abstractmethod
    def collect_trajectory(self, policies: dict[str, str], task: dict) -> Trajectory:
        """Run a full episode with given policies, return trajectory."""
        ...


class BaseAgent(ABC):
    """An agent with a language policy."""

    @abstractmethod
    def act(self, observation: str, policy: str) -> str:
        """Given observation and policy, produce an action (text)."""
        ...


class BaseCritic(ABC):
    """Evaluates trajectories and assigns credit."""

    @abstractmethod
    def evaluate(self, trajectory: Trajectory, policies: dict[str, str]) -> dict:
        """Evaluate a trajectory. Returns evaluation dict with per-agent credits."""
        ...


class BaseReward(ABC):
    """Computes reward signals from trajectories."""

    @abstractmethod
    def compute(self, trajectory: Trajectory) -> float:
        """Compute reward for a trajectory."""
        ...


class BaseOptimizer(ABC):
    """Generates and applies language gradients."""

    @abstractmethod
    def generate_gradient(self, policy: str, evaluation: str, context: str) -> str:
        """Generate a language gradient (improvement instruction)."""
        ...

    @abstractmethod
    def synthesize_policy(self, base_policy, gradients: list[str], agent_name: str = "agent"):
        """Synthesize updated policy (role + skills) from gradients via LLM."""
        ...

    @abstractmethod
    def aggregate_gradients(self, gradients: list[str]) -> str:
        """Aggregate multiple gradients into one."""
        ...
