"""Agent policy with separate role and skills components.

Each agent's policy is split into two parts:
  - **role** (agents.md): Static role description — team position, responsibilities,
    collaboration protocol.  Does NOT change during training.
  - **skill_library** (``K_i``): Trainable task-specific execution knowledge —
    a :class:`~maskills.core.skills.SkillLibrary` of discrete skills, evolved
    via the MASkills operators (refinement / induction / consolidation / pruning).

The combined system prompt sent to the LLM is ``role + skills``.

For backward compatibility the historical single-string ``skills`` attribute
is preserved as a read/write *property* over the library: reading collapses
the library to one markdown blob, writing replaces the library with a single
skill.  Pre-MASkills checkpoints therefore load and run unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .skills import SkillLibrary


class AgentPolicy:
    """Two-part agent policy: static ``role`` + trainable ``skill_library``."""

    def __init__(
        self,
        role: str,
        skills: str = "",
        skill_library: Optional[SkillLibrary] = None,
    ):
        """Construct a policy.

        Args:
            role: static role description.
            skills: legacy single-blob skills string.  Used only when
                ``skill_library`` is not given — wrapped into a one-skill
                library via :meth:`SkillLibrary.from_legacy_body`.
            skill_library: explicit discrete skill library (preferred).
        """
        self.role = role
        if skill_library is not None:
            self.skill_library = skill_library
        else:
            self.skill_library = SkillLibrary.from_legacy_body(skills)

    # ── legacy single-string ``skills`` bridge ───────────────────────────
    @property
    def skills(self) -> str:
        """Library collapsed to one markdown blob (legacy view)."""
        return self.skill_library.combined_body()

    @skills.setter
    def skills(self, value: str) -> None:
        """Replace the library with a single skill built from ``value``."""
        self.skill_library = SkillLibrary.from_legacy_body(value or "")

    @property
    def combined(self) -> str:
        """Merge role and skills into a single system prompt."""
        skills = self.skills
        if not skills.strip():
            return self.role
        return self.role.rstrip() + "\n\n## Skills\n" + skills

    @classmethod
    def from_legacy(cls, policy_text: str) -> "AgentPolicy":
        """Wrap a legacy single-string policy as role-only (empty skills)."""
        return cls(role=policy_text, skills="")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize.  Carries both the discrete library and the legacy
        flat ``skills`` string so older readers stay compatible."""
        return {
            "role": self.role,
            "skills": self.skills,
            "skill_library": self.skill_library.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentPolicy":
        if "role" in data:
            if isinstance(data.get("skill_library"), dict):
                return cls(
                    role=data["role"],
                    skill_library=SkillLibrary.from_dict(data["skill_library"]),
                )
            return cls(role=data["role"], skills=data.get("skills", ""))
        # Legacy format: {"policy": "..."}
        if "policy" in data:
            return cls(role=data["policy"], skills="")
        raise ValueError(f"Cannot parse AgentPolicy from dict: {data}")

    def copy(self) -> "AgentPolicy":
        return AgentPolicy(role=self.role, skill_library=self.skill_library.copy())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AgentPolicy):
            return NotImplemented
        return (
            self.role == other.role
            and self.skill_library.to_dict() == other.skill_library.to_dict()
        )

    def __repr__(self) -> str:
        return (
            f"AgentPolicy(role={self.role[:40]!r}..., "
            f"skills={len(self.skill_library)} skill(s))"
        )

    def __str__(self) -> str:
        return self.combined


_QA_LAST_ANSWER_INSTRUCTION = (
    "- For HotPotQA, your FINAL answer must be the shortest faithful "
    "form of the answer (an entity name, a year, or a short phrase).  "
    "Output ONLY that answer string — do NOT include reasoning, explanations, "
    "restatements of the question, citations, or surrounding sentences."
)


def generate_default_agent_prompt(
    agent_idx: int,
    num_agents: int,
    task_type: str | None = None,
) -> str:
    """Generate a default role prompt for agent at position agent_idx (0-based).

    Describes the shared message pool topology: each agent sees the task and
    all previous responses.  When ``task_type='qa'`` and the agent is the
    last responder, the role explicitly demands a concise answer-only output
    (no explanations) — HotPotQA's judge marks paraphrases CORRECT but
    explanation-heavy responses can mask the actual answer string.
    """
    position = agent_idx + 1

    if num_agents == 1:
        lines = [
            "You are the sole agent in a collaborative task system.",
            "- You receive the task and provide the final answer",
        ]
        if task_type == "qa":
            lines.append(_QA_LAST_ANSWER_INSTRUCTION)
        lines.append("Please provide your response to the task.")
        return "\n".join(lines)

    lines = [f"You are participating in a collaborative task with {num_agents} agents."]
    is_first = (agent_idx == 0)
    is_last = (agent_idx == num_agents - 1)

    if is_first:
        later = ", ".join([f"Agent {j + 1}" for j in range(1, num_agents)])
        lines.append(f"- You are Agent {position}, speaking first")
        lines.append(f"- After you respond, {later} will see the task and your response")
        lines.append(f"- Agent {num_agents}'s output will be the final answer")
    elif is_last:
        prev = ", ".join([f"Agent {j + 1}" for j in range(agent_idx)])
        lines.append(f"- You are Agent {position}, speaking last")
        lines.append(f"- You can see the original task and the responses from {prev}")
        lines.append("- YOUR output is the FINAL answer that will be evaluated")
        if task_type == "qa":
            lines.append(_QA_LAST_ANSWER_INSTRUCTION)
    else:
        prev = ", ".join([f"Agent {j + 1}" for j in range(agent_idx)])
        later = ", ".join([f"Agent {j + 1}" for j in range(position, num_agents)])
        lines.append(f"- You are Agent {position}")
        lines.append(f"- You can see the original task and the responses from {prev}")
        lines.append(f"- {later} will see your response")
        lines.append(f"- Agent {num_agents}'s output will be the final answer")

    lines.append("Please provide your response to the task.")
    return "\n".join(lines)


def default_agent_policy(
    agent_idx: int,
    num_agents: int,
    task_type: str | None = None,
) -> AgentPolicy:
    """Generate a default AgentPolicy for agent at position agent_idx (0-based)."""
    return AgentPolicy(
        role=generate_default_agent_prompt(agent_idx, num_agents, task_type=task_type),
        skills="",
    )
