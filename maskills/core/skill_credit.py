"""Skill-conditioned credit assignment (MASkills §4.2).

Agent-level feedback alone cannot tell *which* invoked skill helped or hurt.
This module evaluates a trajectory at the granularity of individual skill
invocations: for each ``(agent, skill)`` pair recorded in the trajectory's
``skill_trace`` (ξ), a centralized LLM critic produces a structured credit
``C^text(τ, k)`` describing — relative to the counterfactual where that skill
had *not* been used — whether the skill helped coordination, was redundant,
caused a failure, or should be refined.

It additionally emits an *agent-level residual* credit ``C^text(τ)`` capturing
effects not attributable to any single skill; the residual drives the
induction operator (a failure no existing skill explains => a new skill may
be needed).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List

from .critic import CentralizedCritic
from .trajectory import TrajectoryFormatter

# Allowed ``contribution`` verdicts for a single skill invocation.
CONTRIBUTION_VALUES = ("helped", "redundant", "harmful", "neutral")


@dataclass
class SkillCredit:
    """Structured per-skill credit ``C^text(τ, k)`` for one trajectory."""

    agent: str
    skill_id: str
    contribution: str = "neutral"          # one of CONTRIBUTION_VALUES
    evidence: str = ""                      # cited support from the trajectory
    suggested_edit: str = ""                # how the skill should change
    redundant_with: List[str] = field(default_factory=list)
    conflict_with: List[str] = field(default_factory=list)
    utility_delta: float = 0.0              # numeric signal in [-1, 1]

    def to_dict(self) -> Dict:
        return {
            "agent": self.agent,
            "skill_id": self.skill_id,
            "contribution": self.contribution,
            "evidence": self.evidence,
            "suggested_edit": self.suggested_edit,
            "redundant_with": list(self.redundant_with),
            "conflict_with": list(self.conflict_with),
            "utility_delta": self.utility_delta,
        }


@dataclass
class ResidualCredit:
    """Agent-level residual ``C^text(τ)`` — effects not tied to any skill."""

    agent: str
    summary: str = ""
    needs_new_skill: bool = False
    missing_capability: str = ""

    def to_dict(self) -> Dict:
        return {
            "agent": self.agent,
            "summary": self.summary,
            "needs_new_skill": self.needs_new_skill,
            "missing_capability": self.missing_capability,
        }


@dataclass
class TrajectorySkillCredit:
    """All skill-level credit extracted from a single trajectory."""

    skill_credits: List[SkillCredit] = field(default_factory=list)
    residuals: Dict[str, ResidualCredit] = field(default_factory=dict)
    reward: float = 0.0
    raw_response: str = ""


_SKILL_CREDIT_PROMPT = """\
You are evaluating a multi-agent trajectory τ produced by a team of LLM agents.
Each agent owns a library of reusable skills; the skills it actually invoked in
this trajectory are listed below.

## Task
For EACH (agent, skill) pair listed under "Invoked skills", hold the rest of
the team's behavior fixed and judge how invoking that skill influenced the
team's final outcome — a counterfactual contrast against NOT having used the
skill. Be specific and cite at least one concrete observation from the
trajectory. Then, per agent, produce a residual assessment of effects that are
NOT attributable to any single invoked skill (used to decide whether the agent
is missing a skill entirely).

## Trajectory
{trajectory}

## Final outcome
team score = {reward}  ({outcome})
This score is the token-level F1 of the final answer against the gold answer
(0-1).  F1 is precision-sensitive: an answer padded with restated questions,
explanations, citations, or hedging scores LOW even when the correct answer
is buried inside it.  A faithful, minimal answer scores high.
{answer_brevity_hint}
## Invoked skills (ξ)
{invoked_skills}

## Agents
The agent identifiers in this team are: {agent_ids}
Use these EXACT strings (verbatim, case-sensitive) as the "agent" field
values and as the keys of the "residual" object.

## Output
Output ONLY a JSON object, fenced in ```json, with this schema:
```json
{{
  "skills": [
    {{"agent": "<agent>", "skill_id": "<id>",
      "contribution": "helped|redundant|harmful|neutral",
      "evidence": "<concrete evidence from the trajectory>",
      "suggested_edit": "<how this skill should be changed, or empty>",
      "redundant_with": ["<other skill_id>"],
      "conflict_with": ["<other skill_id>"],
      "utility_delta": <float between -1.0 and 1.0>}}
  ],
  "residual": {{
    "<agent>": {{"summary": "<effects not tied to any skill>",
                 "needs_new_skill": true|false,
                 "missing_capability": "<what capability is missing, or empty>"}}
  }}
}}
```"""


# Injected into the credit prompt for QA tasks: HotpotQA-style gold answers
# are tiny, so token-F1 rewards brevity — the optimizer must push the final
# answerer toward the shortest faithful answer.
_QA_BREVITY_HINT = """
## Answer length (critical for this task)
The gold answer here is almost always extremely short — a single entity, a
year, or a one-to-two-word phrase.  Because the score is token-level F1, the
agent that emits the FINAL graded answer must output ONLY that minimal answer
string: no restated question, no explanation, no citations, no hedging, no
surrounding sentences.  Every extra word lowers F1.  When you assess that
agent's skills and propose edits, explicitly push it toward emitting the
shortest faithful answer.
"""


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM response (fenced or bare).

    ``strict=False`` tolerates the literal newlines/tabs LLMs routinely emit
    inside string values (a default ``json.loads`` rejects those).
    """
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    payload = match.group(1) if match else None
    if payload is None:
        # bare object: take from first '{' to last '}'
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in response")
        payload = text[start : end + 1]
    return json.loads(payload, strict=False)


def _as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def _as_float(value, default: float = 0.0) -> float:
    try:
        return max(-1.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _canonical_agent(label: str, valid_agents) -> str:
    """Map an LLM-produced agent label onto a canonical agent id.

    Critics routinely echo the trajectory's display form ("Agent 1") instead
    of the policy key ("agent_1"); without this, every credit/residual for
    that agent is silently dropped.  Matches exact, then space/dash-normalized,
    then by trailing integer index.
    """
    label = str(label).strip()
    if label in valid_agents:
        return label
    norm = label.lower().replace(" ", "_").replace("-", "_")
    if norm in valid_agents:
        return norm
    m = re.search(r"(\d+)\s*$", label)
    if m:
        for a in valid_agents:
            am = re.search(r"(\d+)\s*$", a)
            if am and am.group(1) == m.group(1):
                return a
    return ""


class SkillCreditCritic(CentralizedCritic):
    """Centralized critic that assigns credit per invoked skill + residual.

    Reuses :class:`CentralizedCritic`'s LLM client, prompt loading and
    trajectory formatting; adds :meth:`evaluate_skill_credit` on top.
    """

    def __init__(self, config, prompts_dir=None, tool_library: str = ""):
        super().__init__(config, prompts_dir=prompts_dir, tool_library=tool_library)
        self.logger = logging.getLogger(__name__)
        # Inject the short-answer hint only when the config opts in — correct
        # for HotpotQA, wrong for tasks with varied answer lengths (LOCOMO).
        self.answer_brevity_hint = bool(getattr(config, "answer_brevity_hint", False))

    # ── public API ───────────────────────────────────────────────────────
    def evaluate_skill_credit(self, trajectory, policies: dict) -> TrajectorySkillCredit:
        """Produce skill-level + residual credit for one trajectory."""
        invoked = self._summarize_invoked_skills(trajectory, policies)
        episode = self._trajectory_to_episode(trajectory)
        traj_str = TrajectoryFormatter.format_for_credit_assignment(episode)
        outcome = "SUCCESS" if trajectory.reward > 0.5 else "FAILURE"
        prompt = self._tool_library_block() + _SKILL_CREDIT_PROMPT.format(
            trajectory=traj_str,
            reward=round(float(trajectory.reward), 4),
            outcome=outcome,
            answer_brevity_hint=(_QA_BREVITY_HINT if self.answer_brevity_hint else ""),
            invoked_skills=invoked or "(no skills detected as invoked)",
            agent_ids=", ".join(policies.keys()),
        )
        try:
            raw = self._call_llm(prompt, max_tokens=1800)
            return self._parse(raw, policies, trajectory.reward)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("skill-credit evaluation failed: %s", exc)
            return TrajectorySkillCredit(reward=float(trajectory.reward), raw_response=str(exc))

    # ── helpers ──────────────────────────────────────────────────────────
    def _summarize_invoked_skills(self, trajectory, policies: dict) -> str:
        """Render ξ — the distinct skills each agent invoked, with bodies."""
        trace = getattr(trajectory, "skill_trace", {}) or {}
        blocks: List[str] = []
        for agent, entries in trace.items():
            policy = policies.get(agent)
            library = getattr(policy, "skill_library", None)
            if library is None:
                continue
            steps_by_skill: Dict[str, List[int]] = {}
            for entry in entries:
                for sid in entry.get("skills", []):
                    steps_by_skill.setdefault(sid, []).append(entry.get("step"))
            if not steps_by_skill:
                continue
            blocks.append(f"Agent {agent}:")
            for sid, steps in steps_by_skill.items():
                skill = library.get(sid)
                if skill is None:
                    continue
                body = skill.body.strip()
                if len(body) > 800:
                    body = body[:800] + " …"
                blocks.append(
                    f"  - skill `{skill.name}` (id: {sid}), invoked at steps {steps}\n"
                    f"    description: {skill.description}\n"
                    f"    body: {body}"
                )
        return "\n".join(blocks)

    def _parse(self, raw: str, policies: dict, reward: float) -> TrajectorySkillCredit:
        data = _extract_json(raw)
        valid_agents = set(policies.keys())
        credits: List[SkillCredit] = []
        for item in data.get("skills", []) or []:
            if not isinstance(item, dict):
                continue
            agent = _canonical_agent(item.get("agent", ""), valid_agents)
            skill_id = str(item.get("skill_id", "")).strip()
            if not agent or not skill_id:
                continue
            contribution = str(item.get("contribution", "neutral")).strip().lower()
            if contribution not in CONTRIBUTION_VALUES:
                contribution = "neutral"
            credits.append(
                SkillCredit(
                    agent=agent,
                    skill_id=skill_id,
                    contribution=contribution,
                    evidence=str(item.get("evidence", "")).strip(),
                    suggested_edit=str(item.get("suggested_edit", "")).strip(),
                    redundant_with=_as_list(item.get("redundant_with")),
                    conflict_with=_as_list(item.get("conflict_with")),
                    utility_delta=_as_float(item.get("utility_delta")),
                )
            )
        residuals: Dict[str, ResidualCredit] = {}
        residual_block = data.get("residual", {}) or {}
        if isinstance(residual_block, dict):
            for raw_agent, item in residual_block.items():
                agent = _canonical_agent(raw_agent, valid_agents)
                if not agent or not isinstance(item, dict):
                    continue
                residuals[agent] = ResidualCredit(
                    agent=agent,
                    summary=str(item.get("summary", "")).strip(),
                    needs_new_skill=bool(item.get("needs_new_skill", False)),
                    missing_capability=str(item.get("missing_capability", "")).strip(),
                )
        return TrajectorySkillCredit(
            skill_credits=credits,
            residuals=residuals,
            reward=float(reward),
            raw_response=raw,
        )
