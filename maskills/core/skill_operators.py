"""Skill-evolution operators (MASkills §4.3–§4.4).

``SkillEvolutionOptimizer`` turns skill-level credit into changes to an
agent's discrete :class:`~maskills.core.skills.SkillLibrary`.  It implements:

* **Hierarchical aggregation** (§4.3, Eq.11) — :meth:`aggregate_skill_gradients`
  merges the per-trajectory credits ``{C^text(τ, k)}`` for each skill into one
  stable language gradient ``G(k)``.
* The four **evolution operators** (§4.4):
  - **Refinement** (Eq.13) — :meth:`refine_skill`: localized anchor-based edits
    to an existing skill body.  (The diff-style edit is itself the
    momentum-like mechanism — it preserves validated structure and only
    moves the regions the aggregated gradient implicates.)
  - **Induction** (Eq.14) — :meth:`induce_skill`: propose an entirely new
    skill from hard trajectories the current library does not cover.
  - **Consolidation** (Eq.15) — :meth:`consolidate_skills`: merge functionally
    overlapping skills into one higher-level skill.
  - **Pruning** (Eq.16) — :meth:`select_skills_to_prune`: identify low-utility
    skills for removal.

It subclasses :class:`PolicyGradientOptimizer` purely to reuse its LLM client
(``_llm_call``) and anchor-edit machinery (``_apply_edits``).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .optimizer import PolicyGradientOptimizer
from .skill_credit import SkillCredit
from .skills import Skill, SkillLibrary, parse_skill_md

VERDICT_VALUES = ("keep", "refine", "redundant", "low_utility")


@dataclass
class SkillGradient:
    """Aggregated language gradient ``G(k)`` for a single skill."""

    skill_id: str
    verdict: str = "keep"               # one of VERDICT_VALUES
    summary: str = ""                   # merged usefulness / robustness note
    suggested_edit: str = ""            # consolidated edit direction
    redundant_with: List[str] = field(default_factory=list)
    utility_score: float = 0.0          # merged numeric utility in [-1, 1]
    n_credits: int = 0                  # how many trajectory credits merged

    def to_dict(self) -> Dict:
        return {
            "skill_id": self.skill_id,
            "verdict": self.verdict,
            "summary": self.summary,
            "suggested_edit": self.suggested_edit,
            "redundant_with": list(self.redundant_with),
            "utility_score": self.utility_score,
            "n_credits": self.n_credits,
        }


def _extract_json(text: str):
    """Pull the first JSON value (object or list) out of an LLM response.

    ``strict=False`` tolerates literal control characters (newlines/tabs)
    that LLMs routinely emit inside string values.
    """
    match = re.search(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1), strict=False)
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1], strict=False)
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON value found in response")


def _clamp(x, lo=-1.0, hi=1.0, default=0.0) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return default


_AGGREGATE_PROMPT = """\
You are aggregating {n} per-trajectory credit signals for ONE skill used by an
agent across a batch of rollouts. Merge recurring patterns, resolve
contradictions between trajectories, de-duplicate observations, weight by how
often each pattern recurred, and emit a single stable summary of the skill's
usefulness, robustness, and coordination value.

## Skill
name: {name}
description: {description}
body:
{body}

## Per-trajectory credits (n={n})
{credits}

## Output
Output ONLY a JSON object, fenced in ```json:
```json
{{"verdict": "keep|refine|redundant|low_utility",
  "summary": "<merged assessment>",
  "suggested_edit": "<consolidated edit direction, or empty if verdict=keep>",
  "redundant_with": ["<skill_id this overlaps>"],
  "utility_score": <float between -1.0 and 1.0>}}
```
Use "refine" when the skill is useful but imperfect, "redundant" when it
overlaps another skill, "low_utility" when it is consistently unhelpful."""


_REFINE_PROMPT = """\
You are refining one skill in an agent's library. Apply the aggregated
improvement direction as a SMALL set of anchor-based edits — do NOT rewrite
from scratch. Preserve the skill's abstract goal and every part that already
works; only touch the regions the feedback implicates.

## Current skill
name: {name}
description: {description}
body:
{body}

## Aggregated improvement direction
{summary}

Suggested edit: {suggested_edit}

## Output
Output ONLY a JSON list of edit instructions, fenced in ```json. Each item:
  {{"op": "replace"|"insert_after"|"insert_before"|"delete",
    "anchor": "<verbatim unique substring of the body to locate the edit>",
    "old": "<exact snippet to replace/delete; null for inserts>",
    "new": "<replacement or inserted text; empty string for delete>"}}
An empty list [] means no change is warranted."""


_REFINE_ROLE_PROMPT = """\
You are refining an agent's ROLE — the part of its prompt that says where it
sits in the team and how it must communicate, as distinct from the task
knowledge held in its skill library.

## Current role
{role}

## Failures not attributable to any individual skill
{residual}

## Hard cases from this iteration
{evidence}

## What you may and may not change
The role carries the collaboration protocol that the environment parses: the
handoff format, the tags the agent emits, which agent speaks last, what it is
allowed to see. Breaking any of that breaks every trajectory, and no skill can
repair it.

So: keep every protocol element verbatim — tag syntax, section headers, the
required shape of the final output, the description of who reads this agent's
output. Change only the framing that the residual failures implicate, such as
an unclear division of labour or a missing statement of what this agent is
responsible for deciding.

Prefer no change. Task knowledge belongs in skills, not here.

## Output
Output ONLY a JSON list of edit instructions, fenced in ```json. Each item:
  {{"op": "replace"|"insert_after"|"insert_before"|"delete",
    "anchor": "<verbatim unique substring of the role to locate the edit>",
    "old": "<exact snippet to replace/delete; null for inserts>",
    "new": "<replacement or inserted text; empty string for delete>"}}
An empty list [] means the role should stay as it is — the common case."""


_INDUCE_PROMPT = """\
An agent repeatedly failed on the hard cases below, and its current skill
library does not adequately cover them. Propose AT MOST ONE new skill that
would close this gap.

## Existing skills (do NOT duplicate these)
{existing}

## Hard-case evidence (residual failures with no responsible skill)
{evidence}

## Output
Output ONLY a skill in Anthropic SKILL.md format — YAML frontmatter delimited
by `---` lines containing `name:` and `description:`, followed by a markdown
body. The body must state the skill's abstract goal, the conditions under
which it applies, and the expected coordination behavior with teammates.
If no new skill is genuinely warranted, output the single word NONE."""


_CONSOLIDATE_PROMPT = """\
The skills below are functionally overlapping. Merge them into ONE consolidated,
higher-level skill that synthesizes their shared behavioral structure,
coordination logic, and reusable procedure. The result must subsume every
capability of the inputs without redundancy.

## Skills to merge
{skills}

## Output
Output ONLY the consolidated skill in Anthropic SKILL.md format (`---`
frontmatter with `name:` and `description:`, then a markdown body)."""


class SkillEvolutionOptimizer(PolicyGradientOptimizer):
    """LLM operators that evolve a discrete skill library from credit."""

    def __init__(self, llm_config, tool_library: str = "", prune_utility_threshold: float = -0.3):
        # synthesis_method / momentum are unused here but required by the base
        # __init__; the diff-edit refine operator below is the editing path.
        super().__init__(llm_config, synthesis_method="diff_edit", momentum=0.0,
                          tool_library=tool_library)
        self.prune_utility_threshold = prune_utility_threshold
        self.logger = logging.getLogger(__name__)

    # ── §4.3 hierarchical aggregation ────────────────────────────────────
    def aggregate_skill_gradients(
        self,
        library: SkillLibrary,
        credits_by_skill: Dict[str, List[SkillCredit]],
    ) -> Dict[str, SkillGradient]:
        """Merge per-trajectory credits into one ``G(k)`` per skill.

        A skill with a single credit is folded without an LLM call; skills
        with two or more credits are merged by the aggregation LLM.
        """
        gradients: Dict[str, SkillGradient] = {}
        for skill_id, credits in credits_by_skill.items():
            if not credits:
                continue
            skill = library.get(skill_id)
            if skill is None:
                continue
            if len(credits) == 1:
                gradients[skill_id] = self._gradient_from_single(skill_id, credits[0])
            else:
                gradients[skill_id] = self._aggregate_llm(skill, credits)
        return gradients

    @staticmethod
    def _gradient_from_single(skill_id: str, c: SkillCredit) -> SkillGradient:
        verdict = {
            "helped": "refine" if c.suggested_edit else "keep",
            "redundant": "redundant",
            "harmful": "low_utility",
            "neutral": "refine" if c.suggested_edit else "keep",
        }.get(c.contribution, "keep")
        return SkillGradient(
            skill_id=skill_id,
            verdict=verdict,
            summary=c.evidence,
            suggested_edit=c.suggested_edit,
            redundant_with=list(c.redundant_with),
            utility_score=c.utility_delta,
            n_credits=1,
        )

    def _aggregate_llm(self, skill: Skill, credits: List[SkillCredit]) -> SkillGradient:
        lines = []
        for i, c in enumerate(credits, 1):
            lines.append(
                f"{i}. contribution={c.contribution}; utility_delta={c.utility_delta:+.2f}; "
                f"evidence={c.evidence}; suggested_edit={c.suggested_edit or '(none)'}; "
                f"redundant_with={c.redundant_with or '[]'}"
            )
        body = skill.body.strip()
        prompt = _AGGREGATE_PROMPT.format(
            n=len(credits),
            name=skill.name,
            description=skill.description,
            body=body[:1200] + (" …" if len(body) > 1200 else ""),
            credits="\n".join(lines),
        )
        mean_util = sum(c.utility_delta for c in credits) / len(credits)
        try:
            data = _extract_json(self._llm_call(prompt, max_tokens=600))
            verdict = str(data.get("verdict", "keep")).strip().lower()
            if verdict not in VERDICT_VALUES:
                verdict = "keep"
            return SkillGradient(
                skill_id=skill.skill_id,
                verdict=verdict,
                summary=str(data.get("summary", "")).strip(),
                suggested_edit=str(data.get("suggested_edit", "")).strip(),
                redundant_with=[str(x).strip() for x in data.get("redundant_with", []) if str(x).strip()],
                utility_score=_clamp(data.get("utility_score", mean_util), default=mean_util),
                n_credits=len(credits),
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("skill-gradient aggregation failed for %s: %s",
                                 skill.skill_id, exc)
            # Fall back to a numeric merge of the raw credits.
            merged = SkillGradient(skill_id=skill.skill_id, n_credits=len(credits),
                                   utility_score=mean_util)
            merged.summary = "; ".join(c.evidence for c in credits if c.evidence)[:600]
            merged.suggested_edit = next((c.suggested_edit for c in credits if c.suggested_edit), "")
            merged.verdict = "low_utility" if mean_util <= self.prune_utility_threshold else (
                "refine" if merged.suggested_edit else "keep")
            return merged

    # ── §4.4 Refinement (Eq.13) ──────────────────────────────────────────
    def refine_skill(self, skill: Skill, gradient: SkillGradient) -> Optional[Skill]:
        """Return a refined copy of ``skill``, or None if nothing changed."""
        if not gradient.suggested_edit and not gradient.summary:
            return None
        body = skill.body.strip()
        if not body:
            # Empty body: the suggested edit becomes the body.
            new_body = gradient.suggested_edit.strip()
            if not new_body:
                return None
        else:
            prompt = _REFINE_PROMPT.format(
                name=skill.name,
                description=skill.description,
                body=body,
                summary=gradient.summary or "(see suggested edit)",
                suggested_edit=gradient.suggested_edit or "(see summary)",
            )
            try:
                edits = self._parse_edit_list(self._llm_call(prompt, max_tokens=800))
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("refine parse failed for %s: %s", skill.skill_id, exc)
                return None
            if not edits:
                return None
            new_body = self._apply_edits(body, edits)
            if new_body.strip() == body:
                return None
        refined = skill.copy()
        refined.body = new_body.strip()
        refined.provenance = "refined"
        return refined

    def refine_role(self, role: str, residual: str, evidence: List[str],
                    agent_name: str = "agent") -> Optional[str]:
        """Return an edited role, or None to leave it as it is.

        Off by default. The role holds the collaboration protocol the
        environment parses, so an edit here can invalidate every trajectory at
        once in a way no skill can repair -- which is why the prompt is
        anchor-based rather than a rewrite, and why the trainer puts the result
        through the same validation gate as the library.
        """
        role = (role or "").strip()
        if not role or (not residual and not evidence):
            return None

        prompt = _REFINE_ROLE_PROMPT.format(
            role=role,
            residual=residual or "(none reported)",
            evidence="\n".join(f"- {e}" for e in evidence[:8]) or "(none)",
        )
        try:
            edits = self._parse_edit_list(self._llm_call(prompt, max_tokens=800))
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("role refine parse failed for %s: %s", agent_name, exc)
            return None
        if not edits:
            return None

        new_role = self._apply_edits(role, edits).strip()
        if new_role == role or not new_role:
            return None
        return new_role

    @staticmethod
    def _parse_edit_list(raw: str) -> list:
        match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
        payload = match.group(1) if match else raw[raw.find("[") : raw.rfind("]") + 1]
        data = json.loads(payload, strict=False)
        if not isinstance(data, list):
            raise ValueError("edit instructions must be a JSON list")
        return data

    # ── §4.4 Induction (Eq.14) ───────────────────────────────────────────
    def induce_skill(
        self,
        hard_evidence: List[str],
        existing_library: SkillLibrary,
        residual_summary: str = "",
    ) -> Optional[Skill]:
        """Propose one new skill from hard cases, or None."""
        if not hard_evidence and not residual_summary.strip():
            return None
        existing = "\n".join(
            f"- {s.name} (id: {s.skill_id}): {s.description}" for s in existing_library
        ) or "(none)"
        evidence_lines = []
        if residual_summary.strip():
            evidence_lines.append(f"Residual failure pattern: {residual_summary.strip()}")
        for i, ev in enumerate(hard_evidence[:6], 1):
            evidence_lines.append(f"{i}. {ev}")
        prompt = _INDUCE_PROMPT.format(
            existing=existing,
            evidence="\n".join(evidence_lines),
        )
        try:
            raw = self._llm_call(prompt, max_tokens=900).strip()
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("induction LLM call failed: %s", exc)
            return None
        if raw.upper().startswith("NONE") or "---" not in raw:
            return None
        try:
            skill = parse_skill_md(_strip_fences(raw))
        except ValueError as exc:
            self.logger.warning("induced skill did not parse: %s", exc)
            return None
        skill.provenance = "induced"
        skill.skill_id = ""  # let the library assign a collision-free id
        skill.__post_init__()
        return skill

    # ── §4.4 Consolidation (Eq.15) ───────────────────────────────────────
    @staticmethod
    def detect_redundancy_groups(
        library: SkillLibrary,
        gradients: Dict[str, SkillGradient],
    ) -> List[List[str]]:
        """Cluster skills declared redundant with one another (union-find).

        Returns connected components of size >= 2, restricted to skill ids
        present in ``library``.
        """
        present = set(library.ids())
        parent: Dict[str, str] = {sid: sid for sid in present}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for sid, grad in gradients.items():
            if sid not in present:
                continue
            for other in grad.redundant_with:
                if other in present and other != sid:
                    union(sid, other)
        groups: Dict[str, List[str]] = {}
        for sid in present:
            groups.setdefault(find(sid), []).append(sid)
        return [sorted(g) for g in groups.values() if len(g) >= 2]

    def consolidate_skills(self, library: SkillLibrary, group_ids: List[str]) -> Optional[Skill]:
        """Merge a group of overlapping skills into one consolidated skill."""
        skills = [library.get(sid) for sid in group_ids]
        skills = [s for s in skills if s is not None]
        if len(skills) < 2:
            return None
        blocks = []
        for s in skills:
            body = s.body.strip()
            blocks.append(
                f"### {s.name} (id: {s.skill_id})\n{s.description}\n\n"
                f"{body[:800] + (' …' if len(body) > 800 else '')}"
            )
        prompt = _CONSOLIDATE_PROMPT.format(skills="\n\n".join(blocks))
        try:
            raw = self._llm_call(prompt, max_tokens=900).strip()
            macro = parse_skill_md(_strip_fences(raw))
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("consolidation failed for %s: %s", group_ids, exc)
            return None
        macro.provenance = "consolidated"
        macro.skill_id = ""
        macro.__post_init__()
        # Carry forward accumulated bookkeeping from the merged skills.
        macro.invocations = sum(s.invocations for s in skills)
        macro.utility = sum(s.utility for s in skills) / len(skills)
        return macro

    # ── §4.4 Pruning (Eq.16) ─────────────────────────────────────────────
    def select_skills_to_prune(
        self,
        library: SkillLibrary,
        gradients: Dict[str, SkillGradient],
    ) -> List[str]:
        """Identify low-utility skills to remove (``LowUtility``).

        A skill is pruned when its aggregated gradient verdict is
        ``low_utility``, or its merged utility score is persistently below
        the threshold.  The last remaining skill is never pruned.  Redundant
        skills are left to the consolidation operator, not pruned here.
        """
        doomed: List[str] = []
        for sid in library.ids():
            grad = gradients.get(sid)
            skill = library.get(sid)
            if grad is not None:
                if grad.verdict == "low_utility" or grad.utility_score <= self.prune_utility_threshold:
                    doomed.append(sid)
            elif skill is not None and skill.invocations == 0 and skill.utility <= self.prune_utility_threshold:
                # Never invoked across the batch and historically unhelpful.
                doomed.append(sid)
        # Guard: keep at least one skill if the library is non-empty.
        if doomed and len(doomed) >= len(library):
            doomed = doomed[: len(library) - 1]
        return doomed


def _strip_fences(text: str) -> str:
    """Strip a leading/trailing markdown code fence if present."""
    text = text.strip()
    m = re.match(r"```[a-zA-Z]*\s*\n(.*?)\n```\s*$", text, re.DOTALL)
    return m.group(1).strip() if m else text
