"""LLM-based policy gradient optimizer.

Gradient pipeline:
  1. generate_gradient()        -- per-agent improvement instruction (central_credit)
  2. generate_shared_gradient() -- shared instruction for all agents (central_global)
  3. synthesize_policy()        -- LLM-based rewrite of role + skills from gradients
  4. aggregate_gradients()      -- combine multiple gradients from multiple episodes
  5. parse_credit_response()    -- parse per-agent credit from evaluator response
"""

import json
import logging
import re
from typing import Dict, List, Optional, Union

from openai import OpenAI

from .base import BaseOptimizer
from .policy import AgentPolicy

_TOOL_LIBRARY_HEADER = """\
## Tool Library (visible to you, the coach — the agent is NOT told about
## these directly; the agent only knows what is written in its Skills.
## If a tool would have helped, your improvement note should teach the
## agent to invoke it using the exact syntax documented below.)
{tool_library}
"""


_PER_AGENT_GRADIENT_PROMPT = """\
You are coaching an AI agent to improve its problem-solving strategy.

## Agent's Role (fixed, for context only)
{role}

## Agent's Current Skills/Strategy (this is what you're improving)
{skills}

## Problem Being Solved
{task_context}
{tool_library_block}
## Evaluation of the Agent's Last Attempt
{evaluation}

## Your Task
Write a concise, specific improvement note (2-4 sentences) that:
- Identifies the exact reasoning or strategy mistake made in this attempt
- Gives a concrete, actionable instruction for this type of problem
- Is grounded in the specific case above (not generic advice)

Write the improvement note directly (no preamble or headers):"""


_SYNTHESIZE_POLICY_PROMPT = """\
You are an expert prompt engineer.  Your task is to produce improved versions
of an AI agent's role description and task-specific skills by integrating
concrete improvement feedback from recent problem-solving attempts.

## Agent's Current Role (describes responsibilities and division of labor)
{role}

## Agent's Current Skills/Strategy
{skills}
{tool_library_block}
## Improvement Feedback (from one or more attempts)
{feedback}

## Instructions
Based on the feedback, produce updated role and skills sections.

- The **role** section should refine the agent's responsibilities, division of
  labor, and collaboration protocol based on what worked or didn't work.
  Keep structural info (agent position, who speaks when) but improve the
  strategic guidance for the agent's role in the team.
- The **skills** section should integrate specific improvement suggestions
  into concrete, actionable strategy instructions.

Both sections should be self-contained — do NOT reference "previous attempts",
"feedback", or "improvements".  They should look like they were always written
this way.  Keep each section concise (similar length to the originals).

Output in this exact format (including the markers):

[ROLE]
<updated role description>

[SKILLS]
<updated skills/strategy instructions>"""


_DIFF_EDIT_PROMPT = """\
You are an expert prompt engineer.  Improve an AI agent's role and skills by
emitting a small set of *anchor-based edits* — do NOT rewrite from scratch.

## Agent's Current Role
{role}

## Agent's Current Skills/Strategy
{skills}
{tool_library_block}
## Improvement Feedback
{feedback}
{momentum_section}
## Your Task
Produce a JSON object with two keys, "role" and "skills".  Each value is a
list of edit instructions to apply (in order) to the corresponding section.

Each edit instruction is an object:
  {{"op": "replace" | "insert_after" | "insert_before" | "delete",
    "anchor": "<short unique substring of the current section used to locate the edit>",
    "old":    "<exact existing snippet to replace/delete; null for insert ops>",
    "new":    "<replacement or inserted text; empty string for delete>"}}

Rules:
- The "anchor" MUST be a verbatim, unique substring of the current section.
- For "replace"/"delete", "old" must also appear verbatim in the section.
- Prefer minimal, targeted edits.  An empty list means "no change".
- Do NOT reference "feedback" or "improvements" in the new text — make it
  read as if always written this way.

Output ONLY the JSON object, wrapped in a ```json fenced block:

```json
{{"role": [...], "skills": [...]}}
```"""


_MOMENTUM_BLOCK = """
## Previous Edit Direction (apply momentum: continue this trend)
The previous synthesis step changed the policy as follows; you should keep
moving in the same editorial direction (e.g. "more concise", "more
structured", "more cautious") with weight β={beta:.2f} relative to the new
feedback above, unless the feedback explicitly contradicts it.

### Previous Skills (θ_{{t-1}})
{prev_skills}

### Current Skills (θ_t)
{curr_skills}
"""


_SHARED_GRADIENT_PROMPT = """\
You are coaching a multi-agent AI team to improve their problem-solving strategy.

## Problem Being Solved
{task_context}
{tool_library_block}
## Team Evaluation
{evaluation}

## Your Task
Write a concise, specific improvement note for the whole team (2-4 sentences) that:
- Identifies the exact team-level reasoning or strategy mistake
- Gives a concrete, actionable instruction for this type of problem
- Is grounded in the specific case above (not generic advice)

Write the improvement note directly (no preamble or headers):"""


class PolicyGradientOptimizer(BaseOptimizer):
    """LLM-based policy gradient optimizer."""

    def __init__(
        self,
        llm_config,
        synthesis_method: str = "rewrite",
        momentum: float = 0.0,
        tool_library: str = "",
    ):
        """
        Args:
            llm_config: LLMConfig instance for the optimizer LLM.
            synthesis_method: "rewrite" (LLM rewrites the section) or
                "diff_edit" (LLM emits anchor-based edit instructions that
                are applied locally).
            momentum: float >= 0.  When > 0 and a previous policy is supplied
                to ``synthesize_policy``, the optimizer additionally conditions
                on the previous edit direction (θ_t − θ_{t-1}) with weight β.
            tool_library: optional markdown describing tools available in
                the environment.  Injected into gradient and synthesis
                prompts so the optimizer can propose skills that use them.
                The agent is NOT shown this directly — only skills derived
                from it are.
        """
        from ..config.llm import LLMConfig
        if not isinstance(llm_config, LLMConfig):
            raise TypeError(f"Expected LLMConfig, got {type(llm_config)}")
        if synthesis_method not in ("rewrite", "diff_edit"):
            raise ValueError(f"synthesis_method must be 'rewrite' or 'diff_edit', got {synthesis_method!r}")
        if momentum < 0.0:
            raise ValueError("momentum must be >= 0.0")
        api_key = llm_config.get_api_key()
        if llm_config.base_url:
            self._client = OpenAI(base_url=llm_config.base_url, api_key=api_key)
        else:
            self._client = OpenAI(api_key=api_key)
        self._model = llm_config.model_string
        self.synthesis_method = synthesis_method
        self.momentum = momentum
        self.tool_library = tool_library or ""
        self.logger = logging.getLogger(__name__)

    def _tool_library_block(self) -> str:
        if not self.tool_library.strip():
            return ""
        return "\n" + _TOOL_LIBRARY_HEADER.format(tool_library=self.tool_library.strip())

    def generate_gradient(
        self,
        policy: Union[AgentPolicy, str],
        evaluation: str,
        context: str,
        agent_name: str = "agent",
    ) -> str:
        """Generate a case-specific improvement instruction for one agent."""
        if isinstance(policy, str):
            policy = AgentPolicy.from_legacy(policy)
        skills_display = policy.skills.strip() if policy.skills.strip() else "(No specific skills defined yet)"
        prompt = _PER_AGENT_GRADIENT_PROMPT.format(
            role=policy.role,
            skills=skills_display,
            task_context=context[:8000],
            evaluation=evaluation,
            tool_library_block=self._tool_library_block(),
        )
        try:
            gradient = self._llm_call(prompt, max_tokens=400)
            self.logger.debug(
                "Gradient for %s (%d chars): %.80s ...", agent_name, len(gradient), gradient
            )
            return gradient
        except Exception as exc:
            self.logger.warning(
                "Gradient generation failed for %s, using evaluation as fallback: %s",
                agent_name, exc,
            )
            return evaluation

    def generate_shared_gradient(self, evaluation: str, task_context: str) -> str:
        """Generate one shared gradient for the whole team (central_global)."""
        prompt = _SHARED_GRADIENT_PROMPT.format(
            task_context=task_context[:8000],
            evaluation=evaluation,
            tool_library_block=self._tool_library_block(),
        )
        try:
            gradient = self._llm_call(prompt, max_tokens=400)
            self.logger.debug("Shared gradient (%d chars): %.80s ...", len(gradient), gradient)
            return gradient
        except Exception as exc:
            self.logger.warning(
                "Shared gradient generation failed, using evaluation as fallback: %s", exc
            )
            return evaluation

    def synthesize_policy(
        self,
        base_policy: Union[AgentPolicy, str],
        gradients: List[str],
        agent_name: str = "agent",
        previous_policy: Optional[Union[AgentPolicy, str]] = None,
    ) -> Union[AgentPolicy, str]:
        """Synthesize new role + skills by integrating gradient feedback.

        Dispatches to ``rewrite`` (full LLM rewrite) or ``diff_edit``
        (anchor-based local edits) based on ``self.synthesis_method``.
        If ``self.momentum > 0`` and ``previous_policy`` is provided, the
        prompt additionally describes the previous edit direction so the LLM
        can extend it (textgrad analogue of β·(θ_t − θ_{t-1})).
        """
        if isinstance(base_policy, str):
            base_policy = AgentPolicy.from_legacy(base_policy)
        if isinstance(previous_policy, str):
            previous_policy = AgentPolicy.from_legacy(previous_policy)
        if not gradients:
            return base_policy

        feedback = "\n\n---\n\n".join(gradients)
        try:
            if self.synthesis_method == "diff_edit":
                new_policy = self._synthesize_diff_edit(base_policy, feedback, previous_policy)
            else:
                new_policy = self._synthesize_rewrite(base_policy, feedback, previous_policy)
            self.logger.debug(
                "Synthesized policy for %s (%s, momentum=%.2f): role %d→%d, skills %d→%d",
                agent_name, self.synthesis_method, self.momentum,
                len(base_policy.role), len(new_policy.role),
                len(base_policy.skills), len(new_policy.skills),
            )
            return new_policy
        except Exception as exc:
            self.logger.warning(
                "Policy synthesis failed for %s (%s), falling back to skills-only append: %s",
                agent_name, self.synthesis_method, exc,
            )
            new_skills = base_policy.skills.rstrip() + "\n\n[CASE-SPECIFIC FEEDBACK]\n" + feedback
            return AgentPolicy(role=base_policy.role, skills=new_skills)

    def _momentum_section(self, base_policy: AgentPolicy, previous_policy: Optional[AgentPolicy]) -> str:
        if self.momentum <= 0.0 or previous_policy is None:
            return ""
        prev = previous_policy.skills.strip() or "(empty)"
        curr = base_policy.skills.strip() or "(empty)"
        if prev == curr:
            return ""
        return _MOMENTUM_BLOCK.format(beta=self.momentum, prev_skills=prev, curr_skills=curr)

    def _synthesize_rewrite(
        self,
        base_policy: AgentPolicy,
        feedback: str,
        previous_policy: Optional[AgentPolicy],
    ) -> AgentPolicy:
        skills_display = base_policy.skills.strip() or "(No specific skills defined yet)"
        momentum_section = self._momentum_section(base_policy, previous_policy)
        prompt = _SYNTHESIZE_POLICY_PROMPT.format(
            role=base_policy.role,
            skills=skills_display,
            feedback=feedback + momentum_section,
            tool_library_block=self._tool_library_block(),
        )
        raw = self._llm_call(prompt, max_tokens=1024)
        new_role, new_skills = self._parse_synthesis_output(raw, base_policy)
        return AgentPolicy(role=new_role, skills=new_skills)

    def _synthesize_diff_edit(
        self,
        base_policy: AgentPolicy,
        feedback: str,
        previous_policy: Optional[AgentPolicy],
    ) -> AgentPolicy:
        if not base_policy.skills.strip():
            return self._synthesize_rewrite(base_policy, feedback, previous_policy)
        skills_display = base_policy.skills.strip()
        momentum_section = self._momentum_section(base_policy, previous_policy)
        prompt = _DIFF_EDIT_PROMPT.format(
            role=base_policy.role,
            skills=skills_display,
            feedback=feedback,
            momentum_section=momentum_section,
            tool_library_block=self._tool_library_block(),
        )
        raw = self._llm_call(prompt, max_tokens=1024)
        edits = self._parse_edit_instructions(raw)
        new_role = self._apply_edits(base_policy.role, edits.get("role", []))
        new_skills = self._apply_edits(base_policy.skills, edits.get("skills", []))
        if not new_role.strip():
            new_role = base_policy.role
        return AgentPolicy(role=new_role, skills=new_skills)

    @staticmethod
    def _parse_edit_instructions(raw: str) -> Dict[str, list]:
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
        payload = match.group(1) if match else raw
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("edit instructions must be a JSON object")
        out = {}
        for key in ("role", "skills"):
            v = data.get(key, [])
            if v is None:
                v = []
            if not isinstance(v, list):
                raise ValueError(f"edits for '{key}' must be a list")
            out[key] = v
        return out

    @staticmethod
    def _apply_edits(text: str, instructions: list) -> str:
        def join(left: str, right: str) -> str:
            """Concatenate, adding a newline when neither side separates them.

            An ``insert_after`` whose text does not start with whitespace used
            to be glued straight onto the anchor, producing runs like
            ``"...from Agent 1You are responsible..."``.
            """
            if not left or not right:
                return left + right
            if left[-1].isspace() or right[0].isspace():
                return left + right
            return left + "\n" + right

        for ins in instructions:
            op = ins.get("op")
            anchor = ins.get("anchor", "") or ""
            old = ins.get("old")
            new = ins.get("new", "") or ""
            if not anchor or text.find(anchor) == -1:
                # Skip silently rather than corrupt the text.
                continue
            anchor_idx = text.find(anchor)
            if text.find(anchor, anchor_idx + 1) != -1:
                # ambiguous anchor — skip
                continue
            anchor_end = anchor_idx + len(anchor)
            if op == "insert_after":
                text = join(join(text[:anchor_end], new), text[anchor_end:])
            elif op == "insert_before":
                text = join(join(text[:anchor_idx], new), text[anchor_idx:])
            elif op in ("replace", "delete"):
                if not old:
                    continue
                old_idx = text.find(old, anchor_end)
                if old_idx == -1:
                    old_idx = text.find(old)
                if old_idx == -1:
                    continue
                replacement = "" if op == "delete" else new
                text = text[:old_idx] + replacement + text[old_idx + len(old):]
        return text

    @staticmethod
    def _parse_synthesis_output(raw: str, base_policy: AgentPolicy) -> tuple:
        """Parse [ROLE] and [SKILLS] sections from synthesis LLM output.

        Returns (new_role, new_skills).  Falls back to base_policy values
        for any section that cannot be parsed.
        """
        import re as _re
        role_match = _re.search(r'\[ROLE\]\s*\n(.*?)(?=\[SKILLS\])', raw, _re.DOTALL)
        skills_match = _re.search(r'\[SKILLS\]\s*\n(.*)', raw, _re.DOTALL)

        new_role = role_match.group(1).strip() if role_match else base_policy.role
        new_skills = skills_match.group(1).strip() if skills_match else base_policy.skills

        # Sanity: don't accept empty role
        if not new_role.strip():
            new_role = base_policy.role

        return new_role, new_skills

    @staticmethod
    def aggregate_gradients(gradients: List[str]) -> str:
        """Aggregate multiple gradients (from multiple episodes) into one."""
        if not gradients:
            return ""
        if len(gradients) == 1:
            return gradients[0]
        return "\n\n---\n\n".join(gradients)

    @staticmethod
    def parse_credit_response(response: str, agent_names: List[str]) -> Dict[str, str]:
        """Parse per-agent evaluations from a credit-assignment LLM response.

        Handles:
        - Overcooked: [AGENT 0 EVALUATION] / [AGENT 1 EVALUATION]
        - Language task: JSON dict {"agent_1": "...", "agent_2": "..."}
        - Pistonball: [PISTON_N EVALUATION] markers
        - Fallback: returns full response for all agents.
        """
        if not agent_names:
            return {}

        result: Dict[str, str] = {}

        # Overcooked
        overcooked_names = {'0', '1', 'agent_0', 'agent_1'}
        if set(agent_names).issubset(overcooked_names) or (
            len(agent_names) == 2 and any(n in overcooked_names for n in agent_names)
        ):
            pattern = r'\[AGENT\s+(\d+)\s+EVALUATION\](.*?)(?=\[AGENT\s+\d+\s+EVALUATION\]|$)'
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            if matches:
                idx_to_text = {m[0]: m[1].strip() for m in matches}
                for name in agent_names:
                    idx = name.split('_')[-1] if '_' in name else name
                    if idx in idx_to_text:
                        result[name] = idx_to_text[idx]
                if result:
                    for name in agent_names:
                        if name not in result:
                            result[name] = response
                    return result

        # Language task: JSON dict
        if any(n.startswith('agent_') and n[6:].isdigit() for n in agent_names):
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if not json_match:
                json_match = re.search(r'(\{[^{}]*"agent_\d+"[^{}]*\})', response, re.DOTALL)
            if json_match:
                try:
                    import json as _json
                    parsed = _json.loads(json_match.group(1))
                    for name in agent_names:
                        if name in parsed:
                            result[name] = str(parsed[name])
                    if result:
                        for name in agent_names:
                            if name not in result:
                                result[name] = response
                        return result
                except Exception:
                    pass

        # Pistonball
        if any(n.startswith('piston_') for n in agent_names):
            pattern = r'\[PISTON[_\s]?(\d+)\s+EVALUATION\](.*?)(?=\[PISTON[_\s]?\d+\s+EVALUATION\]|$)'
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            if matches:
                idx_to_text = {m[0]: m[1].strip() for m in matches}
                for name in agent_names:
                    idx = name.split('_')[-1]
                    if idx in idx_to_text:
                        result[name] = idx_to_text[idx]
                if result:
                    for name in agent_names:
                        if name not in result:
                            result[name] = response
                    return result

            third = len(agent_names) // 3
            group_assignments = {}
            for i, name in enumerate(sorted(agent_names, key=lambda x: int(x.split('_')[-1]))):
                if i < third:
                    group_assignments[name] = 'left'
                elif i < 2 * third:
                    group_assignments[name] = 'middle'
                else:
                    group_assignments[name] = 'right'

            for group in ('left', 'middle', 'right'):
                pat = rf'\b{group}\b.{{0,2000}}'
                match = re.search(pat, response, re.DOTALL | re.IGNORECASE)
                if match:
                    group_text = match.group(0)[:500]
                    for name, g in group_assignments.items():
                        if g == group:
                            result[name] = group_text

            if result:
                for name in agent_names:
                    if name not in result:
                        result[name] = response
                return result

        # Fallback
        return {name: response for name in agent_names}

    def _llm_call(self, prompt: str, max_tokens: int = 400) -> str:
        model_lower = self._model.lower()
        params = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if "o1" in model_lower or "o3" in model_lower:
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens
        resp = self._client.chat.completions.create(**params)
        return resp.choices[0].message.content.strip()
