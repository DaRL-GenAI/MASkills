"""GAIA environment adapter for MASkills.

Implements the same 3 topologies as language/locomo so ``eval_topology.py``
can run GAIA alongside the other tasks:

* ``decentralized`` — agent_1 (Researcher) does SEARCH/BROWSE and emits a
  HANDOFF dossier; agent_2 (Solver) does COMPUTE + emits FINAL ANSWER.
* ``centralized``   — agent_2 (main) prompts ``<retrieve>QUERY</retrieve>``;
  each retrieve spawns the Researcher in a FRESH context with the
  question + the main's query. Main has COMPUTE; sub has SEARCH/BROWSE.
* ``hybrid``        — same dispatch as centralized but each Researcher
  invocation sees prior dossiers (so subsequent retrieves can refine).

Tool primitives (SEARCH/BROWSE/COMPUTE), the HANDOFF protocol parsers, the
placeholder/null-kick guards, the answer cleanup regexes, and the official
GAIA scorer come from the sibling :mod:`~maskills.envs.gaia.single_agent`,
:mod:`~maskills.envs.gaia.tools` and
:mod:`~maskills.envs.gaia.decentralized` modules, which double as
standalone evaluators (``python -m maskills.envs.gaia.tools --help``).
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List

from maskills.core.base import BaseEnvironment, Trajectory
from maskills.envs import register_env
from maskills.llm.client import LLMClient

from . import decentralized as _egd
from . import single_agent as _eg
from . import tools as _egt
from ._keys import require_api_keys
from .protocol import prompt_for as _protocol_prompt_for
from .skill_loader import render_agent_prompt
from .task_loader import GaiaTaskLoader

_RETRIEVE_RE = re.compile(r"<retrieve>(.*?)</retrieve>", re.DOTALL | re.IGNORECASE)


@register_env("gaia")
class GaiaEnv(BaseEnvironment):
    """GAIA env with 3 topologies and Tavily/local-Python tool stack."""

    def __init__(self, config):
        require_api_keys(tavily=True)

        self.num_agents = int(getattr(config, "num_agents", 2))
        if self.num_agents != 2:
            raise ValueError("GaiaEnv requires num_agents == 2 (Researcher + Solver)")
        self.agent_names = ["agent_1", "agent_2"]

        self.architecture = getattr(config, "architecture", "decentralized")
        valid = {"decentralized", "centralized", "hybrid"}
        if self.architecture not in valid:
            raise ValueError(f"architecture must be one of {sorted(valid)}")
        self.main_agent = getattr(config, "main_agent", "agent_2")
        self.sub_agents = [a for a in self.agent_names if a != self.main_agent]

        # Per-agent SKILL.md prompts (pre-rendered once, reused per rollout).
        agent_dirs = getattr(config, "agent_skills_dirs", None) or {}
        self.system_prompts: Dict[str, str] = {}
        for agent in self.agent_names:
            d = agent_dirs.get(agent)
            # No library means the no-skills floor, which is the protocol and
            # nothing else -- an agent given a literally empty prompt would not
            # know the tool syntax or how to emit its answer, and would be
            # measuring the wrong thing.
            self.system_prompts[agent] = (
                render_agent_prompt(d) if d else _protocol_prompt_for(agent))

        # LLM client (one shared client; both agents call it).
        llm = getattr(config, "actor_llm", None) or config.llm
        if llm is None:
            raise ValueError("GaiaConfig requires .llm (or .actor_llm) to be set")
        self.llm_client = LLMClient(llm)

        # Rollout budgets
        self.rounds_a1 = int(getattr(config, "rounds_a1", 5))
        self.rounds_a2 = int(getattr(config, "rounds_a2", 3))
        self.budget_a1 = int(getattr(config, "budget_a1", 5))
        self.budget_a2 = int(getattr(config, "budget_a2", 3))
        self.max_retrieves = int(getattr(config, "max_retrieves", 3))
        self.max_tokens = int(getattr(config, "max_tokens", 1500))

        # Task data
        benchmark_path = getattr(config, "benchmark_path", "")
        data_limit = getattr(config, "data_limit", None)
        if benchmark_path:
            self.task_loader = GaiaTaskLoader(benchmark_path, data_limit=data_limit)
        else:
            self.task_loader = None

        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # BaseEnvironment interface
    # ------------------------------------------------------------------

    def reset(self, task: dict) -> dict:
        return {"task": task}

    def step(self, agent_id: str, action: str):
        return {}, 0.0, False, {}

    def collect_trajectory(self, policies: dict, task: dict) -> Trajectory:
        if self.architecture in ("centralized", "hybrid"):
            return self._collect_centralized(task)
        return self._collect_decentralized(task)

    def sample_tasks(self, num_samples: int, seed=None, split: str = "test"):
        if self.task_loader is None:
            raise ValueError("GaiaEnv has no task_loader. Provide benchmark_path.")
        return self.task_loader.sample_tasks(num_samples, seed=seed, split=split)

    def get_skill_library(self) -> list:
        return []

    def get_tool_library(self) -> str:
        return ""

    # ------------------------------------------------------------------
    # LLM call: temperature=0, robust to content-filter Nones.
    # ------------------------------------------------------------------

    def _llm(self, messages: List[dict]) -> tuple[str, dict]:
        """Single chat completion; returns (text, {input, output} tokens)."""
        text, tokens = self.llm_client.chat_messages_with_usage(
            messages, max_tokens=self.max_tokens,
        )
        return text or "", tokens

    # ------------------------------------------------------------------
    # Decentralized topology: HANDOFF protocol
    # ------------------------------------------------------------------

    def _collect_decentralized(self, task: dict) -> Trajectory:

        sys_a1 = self.system_prompts["agent_1"]
        sys_a2 = self.system_prompts["agent_2"]
        usr_parts = _eg.render_user_message(task)

        in_tok = out_tok = 0
        tool_log: List[dict] = []
        raw_turns: List[dict] = []
        handoff_body = ""
        request_more_used = False
        pred = ""
        a1_round = a1_calls = 0
        a2_round = a2_calls = 0
        placeholder_kick_used = False
        null_kick_used = False
        total_rounds = 0
        max_total = self.rounds_a1 + self.rounds_a2 + 3

        msgs_a1 = [
            {"role": "system", "content": sys_a1},
            {"role": "user", "content": usr_parts},
        ]

        def _call(msgs):
            nonlocal in_tok, out_tok
            text, tokens = self._llm(msgs)
            in_tok += tokens.get("input", 0)
            out_tok += tokens.get("output", 0)
            return text

        # ── Phase 1: agent_1 researches and emits HANDOFF ──
        while a1_round < self.rounds_a1 and total_rounds < max_total:
            a1_round += 1
            total_rounds += 1
            content = _call(msgs_a1)
            raw_turns.append({"agent": "agent_1", "round": a1_round, "content": content})
            msgs_a1.append({"role": "assistant", "content": content})

            h = _egd.extract_handoff(content)
            if h:
                handoff_body = h
                break

            calls = _egt.extract_tool_calls(content)
            if not calls:
                if a1_round < self.rounds_a1:
                    msgs_a1.append({"role": "user", "content":
                        "[harness] You emitted neither a tool call "
                        "(SEARCH:/BROWSE:) nor a complete `HANDOFF:` block. "
                        "Emit one of them THIS turn. End HANDOFF with "
                        "`HANDOFF_TO_SOLVER`."})
                    tool_log.append({"agent": "agent_1", "round": a1_round,
                                     "kind": "[KICK]"})
                    continue
                break

            remaining = self.budget_a1 - a1_calls
            if remaining <= 0:
                msgs_a1.append({"role": "user", "content":
                    "[harness] agent_1 tool budget exhausted. Emit HANDOFF "
                    "block now and end with `HANDOFF_TO_SOLVER`."})
                continue
            executed = calls[:remaining]
            tool_msgs = []
            for kind, payload in executed:
                result = _egd.dispatch_a1_tool(kind, payload)
                a1_calls += 1
                tool_log.append({"agent": "agent_1", "round": a1_round,
                                 "kind": kind, "payload": payload[:300],
                                 "result_preview": result[:400]})
                tool_msgs.append(f"=== {kind} result ===\n{result}")
            near = a1_calls >= self.budget_a1 - 1 or a1_round >= self.rounds_a1 - 1
            followup = (
                "\n\n[harness] EMIT THE HANDOFF BLOCK NOW (no more tool "
                "calls). End with `HANDOFF_TO_SOLVER`."
                if near else
                "\n\n[harness] Continue researching, or emit the `HANDOFF:` "
                "block when ready."
            )
            msgs_a1.append({"role": "user", "content":
                            "\n\n".join(tool_msgs) + followup})

        if not handoff_body:
            handoff_body = (
                "HANDOFF:\n"
                "QUESTION_INTERPRETATION: (agent_1 did not produce a structured "
                "handoff; agent_2 should answer from the question + attachment alone)\n"
                "KEY_FACTS_FOUND: (none)\nEVIDENCE_QUOTES: (none)\n"
                "SUGGESTED_COMPUTATION: solve directly from the question and any inline attachment\n"
                "SUGGESTED_ANSWER: null\n"
                "UNCERTAINTY_NOTES: agent_1 failed to produce a handoff\n"
                "HANDOFF_TO_SOLVER"
            )
            tool_log.append({"agent": "harness", "round": 0,
                             "kind": "[SYNTHETIC_HANDOFF]"})

        # ── Phase 2: agent_2 solves ──
        msgs_a2 = [
            {"role": "system", "content": sys_a2},
            {"role": "user", "content": usr_parts},
            {"role": "user", "content":
                "Below is agent_1 (Researcher)'s HANDOFF dossier. Treat "
                "EVIDENCE_QUOTES as your sole external source.\n\n" + handoff_body},
        ]

        while a2_round < self.rounds_a2 and total_rounds < max_total:
            a2_round += 1
            total_rounds += 1
            content = _call(msgs_a2)
            raw_turns.append({"agent": "agent_2", "round": a2_round, "content": content})
            msgs_a2.append({"role": "assistant", "content": content})

            topic = _egd.extract_request_more(content)
            if topic and not request_more_used and _egt.parse_final_answer(content) == "":
                request_more_used = True
                tool_log.append({"agent": "agent_2", "round": a2_round,
                                 "kind": "[REQUEST_MORE]", "payload": topic[:300]})
                msgs_a1.append({"role": "user", "content":
                    f"[from agent_2] agent_2 needs additional evidence: {topic}\n\n"
                    "Run 1-2 more SEARCH/BROWSE calls focused on this gap, "
                    "then emit a REFRESHED `HANDOFF:` block. End with `HANDOFF_TO_SOLVER`."})
                # Up to 2 bonus rounds for agent_1.
                for extra in range(2):
                    if total_rounds >= max_total:
                        break
                    total_rounds += 1
                    c1 = _call(msgs_a1)
                    raw_turns.append({"agent": "agent_1",
                                      "round": a1_round + 1 + extra, "content": c1})
                    msgs_a1.append({"role": "assistant", "content": c1})
                    h = _egd.extract_handoff(c1)
                    if h:
                        handoff_body = h
                        msgs_a2.append({"role": "user", "content":
                            "[harness] agent_1 returned a refreshed HANDOFF:\n\n"
                            + h + "\n\n[harness] Now emit FINAL ANSWER."})
                        break
                    calls1 = _egt.extract_tool_calls(c1)
                    if not calls1:
                        msgs_a1.append({"role": "user", "content":
                            "[harness] Issue SEARCH/BROWSE or emit refreshed HANDOFF NOW."})
                        continue
                    remaining = max(0, self.budget_a1 + 2 - a1_calls)
                    for kind, payload in calls1[:remaining]:
                        result = _egd.dispatch_a1_tool(kind, payload)
                        a1_calls += 1
                        tool_log.append({"agent": "agent_1",
                                         "round": a1_round + 1 + extra,
                                         "kind": kind, "payload": payload[:300],
                                         "result_preview": result[:400]})
                    msgs_a1.append({"role": "user", "content":
                        "[harness] Emit the REFRESHED HANDOFF now (HANDOFF_TO_SOLVER)."})
                continue

            pred = _egt.parse_final_answer(content)
            if pred:
                if (not placeholder_kick_used and a2_round < self.rounds_a2
                        and _egt._has_placeholder(pred)):
                    placeholder_kick_used = True
                    tool_log.append({"agent": "agent_2", "round": a2_round,
                                     "kind": "[PLACEHOLDER_KICK]", "payload": pred})
                    msgs_a2.append({"role": "user", "content":
                        f"[harness] Your FINAL ANSWER contains a literal "
                        f"angle-bracket placeholder ({pred!r}). Substitute "
                        f"concrete values or emit your best guess. Re-emit FINAL ANSWER."})
                    pred = ""
                    continue
                if (not null_kick_used and a2_round < self.rounds_a2
                        and _egd._is_null_like(pred)):
                    null_kick_used = True
                    tool_log.append({"agent": "agent_2", "round": a2_round,
                                     "kind": "[NULL_KICK]", "payload": pred})
                    msgs_a2.append({"role": "user", "content":
                        f"[harness] You emitted {pred!r}. Literal `null`/"
                        f"`unknown`/`[unavailable]` etc. are scored 0. Pick a "
                        f"concrete best guess and re-emit FINAL ANSWER."})
                    pred = ""
                    continue
                break

            calls = _egt.extract_tool_calls(content)
            if not calls:
                if a2_round < self.rounds_a2:
                    msgs_a2.append({"role": "user", "content":
                        "[harness] You emitted neither a tool call (COMPUTE:) "
                        "nor `FINAL ANSWER:` nor `REQUEST_MORE:`. Do ONE this turn."})
                    tool_log.append({"agent": "agent_2", "round": a2_round,
                                     "kind": "[KICK]"})
                    continue
                break
            remaining = self.budget_a2 - a2_calls
            if remaining <= 0:
                msgs_a2.append({"role": "user", "content":
                    "[harness] agent_2 COMPUTE budget exhausted. Emit FINAL ANSWER now."})
                continue
            tool_msgs = []
            for kind, payload in calls[:remaining]:
                result = _egd.dispatch_a2_tool(kind, payload)
                if kind == "COMPUTE":
                    a2_calls += 1
                tool_log.append({"agent": "agent_2", "round": a2_round,
                                 "kind": kind, "payload": payload[:300],
                                 "result_preview": result[:400]})
                tool_msgs.append(f"=== {kind} result ===\n{result}")
            msgs_a2.append({"role": "user", "content":
                            "\n\n".join(tool_msgs) + "\n\n[harness] Emit FINAL ANSWER now."})

        return self._build_trajectory(
            task=task, pred=pred, in_tok=in_tok, out_tok=out_tok,
            tool_log=tool_log, raw_turns=raw_turns,
            architecture="decentralized",
            num_retrieve_calls=0, handoff=handoff_body,
            a1_rounds=a1_round, a2_rounds=a2_round,
            a1_calls=a1_calls, a2_calls=a2_calls,
            request_more_used=request_more_used,
        )

    # ------------------------------------------------------------------
    # Centralized / hybrid topology
    # ------------------------------------------------------------------

    def _collect_centralized(self, task: dict) -> Trajectory:

        main_name = self.main_agent
        sub_name = self.sub_agents[0] if self.sub_agents else None
        is_hybrid = self.architecture == "hybrid"

        main_system = self.system_prompts[main_name]
        sub_system = self.system_prompts[sub_name] if sub_name else ""
        usr_parts = _eg.render_user_message(task)

        in_tok = out_tok = 0
        tool_log: List[dict] = []
        raw_turns: List[dict] = []
        prior_sub_outputs: List[dict] = []

        def _call(msgs):
            nonlocal in_tok, out_tok
            text, tokens = self._llm(msgs)
            in_tok += tokens.get("input", 0)
            out_tok += tokens.get("output", 0)
            return text

        # Single Researcher invocation in a fresh context (centralized) or
        # with prior dossier history surfaced (hybrid).
        def _invoke_sub(query: str) -> str:
            if not sub_name:
                return "[no sub-agent configured]"
            history_block = ""
            if is_hybrid and prior_sub_outputs:
                hist = []
                for i, prev in enumerate(prior_sub_outputs, 1):
                    hist.append(
                        f"--- prior researcher dossier turn {i} (query: "
                        f"{prev['query'] or '[no query]'}) ---\n{prev['output']}"
                    )
                history_block = (
                    "Prior researcher dossiers on this same question "
                    "(extend or correct as needed):\n"
                    + "\n\n".join(hist) + "\n\n"
                )
            sub_user = list(usr_parts)
            sub_user.append({
                "type": "text",
                "text": (
                    f"\n\n[from main agent (Solver)] {history_block}"
                    f"Focused retrieval request:\n{query.strip() or '[no query]'}\n\n"
                    "Issue SEARCH/BROWSE calls (≤ budget) and end with a "
                    "complete `HANDOFF:` block (HANDOFF_TO_SOLVER)."
                ),
            })
            msgs_sub = [
                {"role": "system", "content": sub_system},
                {"role": "user", "content": sub_user},
            ]
            sub_calls = 0
            dossier = ""
            sub_round = 0
            while sub_round < self.rounds_a1:
                sub_round += 1
                c = _call(msgs_sub)
                raw_turns.append({"agent": sub_name, "round": sub_round, "content": c})
                msgs_sub.append({"role": "assistant", "content": c})
                h = _egd.extract_handoff(c)
                if h:
                    dossier = h
                    break
                calls = _egt.extract_tool_calls(c)
                if not calls:
                    if sub_round < self.rounds_a1:
                        msgs_sub.append({"role": "user", "content":
                            "[harness] Issue SEARCH/BROWSE or emit complete "
                            "`HANDOFF:` block (end with HANDOFF_TO_SOLVER)."})
                        tool_log.append({"agent": sub_name, "round": sub_round,
                                         "kind": "[KICK]"})
                        continue
                    break
                remaining = self.budget_a1 - sub_calls
                if remaining <= 0:
                    msgs_sub.append({"role": "user", "content":
                        "[harness] Sub-agent tool budget exhausted. Emit HANDOFF now."})
                    continue
                tool_msgs = []
                for kind, payload in calls[:remaining]:
                    result = _egd.dispatch_a1_tool(kind, payload)
                    sub_calls += 1
                    tool_log.append({"agent": sub_name, "round": sub_round,
                                     "kind": kind, "payload": payload[:300],
                                     "result_preview": result[:400]})
                    tool_msgs.append(f"=== {kind} result ===\n{result}")
                msgs_sub.append({"role": "user", "content":
                    "\n\n".join(tool_msgs) +
                    "\n\n[harness] Emit the HANDOFF block now (HANDOFF_TO_SOLVER)."})
            if not dossier:
                dossier = (
                    "HANDOFF:\nKEY_FACTS_FOUND: (sub-agent failed to produce "
                    "a complete handoff)\nEVIDENCE_QUOTES: (none)\n"
                    "SUGGESTED_ANSWER: null\nHANDOFF_TO_SOLVER"
                )
            if is_hybrid:
                prior_sub_outputs.append({"query": query, "output": dossier})
            return dossier

        # Main agent (Solver) — gets question + attachment, may emit
        # <retrieve> queries or COMPUTE blocks; ends on FINAL ANSWER.
        main_user = list(usr_parts) + [{
            "type": "text",
            "text": (
                "\n\n[harness] You are the Solver. You DO NOT see search "
                "results directly. To get evidence, emit one "
                "`<retrieve>QUERY</retrieve>` per turn — a Researcher sub-agent "
                "will run SEARCH/BROWSE in isolation and return a HANDOFF dossier. "
                "You may also emit `COMPUTE:` blocks. End with `FINAL ANSWER:`."
            ),
        }]
        msgs_main = [
            {"role": "system", "content": main_system},
            {"role": "user", "content": main_user},
        ]
        n_retrieves = 0
        n_compute = 0
        max_main_rounds = self.rounds_a2 + self.max_retrieves + 2
        pred = ""
        placeholder_kick_used = False
        null_kick_used = False
        main_round = 0
        while main_round < max_main_rounds:
            main_round += 1
            c = _call(msgs_main)
            raw_turns.append({"agent": main_name, "round": main_round, "content": c})
            msgs_main.append({"role": "assistant", "content": c})

            pred_candidate = _egt.parse_final_answer(c)
            if pred_candidate:
                if (not placeholder_kick_used and main_round < max_main_rounds
                        and _egt._has_placeholder(pred_candidate)):
                    placeholder_kick_used = True
                    tool_log.append({"agent": main_name, "round": main_round,
                                     "kind": "[PLACEHOLDER_KICK]",
                                     "payload": pred_candidate})
                    msgs_main.append({"role": "user", "content":
                        "[harness] Your FINAL ANSWER contains a literal "
                        "angle-bracket placeholder. Substitute concrete "
                        "values or emit your best guess. Re-emit FINAL ANSWER."})
                    continue
                if (not null_kick_used and main_round < max_main_rounds
                        and _egd._is_null_like(pred_candidate)):
                    null_kick_used = True
                    tool_log.append({"agent": main_name, "round": main_round,
                                     "kind": "[NULL_KICK]", "payload": pred_candidate})
                    msgs_main.append({"role": "user", "content":
                        f"[harness] You emitted {pred_candidate!r}. Literal "
                        f"`null`/`unknown`/`[unavailable]` etc. are scored 0. "
                        f"Pick a concrete best guess and re-emit FINAL ANSWER."})
                    continue
                pred = pred_candidate
                break

            ret_match = _RETRIEVE_RE.search(c)
            calls = _egt.extract_tool_calls(c)
            # Prefer <retrieve> if it appears before any COMPUTE.
            ret_pos = ret_match.start() if ret_match else None
            comp_pos = None
            if calls:
                # find first COMPUTE position
                m = _egt.COMPUTE_RE.search(c)
                if m:
                    comp_pos = m.start()
            use_retrieve = ret_match is not None and (
                comp_pos is None or (ret_pos is not None and ret_pos <= comp_pos)
            )

            if use_retrieve:
                if n_retrieves >= self.max_retrieves:
                    msgs_main.append({"role": "user", "content":
                        "[harness] retrieve budget exhausted. Use COMPUTE on "
                        "the dossiers you already have, or emit FINAL ANSWER."})
                    continue
                q = ret_match.group(1).strip()
                dossier = _invoke_sub(q)
                n_retrieves += 1
                tool_log.append({"agent": main_name, "round": main_round,
                                 "kind": "RETRIEVE", "payload": q[:300],
                                 "result_preview": dossier[:400]})
                msgs_main.append({"role": "user", "content":
                    f"<retrieve_result>\n{dossier}\n</retrieve_result>"})
                continue

            if calls:
                # Solver only owns COMPUTE.
                remaining = self.budget_a2 - n_compute
                if remaining <= 0:
                    msgs_main.append({"role": "user", "content":
                        "[harness] COMPUTE budget exhausted. Emit FINAL ANSWER now."})
                    continue
                tool_msgs = []
                for kind, payload in calls[:remaining]:
                    result = _egd.dispatch_a2_tool(kind, payload)
                    if kind == "COMPUTE":
                        n_compute += 1
                    tool_log.append({"agent": main_name, "round": main_round,
                                     "kind": kind, "payload": payload[:300],
                                     "result_preview": result[:400]})
                    tool_msgs.append(f"=== {kind} result ===\n{result}")
                msgs_main.append({"role": "user", "content":
                    "\n\n".join(tool_msgs) + "\n\n[harness] Emit FINAL ANSWER or another action."})
                continue

            # Neither final answer nor a recognized action — kick once.
            if main_round < max_main_rounds:
                msgs_main.append({"role": "user", "content":
                    "[harness] Emit one of: `<retrieve>QUERY</retrieve>`, "
                    "a `COMPUTE:` python block, or `FINAL ANSWER: <answer>`."})
                tool_log.append({"agent": main_name, "round": main_round,
                                 "kind": "[KICK]"})
                continue
            break

        return self._build_trajectory(
            task=task, pred=pred, in_tok=in_tok, out_tok=out_tok,
            tool_log=tool_log, raw_turns=raw_turns,
            architecture=self.architecture,
            num_retrieve_calls=n_retrieves,
            handoff=prior_sub_outputs[-1]["output"] if prior_sub_outputs else "",
        )

    # ------------------------------------------------------------------
    # Trajectory + reward
    # ------------------------------------------------------------------

    def _build_trajectory(
        self,
        *,
        task: dict,
        pred: str,
        in_tok: int,
        out_tok: int,
        tool_log: list,
        raw_turns: list,
        architecture: str,
        num_retrieve_calls: int = 0,
        handoff: str = "",
        a1_rounds: int = 0,
        a2_rounds: int = 0,
        a1_calls: int = 0,
        a2_calls: int = 0,
        request_more_used: bool = False,
    ) -> Trajectory:
        gold = task.get("Final answer", task.get("ground_truth", "")) or ""
        ok = bool(_egt.is_correct(pred, gold)) if pred else False
        # Build a single aggregate step so eval_topology's token sum works.
        steps = [{
            "agent": "gaia_rollout",
            "agent_id": "gaia_rollout",
            "output": pred,
            "action": pred,
            "tokens": {"input": in_tok, "output": out_tok},
            "tool_log": tool_log,
            "raw_turns": raw_turns,
        }]
        metadata = {
            "task_type": "gaia",
            "category": task.get("Level"),  # use Level as category for grouping
            "final_answer": pred,
            "gold": gold,
            "architecture": architecture,
            "num_retrieve_calls": num_retrieve_calls,
            "qa_metrics": {
                "em": 1.0 if ok else 0.0,
                "f1": 1.0 if ok else 0.0,
                "bleu": 1.0 if ok else 0.0,
            },
            "handoff": handoff[:1000],
            "a1_rounds": a1_rounds, "a2_rounds": a2_rounds,
            "a1_calls": a1_calls, "a2_calls": a2_calls,
            "request_more_used": request_more_used,
        }
        traj = Trajectory(task=task, steps=steps, reward=1.0 if ok else 0.0,
                          metadata=metadata)
        return traj
