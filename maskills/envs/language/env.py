"""Language task environment adapter for MASkills.

Wraps benchmark datasets (HotPotQA, MATH, HumanEval) into the unified
BaseEnvironment interface.

Implements sequential N-agent collaboration via a shared message pool:
    Task Question -> Agent 1 -> Agent 2 -> ... -> Agent N (Final Answer)

Optional per-agent search loop: when ``max_search_calls > 0``, each agent's
turn becomes a multi-step conversation where the agent can emit
``<search>query</search>`` and receive wiki results privately before
producing its contribution to the shared pool.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from maskills.core.base import BaseEnvironment, Trajectory
from maskills.core.policy import AgentPolicy, default_agent_policy
from maskills.core.skills import build_skill_trace, load_skills_dir, render_skill_library
from maskills.envs import register_env
from maskills.llm.client import LLMClient

from .reward import VerifiedRewardGenerator
from .search_tool import _TOOL_RE, parse_tool_call, run_tool
from .task_loader import TaskLoader

_TOOLS_DIR = Path(__file__).parent / "tools"

# Centralized-mode "retrieve" pseudo-tool — the main agent emits
# ``<retrieve>QUERY</retrieve>`` and the env dispatches a fresh sub-agent
# turn in an isolated context.  Kept separate from search_tool's TOOL_NAMES
# so decentralized mode treats stray ``<retrieve>`` tags as plain text.
_RETRIEVE_RE = re.compile(r"<retrieve>(.*?)</retrieve>", re.DOTALL | re.IGNORECASE)


@register_env("language")
class LanguageTaskEnv(BaseEnvironment):
    """Environment for language benchmark tasks (QA, Math, Coding).

    Delegates to benchmark directories for task data.
    """

    def __init__(self, config):
        """
        Args:
            config: LanguageTaskConfig with task_type, benchmark_path, llm, num_agents.
        """
        self.task_type = getattr(config, "task_type", "qa")
        self.num_agents = config.num_agents
        self.agent_names = [f"agent_{i + 1}" for i in range(self.num_agents)]

        # Multi-agent architecture:
        #   "decentralized" — sequential, shared message pool
        #   "centralized"   — main agent invokes sub-agents via <retrieve> in
        #                     isolated contexts (each sub call is fresh)
        #   "hybrid"        — same delegation flow as centralized but each
        #                     sub-agent invocation sees the prior sub-agent
        #                     outputs (sub-agents accumulate state).
        self.architecture = getattr(config, "architecture", "decentralized")
        self.main_agent = getattr(config, "main_agent", "agent_1")
        if self.architecture in ("centralized", "hybrid"):
            if self.main_agent not in self.agent_names:
                raise ValueError(
                    f"main_agent={self.main_agent!r} is not one of "
                    f"agent_names={self.agent_names!r}"
                )
            self.sub_agents = [a for a in self.agent_names if a != self.main_agent]
        else:
            self.sub_agents = []

        llm = getattr(config, "actor_llm", None) or config.llm
        self.llm_client = LLMClient(llm)

        # Reward evaluator
        judge_model = getattr(config, "judge_model", "openai/gpt-5.1")
        code_timeout = getattr(config, "code_timeout", 10.0)
        self.reward_gen = VerifiedRewardGenerator(
            judge_model=judge_model, code_timeout=code_timeout,
            qa_reward_metric=getattr(config, "qa_reward_metric", "f1"),
        )
        self.reward_gen.set_client(self.llm_client.raw_client)

        # Search tool settings.  ``include_context`` is the forward-compatible
        # flag (True = gold HotpotQA context passages are injected into the
        # prompt); the deprecated ``hide_context`` is honoured for back-compat.
        if hasattr(config, "include_context"):
            self.include_context = bool(getattr(config, "include_context"))
        else:
            self.include_context = not bool(getattr(config, "hide_context", False))
        self.max_search_calls = int(getattr(config, "max_search_calls", 0))
        self.search_limit = int(getattr(config, "search_limit", 5))

        # Whether the two non-trainable prompt blocks are appended to each
        # agent's system prompt at rollout time.  ``get_skill_library`` /
        # ``get_tool_library`` still return their content unconditionally
        # so the critic and optimizer can see them.
        self.inject_skill_library = bool(getattr(config, "inject_skill_library", True))
        self.inject_tool_reference = bool(getattr(config, "inject_tool_reference", True))

        # Human-authored skill library (Anthropic SKILL.md format).
        #
        # Two loading modes:
        #   * Global (legacy): ``skills_dir`` -> one library, injected into
        #     every agent's prompt.
        #   * Per-agent (new): ``agent_skills_dirs`` maps agent_name to its
        #     own directory; only that agent sees those skills.  Agents not
        #     listed fall back to the global library (or none).
        #
        # The library is fixed (non-trainable) and invisible to the optimizer.
        skills_dir = getattr(config, "skills_dir", None)
        global_skills = load_skills_dir(skills_dir) if skills_dir else []
        self._global_skill_library_block = render_skill_library(global_skills)

        agent_skills_dirs = getattr(config, "agent_skills_dirs", None) or {}
        self._agent_skill_library_blocks: dict[str, str] = {}
        agent_skills_lists: dict[str, list] = {}
        for agent_name, dir_path in agent_skills_dirs.items():
            agent_skills = load_skills_dir(dir_path) if dir_path else []
            agent_skills_lists[agent_name] = agent_skills
            self._agent_skill_library_blocks[agent_name] = render_skill_library(agent_skills)

        # ``human_skills`` exposes the union of every loaded skill so
        # ``get_skill_library()`` keeps returning a useful list for any
        # callers (critic / optimizer) that ask for it.  Deduplicated by
        # ``Skill.name``.
        merged: dict[str, object] = {}
        for s in global_skills:
            merged[s.name] = s
        for skills in agent_skills_lists.values():
            for s in skills:
                merged.setdefault(s.name, s)
        self.human_skills = list(merged.values())
        # Backward-compatible alias for any external code that read it.
        self._skill_library_block = self._global_skill_library_block

        # Per-agent tool budgets and whitelists.
        agent_max_tool_calls = getattr(config, "agent_max_tool_calls", None) or {}
        agent_allowed_tools = getattr(config, "agent_allowed_tools", None) or {}
        self._agent_max_tool_calls: dict[str, int] = {}
        self._agent_allowed_tools: dict[str, Optional[set]] = {}
        for agent_name in self.agent_names:
            self._agent_max_tool_calls[agent_name] = int(
                agent_max_tool_calls.get(agent_name, self.max_search_calls)
            )
            if agent_name in agent_allowed_tools:
                self._agent_allowed_tools[agent_name] = {
                    t.lower() for t in agent_allowed_tools[agent_name]
                }
            else:
                self._agent_allowed_tools[agent_name] = None  # None = no restriction

        # Tool reference block — the markdown under ``tools/`` that
        # documents the actual tag protocol (``<search>``, ``<grep>``,
        # ``<sympy>``).  Only injected when the tool loop is enabled, so
        # we don't tease the agent with tools it cannot call.  The block
        # is also filtered per-agent so an agent only sees docs for the
        # tools it is actually allowed to invoke.
        self._tool_reference_block = self._build_tool_reference_block()
        self._agent_tool_reference_blocks: dict[str, str] = {
            name: self._build_tool_reference_block(
                allowed=self._agent_allowed_tools[name],
                max_calls=self._agent_max_tool_calls[name],
            )
            for name in self.agent_names
        }

        # Task data
        benchmark_path = getattr(config, "benchmark_path", "")
        data_limit = getattr(config, "data_limit", None)
        train_test_split = getattr(config, "train_test_split", 1.0)
        split_seed = getattr(config, "split_seed", 42)
        n_train = getattr(config, "n_train", None)
        n_val = getattr(config, "n_val", None)
        n_test = getattr(config, "n_test", None)
        if any(n is not None for n in (n_train, n_val, n_test)):
            needed = (n_train or 0) + (n_val or 0) + (n_test or 0)
            if data_limit is None or data_limit < needed:
                data_limit = needed
        if benchmark_path:
            self.task_loader = TaskLoader(
                self.task_type,
                benchmark_path,
                data_limit=data_limit,
                train_test_split=train_test_split,
                split_seed=split_seed,
                n_train=n_train,
                n_val=n_val,
                n_test=n_test,
            )
        else:
            self.task_loader = None

        self.logger = logging.getLogger(__name__)

    def reset(self, task: dict) -> dict:
        return {"task": task}

    def step(self, agent_id: str, action: str) -> tuple[dict, float, bool, dict]:
        return {}, 0.0, False, {}

    def collect_trajectory(self, policies: dict, task: dict) -> Trajectory:
        """Run a single task through the configured multi-agent architecture.

        Dispatches to ``_collect_trajectory_decentralized`` (sequential
        deliberation with a shared message pool) or
        ``_collect_trajectory_centralized`` (main agent reasons, sub-agents
        invoked as tools in isolated contexts) based on
        ``self.architecture``.
        """
        if self.architecture in ("centralized", "hybrid"):
            return self._collect_trajectory_centralized(policies, task)
        return self._collect_trajectory_decentralized(policies, task)

    def _collect_trajectory_decentralized(self, policies: dict, task: dict) -> Trajectory:
        """Original behaviour: N agents speak in sequence into a shared pool."""
        task_prompt = self._get_task_prompt(task)

        steps = []
        message_pool = []

        for i, agent_name in enumerate(self.agent_names):
            is_last = i == self.num_agents - 1

            policy = policies.get(
                agent_name,
                default_agent_policy(i, self.num_agents),
            )
            if isinstance(policy, str):
                policy = AgentPolicy.from_legacy(policy)
            agent_system = self._compose_system_prompt(policy, agent_name=agent_name)

            # Build initial user input: task + all prior-agent messages in the pool
            user_input = f"Task:\n{task_prompt}"
            for msg in message_pool:
                display = msg["agent"].replace("_", " ").title()
                user_input += f"\n\n{display}'s Response:\n{msg['content']}"
            if is_last:
                user_input += "\n\nNow provide your FINAL answer:"

            response, tokens, search_log = self._run_agent_turn(
                agent_system, user_input, agent_name=agent_name,
            )

            # Strip any residual tool tags from the pool-visible output
            pool_text = _TOOL_RE.sub("", response).strip() or response

            steps.append({
                "agent": agent_name,
                "agent_id": agent_name,
                "system_prompt": agent_system,
                "input": user_input,
                "output": pool_text,
                "action": pool_text,
                "tokens": tokens,
                "search_log": search_log,
            })

            message_pool.append({"agent": agent_name, "content": pool_text})

        final_answer = message_pool[-1]["content"]

        # Build trajectory with placeholder reward, then compute verified reward
        metadata = {
            "task_type": self.task_type,
            "final_answer": final_answer,
            "architecture": "decentralized",
        }
        trajectory = Trajectory(task=task, steps=steps, reward=0.0, metadata=metadata)
        reward = self.reward_gen.compute(trajectory)
        trajectory.reward = reward
        metadata["evaluation_feedback"] = f"reward={reward}"
        trajectory.skill_trace = build_skill_trace(steps, policies)

        return trajectory

    def _collect_trajectory_centralized(self, policies: dict, task: dict) -> Trajectory:
        """Centralized orchestration.

        ``main_agent`` (default ``agent_1``) is the reasoner.  It sees the
        QUESTION ONLY (no context passages) and a ``<retrieve>QUERY</retrieve>``
        tool that delegates to a sub-agent in an isolated context.  Each
        sub-agent invocation runs ``_run_agent_turn`` with the FULL task
        (question + context) and the sub-agent's own skills/tools; only
        the sub-agent's final response is returned to main as a
        ``<retrieve_result>`` block.

        The main agent's last text response (with all tool tags stripped)
        is the final answer.
        """
        main_name = self.main_agent
        main_idx = self.agent_names.index(main_name)
        sub_name = self.sub_agents[0] if self.sub_agents else None

        main_policy = policies.get(
            main_name, default_agent_policy(main_idx, self.num_agents),
        )
        if isinstance(main_policy, str):
            main_policy = AgentPolicy.from_legacy(main_policy)

        sub_policy = None
        if sub_name is not None:
            sub_idx = self.agent_names.index(sub_name)
            sub_policy = policies.get(
                sub_name, default_agent_policy(sub_idx, self.num_agents),
            )
            if isinstance(sub_policy, str):
                sub_policy = AgentPolicy.from_legacy(sub_policy)

        main_system = self._compose_system_prompt(main_policy, agent_name=main_name)
        # Main only sees the question — context isolation is the whole point.
        question = task.get("question", task.get("problem", ""))
        if self.task_type == "qa":
            main_user_input = (
                f"Task:\nQuestion: {question}\n\n"
                "Provide your FINAL answer.  Use <retrieve>QUERY</retrieve> "
                "if you need cited evidence from the underlying passages."
            )
        else:
            main_user_input = f"Task:\n{question}\n\nProvide your FINAL answer."

        steps: list[dict] = []
        sub_step_records: list[dict] = []
        # In hybrid mode, sub-agents pool their prior findings so each
        # successive <retrieve> can build on (or correct) what came before.
        prior_sub_outputs: list[dict] = []
        is_hybrid = self.architecture == "hybrid"

        def _invoke_sub(query: str) -> tuple[str, dict, list]:
            if sub_policy is None or sub_name is None:
                return (
                    "Tool error: no sub-agent is configured. "
                    "Configure num_agents>=2 to enable <retrieve>.",
                    {"input": 0, "output": 0},
                    [],
                )
            sub_system = self._compose_system_prompt(sub_policy, agent_name=sub_name)
            full_task_prompt = self._get_task_prompt(task)
            sub_user_input = f"Task:\n{full_task_prompt}"
            if is_hybrid and prior_sub_outputs:
                # Surface prior sub-agent turns so this invocation can use,
                # correct, or extend them rather than restarting from scratch.
                history_parts = []
                for i, prev in enumerate(prior_sub_outputs, 1):
                    history_parts.append(
                        f"--- prior sub-agent turn {i} (query: "
                        f"{prev['query'] or '[no query]'}) ---\n{prev['output']}"
                    )
                sub_user_input += (
                    "\n\nPrior sub-agent findings on this same task "
                    "(extend or correct as needed):\n"
                    + "\n\n".join(history_parts)
                )
            if query:
                sub_user_input += (
                    "\n\nThe main agent is asking you to focus on:\n" + query
                )
            response, tokens, sub_log = self._run_agent_turn(
                sub_system, sub_user_input, agent_name=sub_name,
            )
            response_clean = _TOOL_RE.sub("", response).strip() or response
            sub_step_records.append({
                "agent": sub_name,
                "agent_id": sub_name,
                "system_prompt": sub_system,
                "input": sub_user_input,
                "output": response_clean,
                "action": response_clean,
                "tokens": tokens,
                "search_log": sub_log,
                "main_query": query,
                "isolated_context": not is_hybrid,
                "shared_sub_context": is_hybrid,
            })
            if is_hybrid:
                prior_sub_outputs.append({"query": query, "output": response_clean})
            return response_clean, tokens, sub_log

        main_text, main_tokens, main_log = self._run_main_with_retrieval(
            system_prompt=main_system,
            user_input=main_user_input,
            agent_name=main_name,
            invoke_sub=_invoke_sub,
        )

        # Steps order: sub-agent invocations (in time order), then main's final.
        steps.extend(sub_step_records)
        main_clean = _TOOL_RE.sub("", main_text)
        # Also strip retrieve tags from the final visible output.
        main_clean = _RETRIEVE_RE.sub("", main_clean).strip() or main_text
        steps.append({
            "agent": main_name,
            "agent_id": main_name,
            "system_prompt": main_system,
            "input": main_user_input,
            "output": main_clean,
            "action": main_clean,
            "tokens": main_tokens,
            "search_log": main_log,
        })

        metadata = {
            "task_type": self.task_type,
            "final_answer": main_clean,
            "architecture": self.architecture,
            "main_agent": main_name,
            "sub_agents": list(self.sub_agents),
            "num_retrieve_calls": sum(
                1 for entry in main_log if entry.get("tool") == "retrieve"
            ),
        }
        trajectory = Trajectory(task=task, steps=steps, reward=0.0, metadata=metadata)
        reward = self.reward_gen.compute(trajectory)
        trajectory.reward = reward
        metadata["evaluation_feedback"] = f"reward={reward}"
        trajectory.skill_trace = build_skill_trace(steps, policies)
        return trajectory

    def _run_main_with_retrieval(
        self,
        system_prompt: str,
        user_input: str,
        agent_name: str,
        invoke_sub,
    ):
        """Main-agent turn loop in centralized mode.

        Each iteration the main agent emits one of:
          * ``<retrieve>QUERY</retrieve>`` → ``invoke_sub(QUERY)`` runs the
            sub-agent in an isolated context; its output comes back as
            ``<retrieve_result>...</retrieve_result>``.
          * Any other allowed tool tag (search/grep/sympy) → standard
            tool dispatch via ``run_tool`` with per-agent budget/whitelist.
          * No tool tag → loop ends; the text is the final answer.

        ``<retrieve>`` and other tool calls share the same per-agent budget
        (``self._agent_max_tool_calls[main_name]``).
        """
        max_calls = self._agent_max_tool_calls.get(
            agent_name, self.max_search_calls,
        )
        allowed = self._agent_allowed_tools.get(agent_name)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        total = {"input": 0, "output": 0}
        main_log: list[dict] = []
        last_search_result = ""

        if max_calls <= 0:
            text, tokens = self.llm_client.chat_messages_with_usage(messages)
            total["input"] += tokens.get("input", 0)
            total["output"] += tokens.get("output", 0)
            return text, total, main_log

        for _ in range(max_calls + 1):
            text, tokens = self.llm_client.chat_messages_with_usage(messages)
            total["input"] += tokens.get("input", 0)
            total["output"] += tokens.get("output", 0)

            retrieve_match = _RETRIEVE_RE.search(text)
            tool_match = parse_tool_call(text)

            # Decide which tag fires this turn — whichever appears first in
            # the response wins.  Ties go to retrieve (it is the centralized
            # mode's primary tool).
            retrieve_pos = retrieve_match.start() if retrieve_match else None
            tool_pos = None
            if tool_match is not None:
                from .search_tool import _TOOL_RE as _SEARCH_TOOL_RE
                m = _SEARCH_TOOL_RE.search(text)
                tool_pos = m.start() if m else None

            use_retrieve = retrieve_match is not None and (
                tool_pos is None or retrieve_pos <= tool_pos
            )

            if not use_retrieve and tool_match is None:
                return text, total, main_log

            if len(main_log) >= max_calls:
                return text, total, main_log

            if use_retrieve:
                if allowed is not None and "retrieve" not in allowed:
                    error_msg = (
                        f"Tool 'retrieve' is not enabled for {agent_name}."
                    )
                    main_log.append({
                        "tool": "retrieve",
                        "query": retrieve_match.group(1).strip(),
                        "result": f"Tool error: {error_msg}",
                    })
                    messages.append({"role": "assistant", "content": text})
                    messages.append({
                        "role": "user",
                        "content": f"<tool_error>\n{error_msg}\n</tool_error>",
                    })
                    continue

                query = retrieve_match.group(1).strip()
                sub_response, _sub_tokens, _sub_log = invoke_sub(query)
                main_log.append({
                    "tool": "retrieve",
                    "query": query,
                    "result": sub_response,
                })
                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": f"<retrieve_result>\n{sub_response}\n</retrieve_result>",
                })
                continue

            # Regular tool call (search/grep/sympy).
            tool_name, payload = tool_match
            if allowed is not None and tool_name not in allowed:
                allowed_str = ", ".join(sorted(allowed)) or "none"
                error_msg = (
                    f"Tool '{tool_name}' is not enabled for {agent_name}. "
                    f"Allowed tools: {allowed_str}."
                )
                main_log.append({
                    "tool": tool_name,
                    "query": payload,
                    "result": f"Tool error: {error_msg}",
                })
                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": f"<tool_error>\n{error_msg}\n</tool_error>",
                })
                continue

            result_block = run_tool(
                tool_name,
                payload,
                search_limit=self.search_limit,
                last_search_result=last_search_result,
            )
            if tool_name == "search":
                last_search_result = result_block
            main_log.append({
                "tool": tool_name,
                "query": payload,
                "result": result_block,
            })
            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": (
                    f"<{tool_name}_result>\n{result_block}\n</{tool_name}_result>"
                ),
            })

        return text, total, main_log

    # ------------------------------------------------------------------
    # Per-agent multi-step loop (private search tool calls)
    # ------------------------------------------------------------------

    def _run_agent_turn(
        self,
        system_prompt: str,
        user_input: str,
        agent_name: Optional[str] = None,
    ):
        """Run one agent's turn, allowing up to ``max_search_calls`` tool calls.

        Returns (final_response_text, aggregated_tokens, search_log).
        ``search_log`` records every tool call (search, grep, sympy) as a
        dict ``{"tool", "query", "result"}`` in this agent's private trace
        only; it is not added to the shared pool.

        ``max_search_calls`` is reused as the total tool-call budget so we
        do not silently break callers / configs that only know about the
        original wiki-search loop.

        When ``agent_name`` matches an entry in ``agent_max_tool_calls`` or
        ``agent_allowed_tools`` (config), the per-agent overrides apply
        instead of the global defaults.  A disallowed tool call is reported
        back to the agent as a ``Tool error`` block so it can recover.
        """
        max_calls = self._agent_max_tool_calls.get(
            agent_name, self.max_search_calls,
        ) if agent_name else self.max_search_calls
        allowed = self._agent_allowed_tools.get(agent_name) if agent_name else None

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        total = {"input": 0, "output": 0}
        search_log: list[dict] = []
        last_search_result = ""

        if max_calls <= 0:
            text, tokens = self.llm_client.chat_messages_with_usage(messages)
            total["input"] += tokens.get("input", 0)
            total["output"] += tokens.get("output", 0)
            return text, total, search_log

        for _ in range(max_calls + 1):
            text, tokens = self.llm_client.chat_messages_with_usage(messages)
            total["input"] += tokens.get("input", 0)
            total["output"] += tokens.get("output", 0)

            call = parse_tool_call(text)
            if call is None or len(search_log) >= max_calls:
                return text, total, search_log

            tool_name, payload = call

            if allowed is not None and tool_name not in allowed:
                allowed_str = ", ".join(sorted(allowed)) or "none"
                error_msg = (
                    f"Tool '{tool_name}' is not enabled for {agent_name}. "
                    f"Allowed tools: {allowed_str}.  Either retry with an "
                    f"allowed tool or finish your turn without a tool tag."
                )
                search_log.append({
                    "tool": tool_name,
                    "query": payload,
                    "result": f"Tool error: {error_msg}",
                })
                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": f"<tool_error>\n{error_msg}\n</tool_error>",
                })
                continue

            result_block = run_tool(
                tool_name,
                payload,
                search_limit=self.search_limit,
                last_search_result=last_search_result,
            )
            if tool_name == "search":
                last_search_result = result_block
            search_log.append({
                "tool": tool_name,
                "query": payload,
                "result": result_block,
            })
            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": (
                    f"<{tool_name}_result>\n{result_block}\n</{tool_name}_result>"
                ),
            })

        return text, total, search_log

    def _compose_system_prompt(
        self,
        policy: AgentPolicy,
        agent_name: Optional[str] = None,
    ) -> str:
        """Compose the agent's full system prompt.

        Layout::

            <role>

            ## Skills                       <- trainable (from policy)
            <agent's learned skills>

            ## Skill Library                <- fixed human-authored skills
            <rendered SKILL.md library; per-agent if configured, else global>

            ## Tool Reference               <- fixed tool tag docs
            <docs from maskills/envs/language/tools/*.md, filtered to allowed tools>

        The trainable half (``role`` + ``skills``) is what the optimizer
        sees.  The skill library and tool reference are appended only at
        rollout time, so gradients never touch them.  The tool reference
        is the *agent-facing* counterpart to ``get_tool_library()`` (the
        latter feeds the critic/optimizer).
        """
        base = policy.combined
        extras = []
        if self.inject_skill_library:
            block = ""
            if agent_name and agent_name in self._agent_skill_library_blocks:
                block = self._agent_skill_library_blocks[agent_name]
            else:
                block = self._global_skill_library_block
            if block:
                extras.append(block)
        if self.inject_tool_reference:
            ref_block = ""
            if agent_name and agent_name in self._agent_tool_reference_blocks:
                ref_block = self._agent_tool_reference_blocks[agent_name]
            else:
                ref_block = self._tool_reference_block
            if ref_block:
                extras.append(ref_block)
        if not extras:
            return base
        return base.rstrip() + "\n\n" + "\n\n".join(extras)

    def _build_tool_reference_block(
        self,
        allowed: Optional[set] = None,
        max_calls: Optional[int] = None,
    ) -> str:
        """Concatenate tool-doc markdown for inclusion in the agent prompt.

        ``allowed``: when not None, restrict the docs to tools whose name
        appears in the set (file names match ``<name>.md``; ``search_wiki.md``
        documents the ``search`` tool).
        ``max_calls``: per-agent budget; when 0 the block is omitted entirely
        (no point telling the agent about tags the env will not execute).
        Falls back to ``self.max_search_calls`` when None.
        """
        budget = max_calls if max_calls is not None else self.max_search_calls
        if budget <= 0:
            return ""
        if not _TOOLS_DIR.is_dir():
            return ""
        # Only show docs for tools the agent is allowed to invoke.
        # Filename → tool-name mapping: ``search_wiki.md`` documents
        # the ``search`` tool, others use ``<tool>.md``.
        def _tool_name_for(path: Path) -> str:
            stem = path.stem
            return "search" if stem == "search_wiki" else stem

        chunks = []
        for path in sorted(_TOOLS_DIR.glob("*.md")):
            tname = _tool_name_for(path)
            if allowed is not None and tname not in allowed:
                continue
            try:
                chunks.append(path.read_text())
            except OSError:
                continue
        if not chunks:
            return ""
        body = "\n\n".join(chunks).strip()
        return "## Tool Reference\n\n" + body

    def get_skill_library(self) -> list:
        """Return the loaded human-authored skills (list of Skill)."""
        return list(self.human_skills)

    def get_tool_library(self) -> str:
        """Return the concatenated tool-library markdown for this env.

        Returns an empty string when the tool loop is disabled so that
        downstream components (optimizer, critic) do not receive stale
        tool info in a no-tools run.
        """
        if self.max_search_calls <= 0:
            return ""
        if not _TOOLS_DIR.is_dir():
            return ""
        chunks = []
        for path in sorted(_TOOLS_DIR.glob("*.md")):
            try:
                chunks.append(path.read_text())
            except OSError:
                continue
        return "\n\n".join(chunks).strip()

    def sample_tasks(self, num_samples: int, seed: Optional[int] = None, split: str = "train") -> list[dict]:
        if self.task_loader:
            return self.task_loader.sample_tasks(num_samples, seed=seed, split=split)
        raise ValueError("No task_loader available. Provide benchmark_path in config.")

    def _get_task_prompt(self, task: dict) -> str:
        if self.task_loader and self.include_context:
            return self.task_loader.get_task_prompt(task)

        question = task.get("question", task.get("problem", ""))
        if not self.include_context:
            if self.task_type == "qa":
                return f"Question: {question}"
            if self.task_type == "math":
                return f"Problem: {question}"
            return question

        context = task.get("context", "")
        if context:
            return f"Context: {context}\n\nQuestion: {question}"
        return question
