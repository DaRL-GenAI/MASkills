"""LOCOMO environment adapter for MASkills.

Wraps the LOCOMO long-term conversational memory benchmark
(``env/locomo/data/locomo10.json``) as a multi-agent MASkills env so
``skills.md`` for each agent can be trained via the Monte-Carlo policy
gradient loop.

Recommended multi-agent layouts:

* ``num_agents=2`` (decentralized):
    - ``agent_1`` = **retriever** — sees the full (token-budgeted)
      conversation, the question and category; emits the dialog ID(s)
      and short verbatim excerpts that contain the evidence.
    - ``agent_2`` = **reasoner** — sees ONLY the question, category, and
      the retriever's evidence excerpts; emits the FINAL answer in the
      LOCOMO-expected format (concise string, date for cat 2, comma list
      for cat 1, ``not mentioned`` for cat 5).

* ``num_agents=3`` (decentralized):
    - Add ``agent_3`` = **verifier** that audits the reasoner's answer
      against the evidence and either confirms or rewrites it.  The
      verifier's text is the final answer.

* ``architecture="centralized"`` (with num_agents>=2):
    - ``main_agent`` (default ``agent_2``) reasons and may emit
      ``<retrieve>QUERY</retrieve>`` to delegate retrieval to a
      sub-agent that sees the full conversation.  Main's final text is
      the answer.

Reward is the upstream LOCOMO score for the question's category
(``maskills/envs/locomo/reward.py``).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from maskills.core.base import BaseEnvironment, Trajectory
from maskills.core.policy import AgentPolicy, default_agent_policy
from maskills.core.skills import build_skill_trace, load_skills_dir, render_skill_library
from maskills.envs import register_env
from maskills.llm.client import LLMClient

# ``run_grep`` ships with the language env; LOCOMO reuses it so the regex
# semantics (case-insensitive, max-line-count, error formatting) stay
# consistent across envs.
from ..language.search_tool import run_grep
from .reward import LocomoRewardGenerator
from .task_loader import LocomoTaskLoader

_RETRIEVE_RE = re.compile(r"<retrieve>(.*?)</retrieve>", re.DOTALL | re.IGNORECASE)
_GREP_RE = re.compile(r"<grep>(.*?)</grep>", re.DOTALL | re.IGNORECASE)

_CATEGORY_HINTS = {
    1: (
        "Multi-hop: chain facts across turns/sessions. The gold answer "
        "may be a comma-separated list — match its structure (each item a "
        "short noun-phrase, no ``not mentioned`` filler)."
    ),
    2: (
        "Temporal: the answer is a SINGLE date or year (e.g. ``7 May 2023``, "
        "``2023``). Pick one — do NOT list multiple candidates. If the cited "
        "turn has no explicit date, fall back to its session date stamp."
    ),
    3: (
        "Open-domain reasoning: this is an INFERENCE question (e.g. "
        "``would/could/how old/did they …``) — the answer is not a single "
        "quoted span. The retriever must gather ALL turns that bear on it "
        "(direct and contextual); the reasoner infers a reasoned judgment "
        "from that evidence plus commonsense, in the gold answer's style "
        "(a short claim, often with a brief ``since/because`` justification)."
    ),
    4: (
        "Open-domain single-hop: short phrase answer grounded in one "
        "specific turn."
    ),
    5: (
        "Adversarial: the question may not be answerable from the "
        "conversation. ONLY in this category, if the retriever returns "
        "``NO_EVIDENCE`` or the excerpts do not actually answer the "
        "question, output exactly ``not mentioned``. Never use this "
        "fallback for categories 1-4."
    ),
}

# Human-readable category names, surfaced to the agents in the user prompt
# so each agent knows the question type explicitly.
_CATEGORY_NAMES = {
    1: "multi-hop", 2: "temporal", 3: "open-domain reasoning",
    4: "single-hop", 5: "adversarial",
}


@register_env("locomo")
class LocomoEnv(BaseEnvironment):
    """Multi-agent collaborative QA over LOCOMO conversations."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config):
        self.num_agents = int(getattr(config, "num_agents", 2))
        if self.num_agents < 1:
            raise ValueError("LocomoEnv requires num_agents >= 1")
        self.agent_names = [f"agent_{i + 1}" for i in range(self.num_agents)]

        # Architecture modes:
        #   "decentralized" — retriever -> reasoner [-> verifier] sequential
        #   "centralized"   — reasoner main delegates retrieval via <retrieve>
        #                     in an isolated sub context per call
        #   "hybrid"        — same delegation flow as centralized but each
        #                     sub-agent invocation sees prior sub outputs
        self.architecture = getattr(config, "architecture", "decentralized")
        self.main_agent = getattr(config, "main_agent", "agent_2")
        if self.architecture in ("centralized", "hybrid"):
            if self.main_agent not in self.agent_names:
                # Fall back to the last agent if the configured main is missing.
                self.main_agent = self.agent_names[-1]
            self.sub_agents = [a for a in self.agent_names if a != self.main_agent]
        else:
            self.sub_agents = []

        # Conversation truncation
        self.max_context_tokens = int(getattr(config, "max_context_tokens", 12000))
        self.chars_per_token = int(getattr(config, "chars_per_token", 4))
        self._char_budget = self.max_context_tokens * self.chars_per_token

        # When False, the retriever / solo agent's user prompt carries only
        # a compact session index (not the full transcript); the agent must
        # call ``<grep>`` to fetch any turn content.  Cuts actor input by
        # ~75% per retriever rollout (no more N× conversation duplication
        # across grep-loop rounds).  Default True for backward compat.
        self.retriever_sees_conversation = bool(
            getattr(config, "retriever_sees_conversation", True)
        )

        # Grep tool.  ``max_grep_calls > 0`` enables a multi-turn loop on the
        # agents that see the full conversation (retriever / solo / sub-agent
        # in centralized mode).  Each turn the agent may emit one
        # ``<grep>PATTERN</grep>`` and gets the matching transcript lines
        # back as ``<grep_result>...</grep_result>``; the loop ends on the
        # first response without a grep tag or when the budget is exhausted.
        self.max_grep_calls = int(getattr(config, "max_grep_calls", 0))
        self.grep_max_lines = int(getattr(config, "grep_max_lines", 20))

        # Actor LLM
        llm = getattr(config, "actor_llm", None) or config.llm
        if llm is None:
            raise ValueError("LocomoConfig.llm (or actor_llm) must be set")
        self.llm_client = LLMClient(llm)

        # Reward
        self.reward_gen = LocomoRewardGenerator(
            reward_metric=getattr(config, "locomo_reward_metric", "f1_bleu_mean"),
        )

        # Optional skill library
        self.inject_skill_library = bool(getattr(config, "inject_skill_library", True))
        skills_dir = getattr(config, "skills_dir", None)
        global_skills = load_skills_dir(skills_dir) if skills_dir else []
        self._global_skill_library_block = render_skill_library(global_skills)
        agent_skills_dirs = getattr(config, "agent_skills_dirs", None) or {}
        self._agent_skill_library_blocks: Dict[str, str] = {}
        agent_skills_lists: Dict[str, list] = {}
        for agent_name, dir_path in agent_skills_dirs.items():
            agent_skills = load_skills_dir(dir_path) if dir_path else []
            agent_skills_lists[agent_name] = agent_skills
            self._agent_skill_library_blocks[agent_name] = render_skill_library(agent_skills)
        merged: dict = {}
        for s in global_skills:
            merged[s.name] = s
        for skills in agent_skills_lists.values():
            for s in skills:
                merged.setdefault(s.name, s)
        self.human_skills = list(merged.values())

        # Default role prompts loaded from prompts/ if present, so the
        # retriever / reasoner / verifier roles are explicit even without
        # any trained checkpoint.
        self._role_overrides = self._load_role_overrides()

        # Task data
        benchmark_path = getattr(config, "benchmark_path", "")
        data_limit = getattr(config, "data_limit", None)
        train_test_split = getattr(config, "train_test_split", 1.0)
        split_seed = getattr(config, "split_seed", 42)
        category_filter = getattr(config, "category_filter", None)
        if benchmark_path:
            self.task_loader = LocomoTaskLoader(
                benchmark_path,
                data_limit=data_limit,
                train_test_split=train_test_split,
                split_seed=split_seed,
                category_filter=category_filter,
            )
        else:
            self.task_loader = None

        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # BaseEnvironment interface
    # ------------------------------------------------------------------

    def reset(self, task: dict) -> dict:
        return {"task": task}

    def step(self, agent_id: str, action: str):
        # Language envs orchestrate everything inside collect_trajectory;
        # step is unused but required by BaseEnvironment.
        return {}, 0.0, False, {}

    def collect_trajectory(self, policies: dict, task: dict) -> Trajectory:
        if self.architecture in ("centralized", "hybrid") and self.num_agents >= 2:
            return self._collect_centralized(policies, task)
        return self._collect_decentralized(policies, task)

    def sample_tasks(
        self,
        num_samples: int,
        seed: Optional[int] = None,
        split: str = "train",
    ) -> List[dict]:
        if self.task_loader is None:
            raise ValueError(
                "LocomoEnv has no task_loader. Provide benchmark_path in config."
            )
        return self.task_loader.sample_tasks(num_samples, seed=seed, split=split)

    def get_skill_library(self) -> list:
        return list(self.human_skills)

    def get_tool_library(self) -> str:
        # When the grep loop is enabled, surface its tag protocol to the
        # critic / optimizer so they know agents can call it.  Otherwise
        # LOCOMO has no external tools and we return empty to keep their
        # prompts clean.
        if self.max_grep_calls <= 0:
            return ""
        return _GREP_TOOL_DOC

    # ------------------------------------------------------------------
    # Grep loop (only fired for agents that see the full conversation)
    # ------------------------------------------------------------------

    def _has_grep_access(self, role_kind: str) -> bool:
        """Grep only makes sense for agents that have the conversation."""
        return self.max_grep_calls > 0 and role_kind in ("retriever", "solo")

    def _run_with_grep(
        self,
        system_prompt: str,
        user_input: str,
        conversation_haystack: str,
    ):
        """Multi-turn chat allowing up to ``max_grep_calls`` grep calls.

        Returns ``(final_text, total_tokens, grep_log)``.  ``grep_log`` is
        a list of ``{"query", "result"}`` dicts in call order.  When the
        budget is 0 (or the agent never emits ``<grep>``) this is a
        single-shot chat exactly equivalent to ``chat_with_usage``.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        total = {"input": 0, "output": 0}
        grep_log: List[dict] = []

        budget = max(0, self.max_grep_calls)
        for _ in range(budget + 1):
            text, tokens = self.llm_client.chat_messages_with_usage(messages)
            total["input"] += tokens.get("input", 0)
            total["output"] += tokens.get("output", 0)

            match = _GREP_RE.search(text or "")
            if match is None or len(grep_log) >= budget:
                return text or "", total, grep_log

            pattern = match.group(1).strip()
            # Force the haystack to the conversation: in LOCOMO the agent
            # has no separate "last search result" — the transcript is the
            # only thing worth grepping.  We pass the pattern alone (no
            # ``|||``) and let ``run_grep``'s fallback supply the text.
            result = run_grep(
                pattern, fallback_text=conversation_haystack,
                max_lines=self.grep_max_lines,
            )
            grep_log.append({"query": pattern, "result": result})
            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": f"<grep_result>\n{result}\n</grep_result>",
            })

        return text or "", total, grep_log

    # ------------------------------------------------------------------
    # Decentralized: retriever -> reasoner [-> verifier]
    # ------------------------------------------------------------------

    def _collect_decentralized(self, policies: dict, task: dict) -> Trajectory:
        question = task.get("question", "")
        category = int(task.get("category", 0) or 0)
        cat_hint = _CATEGORY_HINTS.get(category, "")

        full_conv_block = self._render_conversation(task)

        steps: List[dict] = []
        message_pool: List[dict] = []

        for i, agent_name in enumerate(self.agent_names):
            is_last = i == self.num_agents - 1
            role_for = self._role_for(agent_name, i)
            role_kind = self._agent_role_kind(i)

            policy = policies.get(agent_name)
            if policy is None:
                policy = AgentPolicy(role=role_for, skills="")
            elif isinstance(policy, str):
                policy = AgentPolicy.from_legacy(policy)
            elif isinstance(policy, AgentPolicy) and not policy.role.strip():
                # Fill in the default role but KEEP the discrete skill library
                # intact (``skills=`` would collapse it to a single skill).
                policy = AgentPolicy(role=role_for, skill_library=policy.skill_library)

            agent_system = self._compose_system_prompt(policy, agent_name=agent_name)

            user_input = self._build_user_input(
                agent_idx=i,
                agent_name=agent_name,
                question=question,
                category=category,
                cat_hint=cat_hint,
                conversation_block=full_conv_block,
                message_pool=message_pool,
                is_last=is_last,
                task=task,
            )

            grep_log: List[dict] = []
            if self._has_grep_access(role_kind):
                text, tokens, grep_log = self._run_with_grep(
                    agent_system, user_input, full_conv_block,
                )
            else:
                text, tokens = self.llm_client.chat_with_usage(agent_system, user_input)
            # Strip any residual <grep> tags so they don't leak to downstream agents.
            text_clean = _GREP_RE.sub("", text or "").strip() or (text or "").strip()

            step = {
                "agent": agent_name,
                "agent_id": agent_name,
                "system_prompt": agent_system,
                "input": user_input,
                "output": text_clean,
                "action": text_clean,
                "tokens": tokens,
            }
            if grep_log:
                step["grep_log"] = grep_log
            steps.append(step)
            message_pool.append({"agent": agent_name, "content": text_clean})

        final_answer = message_pool[-1]["content"] if message_pool else ""

        metadata = {
            "task_type": "locomo_qa",
            "category": category,
            "final_answer": final_answer,
            "architecture": "decentralized",
            "agent_layout": self._layout_label(),
        }
        trajectory = Trajectory(task=task, steps=steps, reward=0.0, metadata=metadata)
        reward = self.reward_gen.compute(trajectory)
        trajectory.reward = reward
        metadata["evaluation_feedback"] = f"reward={reward}"
        trajectory.skill_trace = build_skill_trace(steps, policies)
        return trajectory

    # ------------------------------------------------------------------
    # Centralized: main reasoner delegates retrieval via <retrieve>
    # ------------------------------------------------------------------

    def _collect_centralized(self, policies: dict, task: dict) -> Trajectory:
        question = task.get("question", "")
        category = int(task.get("category", 0) or 0)
        cat_hint = _CATEGORY_HINTS.get(category, "")

        main_name = self.main_agent
        main_idx = self.agent_names.index(main_name)
        sub_name = self.sub_agents[0]
        sub_idx = self.agent_names.index(sub_name)

        main_role = self._role_for(main_name, main_idx)
        sub_role = self._role_for(sub_name, sub_idx)

        main_policy = policies.get(main_name)
        if main_policy is None:
            main_policy = AgentPolicy(role=main_role, skills="")
        elif isinstance(main_policy, str):
            main_policy = AgentPolicy.from_legacy(main_policy)

        sub_policy = policies.get(sub_name)
        if sub_policy is None:
            sub_policy = AgentPolicy(role=sub_role, skills="")
        elif isinstance(sub_policy, str):
            sub_policy = AgentPolicy.from_legacy(sub_policy)

        main_system = self._compose_system_prompt(main_policy, agent_name=main_name)
        sub_system = self._compose_system_prompt(sub_policy, agent_name=sub_name)

        full_conv_block = self._render_conversation(task)

        sub_steps: List[dict] = []
        is_hybrid = self.architecture == "hybrid"
        prior_sub_outputs: List[dict] = []

        def _invoke_sub(query: str) -> str:
            history_block = ""
            if is_hybrid and prior_sub_outputs:
                history_parts = []
                for i, prev in enumerate(prior_sub_outputs, 1):
                    history_parts.append(
                        f"--- prior sub-agent turn {i} (request: "
                        f"{prev['query'] or '[no query]'}) ---\n{prev['output']}"
                    )
                history_block = (
                    "Prior sub-agent excerpts on this same question "
                    "(extend or correct as needed):\n"
                    + "\n\n".join(history_parts) + "\n\n"
                )
            sub_user = (
                f"You are helping the reasoner answer a question about a long "
                f"conversation.\n\nQuestion: {question}\n"
                f"Category hint: {cat_hint}\n\n"
                f"{history_block}"
                f"Reasoner's retrieval request:\n{query.strip() or '[no query]'}\n\n"
                f"{full_conv_block}\n\n"
                "Return up to 6 short verbatim excerpts (with their D-ids) "
                "that bear on the question. If no turn is relevant, reply "
                "exactly: NO_EVIDENCE."
            )
            # The centralized sub-agent is the retriever — give it grep
            # access when the env was configured with max_grep_calls > 0.
            sub_grep_log: List[dict] = []
            if self.max_grep_calls > 0:
                sub_text, sub_tokens, sub_grep_log = self._run_with_grep(
                    sub_system, sub_user, full_conv_block,
                )
            else:
                sub_text, sub_tokens = self.llm_client.chat_with_usage(sub_system, sub_user)
            sub_text_clean = _GREP_RE.sub("", sub_text or "").strip() or (sub_text or "").strip()
            sub_step = {
                "agent": sub_name,
                "agent_id": sub_name,
                "system_prompt": sub_system,
                "input": sub_user,
                "output": sub_text_clean,
                "action": sub_text_clean,
                "tokens": sub_tokens,
                "main_query": query,
                "isolated_context": not is_hybrid,
                "shared_sub_context": is_hybrid,
            }
            if sub_grep_log:
                sub_step["grep_log"] = sub_grep_log
            sub_steps.append(sub_step)
            if is_hybrid:
                prior_sub_outputs.append({"query": query, "output": sub_text_clean})
            return sub_text_clean

        main_user = (
            f"Question: {question}\n"
            f"Category hint: {cat_hint}\n\n"
            "You do NOT see the conversation directly. To inspect it, emit "
            "exactly one <retrieve>QUERY</retrieve> tag per turn and you will "
            "receive evidence excerpts. When you are ready, write your FINAL "
            "answer in the LOCOMO format (concise, no explanation; for "
            "adversarial / cat-5 questions write exactly ``not mentioned``)."
        )

        messages = [
            {"role": "system", "content": main_system},
            {"role": "user", "content": main_user},
        ]
        max_retrieves = max(1, self.num_agents)  # cap delegation rounds
        total_tokens = {"input": 0, "output": 0}
        retrieve_log = []
        final_text = ""
        for _ in range(max_retrieves + 1):
            text, tokens = self.llm_client.chat_messages_with_usage(messages)
            total_tokens["input"] += tokens.get("input", 0)
            total_tokens["output"] += tokens.get("output", 0)
            m = _RETRIEVE_RE.search(text or "")
            if m is None or len(retrieve_log) >= max_retrieves:
                final_text = (text or "").strip()
                break
            query = m.group(1).strip()
            sub_response = _invoke_sub(query)
            retrieve_log.append({"query": query, "result": sub_response})
            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": f"<retrieve_result>\n{sub_response}\n</retrieve_result>",
            })

        final_clean = _RETRIEVE_RE.sub("", final_text).strip() or final_text

        steps: List[dict] = list(sub_steps)
        steps.append({
            "agent": main_name,
            "agent_id": main_name,
            "system_prompt": main_system,
            "input": main_user,
            "output": final_clean,
            "action": final_clean,
            "tokens": total_tokens,
            "retrieve_log": retrieve_log,
        })

        metadata = {
            "task_type": "locomo_qa",
            "category": category,
            "final_answer": final_clean,
            "architecture": self.architecture,
            "main_agent": main_name,
            "sub_agents": list(self.sub_agents),
            "num_retrieve_calls": len(retrieve_log),
        }
        trajectory = Trajectory(task=task, steps=steps, reward=0.0, metadata=metadata)
        reward = self.reward_gen.compute(trajectory)
        trajectory.reward = reward
        metadata["evaluation_feedback"] = f"reward={reward}"
        trajectory.skill_trace = build_skill_trace(steps, policies)
        return trajectory

    # ------------------------------------------------------------------
    # Prompt assembly helpers
    # ------------------------------------------------------------------

    def _layout_label(self) -> str:
        if self.num_agents == 1:
            return "solo"
        if self.num_agents == 2:
            return "retriever+reasoner"
        if self.num_agents == 3:
            return "retriever+reasoner+verifier"
        return f"{self.num_agents}-agents"

    def _agent_role_kind(self, agent_idx: int) -> str:
        """Map position → semantic role.  Stable for ``num_agents`` in 1..N."""
        if self.num_agents == 1:
            return "solo"
        if self.num_agents == 2:
            return "retriever" if agent_idx == 0 else "reasoner"
        # 3+ agents: retriever, reasoner, verifier(s)
        if agent_idx == 0:
            return "retriever"
        if agent_idx == self.num_agents - 1:
            return "verifier"
        return "reasoner"

    def _role_for(self, agent_name: str, agent_idx: int) -> str:
        kind = self._agent_role_kind(agent_idx)
        override = self._role_overrides.get(kind)
        if override:
            return override
        return _DEFAULT_ROLES.get(kind, default_agent_policy(agent_idx, self.num_agents).role)

    def _load_role_overrides(self) -> Dict[str, str]:
        prompts_dir = Path(__file__).parent / "prompts"
        overrides: Dict[str, str] = {}
        if not prompts_dir.is_dir():
            return overrides
        for name in ("retriever", "reasoner", "verifier", "solo"):
            path = prompts_dir / f"{name}.md"
            if path.exists():
                try:
                    overrides[name] = path.read_text()
                except OSError:
                    continue
        return overrides

    def _compose_system_prompt(
        self, policy: AgentPolicy, agent_name: Optional[str] = None,
    ) -> str:
        base = policy.combined
        if not self.inject_skill_library:
            return base
        block = ""
        if agent_name and agent_name in self._agent_skill_library_blocks:
            block = self._agent_skill_library_blocks[agent_name]
        else:
            block = self._global_skill_library_block
        if not block:
            return base
        return base.rstrip() + "\n\n" + block

    def _build_user_input(
        self,
        *,
        agent_idx: int,
        agent_name: str,
        question: str,
        category: int,
        cat_hint: str,
        conversation_block: str,
        message_pool: List[dict],
        is_last: bool,
        task: Optional[dict] = None,
    ) -> str:
        kind = self._agent_role_kind(agent_idx)
        cat_name = _CATEGORY_NAMES.get(category, "unknown")
        header = (
            f"Question: {question}\n"
            f"Question type: category {category} ({cat_name})\n"
            f"Category hint: {cat_hint}"
        )

        # When retriever_sees_conversation=False, the agent gets only the
        # session INDEX (sessions + dates + #turns), not the transcript.
        # It must call ``<grep>`` to fetch any turn content.
        use_index = (not self.retriever_sees_conversation
                     and self.max_grep_calls > 0
                     and kind in ("solo", "retriever")
                     and task is not None)
        body_block = self._render_session_index(task) if use_index else conversation_block

        if kind == "solo":
            base = f"{header}\n\n{body_block}\n\nProvide your FINAL answer."
            if self.max_grep_calls > 0:
                src = ("transcript that holds the conversation (call grep to fetch turns)"
                       if use_index else "transcript above")
                base += (
                    f"\n\nYou may run up to {self.max_grep_calls} "
                    f"``<grep>PATTERN</grep>`` call(s) on the {src} "
                    "before answering (case-insensitive regex, returns at "
                    f"most {self.grep_max_lines} matching lines; each line "
                    "is prefixed with its dia id and session date)."
                )
            return base

        if kind == "retriever":
            base = (
                f"{header}\n\n{body_block}\n\n"
                "Return up to 6 short verbatim excerpts (with their D-ids) "
                "that bear on the question.  Format each line as ``[D<sess>:"
                "<turn>] (SESSION_DATE) SPEAKER said: \"...\"``.  If no turn "
                "is relevant, reply exactly: NO_EVIDENCE."
            )
            if self.max_grep_calls > 0:
                src = ("the transcript (NOT shown — call grep to fetch turns)"
                       if use_index else "the transcript above")
                base += (
                    f"\n\nYou may run up to {self.max_grep_calls} "
                    f"``<grep>PATTERN</grep>`` call(s) on {src} BEFORE "
                    "writing your excerpts (case-insensitive regex, returns "
                    f"at most {self.grep_max_lines} matching lines; each "
                    "line carries its dia id and session date — copy the "
                    "date verbatim into your ``(SESSION_DATE)`` slot)."
                )
            return base

        # reasoner / verifier see the retriever's evidence excerpts (and any
        # prior agents' messages), but NOT the full conversation.
        prior = []
        for msg in message_pool:
            display = msg["agent"].replace("_", " ").title()
            prior.append(f"{display} said:\n{msg['content']}")
        prior_block = "\n\n".join(prior) if prior else "(no prior agent messages)"

        if kind == "reasoner":
            body = (
                f"{header}\n\nEvidence from upstream agents:\n{prior_block}"
                "\n\nUsing ONLY the evidence above, write your FINAL answer "
                "in the LOCOMO format (concise; date for cat 2; comma-list "
                "for cat 1; exactly ``not mentioned`` for cat 5)."
            )
            if is_last:
                body += "\n\nYour reply WILL be evaluated as the final answer."
            return body

        # verifier
        return (
            f"{header}\n\nPrior agents' outputs:\n{prior_block}\n\n"
            "Audit the reasoner's answer against the retriever's evidence. "
            "If the answer is correct and well-formatted, output it verbatim. "
            "Otherwise output a corrected FINAL answer in the LOCOMO format. "
            "Your reply WILL be evaluated as the final answer."
        )

    # ------------------------------------------------------------------
    # Conversation rendering / truncation
    # ------------------------------------------------------------------

    def _render_conversation(self, task: dict) -> str:
        """Render the conversation newest-session-first under a char budget.

        Format mirrors upstream LOCOMO ``get_input_context``: turns become
        ``D<sess>:<turn> [date] SPEAKER said, "text"`` so the retriever
        can cite ``dia_id``s back.  Image turns include ``blip_caption``
        as ``[image: ...]``.

        LOCOMO evidence is uniformly distributed across sessions, so when
        the budget IS exceeded we surface that loss to the agent with an
        explicit ``[earlier sessions truncated]`` marker — otherwise the
        retriever silently misses the early sessions where many cat-2/3
        answers live.
        """
        session_meta = task.get("session_meta") or []
        if not session_meta:
            return "Conversation: (empty)"

        budget = self._char_budget
        included = []  # list of (session_dict, rendered_str)
        truncated = False
        for session in reversed(session_meta):
            chunk = self._render_session(session)
            if not chunk:
                continue
            if budget - len(chunk) < 0 and included:
                truncated = True
                break
            included.append((session, chunk))
            budget -= len(chunk)
            if budget <= 0:
                truncated = len(included) < len(session_meta)
                break

        # Display oldest-first so dia_id ordering reads naturally.
        included.reverse()
        body_parts = []
        if truncated:
            n_dropped = len(session_meta) - len(included)
            body_parts.append(
                f"[earlier sessions truncated: {n_dropped} session(s) removed]"
            )
        body_parts.extend(chunk for _, chunk in included)
        body = "\n\n".join(body_parts)
        header = "Conversation (older to newer)\n"
        return header + body

    def _render_session(self, session: dict) -> str:
        name = session.get("name", "session_?")
        date = session.get("date", "")
        header = f"--- {name} ({date}) ---" if date else f"--- {name} ---"
        lines = [header]
        # Inline the session date on every turn line so any grep hit
        # carries the date — without this the agent has no way to attach
        # ``(SESSION_DATE)`` to its excerpts when running in
        # retriever_sees_conversation=False mode (no separate header
        # lookup is possible).
        date_inline = f" ({date})" if date else ""
        for turn in session.get("turns", []):
            dia_id = turn.get("dia_id", "")
            speaker = turn.get("speaker", "")
            text = (turn.get("text") or "").strip()
            caption = turn.get("blip_caption")
            tag = f"[{dia_id}]" if dia_id else ""
            line = f"{tag}{date_inline} {speaker} said, \"{text}\"".strip()
            if caption:
                line += f"  [image: {caption.strip()}]"
            lines.append(line)
        return "\n".join(lines)

    def _render_session_index(self, task: dict) -> str:
        """Render the conversation INDEX (session list, no turn content).

        Used in retriever_sees_conversation=False mode so the agent knows
        the search space (number of sessions, dates, speakers, #turns per
        session) without paying for the full transcript.  The agent then
        uses ``<grep>`` to fetch any actual turn content.
        """
        session_meta = task.get("session_meta") or []
        if not session_meta:
            return "Conversation index: (empty)"
        speakers = task.get("speakers") or []
        first_date = session_meta[0].get("date", "?")
        last_date = session_meta[-1].get("date", "?")
        total_turns = sum(len(s.get("turns", [])) for s in session_meta)
        lines = [
            f"Conversation index — {len(session_meta)} sessions, "
            f"{total_turns} turns total, "
            f"{first_date} → {last_date}",
            f"Speakers: {', '.join(speakers) if speakers else '(unknown)'}",
            "",
            "Per-session metadata (no turn content shown — use `<grep>` to fetch turns):",
        ]
        for s in session_meta:
            name = s.get("name", "session_?")
            date = s.get("date", "")
            n = len(s.get("turns", []))
            lines.append(
                f"- {name} ({date}) — {n} turns"
            )
        return "\n".join(lines)


# Built-in default roles used when prompts/<kind>.md is not present.
_DEFAULT_ROLES: Dict[str, str] = {
    "solo": (
        "You are answering a question about a long multi-session "
        "conversation between two speakers. Read the conversation, then "
        "give a concise FINAL answer."
    ),
    "retriever": (
        "You are the **retriever** in a 2-agent QA team for the LOCOMO "
        "benchmark.\n"
        "- You see the full conversation (token-budgeted) and the question.\n"
        "- Your job is to surface the evidence the reasoner will use; you "
        "do not answer the question.\n"
        "- Output up to 6 short verbatim excerpts with their dia_id "
        "(e.g. ``D3:7``).\n"
        "- Reply ``NO_EVIDENCE`` when nothing in the conversation answers "
        "the question — this is critical for adversarial (category 5) "
        "questions."
    ),
    "reasoner": (
        "You are the **reasoner** in a 2-agent QA team for the LOCOMO "
        "benchmark.\n"
        "- You see ONLY the question, the category hint, and the "
        "retriever's evidence excerpts.\n"
        "- Compose the FINAL answer in the LOCOMO format: concise, no "
        "explanation; a date for category 2; a comma-separated list for "
        "category 1; the exact string ``not mentioned`` when the "
        "evidence is ``NO_EVIDENCE`` or does not actually answer the "
        "question (category 5)."
    ),
    "verifier": (
        "You are the **verifier** in a 3-agent QA team for the LOCOMO "
        "benchmark.\n"
        "- You see the question, the retriever's evidence and the "
        "reasoner's draft answer.\n"
        "- If the reasoner's answer is correct AND formatted per the "
        "category rules, output it verbatim.\n"
        "- Otherwise output a corrected FINAL answer in the LOCOMO "
        "format. Do NOT add explanation."
    ),
}


# Surfaced via ``get_tool_library`` when grep is enabled, so the optimizer
# and critic know the agent has a ``<grep>`` action available.
_GREP_TOOL_DOC = (
    "## Tool: grep\n"
    "Tag protocol: ``<grep>PATTERN</grep>``\n"
    "- Filters the rendered conversation transcript using a Python\n"
    "  case-insensitive regex (``re.search`` line-by-line).\n"
    "- Returns up to a fixed number of matching lines as\n"
    "  ``<grep_result>...</grep_result>``; ``No lines match /<PATTERN>/.`` if\n"
    "  nothing hits.\n"
    "- Only the agents that already see the full transcript (retriever in\n"
    "  decentralized mode, sub-agent in centralized mode, solo agent in\n"
    "  1-agent mode) may call it; the reasoner / verifier see only upstream\n"
    "  excerpts and cannot grep.\n"
    "- Useful for pulling every turn that names a specific person, date, or\n"
    "  topic out of long, multi-session conversations."
)
