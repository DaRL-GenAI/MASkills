"""Unified configuration system with hierarchical configs."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from typing import Dict, List, Optional

from .llm import LLMConfig, get_llm_config


@dataclass
class BaseConfig:
    """Shared configuration fields across all environments."""

    # Experiment
    exp_name: str = "experiment"
    paradigm: str = "central_credit"  # "central_global" | "central_credit"

    # Training
    # Iterations are 1-indexed: ``iteration=0`` is reserved for the empty-skills
    # baseline test round (no training, no checkpoint), and ``iteration=1..N``
    # are the actual training rounds.  Checkpoint ``iter_N`` is the policy
    # produced by the N-th training round.
    num_iterations: int = 5
    trajectories_per_iteration: int = 10
    mini_batch_size: Optional[int] = None
    start_iteration: int = 1
    # When True, the held-out test set is evaluated every iteration (for a
    # per-iteration test curve); when False, only at baseline + final iter.
    eval_test_every_iter: bool = False

    # Optimizer behavior
    synthesis_method: str = "rewrite"  # "rewrite" | "diff_edit"
    momentum: float = 0.0  # 0.0 disables; typical 0.3-0.9

    # LLM backends
    llm: Optional[LLMConfig] = None
    actor_llm: Optional[LLMConfig] = None
    critic_llm: Optional[LLMConfig] = None
    optimizer_llm: Optional[LLMConfig] = None

    # Agents
    num_agents: int = 2

    # I/O
    experiment_dir: str = "./experiments"
    checkpoint_dir: str = "./ckpt_policy"

    # Parallelism
    max_workers: int = 64

    # Logging
    log_level: str = "INFO"

    # Human-authored skill library (Anthropic SKILL.md format).
    # Path to a directory containing one sub-folder per skill, each with a
    # SKILL.md file.  When set, every agent's system prompt is prefixed with
    # a "Skill Library" block that lists and inlines these skills.  The
    # library is fixed (non-trainable); the optimizer never sees it.
    skills_dir: Optional[str] = None

    # ── MASkills continual skill evolution ───────────────────────────────
    # When True, training evolves each agent's discrete SkillLibrary via the
    # four operators (refinement / induction / consolidation / pruning) under
    # skill-conditioned credit assignment, instead of the monolithic
    # role+skills rewrite.  Requires the SkillEvolutionTrainer.
    skill_evolution: bool = False
    # Validation rollback tolerance δ (Eq.18): a candidate skill library is
    # accepted only if its held-out reward is >= the current library's minus
    # δ.  Requires n_val > 0; with n_val == 0 the gate is disabled.
    skill_eval_delta: float = 0.02
    # Multi-timescale operator cadence (in iterations).  Refinement is cheap
    # and runs often; induction/consolidation/pruning act on longer horizons
    # once evidence has accumulated.  0 disables that operator entirely.
    refine_every: int = 1
    induct_every: int = 2
    consolidate_every: int = 3
    prune_every: int = 3
    # Library-size guard: induction will not grow a library beyond this.
    max_skills_per_agent: int = 12
    # Whether the agent's ROLE evolves alongside its skills. Off by default:
    # the role carries the collaboration protocol the environment parses, so an
    # edit there can invalidate every trajectory in a way no skill can repair.
    # When on, a proposed role goes through the same validation gate as the
    # library, so a harmful edit is rolled back with it.
    evolve_role: bool = False
    # A trajectory counts as a "hard case" (eligible as induction evidence
    # H_i) when its reward is <= this threshold.
    hard_trajectory_threshold: float = 0.5
    # Inject the HotpotQA-style "gold answers are 1-2 words, keep the final
    # answer minimal" hint into the skill-credit prompt.  Correct for
    # short-answer QA; leave False for tasks with varied answer lengths.
    answer_brevity_hint: bool = False

    def __post_init__(self):
        valid_paradigms = ['central_global', 'central_credit']
        if self.paradigm not in valid_paradigms:
            raise ValueError(f"Invalid paradigm '{self.paradigm}'. Must be one of {valid_paradigms}")
        if self.num_agents < 1:
            raise ValueError("num_agents must be at least 1")
        valid_methods = ['rewrite', 'diff_edit']
        if self.synthesis_method not in valid_methods:
            raise ValueError(f"Invalid synthesis_method '{self.synthesis_method}'. Must be one of {valid_methods}")
        if self.momentum < 0.0:
            raise ValueError("momentum must be >= 0.0")

    def get_actor_llm(self) -> LLMConfig:
        """Get LLM config for actors (fallback: llm)."""
        return self.actor_llm or self.llm

    def get_critic_llm(self) -> LLMConfig:
        """Get LLM config for critic (fallback: llm)."""
        return self.critic_llm or self.llm

    def get_optimizer_llm(self) -> LLMConfig:
        """Get LLM config for optimizer (fallback: llm)."""
        return self.optimizer_llm or self.llm

    @classmethod
    def from_dict(cls, data: dict) -> 'BaseConfig':
        """Build a config from a dict.  Resolves LLM configs and filters unknown keys."""
        data = dict(data)  # don't mutate caller

        # Resolve LLM configs
        for llm_field in ['llm', 'actor_llm', 'critic_llm', 'optimizer_llm']:
            if llm_field in data:
                val = data[llm_field]
                if isinstance(val, str):
                    data[llm_field] = get_llm_config(val)
                elif isinstance(val, dict):
                    if 'preset' in val:
                        data[llm_field] = get_llm_config(val['preset'])
                    else:
                        data[llm_field] = LLMConfig.from_dict(val)

        # Filter to valid fields for this class
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str, overrides: dict = None) -> 'BaseConfig':
        """Load config from JSON file with optional overrides."""
        with open(path) as f:
            data = json.load(f)
        if overrides:
            data.update(overrides)
        return cls.from_dict(data)

    def to_json(self, path: str):
        """Save config to JSON file."""
        data = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, LLMConfig):
                val = val.to_dict()
            data[f.name] = val
        with open(path, 'w') as fh:
            json.dump(data, fh, indent=2)


@dataclass
class LanguageTaskConfig(BaseConfig):
    """Language task specific config."""

    task_type: str = "qa"  # "qa" | "math" | "coding"
    benchmark_path: str = ""
    data_limit: Optional[int] = None
    judge_model: str = "openai/gpt-5.1"
    code_timeout: float = 10.0
    optimizer_workers: int = 1

    # QA reward signal — the quantity ``traj.reward`` carries and that every
    # optimization target (validation gate, skill credit, hard-case
    # threshold, stats) keys off.  "f1" = token F1 (precision-sensitive, the
    # default target); "em" = exact match; "judge" = binary LLM-as-judge.
    qa_reward_metric: str = "f1"

    # Train-test split
    train_test_split: float = 1.0  # 1.0 = all training (backward compat)
    split_seed: int = 42
    eval_samples: Optional[int] = None  # subsample test set; None = use all

    # Explicit sequential train/val/test split.  When any of these is set,
    # the task loader takes the first ``n_train`` tasks as training, the
    # next ``n_val`` as validation, the next ``n_test`` as held-out test,
    # and ignores ``train_test_split`` / ``split_seed``.
    n_train: Optional[int] = None
    n_val: Optional[int] = None
    n_test: Optional[int] = None

    # Cadence controls for held-out evaluation passes.
    # When ``n_val`` > 0 the validation set is evaluated every iteration
    # (matching the existing test cadence).  ``eval_test_every_iter``
    # controls whether the larger held-out test pool runs every iteration
    # (True) or only at the baseline + final iteration (False, default).
    eval_test_every_iter: bool = False

    # Whether to include the gold HotpotQA context passages in the task
    # prompt.  When False, gold passages are withheld and agents must call
    # the search_wiki tool (see ``max_search_calls``) to recover them.
    include_context: bool = True
    max_search_calls: int = 0  # 0 disables the tool loop
    search_limit: int = 5  # results per search call

    # Whether to append the human-authored ``Skill Library`` block (rendered
    # from ``skills_dir``) to each agent's system prompt at rollout time.
    # ``get_skill_library()`` still returns the loaded skills regardless.
    inject_skill_library: bool = True
    # Whether to append the ``Tool Reference`` block (rendered from
    # ``maskills/envs/language/tools/*.md``) to each agent's system prompt.
    # ``get_tool_library()`` still returns the docs for the optimizer / critic.
    inject_tool_reference: bool = True

    # Multi-agent collaboration architecture.
    #   - "decentralized": agents speak in sequence into a shared message
    #     pool; the last agent's output is the final answer.  This is the
    #     original behaviour.
    #   - "centralized": one main agent (the reasoner) drives the
    #     conversation and may invoke sub-agents via <retrieve>QUERY</retrieve>.
    #     Each sub-agent invocation runs in an ISOLATED context (the sub-agent
    #     sees the full task plus the main's query, but not main's prior
    #     reasoning); only the sub-agent's final response is returned to
    #     main as a <retrieve_result> tool block.  The main agent's own
    #     output is the final answer.
    architecture: str = "decentralized"
    # In centralized mode, this is the agent_name that plays the main /
    # reasoner role.  All other agents are sub-agents reachable via
    # <retrieve>.  Defaults to ``agent_1``.
    main_agent: str = "agent_1"

    # Per-agent skill library overrides.  Maps agent_name (e.g. ``agent_1``)
    # to a directory of SKILL.md folders.  When set, the named agent's
    # system prompt receives only that directory's skills instead of the
    # global ``skills_dir`` library.  Agents not listed fall back to the
    # global ``skills_dir`` (or no library at all if neither is set).
    agent_skills_dirs: Optional[Dict[str, str]] = None
    # Per-agent tool budgets.  Maps agent_name to the per-turn
    # ``max_search_calls`` budget (0 = no tool loop).  Agents not listed
    # use the global ``max_search_calls``.
    agent_max_tool_calls: Optional[Dict[str, int]] = None
    # Per-agent tool whitelist.  Maps agent_name to a list of tool names
    # the agent may invoke (``search``, ``grep``, ``sympy``).  Agents not
    # listed may invoke any tool whose budget is non-zero.  An empty list
    # forbids every tool for that agent.
    agent_allowed_tools: Optional[Dict[str, List[str]]] = None

    def __post_init__(self):
        super().__post_init__()
        valid_task_types = ['qa', 'math', 'coding']
        if self.task_type not in valid_task_types:
            raise ValueError(f"Invalid task_type '{self.task_type}'. Must be one of {valid_task_types}")
        valid_archs = ['decentralized', 'centralized', 'hybrid']
        if self.architecture not in valid_archs:
            raise ValueError(
                f"Invalid architecture '{self.architecture}'. Must be one of {valid_archs}"
            )

    @classmethod
    def from_dict(cls, data: dict) -> 'LanguageTaskConfig':
        """Back-compat: translate deprecated ``hide_context`` to ``include_context``."""
        data = dict(data)
        if 'hide_context' in data:
            hide = bool(data.pop('hide_context'))
            data.setdefault('include_context', not hide)
        return super().from_dict(data)


@dataclass
class LocomoConfig(BaseConfig):
    """LOCOMO long-term conversational memory benchmark config.

    The dataset is ``env/locomo/data/locomo10.json`` (10 conversations,
    ~1990 QA items across categories 1/2/3/4/5).  Each task is
    one (conversation, qa_item) pair; multi-agent rollouts collaborate
    over the long conversation context.
    """

    # Data
    benchmark_path: str = ""
    data_limit: Optional[int] = None
    train_test_split: float = 1.0
    split_seed: int = 42
    eval_samples: Optional[int] = None
    # Per-split sample sizes.  ``n_val`` drives both the validation-rollback
    # gate and the per-iteration val metrics; ``n_test`` the held-out test
    # sweep.  ``None`` falls back to the legacy ``eval_samples`` behaviour.
    n_train: Optional[int] = None
    n_val: Optional[int] = None
    n_test: Optional[int] = None
    # LOCOMO reward signal — what ``traj.reward`` carries and every
    # optimization target keys off.  "f1" = category-routed token F1,
    # "bleu" = category-routed BLEU, "f1_bleu_mean" = their mean (default,
    # so both F1 and BLEU drive the optimization).
    locomo_reward_metric: str = "f1_bleu_mean"
    # Restrict the loaded task pool to a subset of categories (1,2,3,4,5).
    # ``None`` keeps every category.  Useful for curriculum training
    # (e.g. start with ``[2, 3]``, add ``[1, 5]`` once base policy is stable).
    category_filter: Optional[List[int]] = None

    # Conversation truncation: hard token budget for the retriever input.
    # The 10 LOCOMO conversations span 19–32 sessions and 16k–32k tokens of
    # dialogue; evidence is uniformly distributed across sessions, so a
    # budget below ~35k tokens systematically drops required evidence for
    # the oldest sessions.  Default 40k tokens fits the largest conversation
    # (~32k tokens) comfortably and stays well within gpt-4o-mini's 128k
    # context window.  Lower the budget only with a smaller (e.g.
    # session-summarised) representation.
    max_context_tokens: int = 40000
    # Approximate chars-per-token used for the budget heuristic (the env
    # does not pull in a real tokenizer to keep the dep surface small).
    chars_per_token: int = 4

    # Multi-agent collaboration architecture.  ``decentralized`` runs all
    # agents sequentially over a shared message pool: retriever (agent_1)
    # produces evidence, reasoner (agent_2) composes the final answer,
    # optional verifier (agent_3) audits.  ``centralized`` uses agent_1 as
    # a reasoner that delegates retrieval via <retrieve> sub-calls.
    architecture: str = "decentralized"
    main_agent: str = "agent_2"  # the reasoner is final by default

    # Whether to append the human-authored ``Skill Library`` block (rendered
    # from ``skills_dir``) to each agent's system prompt at rollout time.
    inject_skill_library: bool = True

    # Per-agent skill library overrides (agent_name -> skills_dir path).
    agent_skills_dirs: Optional[Dict[str, str]] = None

    # Grep tool over the rendered conversation transcript.  When
    # ``max_grep_calls > 0``, the agents that already see the full
    # transcript (retriever in decentralized mode, sub-agent in
    # centralized mode, solo agent in 1-agent mode) get a multi-turn
    # ``<grep>PATTERN</grep>`` loop with that budget.  Other agents
    # (reasoner / verifier) are unaffected — they have no transcript to
    # grep.  ``grep_max_lines`` caps the per-call result size.
    max_grep_calls: int = 0
    grep_max_lines: int = 20

    # When True (legacy default), the retriever / solo agent's user prompt
    # carries the entire rendered conversation in addition to having the
    # ``<grep>`` tool — the conversation is duplicated into every chat-loop
    # round, costing roughly (max_grep_calls + 1) × conversation_tokens
    # per rollout.  When False, the user prompt carries ONLY a compact
    # session index (session_<s> (DATE) + speakers + #turns) and the agent
    # must call ``<grep>`` to fetch any turn content; the conversation
    # itself is held only inside the env as the grep haystack.  This cuts
    # actor input tokens by ~75% per retriever rollout with no functional
    # change (the agent had no other use for the full transcript besides
    # grepping it).
    retriever_sees_conversation: bool = True

    def __post_init__(self):
        super().__post_init__()
        valid_archs = ["decentralized", "centralized", "hybrid"]
        if self.architecture not in valid_archs:
            raise ValueError(
                f"Invalid architecture '{self.architecture}'. Must be one of {valid_archs}"
            )
        if self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be > 0")
        if self.chars_per_token <= 0:
            raise ValueError("chars_per_token must be > 0")
        if self.max_grep_calls < 0:
            raise ValueError("max_grep_calls must be >= 0")
        if self.grep_max_lines <= 0:
            raise ValueError("grep_max_lines must be > 0")


@dataclass
class GaiaConfig(BaseConfig):
    """GAIA benchmark config (multi-agent with real web/compute tools).

    Data is a JSONL of GAIA items (e.g. ``data/gaia/test_text103.jsonl``).
    Skills live as SKILL.md folders under ``agent_skills_dirs``; the env
    loads them into each agent's prompt rather than via a checkpoint
    ``skill_library.json``.

    Tools:
      * SEARCH / BROWSE (Tavily or Perplexity Sonar) — agent_1 only
      * COMPUTE (local Python subprocess) — agent_2 only

    Topologies:
      * decentralized — Researcher emits HANDOFF, Solver emits FINAL ANSWER
      * centralized   — Solver is main; <retrieve> spawns isolated Researcher
      * hybrid        — same as centralized but Researcher sees prior dossiers
    """

    # Data
    benchmark_path: str = ""
    data_limit: Optional[int] = None
    split_seed: int = 42

    # Multi-agent topology
    architecture: str = "decentralized"
    main_agent: str = "agent_2"  # Solver is main in centralized/hybrid

    # Per-agent SKILL.md roots (agent_name -> directory with SKILL.md +
    # subskill folders).  The env reads SKILL.md bodies directly.
    agent_skills_dirs: Optional[Dict[str, str]] = None

    # Rollout budgets
    rounds_a1: int = 5
    rounds_a2: int = 3
    budget_a1: int = 5
    budget_a2: int = 3
    max_retrieves: int = 3
    max_tokens: int = 1500

    # Whether to append a Skill Library block — unused for GAIA (SKILL.md
    # bodies ARE the skill library; agents get them via agent_skills_dirs).
    inject_skill_library: bool = False

    def __post_init__(self):
        super().__post_init__()
        valid_archs = ["decentralized", "centralized", "hybrid"]
        if self.architecture not in valid_archs:
            raise ValueError(
                f"Invalid architecture '{self.architecture}'. Must be one of {valid_archs}"
            )


