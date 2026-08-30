"""MASkills — continual skills optimization for multi-agent LLM systems.

An agent's task knowledge is kept as a *discrete, persistent skill library*: a
directory of ``SKILL.md`` files, each individually addressable. Training does
not rewrite one monolithic prompt; it applies four typed operators to that
library — **induct**, **refine**, **consolidate**, **prune** — with per-skill
credit assignment attributing an episode's outcome back to the skills the
trajectory actually invoked, and a held-out validation gate that rolls back a
candidate library that does not hold up.

Two entry points, matching the two halves of the paper:

* **In-memory library evolution** (LOCOMO, HotpotQA) — :class:`SkillLibrary`
  lives inside the policy and is evolved by :class:`SkillEvolutionTrainer`
  under :class:`SkillCreditCritic` and :class:`SkillEvolutionOptimizer`::

      import maskills

      config = maskills.LanguageTaskConfig(
          task_type="qa",
          paradigm="central_credit",
          skill_evolution=True,
          llm=maskills.LLMConfig.from_preset("gpt-4o-mini"),
      )
      env = maskills.make_env("language", config)
      trainer = maskills.SkillEvolutionTrainer(
          config=config,
          env=env,
          critic=maskills.SkillCreditCritic(config),
          optimizer=maskills.SkillEvolutionOptimizer(config.get_optimizer_llm()),
      )
      trainer.train()

* **On-disk ``SKILL.md`` evolution** (GAIA) — the library is a directory the
  agent reads and the optimizer edits; see :mod:`maskills.skill_lib` and
  ``scripts/maskills_train_iter.py``.

The CTDE substrate underneath (trajectory collection, centralized critic,
language policy-gradient optimizer) originates in LangMARL
(https://github.com/DaRL-GenAI/LangMARL) and is vendored here under
``maskills.core`` / ``maskills.trainer`` so this repository reproduces the
paper's results standalone. See README.md for what MASkills adds on top.
"""

__version__ = "1.0.0"

# Configuration
from maskills.config.base import (
    BaseConfig,
    GaiaConfig,
    LanguageTaskConfig,
    LocomoConfig,
)
from maskills.config.llm import LLMConfig, get_llm_config, list_available_models

# Core abstractions
from maskills.core.base import (
    BaseAgent,
    BaseCritic,
    BaseEnvironment,
    BaseOptimizer,
    BaseReward,
    Trajectory,
)
from maskills.core.critic import CentralizedCritic
from maskills.core.optimizer import PolicyGradientOptimizer
from maskills.core.policy import AgentPolicy, default_agent_policy
from maskills.core.skill_credit import SkillCreditCritic
from maskills.core.skill_operators import SkillEvolutionOptimizer
from maskills.core.skills import (
    Skill,
    SkillLibrary,
    build_skill_trace,
    detect_invoked_skills,
    load_skills_dir,
    render_skill_library,
)
from maskills.core.trajectory import TrajectoryFormatter

# Environment registry
from maskills.envs import list_envs, make_env, register_env

# LLM client
from maskills.llm.client import LLMClient
from maskills.llm.token_tracker import TokenTracker

# On-disk SKILL.md library pipeline (GAIA)
from maskills.skill_lib import apply_ops, load_lib, snapshot_lib

# Store
from maskills.store import LocalStore, PolicyCheckpoint, RunLogger, TrajectoryStore

# Trainer
from maskills.trainer.callbacks import (
    Callback,
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
)
from maskills.trainer.monte_carlo import MonteCarloTrainer
from maskills.trainer.skill_evolution import OperatorSchedule, SkillEvolutionTrainer

#: ``env`` field in a config file -> the config class that parses it.
_CONFIG_CLASSES = {
    "language": LanguageTaskConfig,
    "locomo": LocomoConfig,
    "gaia": GaiaConfig,
}


def load_config(path: str, overrides: dict = None):
    """Load a JSON config, dispatching on its ``env`` field.

    Args:
        path: Path to the JSON config file.
        overrides: Optional key/value overrides applied on top of the file.

    Returns:
        A :class:`BaseConfig` subclass instance matching ``env``.
    """
    import json

    with open(path) as f:
        env_name = json.load(f).get("env", "language")
    cls = _CONFIG_CLASSES.get(env_name, BaseConfig)
    return cls.from_json(path, overrides=overrides)


def train(config_path: str, **overrides):
    """One-line training: load a config, build the components, run the loop.

    Selects the skill-evolution stack when ``skill_evolution`` is set on the
    config, and the flat policy-gradient stack otherwise.

    Example:
        maskills.train("configs/language_task/qa_hotpot_decentralized.json")
    """
    import json

    with open(config_path) as f:
        env_name = json.load(f).get("env", "language")
    config = load_config(config_path, overrides=overrides)
    env = make_env(env_name, config)

    if getattr(config, "skill_evolution", False):
        critic = SkillCreditCritic(config)
        optimizer = SkillEvolutionOptimizer(config.get_optimizer_llm())
        trainer_cls = SkillEvolutionTrainer
    else:
        critic = CentralizedCritic(config)
        optimizer = PolicyGradientOptimizer(
            config.get_optimizer_llm(),
            synthesis_method=getattr(config, "synthesis_method", "rewrite"),
            momentum=getattr(config, "momentum", 0.0),
        )
        trainer_cls = MonteCarloTrainer

    return trainer_cls(
        config=config, env=env, critic=critic, optimizer=optimizer
    ).train()


__all__ = [
    # Skill library and operators — what MASkills adds
    "Skill",
    "SkillLibrary",
    "SkillCreditCritic",
    "SkillEvolutionOptimizer",
    "SkillEvolutionTrainer",
    "OperatorSchedule",
    "load_skills_dir",
    "render_skill_library",
    "detect_invoked_skills",
    "build_skill_trace",
    "load_lib",
    "apply_ops",
    "snapshot_lib",
    # Core abstractions
    "Trajectory",
    "BaseEnvironment",
    "BaseAgent",
    "BaseCritic",
    "BaseOptimizer",
    "BaseReward",
    "AgentPolicy",
    "default_agent_policy",
    "PolicyGradientOptimizer",
    "CentralizedCritic",
    "TrajectoryFormatter",
    # Config
    "BaseConfig",
    "LanguageTaskConfig",
    "LocomoConfig",
    "GaiaConfig",
    "LLMConfig",
    "get_llm_config",
    "list_available_models",
    # LLM
    "LLMClient",
    "TokenTracker",
    # Trainer
    "MonteCarloTrainer",
    "Callback",
    "LoggingCallback",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    # Envs
    "make_env",
    "register_env",
    "list_envs",
    # Store
    "LocalStore",
    "PolicyCheckpoint",
    "TrajectoryStore",
    "RunLogger",
    # Convenience
    "train",
    "load_config",
]
