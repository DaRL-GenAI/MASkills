"""Thin convenience wrapper over maskills for language task experiments.

Usage:
    from maskills.experiments.language import run_experiment

    results = run_experiment("configs/language_task/qa_central_credit.json")

Or with overrides:

    results = run_experiment(
        "configs/language_task/qa_central_credit.json",
        overrides={"num_iterations": 3, "data_limit": 50},
    )

Or programmatically:

    exp = LanguageExperiment(config)
    exp.setup_manual_split(train_range=(100, 200), test_range=(0, 100))
    results = exp.run()
"""

from __future__ import annotations

from typing import Optional

import maskills
from maskills.config.base import LanguageTaskConfig

# Side-effect import: registers the "language" env in the MASkills registry
from maskills.envs.language import env as _env_module  # noqa: F401


class LanguageExperiment:
    """One-stop experiment runner for language tasks (QA, Math, Coding).

    Wraps the env, critic, optimizer and trainer into a single object.
    """

    def __init__(self, config: LanguageTaskConfig):
        self.config = config

        self.env = maskills.make_env("language", config)
        tool_library = self.env.get_tool_library() if hasattr(self.env, "get_tool_library") else ""
        self.callbacks = [
            maskills.LoggingCallback(),
            maskills.CheckpointCallback(),
        ]

        # When ``skill_evolution`` is set, run the MASkills continual
        # skill-evolution pipeline (skill-level credit + four operators +
        # validation rollback) instead of the monolithic role+skills rewrite.
        if getattr(config, "skill_evolution", False):
            self.critic = maskills.SkillCreditCritic(config, tool_library=tool_library)
            self.optimizer = maskills.SkillEvolutionOptimizer(
                config.get_optimizer_llm(),
                tool_library=tool_library,
            )
            trainer_cls = maskills.SkillEvolutionTrainer
        else:
            self.critic = maskills.CentralizedCritic(config, tool_library=tool_library)
            self.optimizer = maskills.PolicyGradientOptimizer(
                config.get_optimizer_llm(),
                synthesis_method=getattr(config, "synthesis_method", "rewrite"),
                momentum=getattr(config, "momentum", 0.0),
                tool_library=tool_library,
            )
            trainer_cls = maskills.MonteCarloTrainer

        self.trainer = trainer_cls(
            config=config,
            env=self.env,
            critic=self.critic,
            optimizer=self.optimizer,
            callbacks=self.callbacks,
        )

    def setup_manual_split(
        self,
        train_range: tuple[int, int],
        test_range: tuple[int, int],
    ):
        """Manually split tasks by index ranges (sequential, no shuffle).

        Args:
            train_range: (start, end) indices for training tasks.
            test_range: (start, end) indices for test tasks.
        """
        all_tasks = self.env.task_loader.tasks
        t0, t1 = train_range
        e0, e1 = test_range
        self.env.task_loader.train_tasks = all_tasks[t0:t1]
        self.env.task_loader.test_tasks = all_tasks[e0:e1]
        # Needs to be < 1.0 to trigger test evaluation in trainer
        self.config.train_test_split = 0.5

    def run(self) -> list[dict]:
        """Run training and return metrics."""
        return self.trainer.train()

    def get_final_policies(self) -> dict:
        """Return final trained policies."""
        return self.trainer.checkpoint.get_policies()

    @classmethod
    def from_json(
        cls,
        config_path: str,
        overrides: Optional[dict] = None,
    ) -> "LanguageExperiment":
        """Create experiment from a JSON config file."""
        config = LanguageTaskConfig.from_json(config_path, overrides=overrides)
        return cls(config)


def run_experiment(
    config_path: str,
    overrides: Optional[dict] = None,
    train_range: Optional[tuple[int, int]] = None,
    test_range: Optional[tuple[int, int]] = None,
) -> list[dict]:
    """One-liner to run a language task experiment.

    Args:
        config_path: Path to JSON config file.
        overrides: Optional dict of config field overrides.
        train_range: Optional (start, end) for manual train split.
        test_range: Optional (start, end) for manual test split.

    Returns:
        Training metrics from the run.
    """
    exp = LanguageExperiment.from_json(config_path, overrides=overrides)

    if train_range and test_range:
        exp.setup_manual_split(train_range, test_range)

    return exp.run()
