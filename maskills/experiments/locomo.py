"""Thin convenience wrapper over maskills for LOCOMO experiments.

Usage:

    from maskills.experiments.locomo import run_experiment

    results = run_experiment("configs/locomo/qa_central_credit.json")

Or programmatically:

    exp = LocomoExperiment.from_json("configs/locomo/qa_central_credit.json")
    exp.setup_manual_split(train_range=(0, 400), test_range=(400, 600))
    results = exp.run()
"""

from __future__ import annotations

from typing import Optional

import maskills
from maskills.config.base import LocomoConfig

# Side-effect import: registers the "locomo" env in the MASkills registry
from maskills.envs.locomo import env as _env_module  # noqa: F401


class LocomoExperiment:
    """One-stop experiment runner for the LOCOMO benchmark."""

    def __init__(self, config: LocomoConfig):
        self.config = config

        self.env = maskills.make_env("locomo", config)
        tool_library = ""
        if hasattr(self.env, "get_tool_library"):
            tool_library = self.env.get_tool_library()

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
                config.get_optimizer_llm(), tool_library=tool_library,
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
        """Manually split tasks by index ranges (sequential, no shuffle)."""
        all_tasks = self.env.task_loader.tasks
        t0, t1 = train_range
        e0, e1 = test_range
        self.env.task_loader.train_tasks = all_tasks[t0:t1]
        self.env.task_loader.test_tasks = all_tasks[e0:e1]
        # train_test_split < 1.0 triggers test evaluation in trainer.
        self.config.train_test_split = 0.5

    def setup_conversation_split(
        self,
        n_train: int = 2,
        n_val: int = 2,
        n_test: int = 6,
        seed: int = 42,
    ) -> dict:
        """Split by long conversation: disjoint conversations per split.

        Returns the ``sample_id`` assignment per split.  ``n_val`` / ``n_test``
        are set to the full pool sizes so the trainer evaluates every val /
        test task (needed for stable per-category breakdowns).
        """
        assignment = self.env.task_loader.split_by_conversation(
            n_train, n_val, n_test, seed
        )
        tl = self.env.task_loader
        # train_test_split < 1.0 triggers test evaluation in trainer.
        self.config.train_test_split = 0.5
        self.config.n_val = len(tl.val_tasks)
        self.config.n_test = len(tl.test_tasks)
        return assignment

    def run(self) -> list[dict]:
        """Run training and return per-iteration metrics."""
        return self.trainer.train()

    def get_final_policies(self) -> dict:
        return self.trainer.checkpoint.get_policies()

    @classmethod
    def from_json(
        cls,
        config_path: str,
        overrides: Optional[dict] = None,
    ) -> "LocomoExperiment":
        config = LocomoConfig.from_json(config_path, overrides=overrides)
        return cls(config)


def run_experiment(
    config_path: str,
    overrides: Optional[dict] = None,
    train_range: Optional[tuple[int, int]] = None,
    test_range: Optional[tuple[int, int]] = None,
) -> list[dict]:
    """One-liner to run a LOCOMO experiment.

    Args:
        config_path: Path to JSON config file.
        overrides: Optional dict of config field overrides.
        train_range: Optional (start, end) for manual train split.
        test_range: Optional (start, end) for manual test split.
    """
    exp = LocomoExperiment.from_json(config_path, overrides=overrides)
    if train_range and test_range:
        exp.setup_manual_split(train_range, test_range)
    return exp.run()
