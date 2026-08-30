"""Generic Monte Carlo trainer for any MASkills environment."""

from __future__ import annotations

import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

from tqdm import tqdm

from ..config.base import BaseConfig
from ..core.base import BaseCritic, BaseEnvironment, BaseOptimizer, BaseReward, Trajectory
from ..core.policy import AgentPolicy
from ..llm.token_tracker import TokenTracker
from ..store.base import BaseStore
from ..store.checkpoint import PolicyCheckpoint
from ..store.local import LocalStore
from ..store.run_logger import RunLogger
from ..store.trajectory_store import TrajectoryStore
from .callbacks import Callback


class MonteCarloTrainer:
    """Generic Monte Carlo trainer for any MASkills environment.

    Five-phase iteration:
      1. Load policies from checkpoint
      2. Generate trajectories (parallel)
      3. Evaluate & generate gradients
      4. Aggregate & apply gradients
      5. Save updated policies
    """

    def __init__(
        self,
        config: BaseConfig,
        env: BaseEnvironment,
        critic: BaseCritic,
        optimizer: BaseOptimizer,
        reward_fn: Optional[BaseReward] = None,
        store: Optional[BaseStore] = None,
        callbacks: Optional[list[Callback]] = None,
    ):
        self.config = config
        self.env = env
        self.critic = critic
        self.optimizer = optimizer
        self.reward_fn = reward_fn
        self.callbacks = callbacks or []

        # Storage
        self.store = store or LocalStore(config.experiment_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = self.store.create_run(f"{config.exp_name}_{timestamp}", config)
        self.checkpoint = PolicyCheckpoint(
            self.store, self.run_id, config.num_agents,
            task_type=getattr(config, "task_type", None),
        )
        self.trajectory_store = TrajectoryStore(self.store, self.run_id)
        self.run_logger = RunLogger(self.store, self.run_id)

        # Token tracking
        llm = config.llm
        self.token_tracker = TokenTracker(
            model=llm.model_string if llm else "gpt-4o-mini",
            input_price=llm.input_price_per_million if llm else None,
            output_price=llm.output_price_per_million if llm else None,
        )

        self._stats_lock = threading.Lock()
        self._should_stop = False

    def train(self, num_iterations: Optional[int] = None):
        """Main training loop.

        Iteration convention: 1-indexed.  ``iteration=0`` is the empty-skills
        baseline test round (no training, no checkpoint).  Training iterations
        are ``1..num_iterations`` and produce checkpoints ``iter_1..iter_N``.
        """
        n = num_iterations or self.config.num_iterations

        # Auto-detect resume point
        latest = self.checkpoint.store.latest_checkpoint(self.run_id)
        start = (latest + 1) if latest is not None else self.config.start_iteration
        end = n + 1  # exclusive upper bound: iters 1..n inclusive

        self.run_logger.info(f"Starting training from iteration {start} to {n}")

        # Baseline round: evaluate empty-skills policies on the test set
        # before any training begins. Only runs on a fresh run (no checkpoints yet).
        if latest is None:
            self._run_baseline_round()

        for i in range(start, end):
            if self._should_stop:
                self.run_logger.info("Training stopped early")
                break

            for cb in self.callbacks:
                cb.on_iteration_start(i, self)

            stats = self.train_one_iteration(i)

            for cb in self.callbacks:
                cb.on_iteration_end(i, stats, self)

        self.run_logger.info("Training complete!")
        return self.store.load_metrics(self.run_id)

    def train_one_iteration(self, iteration: int) -> dict:
        """Run a single training iteration."""
        self.run_logger.info(f"\n{'='*60}\nIteration {iteration}\n{'='*60}")

        # Phase 1: Load policies
        policies = self.checkpoint.get_policies()
        base_policies = dict(policies)
        self.run_logger.iteration_start(iteration, {k: str(v) for k, v in policies.items()})

        # Phase 2: Collect trajectories (from training set)
        num_traj = self.config.trajectories_per_iteration
        existing = self.trajectory_store.count(iteration)

        if existing >= num_traj:
            self.run_logger.info(f"Loading {num_traj} existing trajectories")
            trajectories = self.trajectory_store.load(iteration, limit=num_traj)
        else:
            self.run_logger.info(f"Generating {num_traj} trajectories")
            trajectories = self._collect_trajectories(policies, iteration, split="train")

        # Phase 2b: Compute verified rewards if available
        if self.reward_fn:
            for traj in trajectories:
                traj.reward = self.reward_fn.compute(traj)

        # Phase 3: Evaluate & generate gradients
        gradient_trajectories = trajectories
        if self.config.mini_batch_size and self.config.mini_batch_size < len(trajectories):
            gradient_trajectories = random.sample(trajectories, self.config.mini_batch_size)

        gradients = self._evaluate_and_generate_gradients(gradient_trajectories, base_policies)

        # Phase 4: Synthesize updated policies via LLM (updates both role & skills)
        # If momentum is enabled, also load θ_{t-1} so the optimizer can extend
        # the previous edit direction.
        prev_policies: dict = {}
        if getattr(self.config, "momentum", 0.0) > 0.0 and iteration > 0:
            try:
                prev_policies = self.checkpoint.get_policies(iteration - 1)
            except Exception as exc:
                self.run_logger.info(f"momentum: prev checkpoint unavailable ({exc}); skipping")
                prev_policies = {}

        synthesis_method = getattr(self.optimizer, "synthesis_method", "rewrite")
        momentum = getattr(self.optimizer, "momentum", 0.0)
        new_policies = {}
        for agent, grads in gradients.items():
            if not grads:
                new_policies[agent] = base_policies[agent]
            else:
                aggregated = self.optimizer.aggregate_gradients(grads)
                prev = prev_policies.get(agent)
                new_policies[agent] = self.optimizer.synthesize_policy(
                    base_policies[agent], grads, agent_name=agent,
                    previous_policy=prev,
                )
                self.store.save_gradients(self.run_id, iteration, agent, grads, aggregated)
                self.run_logger.gradient_saved(
                    iteration, agent, len(grads),
                    synthesis_method=synthesis_method,
                    momentum=momentum,
                    momentum_applied=(momentum > 0.0 and prev is not None),
                )

        # Phase 5: Save checkpoint.  In the 1-indexed convention checkpoint
        # ``iter_N`` IS the result of training iteration N, so we save at
        # ``iteration`` (not ``iteration+1``).
        stats = self._collect_stats(trajectories, prefix="train")
        self.checkpoint.save_policies(iteration, new_policies, stats)
        self._export_autogen_skills(iteration, new_policies)

        # Phase 6a: Evaluate on validation set (every iteration when n_val>0)
        val_stats = self._evaluate_on_val_set(new_policies, iteration)
        if val_stats:
            stats.update(val_stats)

        # Phase 6b: Evaluate on held-out test set.  When ``eval_test_every_iter``
        # is False (default) the larger test pool is only swept at the
        # baseline round and at the FINAL training iteration; otherwise
        # every iteration.
        eval_test_every_iter = getattr(self.config, "eval_test_every_iter", False)
        is_final = iteration >= (self.config.num_iterations or 0)
        if eval_test_every_iter or is_final:
            test_stats = self._evaluate_on_test_set(new_policies, iteration)
            if test_stats:
                stats.update(test_stats)

        self.run_logger.iteration_end(iteration, stats)
        return stats

    def _collect_trajectories(
        self,
        policies: dict[str, AgentPolicy],
        iteration: int,
        split: str = "train",
    ) -> list[Trajectory]:
        """Generate trajectories using the environment."""
        num_traj = self.config.trajectories_per_iteration
        trajectories = []

        # Sample tasks from the environment
        if hasattr(self.env, 'sample_tasks'):
            tasks = self.env.sample_tasks(num_traj, split=split)
        else:
            tasks = [{"task_id": f"task_{i}", "question": f"Task {i}"} for i in range(num_traj)]

        def _run_one(idx, task):
            traj = self.env.collect_trajectory(policies, task)
            self.trajectory_store.save(iteration, idx, traj)
            self.run_logger.episode_saved(iteration, idx, traj.reward)
            return traj

        max_workers = self.config.max_workers
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_run_one, i, task): i
                    for i, task in enumerate(tasks)
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
                    trajectories.append(future.result())
        else:
            for i, task in enumerate(tqdm(tasks, desc="Generating")):
                trajectories.append(_run_one(i, task))

        return trajectories

    def _evaluate_and_generate_gradients(
        self,
        trajectories: list[Trajectory],
        base_policies: dict[str, AgentPolicy],
    ) -> dict[str, list[str]]:
        """Evaluate trajectories and generate per-agent gradients."""
        accumulated = {agent: [] for agent in base_policies}

        def _eval_one(traj: Trajectory):
            eval_result = self.critic.evaluate(traj, base_policies)
            raw_response = eval_result.get("raw_response", "")
            episode_id = traj.metadata.get("episode_id", 0)
            self.run_logger.evaluation_done(
                0, episode_id, self.config.paradigm, raw_response
            )

            # Include supporting passages (e.g. HotpotQA gold context) when
            # the task carries them; the optimizer needs the full evidence
            # base to coach the agent.  Math / coding tasks have empty
            # ``context`` so they fall back to question-only as before.
            _question = traj.task.get("question", traj.task.get("problem", ""))
            _context = traj.task.get("context", "")
            if _context:
                task_context = f"Context:\n{_context}\n\nQuestion: {_question}"
            else:
                task_context = _question
            per_agent = eval_result.get("per_agent", {})

            gradients = {}
            if self.config.paradigm == "central_global":
                shared_grad = self.optimizer.generate_shared_gradient(raw_response, task_context)
                for agent in base_policies:
                    gradients[agent] = shared_grad
            else:
                for agent, agent_eval in per_agent.items():
                    gradients[agent] = self.optimizer.generate_gradient(
                        base_policies[agent], agent_eval, task_context, agent
                    )
            return gradients

        max_workers = getattr(self.config, 'optimizer_workers', 1)
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_eval_one, t): t for t in trajectories}
                for f in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                    ep_grads = f.result()
                    with self._stats_lock:
                        for agent, grad in ep_grads.items():
                            accumulated[agent].append(grad)
        else:
            for traj in tqdm(trajectories, desc="Evaluating"):
                ep_grads = _eval_one(traj)
                for agent, grad in ep_grads.items():
                    accumulated[agent].append(grad)

        return accumulated

    def _run_baseline_round(self):
        """Evaluate untrained (empty-skills) policies on the test set as iter 0.

        No checkpoint is saved at iteration 0 — it is the pre-training reference
        point.  The first training round is iteration 1.
        """
        from ..core.policy import AgentPolicy
        policies = self.checkpoint.get_policies()
        empty_policies = {
            agent: AgentPolicy(role=p.role, skills="") for agent, p in policies.items()
        }
        self.run_logger.info(f"{'='*60}\nBaseline round (empty skills, iteration=0)\n{'='*60}")
        val_stats = self._evaluate_on_val_set(empty_policies, iteration=0)
        test_stats = self._evaluate_on_test_set(empty_policies, iteration=0)
        combined = {}
        if val_stats:
            combined.update(val_stats)
        if test_stats:
            combined.update(test_stats)
        if combined:
            entry = {"type": "baseline", "iteration": 0, **combined}
            self.store.append_metrics(self.run_id, entry)
            msg_parts = []
            if val_stats:
                msg_parts.append(
                    f"val avg_reward={val_stats.get('val_avg_reward', 0):.3f} "
                    f"f1={val_stats.get('val_avg_f1', 0):.3f} n={val_stats.get('val_num_episodes', 0)}"
                )
            if test_stats:
                msg_parts.append(
                    f"test avg_reward={test_stats.get('test_avg_reward', 0):.3f} "
                    f"f1={test_stats.get('test_avg_f1', 0):.3f} n={test_stats.get('test_num_episodes', 0)}"
                )
            self.run_logger.info("Baseline | " + " | ".join(msg_parts))

    def _evaluate_on_test_set(
        self,
        policies: dict[str, AgentPolicy],
        iteration: int,
    ) -> dict:
        """Evaluate policies on the test set (no gradient generation)."""
        train_test_split = getattr(self.config, 'train_test_split', 1.0)
        n_test = getattr(self.config, 'n_test', None) or 0
        # If an explicit sequential split provided n_test, honor it even
        # when train_test_split == 1.0 (the legacy fallback path).
        if train_test_split >= 1.0 and n_test <= 0:
            return {}  # No test set configured

        eval_samples = getattr(self.config, 'eval_samples', None)
        if not hasattr(self.env, 'sample_tasks'):
            return {}

        # Prefer the explicit ``n_test`` count when set; otherwise fall back
        # to ``eval_samples`` or the per-iter trajectory budget.
        num_test = n_test if n_test > 0 else (eval_samples or self.config.trajectories_per_iteration)
        test_tasks = self.env.sample_tasks(num_test, split="test")
        if not test_tasks:
            return {}

        self.run_logger.info(f"Evaluating on {len(test_tasks)} test tasks")
        test_trajectories = []

        def _run_test(idx, task):
            traj = self.env.collect_trajectory(policies, task)
            self.trajectory_store.save(iteration, idx, traj, split="test")
            self.run_logger.episode_saved(iteration, idx, traj.reward)
            return traj

        max_workers = self.config.max_workers
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_run_test, i, task): i
                    for i, task in enumerate(test_tasks)
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc="Test eval"):
                    test_trajectories.append(future.result())
        else:
            for i, task in enumerate(tqdm(test_tasks, desc="Test eval")):
                test_trajectories.append(_run_test(i, task))

        return self._collect_stats(test_trajectories, prefix="test")

    def _evaluate_on_val_set(
        self,
        policies: dict[str, AgentPolicy],
        iteration: int,
    ) -> dict:
        """Evaluate policies on the validation set (no gradient generation).

        Activated whenever the env's task loader has a non-empty
        ``val_tasks`` pool (typically when ``n_val`` is set on the config).
        """
        n_val = getattr(self.config, 'n_val', None) or 0
        if n_val <= 0:
            return {}
        if not hasattr(self.env, 'sample_tasks'):
            return {}

        val_tasks = self.env.sample_tasks(n_val, split="val")
        if not val_tasks:
            return {}

        self.run_logger.info(f"Evaluating on {len(val_tasks)} validation tasks")
        val_trajectories = []

        def _run_val(idx, task):
            traj = self.env.collect_trajectory(policies, task)
            self.trajectory_store.save(iteration, idx, traj, split="val")
            self.run_logger.episode_saved(iteration, idx, traj.reward)
            return traj

        max_workers = self.config.max_workers
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_run_val, i, task): i
                    for i, task in enumerate(val_tasks)
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc="Val eval"):
                    val_trajectories.append(future.result())
        else:
            for i, task in enumerate(tqdm(val_tasks, desc="Val eval")):
                val_trajectories.append(_run_val(i, task))

        return self._collect_stats(val_trajectories, prefix="val")

    def _export_autogen_skills(self, iteration: int, policies: dict[str, AgentPolicy]):
        """Mirror each agent's learned skill library under the run dir.

        Layout::

            <experiment_dir>/<run_id>/skills_autogen/iter_<i>/<agent>/<skill_id>/SKILL.md

        One folder per discrete skill, the same layout a seed library uses,
        so an auto-generated skill can be copied into any other run's
        ``skills_dir`` and loaded unchanged.
        """
        try:
            run_dir = self.store.get_run_dir(self.run_id)
        except AttributeError:
            return  # store without a filesystem backing — skip silently
        from pathlib import Path as _Path
        out_root = _Path(run_dir) / "skills_autogen" / f"iter_{iteration}"
        for agent, policy in policies.items():
            if not policy.skill_library:
                continue
            policy.skill_library.to_skill_md_dir(out_root / agent)

    def _collect_stats(self, trajectories: list[Trajectory], prefix: str = "") -> dict:
        rewards = [t.reward for t in trajectories]
        token_stats = self.token_tracker.get_stats()
        p = f"{prefix}_" if prefix else ""
        stats = {
            "paradigm": self.config.paradigm,
            f"{p}num_episodes": len(trajectories),
            f"{p}avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            f"{p}min_reward": min(rewards, default=0.0),
            f"{p}max_reward": max(rewards, default=0.0),
            f"{p}rewards": rewards,
            **{k: token_stats[k] for k in ['input_tokens', 'output_tokens', 'total_tokens', 'cost_usd']},
        }
        # Aggregate QA token-level metrics (F1 / precision / recall / EM /
        # BLEU) when the env populated ``trajectory.metadata['qa_metrics']``.
        qa_metric_keys = ("f1", "precision", "recall", "em", "bleu")
        accumulators: dict[str, list[float]] = {k: [] for k in qa_metric_keys}
        for traj in trajectories:
            qm = traj.metadata.get("qa_metrics") if hasattr(traj, "metadata") else None
            if not qm:
                continue
            for k in qa_metric_keys:
                if k in qm:
                    accumulators[k].append(float(qm[k]))
        for k, vals in accumulators.items():
            if vals:
                stats[f"{p}avg_{k}"] = sum(vals) / len(vals)

        # Per-category breakdown (LOCOMO sets ``category`` on traj metadata).
        by_cat: dict[int, list[Trajectory]] = {}
        for traj in trajectories:
            c = traj.metadata.get("category") if hasattr(traj, "metadata") else None
            if c:
                by_cat.setdefault(int(c), []).append(traj)
        for c in sorted(by_cat):
            trajs = by_cat[c]
            cr = [t.reward for t in trajs]
            stats[f"{p}cat{c}_n"] = len(trajs)
            stats[f"{p}cat{c}_avg_reward"] = sum(cr) / len(cr) if cr else 0.0
            for metric in ("f1", "bleu"):
                vals = [
                    float(t.metadata["qa_metrics"][metric])
                    for t in trajs
                    if t.metadata.get("qa_metrics")
                    and metric in t.metadata["qa_metrics"]
                ]
                if vals:
                    stats[f"{p}cat{c}_avg_{metric}"] = sum(vals) / len(vals)
        return stats
