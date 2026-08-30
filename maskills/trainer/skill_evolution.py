"""Continual skill-evolution trainer (MASkills §4).

``SkillEvolutionTrainer`` replaces the monolithic role+skills rewrite of
:class:`MonteCarloTrainer` with the MASkills closed loop:

  1. collect trajectories (each carries a skill-invocation trace ξ);
  2. **skill-conditioned credit assignment** — a per-skill + residual credit
     per trajectory (:class:`SkillCreditCritic`);
  3. **hierarchical aggregation** — merge per-trajectory credits into one
     stable language gradient ``G(k)`` per skill;
  4. **skill-evolution operators** — refinement / induction / consolidation /
     pruning, each on its own multi-timescale cadence (:class:`OperatorSchedule`);
  5. **validation rollback** — a candidate library is committed only if it does
     not regress held-out reward by more than δ (a trust-region constraint).

Only :meth:`train_one_iteration` is overridden; the base ``train()`` loop,
baseline round and held-out evaluation are reused unchanged.
"""

from __future__ import annotations

import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Tuple

from tqdm import tqdm

from ..core.base import Trajectory
from ..core.policy import AgentPolicy
from ..core.skill_credit import ResidualCredit, SkillCredit, TrajectorySkillCredit
from ..core.skill_operators import SkillGradient
from ..core.skills import SkillLibrary
from .monte_carlo import MonteCarloTrainer


@dataclass
class OperatorSchedule:
    """Multi-timescale cadence for the four skill-evolution operators.

    ``every == 0`` disables an operator.  ``every == 1`` runs it every
    iteration.  Refinement is typically frequent; induction / consolidation /
    pruning act on longer horizons once evidence has accumulated.
    """

    refine_every: int = 1
    induct_every: int = 2
    consolidate_every: int = 3
    prune_every: int = 3

    def active(self, iteration: int) -> Dict[str, bool]:
        def on(every: int) -> bool:
            return every > 0 and iteration % every == 0
        return {
            "refine": on(self.refine_every),
            "induct": on(self.induct_every),
            "consolidate": on(self.consolidate_every),
            "prune": on(self.prune_every),
        }


class SkillEvolutionTrainer(MonteCarloTrainer):
    """Trainer that evolves discrete skill libraries under skill-level credit.

    Expects ``critic`` to be a :class:`SkillCreditCritic` and ``optimizer`` a
    :class:`SkillEvolutionOptimizer`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.schedule = OperatorSchedule(
            refine_every=getattr(self.config, "refine_every", 1),
            induct_every=getattr(self.config, "induct_every", 2),
            consolidate_every=getattr(self.config, "consolidate_every", 3),
            prune_every=getattr(self.config, "prune_every", 3),
        )
        self.hard_threshold = getattr(self.config, "hard_trajectory_threshold", 0.5)
        self.max_skills = getattr(self.config, "max_skills_per_agent", 12)
        self.delta = getattr(self.config, "skill_eval_delta", 0.02)

    # ── main iteration ───────────────────────────────────────────────────
    def train_one_iteration(self, iteration: int) -> dict:
        self.run_logger.info(f"\n{'='*60}\nIteration {iteration} (skill evolution)\n{'='*60}")

        # Phase 1: load policies (discrete skill libraries).
        policies = self.checkpoint.get_policies()
        base_policies = {a: p.copy() for a, p in policies.items()}
        self.run_logger.iteration_start(iteration, {k: str(v) for k, v in policies.items()})

        # Phase 2: collect (or reload) trajectories with skill traces.
        num_traj = self.config.trajectories_per_iteration
        if self.trajectory_store.count(iteration) >= num_traj:
            self.run_logger.info(f"Loading {num_traj} existing trajectories")
            trajectories = self.trajectory_store.load(iteration, limit=num_traj)
        else:
            self.run_logger.info(f"Generating {num_traj} trajectories")
            trajectories = self._collect_trajectories(policies, iteration, split="train")
        if self.reward_fn:
            for traj in trajectories:
                traj.reward = self.reward_fn.compute(traj)

        grad_trajs = trajectories
        if self.config.mini_batch_size and self.config.mini_batch_size < len(trajectories):
            grad_trajs = random.sample(trajectories, self.config.mini_batch_size)

        # Phase 3: skill-conditioned credit assignment.
        credits = self._collect_skill_credit(grad_trajs, base_policies)

        # Phase 3b: update per-skill utility / invocation bookkeeping.
        self._update_skill_stats(base_policies, grad_trajs, credits)

        # Phase 3c: hierarchical aggregation -> G(k) per skill, per agent.
        skill_grads, residuals, hard_evidence = self._aggregate(
            credits, base_policies, grad_trajs
        )

        # Phase 4: apply scheduled skill-evolution operators -> candidates.
        schedule = self.schedule.active(iteration)
        self.run_logger.info(
            "Operators active this iteration: "
            + ", ".join(k for k, v in schedule.items() if v) or "(none)"
        )
        candidate_policies: Dict[str, AgentPolicy] = {}
        op_stats: Dict[str, dict] = {}
        evolve_role = getattr(self.config, "evolve_role", False)
        for agent, policy in base_policies.items():
            cand_lib, log = self._evolve_library(
                agent, policy,
                skill_grads.get(agent, {}),
                residuals.get(agent, ""),
                hard_evidence.get(agent, []),
                schedule,
            )
            # The role is held fixed unless the run opted in. What drives an
            # edit is the residual -- the failures credit assignment could not
            # attribute to any skill, which is exactly the part of the prompt
            # the library does not cover.
            role = policy.role
            if evolve_role:
                new_role = self.optimizer.refine_role(
                    policy.role, residuals.get(agent, ""),
                    hard_evidence.get(agent, []), agent_name=agent)
                if new_role:
                    role = new_role
                    log["role_edited"] = True
                    self.run_logger.info(
                        f"  [{agent}] role edited "
                        f"({len(policy.role)} -> {len(new_role)} chars)")
            candidate_policies[agent] = AgentPolicy(role=role, skill_library=cand_lib)
            op_stats[agent] = log

        # Phase 5: validation rollback (trust-region commit).
        accepted, gate = self._validation_gate(base_policies, candidate_policies)

        # Phase 6: checkpoint + autogen export.
        stats = self._collect_stats(trajectories, prefix="train")
        stats.update(self._operator_metrics(op_stats, accepted, base_policies, gate))
        self.checkpoint.save_policies(iteration, accepted, stats)
        self._export_autogen_skills(iteration, accepted)
        self._log_operator_summary(iteration, op_stats, gate, accepted)

        # Phase 6b: held-out validation / test passes (reuse base helpers).
        val_stats = self._evaluate_on_val_set(accepted, iteration)
        if val_stats:
            stats.update(val_stats)
        eval_test_every_iter = getattr(self.config, "eval_test_every_iter", False)
        is_final = iteration >= (self.config.num_iterations or 0)
        if eval_test_every_iter or is_final:
            test_stats = self._evaluate_on_test_set(accepted, iteration)
            if test_stats:
                stats.update(test_stats)

        self.run_logger.iteration_end(iteration, stats)
        return stats

    # ── Phase 3: skill credit ────────────────────────────────────────────
    def _collect_skill_credit(
        self, trajectories: List[Trajectory], policies: Dict[str, AgentPolicy]
    ) -> List[TrajectorySkillCredit]:
        results: List[TrajectorySkillCredit] = []
        workers = getattr(self.config, "optimizer_workers", 1)

        def _one(traj: Trajectory) -> TrajectorySkillCredit:
            return self.critic.evaluate_skill_credit(traj, policies)

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_one, t) for t in trajectories]
                for f in tqdm(as_completed(futures), total=len(futures), desc="Skill credit"):
                    results.append(f.result())
        else:
            for traj in tqdm(trajectories, desc="Skill credit"):
                results.append(_one(traj))
        return results

    def _update_skill_stats(
        self,
        policies: Dict[str, AgentPolicy],
        trajectories: List[Trajectory],
        credits: List[TrajectorySkillCredit],
    ) -> None:
        """Accumulate invocation counts (from ξ) and a smoothed utility
        estimate (from credit ``utility_delta``) onto each skill in place."""
        # Invocation counts from the skill traces.
        inv: Dict[str, Dict[str, int]] = {}
        for traj in trajectories:
            for agent, entries in (getattr(traj, "skill_trace", {}) or {}).items():
                for entry in entries:
                    for sid in entry.get("skills", []):
                        inv.setdefault(agent, {}).setdefault(sid, 0)
                        inv[agent][sid] += 1
        # Utility deltas from the per-skill credits.
        deltas: Dict[str, Dict[str, List[float]]] = {}
        for tc in credits:
            for c in tc.skill_credits:
                deltas.setdefault(c.agent, {}).setdefault(c.skill_id, []).append(c.utility_delta)
        for agent, policy in policies.items():
            for skill in policy.skill_library:
                skill.invocations += inv.get(agent, {}).get(skill.skill_id, 0)
                ds = deltas.get(agent, {}).get(skill.skill_id, [])
                if ds:
                    mean_d = sum(ds) / len(ds)
                    # exponential moving average toward the batch mean
                    skill.utility = round(0.6 * skill.utility + 0.4 * mean_d, 4)

    def _aggregate(
        self,
        credits: List[TrajectorySkillCredit],
        policies: Dict[str, AgentPolicy],
        trajectories: List[Trajectory],
    ) -> Tuple[Dict[str, Dict[str, SkillGradient]], Dict[str, str], Dict[str, List[str]]]:
        """Aggregate credits into per-agent {skill_id: G(k)}, residual
        summaries, and hard-case evidence for induction."""
        # Bucket skill credits by agent -> skill_id.
        by_agent_skill: Dict[str, Dict[str, List[SkillCredit]]] = {}
        residuals_raw: Dict[str, List[ResidualCredit]] = {}
        for tc in credits:
            for c in tc.skill_credits:
                by_agent_skill.setdefault(c.agent, {}).setdefault(c.skill_id, []).append(c)
            for agent, r in tc.residuals.items():
                residuals_raw.setdefault(agent, []).append(r)

        skill_grads: Dict[str, Dict[str, SkillGradient]] = {}
        for agent, policy in policies.items():
            cbs = by_agent_skill.get(agent, {})
            skill_grads[agent] = self.optimizer.aggregate_skill_gradients(
                policy.skill_library, cbs
            )

        # Residual summary per agent: merge residuals that flag a missing skill.
        residual_summary: Dict[str, str] = {}
        for agent, rs in residuals_raw.items():
            flagged = [r for r in rs if r.needs_new_skill and r.summary]
            seen, parts = set(), []
            for r in flagged:
                key = r.summary[:80]
                if key in seen:
                    continue
                seen.add(key)
                parts.append(r.summary + (f" (missing: {r.missing_capability})"
                                          if r.missing_capability else ""))
            residual_summary[agent] = " | ".join(parts[:5])

        # Hard-case evidence H_i (paper §4.4): every trajectory whose reward is
        # <= threshold is a hard case for each participating agent.  We do NOT
        # gate on the critic's ``needs_new_skill`` self-report — an agent that
        # repeatedly fails evidently lacks coverage; the residual text, when
        # present, only enriches the evidence handed to the induction LLM.
        hard_evidence: Dict[str, List[str]] = {}
        for tc in credits:
            if tc.reward > self.hard_threshold:
                continue
            for agent in policies:
                r = tc.residuals.get(agent)
                ev = f"reward={tc.reward:.2f}"
                if r and r.summary:
                    ev += f"; residual={r.summary}"
                    if r.missing_capability:
                        ev += f"; missing={r.missing_capability}"
                hard_evidence.setdefault(agent, []).append(ev)
        return skill_grads, residual_summary, hard_evidence

    # ── Phase 4: operators ───────────────────────────────────────────────
    def _evolve_library(
        self,
        agent: str,
        base_policy: AgentPolicy,
        skill_grads: Dict[str, SkillGradient],
        residual_summary: str,
        hard_evidence: List[str],
        schedule: Dict[str, bool],
    ) -> Tuple[SkillLibrary, dict]:
        """Apply the scheduled operators to a copy of the agent's library."""
        lib = base_policy.skill_library.copy()
        log = {"refined": 0, "induced": 0, "consolidated": 0, "pruned": 0}

        # Refinement — localized edits to skills the gradient says to refine.
        if schedule["refine"]:
            for sid, grad in skill_grads.items():
                if grad.verdict != "refine":
                    continue
                skill = lib.get(sid)
                if skill is None:
                    continue
                refined = self.optimizer.refine_skill(skill, grad)
                if refined is not None:
                    lib.replace(refined)
                    log["refined"] += 1

        # Consolidation — merge functionally overlapping skills.
        if schedule["consolidate"]:
            groups = self.optimizer.detect_redundancy_groups(lib, skill_grads)
            for group in groups:
                macro = self.optimizer.consolidate_skills(lib, group)
                if macro is not None and lib.merge(group, macro) is not None:
                    log["consolidated"] += 1

        # Pruning — drop low-utility skills.
        if schedule["prune"]:
            for sid in self.optimizer.select_skills_to_prune(lib, skill_grads):
                if lib.remove(sid):
                    log["pruned"] += 1

        # Induction — propose a new skill for uncovered hard cases.
        if schedule["induct"] and len(lib) < self.max_skills:
            if hard_evidence or residual_summary:
                new_skill = self.optimizer.induce_skill(hard_evidence, lib, residual_summary)
                if new_skill is not None:
                    lib.add(new_skill)
                    log["induced"] += 1
        return lib, log

    # ── Phase 5: validation rollback ─────────────────────────────────────
    def _validation_gate(
        self,
        base_policies: Dict[str, AgentPolicy],
        candidate_policies: Dict[str, AgentPolicy],
    ) -> dict:
        """Commit each agent's candidate library only if it does not regress
        held-out reward by more than δ (Eq.18).  Returns (accepted, gate)."""
        n_val = getattr(self.config, "n_val", 0) or 0
        if n_val <= 0 or not hasattr(self.env, "sample_tasks"):
            return dict(candidate_policies), {"status": "disabled"}
        val_tasks = self.env.sample_tasks(n_val, split="val")
        if not val_tasks:
            return dict(candidate_policies), {"status": "no_val_tasks"}

        base_score = self._val_score(base_policies, val_tasks)
        accepted: Dict[str, AgentPolicy] = {}
        per_agent: Dict[str, dict] = {}
        for agent, candidate in candidate_policies.items():
            base = base_policies[agent]
            # The role counts as a change too: when evolve_role is on, an
            # iteration can edit the role and leave the library alone, and
            # comparing only the library would silently discard that edit.
            changed = (
                candidate.skill_library.to_dict() != base.skill_library.to_dict()
                or candidate.role != base.role
            )
            if not changed:
                accepted[agent] = base
                per_agent[agent] = {"changed": False, "accepted": True}
                continue
            trial = dict(base_policies)
            trial[agent] = candidate
            score = self._val_score(trial, val_tasks)
            ok = score >= base_score - self.delta
            accepted[agent] = candidate if ok else base
            per_agent[agent] = {
                "changed": True, "accepted": ok,
                "val_score": round(score, 4), "base_score": round(base_score, 4),
            }
        return accepted, {"status": "active", "base_score": round(base_score, 4),
                          "per_agent": per_agent}

    def _val_score(self, policies: Dict[str, AgentPolicy], tasks: list) -> float:
        """Mean reward of ``policies`` over a fixed list of validation tasks."""
        trajs: List[Trajectory] = []

        def _run(task: dict) -> Trajectory:
            traj = self.env.collect_trajectory(policies, task)
            if self.reward_fn:
                traj.reward = self.reward_fn.compute(traj)
            return traj

        workers = self.config.max_workers
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_run, t) for t in tasks]
                for f in as_completed(futures):
                    trajs.append(f.result())
        else:
            trajs = [_run(t) for t in tasks]
        rewards = [t.reward for t in trajs]
        return sum(rewards) / len(rewards) if rewards else 0.0

    # ── metrics / logging ────────────────────────────────────────────────
    def _operator_metrics(
        self,
        op_stats: Dict[str, dict],
        accepted: Dict[str, AgentPolicy],
        base_policies: Dict[str, AgentPolicy],
        gate: dict,
    ) -> dict:
        totals = {"refined": 0, "induced": 0, "consolidated": 0, "pruned": 0}
        for log in op_stats.values():
            for k in totals:
                totals[k] += log.get(k, 0)
        rollbacks = 0
        if gate.get("status") == "active":
            rollbacks = sum(
                1 for a in gate["per_agent"].values()
                if a.get("changed") and not a.get("accepted")
            )
        return {
            "skill_refined": totals["refined"],
            "skill_induced": totals["induced"],
            "skill_consolidated": totals["consolidated"],
            "skill_pruned": totals["pruned"],
            "skill_rollbacks": rollbacks,
            "skill_library_size": {a: len(p.skill_library) for a, p in accepted.items()},
            "skill_validation_gate": gate.get("status"),
        }

    def _log_operator_summary(
        self, iteration: int, op_stats: Dict[str, dict], gate: dict,
        accepted: Dict[str, AgentPolicy],
    ) -> None:
        for agent, log in op_stats.items():
            ga = gate.get("per_agent", {}).get(agent, {}) if gate.get("status") == "active" else {}
            verdict = ""
            if ga.get("changed"):
                verdict = " ACCEPTED" if ga.get("accepted") else " ROLLED-BACK"
                verdict += f" (val {ga.get('val_score')} vs base {ga.get('base_score')})"
            self.run_logger.info(
                f"  [{agent}] refine={log['refined']} induct={log['induced']} "
                f"consolidate={log['consolidated']} prune={log['pruned']} "
                f"-> {len(accepted[agent].skill_library)} skills{verdict}"
            )
