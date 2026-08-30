"""HotpotQA skill-evolution run.

Unlike GAIA, the library here lives in memory: each agent's ``SkillLibrary``
is part of its policy, and :class:`~maskills.trainer.skill_evolution.SkillEvolutionTrainer`
runs all the iterations in one process, applying the four operators under
skill-level credit with a validation gate between them.

The starting library is what the run is usually varying:

* a directory of seed skills, the normal setting;
* ``empty``, which asks whether the operators can build a useful library from
  nothing;
* a deliberately unhelpful directory, which asks whether they can recover
  from a harmful one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from maskills.core.skills import SkillLibrary, load_skills_dir
from maskills.experiments.language import LanguageExperiment

#: Written in when a delegating topology is seeded. The default role generator
#: only describes the decentralized chain, and centralized and hybrid both run
#: a main agent that queries a sub-agent through ``<retrieve>``.
_CENTRALIZED_ROLES = {
    "agent_1": (
        "You are the MAIN agent in a centralized two-agent QA system.\n"
        "- You see only the QUESTION, not the underlying context passages.\n"
        "- Call <retrieve>QUERY</retrieve> to delegate evidence gathering to the "
        "retrieval sub-agent; it returns a <retrieve_result> block of cited quotes.\n"
        "- YOUR final text response IS the final answer that will be evaluated.\n"
        "- For HotPotQA, your FINAL answer must be the shortest faithful form of the "
        "answer. Output ONLY that answer string."
    ),
    "agent_2": (
        "You are the RETRIEVAL SUB-AGENT in a centralized two-agent QA system.\n"
        "- You are invoked as a tool by the main agent, in an isolated context.\n"
        "- You see the full task (question + context passages) plus the main "
        "agent's focus query.\n"
        "- You do NOT answer the question. You return a small cited evidence pack.\n"
        "Please provide your evidence pack."
    ),
}


def _print_metrics_table(metrics: list, tag: str) -> None:
    header = (f"{'iter':<8}{'split':<7}{'reward':<9}{'F1':<9}{'EM':<9}"
              f"{'refined':<9}{'induced':<9}{'consol.':<9}{'pruned':<8}{'rollbk':<8}")
    print("\n" + "=" * 72)
    print(f"PER-ITERATION METRICS — {tag}")
    print("=" * 72)
    print(header)
    print("-" * len(header))

    def cell(m, key):
        value = m.get(key)
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value) if value is not None else "-"

    seen = set()
    for m in metrics:
        iteration = m.get("iteration", m.get("type", "?"))
        for split in ("train", "test"):
            if not any(k.startswith(f"{split}_") for k in m):
                continue
            if (iteration, split) in seen:
                continue
            seen.add((iteration, split))
            print(f"{str(iteration):<8}{split:<7}"
                  f"{cell(m, f'{split}_avg_reward'):<9}{cell(m, f'{split}_avg_f1'):<9}"
                  f"{cell(m, f'{split}_avg_em'):<9}"
                  f"{cell(m, 'skill_refined'):<9}{cell(m, 'skill_induced'):<9}"
                  f"{cell(m, 'skill_consolidated'):<9}{cell(m, 'skill_pruned'):<8}"
                  f"{cell(m, 'skill_rollbacks'):<8}")


def run(args, project_root: Path) -> list:
    """Train a HotpotQA library end to end; return the per-iteration metrics."""
    from_empty = args.init_skills == "empty"
    tag = f"{args.topology}_{args.tag}" if args.tag else args.topology

    config_path = project_root / "configs" / "language_task" / "qa_hotpot_decentralized.json"
    experiment = LanguageExperiment.from_json(str(config_path), overrides={
        "benchmark_path": args.benchmark_path,
        "experiment_dir": str(project_root / "experiments"),
        "checkpoint_dir": str(project_root / "experiments" / f"ckpt_skillevo_{tag}"),
        "exp_name": f"skillevo_{tag}",
        "architecture": args.topology,
        "num_iterations": args.iters,
        "trajectories_per_iteration": args.n_train,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "n_test": args.n_test,
        "max_workers": args.workers,
        "optimizer_workers": args.workers,
        **({"llm": args.actor_model, "actor_llm": args.actor_model}
           if args.actor_model else {}),
        **({"optimizer_llm": args.optimizer_model} if args.optimizer_model else {}),
        "eval_test_every_iter": True,
        "include_context": True,
        "max_search_calls": args.max_search_calls,
        "inject_skill_library": False,  # seed skills live in the trainable library
        # ── skill-evolution knobs ──
        "skill_evolution": True,
        "skill_eval_delta": args.gate_tolerance,
        "refine_every": 1,
        # From empty there is nothing to refine early on, so induct every
        # iteration to let the library bootstrap; otherwise the paper's cadence.
        "induct_every": 1 if from_empty else 2,
        "consolidate_every": 2,
        "prune_every": 2,
        "max_skills_per_agent": args.max_skills,
        "evolve_role": args.evolve_role,
        "hard_trajectory_threshold": 0.5,
        "answer_brevity_hint": True,  # HotpotQA gold answers are 1-2 words
    })

    print("=" * 72)
    print(f"MASKILLS HotpotQA — topology={args.topology}, init-skills={args.init_skills}")
    print(f"split: {args.n_train} train / {args.n_val} val / {args.n_test} test "
          f"| iters={args.iters}")

    if from_empty:
        # No seed checkpoint: iteration 1 starts from empty libraries and the
        # trainer runs an empty-skills baseline round (iteration 0) first.
        print("starting from EMPTY libraries; inducting every iteration")
        print("=" * 72, flush=True)
    else:
        seed_root = Path(args.init_skills)
        policies = experiment.trainer.checkpoint._generate_defaults()
        for agent in ("agent_1", "agent_2"):
            seed_dir = seed_root / agent
            skills = load_skills_dir(seed_dir)
            if not skills:
                raise SystemExit(f"No SKILL.md found under {seed_dir}")
            policies[agent].skill_library = SkillLibrary(skills=skills)
            if args.topology in ("centralized", "hybrid"):
                policies[agent].role = _CENTRALIZED_ROLES[agent]

        # Saved as the iter-0 checkpoint, so training iteration 1 starts here.
        experiment.trainer.checkpoint.save_policies(
            0, policies, stats={"note": f"seed:{seed_root}"})

        print(f"seed skills: {seed_root}/agent_1, agent_2")
        for agent in ("agent_1", "agent_2"):
            library = policies[agent].skill_library
            print(f"  {agent}: {len(library)} seed skill(s) -> {library.ids()}")
        print("=" * 72, flush=True)

    metrics = experiment.run()
    _print_metrics_table(metrics, tag)

    out_path = project_root / "experiments" / f"ckpt_skillevo_{tag}" / "metrics_summary.json"
    os.makedirs(out_path.parent, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nSaved: {out_path}")
    return metrics
