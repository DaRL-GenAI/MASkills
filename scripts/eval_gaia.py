#!/usr/bin/env python3
"""Evaluate a GAIA skill library on a held-out test set.

Two engines, because GAIA has two rollout implementations and mixing their
numbers would not be a controlled comparison:

  --engine library  (default)  The on-disk SKILL.md pipeline the trainer uses,
                               so the score is directly comparable to training.
                               Topologies: centralized, decentralized.
  --engine env                 The GaiaEnv rollout, the same code path the other
                               benchmarks use. Adds hybrid, so this is the engine
                               for comparing topologies against each other.

Examples:
    python scripts/eval_gaia.py --topology decentralized \\
        --skills <library dir> --input data/gaia/test65.jsonl

    # the no-skills floor: protocol only, no task knowledge
    python scripts/eval_gaia.py --skills empty --input data/gaia/test65.jsonl

    python scripts/eval_gaia.py --engine env --topology hybrid \\
        --skills <qwen library dir> \\
        --input data/gaia/test_text103.jsonl --model qwen/qwen-2.5-7b-instruct
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maskills.envs.gaia._keys import require_api_keys  # noqa: E402
from maskills.experiments import env_eval  # noqa: E402
from maskills.experiments.gaia import EMPTY, TOPOLOGIES, evaluate, is_empty  # noqa: E402

#: The env engine adds hybrid on top of what the library pipeline implements.
ENV_TOPOLOGIES = env_eval.TOPOLOGIES


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", choices=["library", "env"], default="library")
    p.add_argument("--topology", choices=sorted(set(TOPOLOGIES) | set(ENV_TOPOLOGIES)),
                   default="decentralized")
    p.add_argument("--skills", default=EMPTY,
                   help="the library to evaluate; for decentralized and hybrid, "
                        "the directory holding agent_1/ and agent_2/. Defaults "
                        "to '{EMPTY}': the protocol-only floor, with the tool "
                        "syntax and answer contract but no task knowledge."
                   .format(EMPTY=EMPTY))
    p.add_argument("--input", default="data/gaia/test65.jsonl")
    p.add_argument("--model", default="openai/gpt-4o")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0,
                   help="cap the test pool to this many tasks (deterministic head)")
    p.add_argument("--out", default="", help="results JSONL (library engine)")
    p.add_argument("--out-dir", default="analysis/eval_gaia",
                   help="results directory (env engine)")
    p.add_argument("--max-tokens", type=int, default=1500)

    central = p.add_argument_group("centralized rollout (library engine)")
    central.add_argument("--max-rounds", type=int, default=6)
    central.add_argument("--tool-budget", type=int, default=6)

    dec = p.add_argument_group("decentralized rollout")
    dec.add_argument("--rounds-a1", type=int, default=5)
    dec.add_argument("--rounds-a2", type=int, default=3)
    dec.add_argument("--budget-a1", type=int, default=5)
    dec.add_argument("--budget-a2", type=int, default=3)

    env = p.add_argument_group("env engine")
    env.add_argument("--max-retrieves", type=int, default=3)
    env.add_argument("--max-retries", type=int, default=4)
    env.add_argument("--report-only", action="store_true",
                     help="re-summarize the rows already on disk, run nothing")
    return p


def _run_env_engine(args) -> None:
    """Evaluate through GaiaEnv, the engine that also implements hybrid."""
    import maskills
    from maskills.config.base import GaiaConfig, LLMConfig

    env_eval.use_deterministic_rollouts()

    empty = is_empty(args.skills)
    skills = None if empty else Path(args.skills)
    config = GaiaConfig(
        exp_name=f"eval_gaia_{args.topology}",
        architecture=args.topology,
        main_agent="agent_2",
        num_agents=2,
        benchmark_path=args.input,
        # GaiaEnv renders the SKILL.md directories into the agent prompts
        # itself, so the policies handed to it stay empty.
        # None leaves GaiaEnv on its protocol-only prompts: the no-skills floor.
        agent_skills_dirs=None if empty else {
            "agent_1": str(skills / "agent_1"),
            "agent_2": str(skills / "agent_2")},
        inject_skill_library=False,
        llm=LLMConfig(name=args.model, model_string=args.model,
                      api_key_env_var="OPENAI_API_KEY", max_tokens=4096),
        split_seed=42,
        rounds_a1=args.rounds_a1, rounds_a2=args.rounds_a2,
        budget_a1=args.budget_a1, budget_a2=args.budget_a2,
        max_retrieves=args.max_retrieves, max_tokens=args.max_tokens,
    )

    env = maskills.make_env("gaia", config)
    tasks = list(env.task_loader.test_tasks)
    if args.limit > 0:
        tasks = tasks[: args.limit]
    print(f"=== gaia · {args.topology} · env engine · {len(tasks)} tasks ===")
    print(f"  skills: {args.skills}")
    env_eval.run_eval(
        env, {}, tasks,
        out_dir=Path(args.out_dir), tag=f"gaia_{args.topology}",
        workers=args.workers, max_retries=args.max_retries,
        report_only=args.report_only,
        extra={"task": "gaia", "topology": args.topology, "skills": args.skills},
    )


def main():
    args = build_parser().parse_args()
    require_api_keys(tavily=True)

    if args.engine == "env":
        _run_env_engine(args)
        return

    if args.topology not in TOPOLOGIES:
        raise SystemExit(
            f"--topology {args.topology} is not implemented by the library "
            f"engine (it has {', '.join(sorted(TOPOLOGIES))}). "
            "Re-run with --engine env."
        )
    evaluate(args)


if __name__ == "__main__":
    main()
