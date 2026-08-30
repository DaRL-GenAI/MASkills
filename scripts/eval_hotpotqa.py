#!/usr/bin/env python3
"""Evaluate a HotpotQA skill library on the held-out test split.

Pass --skills empty for the no-skills floor, or a training checkpoint
(experiments/runs/<run_id>/checkpoints/iter_<i>/) to score a learned library.
Any of the three topologies can be used with any library, which is what makes
the topology comparison a controlled one.

Examples:
    python scripts/eval_hotpotqa.py --skills empty
    python scripts/eval_hotpotqa.py --topology hybrid \\
        --skills experiments/runs/skillevo_decentralized_20260101_120000/checkpoints/iter_4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

import maskills  # noqa: E402
from maskills.config.base import LanguageTaskConfig, LLMConfig  # noqa: E402
from maskills.experiments import env_eval  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--topology", choices=env_eval.TOPOLOGIES, default="decentralized")
    p.add_argument("--skills", default=env_eval.EMPTY,
                   help=f"library directory with agent_1/ and agent_2/, "
                        f"or '{env_eval.EMPTY}' for the no-skills floor")
    p.add_argument("--benchmark-path",
                   default=str(PROJECT_ROOT / "env" / "lang_benchmark" / "HotPotQA"))
    p.add_argument("--model", default="openai/gpt-4o-mini")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--limit", type=int, default=0,
                   help="cap the test pool to this many tasks (deterministic head)")
    p.add_argument("--out-dir", default="analysis/eval_hotpotqa")
    p.add_argument("--label", default="",
                   help="output tag (default: hotpotqa_<topology>)")
    p.add_argument("--max-search-calls", type=int, default=3)
    p.add_argument("--search-limit", type=int, default=5)
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--report-only", action="store_true",
                   help="re-summarize the rows already on disk, run nothing")
    return p


def main():
    args = build_parser().parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    env_eval.use_deterministic_rollouts()

    llm = LLMConfig(
        name=args.model, model_string=args.model,
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key_env_var="OPENAI_API_KEY",
        is_multimodal=True, max_tokens=4096,
    )
    config = LanguageTaskConfig(
        exp_name=f"eval_hotpotqa_{args.topology}",
        task_type="qa",
        architecture=args.topology,
        # The libraries are trained decentralized with agent_1 speaking first
        # and agent_2 producing the final answer. Centralized and hybrid keep
        # that division by making agent_2 the main and agent_1 the one it
        # queries for grounding.
        main_agent="agent_2",
        num_agents=2,
        benchmark_path=args.benchmark_path,
        include_context=True,
        max_search_calls=args.max_search_calls,
        search_limit=args.search_limit,
        inject_skill_library=False,
        inject_tool_reference=False,
        qa_reward_metric="f1",
        llm=llm, actor_llm=llm,
        n_train=500, n_val=200, n_test=800,
        train_test_split=1.0, split_seed=42,
        max_workers=args.workers,
    )

    env = maskills.make_env("language", config)
    tasks = list(env.task_loader.test_tasks)
    if args.limit > 0:
        tasks = tasks[: args.limit]

    policies = env_eval.load_policies(args.skills, num_agents=2, task_type="qa")

    print("=" * 78)
    print(f"HotpotQA · {args.topology} · {len(tasks)} test tasks")
    print(f"  actor  : {config.get_actor_llm().model_string}")
    print(f"  skills : {args.skills}  ({env_eval.describe_policies(policies)})")
    print(f"  search : {config.max_search_calls} calls, k={config.search_limit}")
    print("=" * 78, flush=True)

    env_eval.run_eval(
        env, policies, tasks,
        out_dir=Path(args.out_dir),
        tag=args.label or f"hotpotqa_{args.topology}",
        workers=args.workers, max_retries=args.max_retries,
        report_only=args.report_only,
        extra={"task": "hotpotqa", "topology": args.topology, "skills": args.skills},
    )


if __name__ == "__main__":
    main()
