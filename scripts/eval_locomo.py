#!/usr/bin/env python3
"""Evaluate a LOCOMO skill library on one question category's test split.

Categories are evaluated separately because the libraries are trained per
category: 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop. The
conversation split is the same 2/2/6 the trainer uses (seed 42), so the test
conversations are ones no library was trained on.

Pass --skills empty for the no-skills floor, or a training checkpoint
(experiments/runs/<run_id>/checkpoints/iter_<i>/) to score a learned library.

Examples:
    python scripts/eval_locomo.py --category 1 --skills empty
    python scripts/eval_locomo.py --category 1 --topology hybrid \\
        --skills experiments/runs/locomo_percat_cat1_multihop_.../checkpoints/iter_3
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
from maskills.config.base import LLMConfig, LocomoConfig  # noqa: E402
from maskills.experiments import env_eval  # noqa: E402

CATEGORY_NAMES = {1: "multihop", 2: "temporal", 3: "opendomain", 4: "singlehop"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--category", type=int, required=True, choices=sorted(CATEGORY_NAMES))
    p.add_argument("--topology", choices=env_eval.TOPOLOGIES, default="decentralized")
    p.add_argument("--skills", default=env_eval.EMPTY,
                   help=f"library directory with agent_1/ and agent_2/, "
                        f"or '{env_eval.EMPTY}' for the no-skills floor")
    p.add_argument("--benchmark-path",
                   default=str(PROJECT_ROOT / "env" / "locomo" / "data" / "locomo10.json"))
    p.add_argument("--model", default="openai/gpt-4o-mini")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--limit", type=int, default=0,
                   help="cap the test pool to this many tasks (deterministic head)")
    p.add_argument("--out-dir", default="analysis/eval_locomo")
    p.add_argument("--label", default="",
                   help="output tag (default: locomo_<category>_<topology>)")
    p.add_argument("--max-grep-calls", type=int, default=3)
    p.add_argument("--retriever-sees-conversation", action="store_true",
                   help="give the retriever the raw conversation as well as grep; "
                        "off by default, matching the training runs")
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
    category = CATEGORY_NAMES[args.category]
    config = LocomoConfig(
        exp_name=f"eval_locomo_cat{args.category}_{args.topology}",
        architecture=args.topology,
        main_agent="agent_2",  # the reasoner produces the final answer
        num_agents=2,
        benchmark_path=args.benchmark_path,
        category_filter=[args.category],
        max_context_tokens=40000,
        chars_per_token=4,
        max_grep_calls=args.max_grep_calls,
        grep_max_lines=20,
        retriever_sees_conversation=args.retriever_sees_conversation,
        inject_skill_library=False,
        locomo_reward_metric="f1_bleu_mean",
        llm=llm, actor_llm=llm,
        split_seed=42, train_test_split=1.0,
        max_workers=args.workers,
    )

    env = maskills.make_env("locomo", config)
    # The same conversation split the trainers use, so the test conversations
    # are ones no library has seen.
    env.task_loader.split_by_conversation(n_train=2, n_val=2, n_test=6, seed=42)
    tasks = list(env.task_loader.test_tasks)
    if args.limit > 0:
        tasks = tasks[: args.limit]

    # Roles stay blank: the env substitutes its own retriever/reasoner prompts.
    policies = env_eval.load_policies(args.skills, num_agents=2,
                                      fill_default_role=False)

    print("=" * 78)
    print(f"LOCOMO cat{args.category} ({category}) · {args.topology} · "
          f"{len(tasks)} test tasks")
    print(f"  actor  : {config.get_actor_llm().model_string}")
    print(f"  skills : {args.skills}  ({env_eval.describe_policies(policies)})")
    print(f"  grep   : {config.max_grep_calls} calls")
    print("=" * 78, flush=True)

    env_eval.run_eval(
        env, policies, tasks,
        out_dir=Path(args.out_dir),
        tag=args.label or f"locomo_cat{args.category}_{args.topology}",
        workers=args.workers, max_retries=args.max_retries,
        report_only=args.report_only,
        extra={"task": f"locomo_{category}", "category": args.category,
               "topology": args.topology, "skills": args.skills},
    )


if __name__ == "__main__":
    main()
