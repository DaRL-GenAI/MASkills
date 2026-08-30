#!/usr/bin/env python3
"""Train a LOCOMO skill library for one question category.

Categories are trained separately because what a good skill looks like differs
sharply between them — a temporal question needs the session date carried
through the handoff, a multi-hop one needs a grep plan per hop. The retriever
(agent_1) and reasoner (agent_2) each evolve their own library.

The grep tool is enabled from the first iteration while the libraries start
empty, so the agent can call it but does not yet know how: teaching that is the
first thing the operators have to do.

Examples:
    python scripts/train_locomo.py --category 1 --iters 5
    python scripts/train_locomo.py --category 2 --init-skills <seed dir>
    python scripts/train_locomo.py --category 1 --ablation credit
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

from maskills.experiments import locomo_percat  # noqa: E402

ABLATIONS = ["none", "credit", "momentum", "rollback", "consolprune"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--category", type=int, required=True,
                   choices=sorted(locomo_percat.CATEGORY_NAMES),
                   help="1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop")
    p.add_argument("--topology", choices=["decentralized", "centralized", "hybrid"],
                   default="decentralized")
    p.add_argument("--init-skills", default="empty",
                   help="root holding agent_{1,2}/cat<N>_<name>/ seed skills. "
                        "Defaults to 'empty': no library ships with the "
                        "repository, so training starts from no skills unless "
                        "you point this at one of your own.")
    p.add_argument("--benchmark-path",
                   default=str(PROJECT_ROOT / "env" / "locomo" / "data" / "locomo10.json"))

    data = p.add_argument_group("data and schedule")
    data.add_argument("--iters", type=int, default=5)
    data.add_argument("--train-per-iter", type=int, default=24)
    data.add_argument("--n-val", type=int, default=0,
                      help="cap validation tasks; 0 = the full per-category pool")
    data.add_argument("--n-test", type=int, default=0,
                      help="cap test tasks; 0 = the full per-category pool")
    data.add_argument("--workers", type=int, default=32)
    data.add_argument("--optimizer-workers", type=int, default=16)

    models = p.add_argument_group("models")
    models.add_argument("--actor-model", default="gpt-4o-mini",
                        help="LLM preset the agents roll out with")
    models.add_argument("--optimizer-model", default="gpt-4o-mini",
                        help="LLM preset for the critic and the operators")

    gate = p.add_argument_group("operators and the validation gate")
    gate.add_argument("--gate-tolerance", type=float, default=0.03,
                      help="accept a candidate library while its held-out reward "
                           "is within this margin of the current one")
    gate.add_argument("--max-skills", type=int, default=4,
                      help="induction will not grow a library beyond this")
    gate.add_argument("--ablation", choices=ABLATIONS, default="none",
                      help="disable one mechanism: credit (uniform per-skill "
                           "credit), momentum (no EMA on skill utility), "
                           "rollback (never reject a candidate), consolprune "
                           "(no consolidation or pruning)")

    gate.add_argument("--evolve-role", action="store_true",
                      help="also evolve each agent's role prompt, not just its "
                           "skills. Off by default: the role carries the "
                           "collaboration protocol the environment parses, so an "
                           "edit there can invalidate every trajectory in a way "
                           "no skill can repair. A proposed role goes through the "
                           "same validation gate as the library.")

    env = p.add_argument_group("environment")
    env.add_argument("--retriever-sees-conversation", action="store_true",
                     help="give the retriever the full conversation instead of a "
                          "session index plus grep; ~4x the input tokens per "
                          "rollout, and off in the paper's runs")
    p.add_argument("--resume-run-id", default="",
                   help="reuse an existing run_id under experiments/runs/ so "
                        "training continues from its latest checkpoint")
    return p


def main():
    args = build_parser().parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    if args.init_skills and args.init_skills != "empty":
        seed_root = Path(args.init_skills)
        if not seed_root.is_absolute():
            seed_root = PROJECT_ROOT / seed_root
        if not seed_root.is_dir():
            raise SystemExit(
                f"No seed library at {seed_root}. Pass a directory holding "
                "agent_1/ and agent_2/, or --init-skills empty."
            )
        args.init_skills = str(seed_root)

    locomo_percat.run(args, PROJECT_ROOT)


if __name__ == "__main__":
    main()
