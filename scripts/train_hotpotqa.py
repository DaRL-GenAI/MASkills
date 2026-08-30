#!/usr/bin/env python3
"""Train a HotpotQA skill library with the MASkills evolution loop.

One invocation runs every iteration: each rolls the current libraries out on a
training batch, assigns credit to the skills the trajectories invoked, applies
the four operators, and keeps the result only if it holds up on the validation
split.

Examples:
    # the paper's run
    python scripts/train_hotpotqa.py --n-train 100 --n-val 50 --n-test 100 --iters 5

    # seed from a library of your own
    python scripts/train_hotpotqa.py --init-skills <seed dir> --iters 5

The defaults are a deliberately tiny smoke run (4 train / 3 val / 3 test /
2 iterations) so the whole pipeline can be exercised against the real API for
a few cents before committing to a full one.
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

from maskills.experiments import hotpotqa  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--topology", choices=["decentralized", "centralized", "hybrid"],
                   default="decentralized")
    p.add_argument("--init-skills", default="empty",
                   help="seed library directory holding agent_1/ and agent_2/. "
                        "Defaults to 'empty': no library ships with the "
                        "repository, so training starts from no skills unless "
                        "you point this at one of your own.")
    p.add_argument("--benchmark-path",
                   default=str(PROJECT_ROOT / "env" / "lang_benchmark" / "HotPotQA"))

    data = p.add_argument_group("data and schedule")
    data.add_argument("--n-train", type=int, default=4)
    data.add_argument("--n-val", type=int, default=3)
    data.add_argument("--n-test", type=int, default=3)
    data.add_argument("--iters", type=int, default=2)
    data.add_argument("--workers", type=int, default=8)
    data.add_argument("--max-search-calls", type=int, default=3)

    models = p.add_argument_group("models")
    models.add_argument("--actor-model", default="",
                        help="LLM preset the agents roll out with "
                             "(default: whatever the config names)")
    models.add_argument("--optimizer-model", default="",
                        help="LLM preset for the critic and the operators "
                             "(default: the actor's model)")

    gate = p.add_argument_group("operators and the validation gate")
    gate.add_argument("--gate-tolerance", type=float, default=0.03,
                      help="accept a candidate library while its held-out reward "
                           "is within this margin of the current one")
    gate.add_argument("--max-skills", type=int, default=8,
                      help="induction will not grow a library beyond this")

    gate.add_argument("--evolve-role", action="store_true",
                      help="also evolve each agent's role prompt, not just its "
                           "skills. Off by default: the role carries the "
                           "collaboration protocol the environment parses, so an "
                           "edit there can invalidate every trajectory in a way "
                           "no skill can repair. A proposed role goes through the "
                           "same validation gate as the library.")

    p.add_argument("--tag", default="",
                   help="suffix for the run name and checkpoint directory")
    return p


def main():
    args = build_parser().parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    if args.init_skills != "empty" and not Path(args.init_skills).is_dir():
        raise SystemExit(
            f"No seed library at {args.init_skills}. Pass a directory holding "
            "agent_1/ and agent_2/, or --init-skills empty."
        )
    if not args.tag and args.init_skills == "empty":
        args.tag = "empty"

    hotpotqa.run(args, PROJECT_ROOT)


if __name__ == "__main__":
    main()
