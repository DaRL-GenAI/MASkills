#!/usr/bin/env python3
"""Train a GAIA skill library for one iteration: K_i -> K_{i+1}.

An iteration rolls the actor out on the training batch under the current
library, sends the failures to the optimizer, applies the operations it
proposes to a candidate copy, and keeps that copy only if it holds up on a
held-out slice. Run it once per iteration, advancing --init-skills each time:

    python scripts/train_gaia.py --topology decentralized --iter 1 \\
        --init-skills <K_0 dir> \\
        --out-skills  <K_1 dir>

    python scripts/train_gaia.py --topology decentralized --iter 2 \\
        --init-skills <K_1 dir> \\
        --out-skills  <K_2 dir>

Topologies:
    centralized    one agent, one library
    decentralized  agent_1 Researcher + agent_2 Solver, one library each
                   (--init-skills points at the directory holding both)

GAIA has no hybrid training: hybrid exists only in the environment used for
evaluation. See scripts/eval_gaia.py --engine env.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maskills.envs.gaia._keys import require_api_keys  # noqa: E402
from maskills.experiments.gaia import ABLATIONS, TOPOLOGIES, train_iteration  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--topology", choices=list(TOPOLOGIES), default="decentralized")
    p.add_argument("--iter", type=int, required=True,
                   help="iteration number i, producing K_i -> K_{i+1}")

    skills = p.add_argument_group("skill library")
    skills.add_argument("--init-skills", default="empty",
                        help="the current library K_i to start from, or 'empty' "
                             "to start from the protocol-only floor")
    skills.add_argument("--out-skills", required=True,
                        help="where the K_{i+1} candidate is written")

    data = p.add_argument_group("data")
    data.add_argument("--train", default="data/gaia/train100.jsonl")
    data.add_argument("--train-n", type=int, default=100,
                      help="training items to roll out (<= file size)")
    data.add_argument("--val-n", type=int, default=20,
                      help="validation-gate subset size, drawn from the train batch")
    data.add_argument("--seed", type=int, default=42)
    data.add_argument("--out-dir", default="",
                      help="iteration metadata directory "
                           "(default: data/maskills_<topology>)")

    models = p.add_argument_group("models")
    models.add_argument("--actor-model", default="openai/gpt-4o")
    models.add_argument("--optimizer-model", default="openai/gpt-5.1")
    models.add_argument("--optimizer-temp", type=float, default=0.5)
    models.add_argument("--workers", type=int, default=4)
    models.add_argument("--max-tokens", type=int, default=1500)

    # Centralized rollout budget: one agent, one tool loop.
    central = p.add_argument_group("centralized rollout")
    central.add_argument("--max-rounds", type=int, default=6)
    central.add_argument("--tool-budget", type=int, default=6)

    # Decentralized rollout budget: per-agent rounds and tool calls.
    dec = p.add_argument_group("decentralized rollout")
    dec.add_argument("--rounds-a1", type=int, default=5)
    dec.add_argument("--rounds-a2", type=int, default=3)
    dec.add_argument("--budget-a1", type=int, default=5)
    dec.add_argument("--budget-a2", type=int, default=3)

    gate = p.add_argument_group("operators and the validation gate")
    gate.add_argument("--max-ops", type=int, default=5)
    gate.add_argument("--gate-tolerance", type=int, default=1,
                      help="accept while K_new >= K_i - tolerance; the default of "
                           "1 absorbs single-case noise on a 20-task validation set")
    gate.add_argument("--ablation", choices=list(ABLATIONS), default="none",
                      help="disable one mechanism: credit (hide which tasks were "
                           "correct), rollback (never reject a candidate), "
                           "consolprune (keep only induct and refine)")
    gate.add_argument("--evolve-role", action="store_true",
                      help="also evolve each agent's role prompt, not just its "
                           "skills. Off by default: the role carries the "
                           "collaboration protocol the environment parses, so an "
                           "edit there can invalidate every trajectory in a way "
                           "no skill can repair. A proposed role goes through the "
                           "same validation gate as the library. The paper's GAIA libraries were "
                           "produced with this ON.")
    gate.add_argument("--prior-ops-files", nargs="*", default=[],
                      help="optimizer_raw.txt files from earlier rejected "
                           "iterations, fed back as 'do not re-propose'")

    p.add_argument("--train-rollout-cache", default="",
                   help="reuse this K_i train rollout instead of re-running step 1")
    return p


def main():
    args = build_parser().parse_args()
    require_api_keys(tavily=True)
    if not args.out_dir:
        args.out_dir = f"data/maskills_{args.topology}"
    train_iteration(args)


if __name__ == "__main__":
    main()
