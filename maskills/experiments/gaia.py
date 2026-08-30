"""GAIA training and evaluation over an on-disk ``SKILL.md`` library.

This is the pipeline the paper's GAIA tables come from. Unlike LOCOMO and
HotpotQA, the library never becomes an in-memory ``SkillLibrary``: it stays a
directory that the actor reads and the optimizer edits, so an iteration is a
directory-to-directory transformation ``K_i -> K_{i+1}``.

One iteration, identical in both topologies:

1. Roll the actor out on the training batch under ``K_i``.
2. Send the failed trajectories to the optimizer, which proposes typed
   operations (induct / refine / consolidate / prune).
3. Snapshot ``K_i`` to a candidate directory and apply the operations there.
4. Re-roll the candidate on a held-out slice of the same batch. Keep it only
   if it does not lose more than ``gate_tolerance`` cases; otherwise restore
   ``K_i`` over the candidate.

The two topologies differ only in how a library is loaded, rolled out,
optimized and written back, which is what :class:`Centralized` and
:class:`Decentralized` supply. Hybrid is not available here — it exists only
in :class:`~maskills.envs.gaia.env.GaiaEnv`, which drives the agents through
the environment rather than through this directory pipeline. See
``scripts/eval_gaia.py --engine env``.
"""

from __future__ import annotations

import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from openai import OpenAI

from maskills.envs.gaia import decentralized as _dec_rollout
from maskills.envs.gaia import protocol as _protocol
from maskills.envs.gaia import tools as _tool_rollout
from maskills.lib_optimizer import propose_ops
from maskills.lib_optimizer_dec import propose_ops_dec
from maskills.skill_lib import apply_ops, load_lib, snapshot_lib, write_skill_file

#: Ablations, each disabling one mechanism while leaving the rest intact.
ABLATIONS = ("none", "credit", "rollback", "consolprune")

#: The slug an operation uses to target a library's root SKILL.md -- the
#: agent's role, as opposed to one of its skills.
ROOT_SLUG = "_root_"

#: Passed as a library path to mean "protocol only, no skills" -- GAIA's
#: no-skills floor. See :mod:`maskills.envs.gaia.protocol` for why the floor
#: is not a literally empty prompt.
EMPTY = "empty"


def is_empty(lib_dir) -> bool:
    return str(lib_dir) == EMPTY


def _targets_role(op: dict) -> bool:
    """True when this operation would edit or remove a library's root."""
    return op.get("op") in ("refine", "prune") and op.get("slug") == ROOT_SLUG


# ── centralized: one agent, one library ────────────────────────────────


def render_system_prompt_for_lib(bundle: dict) -> str:
    """Concatenate root + every sub-skill body as Tier A.

    No progressive disclosure: these libraries are small (<=15 skills), so the
    actor simply gets all of them on every task.
    """
    parts = [bundle["root"]["body"].strip(),
             "\n\n---\n\n# Skill bodies (all loaded as Tier A)\n"]
    for skill in bundle["skills"].values():
        parts.append(f"\n## skill: `{skill['name']}`\n{skill['body'].strip()}\n")
    return "".join(parts)


class Centralized:
    """One agent reading one library, with the SEARCH / BROWSE / COMPUTE stack."""

    name = "centralized"
    #: Rollout knobs this topology accepts, and their defaults.
    rollout_defaults = {"max_tokens": 1500, "max_rounds": 6, "tool_budget": 6}

    @staticmethod
    def load(lib_dir):
        if is_empty(lib_dir):
            return _protocol.bundle(_protocol.CENTRALIZED, "agent")
        return load_lib(lib_dir)

    @staticmethod
    def snapshot(src, dst: Path) -> None:
        """Copy ``src`` over ``dst``, materializing the protocol floor if asked."""
        if is_empty(src):
            write_skill_file(dst / "SKILL.md", "agent",
                             "protocol only, no skills", _protocol.CENTRALIZED)
            return
        snapshot_lib(src, dst)

    @staticmethod
    def describe(state) -> str:
        return f"1 root + {len(state['skills'])} sub-skills"

    @staticmethod
    def rollout(items: list, state, model: str, workers: int,
                on_done=None, **kw) -> list:
        """Run the actor on ``items`` in parallel; one result dict per item.

        ``run_one`` resolves ``render_system_prompt`` and ``select_relevant_b``
        through the tools module's own namespace, so pointing it at a MASkills
        library means swapping those two names there for the duration of the
        call. Rollouts run one after another across iterations, so the swap is
        never concurrent with a different library.
        """
        prompt = render_system_prompt_for_lib(state)
        original = (_tool_rollout.render_system_prompt,
                    _tool_rollout.select_relevant_b)
        _tool_rollout.render_system_prompt = lambda bundle, relevant: prompt
        _tool_rollout.select_relevant_b = lambda item: []

        client = OpenAI()
        results: list = [None] * len(items)
        try:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {
                    ex.submit(_tool_rollout.run_one, client, item, state, model,
                              kw["max_tokens"], kw["max_rounds"], kw["tool_budget"]): i
                    for i, item in enumerate(items)
                }
                for done, future in enumerate(as_completed(futures), start=1):
                    i = futures[future]
                    result = future.result()
                    result["question"] = items[i]["Question"]
                    results[i] = result
                    if on_done:
                        on_done(done, len(items), result)
        finally:
            (_tool_rollout.render_system_prompt,
             _tool_rollout.select_relevant_b) = original
        return results

    @staticmethod
    def propose(state, failed: list, **kw) -> dict:
        return propose_ops(state, failed, **kw)

    @staticmethod
    def apply(lib_dir: Path, ops: list) -> list:
        return apply_ops(lib_dir, ops)

    @staticmethod
    def op_label(op: dict) -> str:
        return f"{op.get('op')}({op.get('slug', op.get('slug_new', '?'))})"

    @staticmethod
    def format_row(r: dict) -> str:
        return (f"L{r.get('Level', '?')} {r.get('task_id', '?')[:8]} "
                f"pred={r.get('pred', '')!r:20s} gold={r.get('gold', '')!r:20s}")


# ── decentralized: Researcher + Solver, one library each ───────────────


class Decentralized:
    """``agent_1`` Researcher (SEARCH / BROWSE) hands off to ``agent_2`` Solver."""

    name = "decentralized"
    rollout_defaults = {"max_tokens": 1500, "rounds_a1": 5, "rounds_a2": 3,
                        "budget_a1": 5, "budget_a2": 3}

    @staticmethod
    def load(lib_dir):
        if is_empty(lib_dir):
            return (_protocol.bundle(_protocol.RESEARCHER, "agent_1"),
                    _protocol.bundle(_protocol.SOLVER, "agent_2"))
        lib_dir = Path(lib_dir)
        return load_lib(lib_dir / "agent_1"), load_lib(lib_dir / "agent_2")

    @staticmethod
    def snapshot(src, dst: Path) -> None:
        dst.mkdir(parents=True, exist_ok=True)
        if is_empty(src):
            for agent, body in _protocol.BY_AGENT.items():
                write_skill_file(dst / agent / "SKILL.md", agent,
                                 "protocol only, no skills", body)
            return
        src = Path(src)
        snapshot_lib(src / "agent_1", dst / "agent_1")
        snapshot_lib(src / "agent_2", dst / "agent_2")

    @staticmethod
    def describe(state) -> str:
        a, b = state
        return (f"K_a: 1 root + {len(a['skills'])} sub-skills, "
                f"K_b: 1 root + {len(b['skills'])} sub-skills")

    @staticmethod
    def rollout(items: list, state, model: str, workers: int,
                on_done=None, **kw) -> list:
        """Run the two-agent system on ``items``.

        ``run_one`` already takes both bundles, so unlike the centralized case
        nothing has to be swapped. A worker that raises is recorded as an
        incorrect result rather than failing the whole rollout — one bad task
        should not cost an iteration's worth of API calls.
        """
        bundle_a, bundle_b = state
        client = OpenAI()
        results: list = [None] * len(items)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(_dec_rollout.run_one, client, item, bundle_a, bundle_b,
                          model, kw["max_tokens"], kw["rounds_a1"], kw["rounds_a2"],
                          kw["budget_a1"], kw["budget_a2"]): i
                for i, item in enumerate(items)
            }
            for done, future in enumerate(as_completed(futures), start=1):
                i = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - one task must not sink the run
                    result = {
                        "task_id": items[i]["task_id"],
                        "Level": items[i]["Level"], "kind": items[i]["kind"],
                        "gold": items[i]["Final answer"], "pred": "",
                        "correct": False, "in_tok": 0, "out_tok": 0,
                        "n_rounds": 0, "n_tool_calls": 0,
                        "a1_rounds": 0, "a2_rounds": 0, "a1_calls": 0, "a2_calls": 0,
                        "request_more_used": False, "handoff": "",
                        "tool_log": [], "raw_turns": [],
                        "error": f"WorkerCrash: {type(exc).__name__}: {exc}",
                    }
                result["question"] = items[i]["Question"]
                results[i] = result
                if on_done:
                    on_done(done, len(items), result)
        return results

    @staticmethod
    def propose(state, failed: list, **kw) -> dict:
        bundle_a, bundle_b = state
        return propose_ops_dec(bundle_a, bundle_b, failed, **kw)

    @staticmethod
    def apply(lib_dir: Path, ops: list) -> list:
        """Route each op to the library of the agent it is tagged for."""
        log = []
        for op in ops:
            agent = op.get("agent", "?")
            if agent not in ("a", "b"):
                log.append({"op": op.get("op"), "status": "missing_agent_tag",
                            "detail": str(op)[:200]})
                continue
            target = lib_dir / ("agent_1" if agent == "a" else "agent_2")
            for entry in apply_ops(target, [op]):
                entry["agent"] = agent
                log.append(entry)
        return log

    @staticmethod
    def op_label(op: dict) -> str:
        return (f"{op.get('op')}(agent={op.get('agent', '?')},"
                f"{op.get('slug', op.get('slug_new', '?'))})")

    @staticmethod
    def format_row(r: dict) -> str:
        return (f"L{r.get('Level', '?')} {r.get('task_id', '?')[:8]} "
                f"a1={r.get('a1_rounds', 0)}/{r.get('a1_calls', 0)} "
                f"a2={r.get('a2_rounds', 0)}/{r.get('a2_calls', 0)} "
                f"pred={r.get('pred', '')!r:25s} gold={r.get('gold', '')!r:25s}")


TOPOLOGIES = {"centralized": Centralized, "decentralized": Decentralized}


# ── shared helpers ─────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.open() if line.strip()]


def _load_prior_ops(paths: list) -> list:
    """Parse ``optimizer_raw.txt`` files from earlier rejected iterations.

    They are fed back to the optimizer as "you already tried these and the
    validation gate threw them out", which stops it re-proposing the same edit
    every iteration.
    """
    prior = []
    for candidate in paths or []:
        path = Path(candidate)
        if not path.exists():
            continue
        try:
            raw = path.read_text().strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
                raw = re.sub(r"\n?```\s*$", "", raw)
            ops = json.loads(raw) if raw else []
            if not isinstance(ops, list):
                continue
            iteration = next((p.split("_")[-1] for p in path.parts
                              if p.startswith("iter_")), "?")
            prior.append({"iter": iteration, "ops": ops})
        except (OSError, ValueError) as exc:
            print(f"  ⚠ could not parse prior-ops file {path}: {exc}")
    return prior


def _reuse_cached_rollout(cache: Path, train_items: list) -> list:
    """Load a previous ``K_i`` rollout, restricted and reordered to this batch."""
    rollouts = _read_jsonl(cache)
    wanted = {item["task_id"] for item in train_items}
    by_id = {r["task_id"]: r for r in rollouts if r["task_id"] in wanted}
    return [by_id[item["task_id"]] for item in train_items if item["task_id"] in by_id]


def _rollout_kwargs(topology, args) -> dict:
    """Pull this topology's rollout knobs off the parsed CLI arguments."""
    return {key: getattr(args, key, default)
            for key, default in topology.rollout_defaults.items()}


# ── one training iteration ─────────────────────────────────────────────


def train_iteration(args) -> dict:
    """Run one ``K_i -> K_{i+1}`` iteration; return the iteration metadata."""
    topology = TOPOLOGIES[args.topology]
    rollout_kw = _rollout_kwargs(topology, args)

    out_dir = Path(args.out_dir) / f"iter_{args.iter}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}\n[MASkills GAIA · {topology.name} · iter {args.iter}]")
    print(f"  cur lib    : {args.init_skills}")
    print(f"  new lib    : {args.out_skills}")
    print(f"  actor      : {args.actor_model}")
    print(f"  optimizer  : {args.optimizer_model}")
    if args.ablation != "none":
        print(f"  ablation   : {args.ablation}")
    print(f"  out_dir    : {out_dir}")
    print(f"{'=' * 70}\n")

    rng = random.Random(args.seed + args.iter)
    items = _read_jsonl(Path(args.train))
    rng.shuffle(items)
    train_items = items[: args.train_n]
    val_subset = rng.sample(train_items, k=min(args.val_n, len(train_items)))
    val_ids = {item["task_id"] for item in val_subset}

    # ── Step 1: roll out K_i ──
    cur_dir = args.init_skills if is_empty(args.init_skills) else Path(args.init_skills)
    state = topology.load(cur_dir)
    print(f"Loaded K_i: {topology.describe(state)}")
    t0 = time.time()

    cache = Path(args.train_rollout_cache) if args.train_rollout_cache else None
    if cache and cache.exists():
        print(f"\n── Step 1: loading train rollout from cache {cache} ──")
        rollouts = _reuse_cached_rollout(cache, train_items)
        print(f"  Loaded {len(rollouts)} cached rollouts (skipped actor calls)")
    else:
        print("\n── Step 1: rollout actor on train ──")
        rollouts = topology.rollout(
            train_items, state, args.actor_model, args.workers,
            on_done=lambda d, n, r: print(
                f"  [actor {d:3d}/{n}] {'✓' if r['correct'] else '✗'} "
                f"{topology.format_row(r)}  "
                f"({r.get('in_tok', 0)}+{r.get('out_tok', 0)} tok)"),
            **rollout_kw)

    correct = sum(1 for r in rollouts if r["correct"])
    print(f"\nK_i train score: {correct}/{len(rollouts)} = "
          f"{correct / max(len(rollouts), 1) * 100:.1f}%   in {time.time() - t0:.1f}s")
    (out_dir / "rollout_train_Ki.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rollouts))

    ki_val_correct = sum(1 for r in rollouts if r["task_id"] in val_ids and r["correct"])
    ki_val_total = sum(1 for r in rollouts if r["task_id"] in val_ids)
    print(f"K_i validation score: {ki_val_correct}/{ki_val_total} "
          f"= {ki_val_correct / max(ki_val_total, 1) * 100:.1f}%")

    # ── Step 2: optimizer proposes operations ──
    print(f"\n── Step 2: optimizer ({args.optimizer_model}) proposes ops ──")
    if args.ablation == "credit":
        # Hand over every rollout, so the optimizer cannot tell which tasks
        # were graded correct: that removes the per-task credit signal.
        failed = list(rollouts)
        print(f"[ablation:credit] feeding ALL {len(failed)} rollouts, "
              "not just the failures — no per-task credit signal")
    else:
        failed = [r for r in rollouts if not r["correct"]]
        print(f"Sending {len(failed)} failed trajectories to the optimizer")

    prior_rejected = _load_prior_ops(args.prior_ops_files)
    if prior_rejected:
        print(f"Including {sum(len(p['ops']) for p in prior_rejected)} "
              "prior-rejected ops as 'do not re-propose' context")

    proposal = topology.propose(
        state, failed,
        model=args.optimizer_model,
        max_ops=args.max_ops,
        prior_rejected=prior_rejected,
        temperature=args.optimizer_temp,
    )
    ops = proposal["ops"]
    if not getattr(args, "evolve_role", False):
        kept = [op for op in ops if not _targets_role(op)]
        if len(kept) != len(ops):
            print(f"[role static] dropped {len(ops) - len(kept)} op(s) targeting "
                  f"the root SKILL.md; pass --evolve-role to allow them")
            ops = kept
    if args.ablation == "consolprune":
        before = len(ops)
        ops = [op for op in ops if op.get("op") in ("induct", "refine")]
        print(f"[ablation:consolprune] dropped {before - len(ops)} "
              f"consolidate/prune ops, kept {len(ops)} induct/refine")

    (out_dir / "optimizer_raw.txt").write_text(proposal["raw"])
    (out_dir / "optimizer_meta.json").write_text(json.dumps({
        "model": proposal["model"],
        "usage": proposal["usage"],
        "parse_error": proposal["parse_error"],
        "n_failures_shown": proposal["n_failures_shown"],
        "n_ops_proposed": len(ops),
    }, indent=2))
    if proposal["parse_error"]:
        print(f"  ⚠ optimizer JSON parse error: {proposal['parse_error']}")
        print(f"  raw output saved to {out_dir / 'optimizer_raw.txt'}")
    print(f"Proposed {len(ops)} ops: " + ", ".join(topology.op_label(o) for o in ops))

    # ── Step 3: snapshot K_i and apply the operations to the candidate ──
    print("\n── Step 3: snapshot + apply ops ──")
    new_dir = Path(args.out_skills)
    topology.snapshot(cur_dir, new_dir)
    apply_log = topology.apply(new_dir, ops)
    (out_dir / "apply_log.json").write_text(json.dumps(apply_log, indent=2))
    applied_ok = sum(1 for entry in apply_log if entry.get("status") == "ok")
    print(f"Applied: {applied_ok} ok, {len(apply_log) - applied_ok} other")
    for entry in apply_log:
        print(f"  - {entry}")

    # ── Step 4: validation gate ──
    print(f"\n── Step 4: validation gate ({len(val_subset)} held-out) ──")
    val_rollouts = topology.rollout(
        val_subset, topology.load(new_dir), args.actor_model, args.workers,
        on_done=lambda d, n, r: print(
            f"  [val   {d:3d}/{n}] {'✓' if r['correct'] else '✗'} "
            f"{topology.format_row(r)}"),
        **rollout_kw)
    (out_dir / "val_rollouts_Knew.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in val_rollouts))

    new_val_correct = sum(1 for r in val_rollouts if r["correct"])
    new_val_total = len(val_rollouts)
    print(f"\nK_{{i+1}} validation score: {new_val_correct}/{new_val_total} "
          f"= {new_val_correct / max(new_val_total, 1) * 100:.1f}%  "
          f"(was {ki_val_correct}/{ki_val_total} = "
          f"{ki_val_correct / max(ki_val_total, 1) * 100:.1f}% under K_i)")

    delta = new_val_correct - ki_val_correct
    sign = "+" if delta >= 0 else ""
    if args.ablation == "rollback":
        print(f"  [ablation:rollback] gate disabled — accepting unconditionally "
              f"(Δ={sign}{delta})")
        decision = "accepted_ablation"
    elif new_val_correct >= ki_val_correct - args.gate_tolerance:
        print(f"  ✓ Validation gate PASSED (Δ={sign}{delta}, "
              f"within tol={args.gate_tolerance}) — keeping the new library.")
        decision = "accepted"
    else:
        print(f"  ⚠ Validation gate FAILED (drop > tol={args.gate_tolerance}) "
              "— reverting.")
        topology.snapshot(cur_dir, new_dir)
        decision = "rejected"

    meta = {
        "iter": args.iter,
        "topology": topology.name,
        "ablation": args.ablation,
        "evolve_role": bool(getattr(args, "evolve_role", False)),
        "lib_dir_cur": str(cur_dir),
        "lib_dir_new": str(new_dir),
        "actor_model": args.actor_model,
        "optimizer_model": args.optimizer_model,
        "train_n": args.train_n,
        "Ki_train_correct": correct,
        "Ki_train_total": len(rollouts),
        "Ki_val_correct": ki_val_correct,
        "Ki_val_total": ki_val_total,
        "Knew_val_correct": new_val_correct,
        "Knew_val_total": new_val_total,
        "decision": decision,
        "gate_tolerance": args.gate_tolerance,
        "n_ops_proposed": len(ops),
        "n_ops_applied_ok": applied_ok,
        "wall_seconds": time.time() - t0,
    }
    (out_dir / "iter_meta.json").write_text(json.dumps(meta, indent=2))

    print("\n" + "=" * 70)
    print(f"[iter {args.iter}] {decision.upper()} — "
          f"train K_i: {correct / max(len(rollouts), 1) * 100:.1f}%   "
          f"val K_i→K_new: {ki_val_correct / max(ki_val_total, 1) * 100:.1f}%"
          f" → {new_val_correct / max(new_val_total, 1) * 100:.1f}%")
    print(f"  metadata → {out_dir / 'iter_meta.json'}")
    print(f"  new lib  → {new_dir}")
    print("=" * 70)
    return meta


# ── evaluation ─────────────────────────────────────────────────────────


def _breakdown(rollouts: list, key: str) -> dict:
    groups: dict = {}
    for r in rollouts:
        correct, total = groups.setdefault(r.get(key), [0, 0])
        groups[r.get(key)] = [correct + int(r["correct"]), total + 1]
    return groups


def evaluate(args, out_path: Optional[Path] = None) -> dict:
    """Score one library on a test set; return the summary."""
    topology = TOPOLOGIES[args.topology]
    rollout_kw = _rollout_kwargs(topology, args)

    lib_dir = args.skills if is_empty(args.skills) else Path(args.skills)
    state = topology.load(lib_dir)
    items = _read_jsonl(Path(args.input))
    if getattr(args, "limit", 0) > 0:
        items = items[: args.limit]

    if out_path is None:
        out_path = Path(args.out) if args.out else (
            Path("data") / f"maskills_{topology.name}"
            / f"eval_{EMPTY if is_empty(lib_dir) else Path(lib_dir).name}"
              f"_{Path(args.input).stem}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Library: {lib_dir}  ({topology.describe(state)})")
    print(f"Input  : {args.input} ({len(items)} items)")
    print(f"Out    : {out_path}\n")

    t0 = time.time()
    rollouts = topology.rollout(
        items, state, args.model, args.workers,
        on_done=lambda d, n, r: print(
            f"  [{d:3d}/{n}] {'✓' if r['correct'] else '✗'} "
            f"{topology.format_row(r)}  "
            f"({r.get('in_tok', 0)}+{r.get('out_tok', 0)} tok)"),
        **rollout_kw)
    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rollouts))

    correct = sum(1 for r in rollouts if r["correct"])
    in_tok = sum(r.get("in_tok", 0) for r in rollouts)
    out_tok = sum(r.get("out_tok", 0) for r in rollouts)

    print("\n" + "=" * 72)
    print(f"OVERALL  {correct}/{len(rollouts)} = "
          f"{correct / max(len(rollouts), 1) * 100:.1f}%   in {time.time() - t0:.1f}s")
    print(f"Tokens   {in_tok:,} in + {out_tok:,} out")
    for label, key in (("Level", "Level"), ("kind", "kind")):
        print(f"\nBy {label}:")
        for group, (c, total) in sorted(_breakdown(rollouts, key).items(),
                                        key=lambda kv: str(kv[0])):
            print(f"  {str(group):8s}: {c:2d}/{total:2d}  ({c / total * 100:.1f}%)")
    print(f"\nResults → {out_path}")

    return {
        "topology": topology.name,
        "skills": str(lib_dir),
        "input": args.input,
        "n": len(rollouts),
        "correct": correct,
        "accuracy": correct / max(len(rollouts), 1),
        "in_tok": in_tok,
        "out_tok": out_tok,
        "out_path": str(out_path),
    }
