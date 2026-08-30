"""The training loop and policy loading behind the six scripts/ entry points.

``scripts/train_gaia.py`` and ``scripts/eval_gaia.py`` run one loop for both
topologies, so the loop is exercised here with the actor and the optimizer
stubbed out: what is under test is the iteration's control flow -- which
rollouts reach the optimizer, where the operations land, and whether the
validation gate keeps or reverts the candidate -- not the LLM behind it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maskills.core.skills import Skill, SkillLibrary  # noqa: E402
from maskills.experiments import env_eval  # noqa: E402
from maskills.experiments.gaia import (  # noqa: E402
    TOPOLOGIES,
    Centralized,
    Decentralized,
    train_iteration,
)
from maskills.skill_lib import load_lib, write_skill_file  # noqa: E402

# ── fixtures ───────────────────────────────────────────────────────────


def _write_library(root: Path, slugs=("alpha", "beta")) -> Path:
    write_skill_file(root / "SKILL.md", "root", "root identity", "Be careful.")
    for slug in slugs:
        write_skill_file(root / slug / "SKILL.md", f"skill-{slug}",
                         f"the {slug} skill", f"Body of {slug}.")
    return root


def _task(task_id: str) -> dict:
    return {"task_id": task_id, "Question": f"Q{task_id}", "Final answer": "42",
            "Level": 1, "kind": "text"}


def _train_args(tmp_path: Path, topology: str, **overrides) -> argparse.Namespace:
    train_file = tmp_path / "train.jsonl"
    train_file.write_text("\n".join(json.dumps(_task(str(i))) for i in range(6)))
    args = argparse.Namespace(
        topology=topology, iter=1,
        init_skills=str(tmp_path / "cur"), out_skills=str(tmp_path / "new"),
        train=str(train_file), train_n=6, val_n=3, seed=0,
        out_dir=str(tmp_path / "meta"),
        actor_model="stub-actor", optimizer_model="stub-optimizer",
        optimizer_temp=0.0, workers=2, max_tokens=10,
        max_rounds=1, tool_budget=1,
        rounds_a1=1, rounds_a2=1, budget_a1=1, budget_a2=1,
        max_ops=5, gate_tolerance=0, ablation="none",
        prior_ops_files=[], train_rollout_cache="",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class _StubRollout:
    """Stands in for the actor: scores are scripted per call."""

    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = []

    def __call__(self, items, state, model, workers, on_done=None, **kw):
        correct = self.scores.pop(0)
        self.calls.append({"n": len(items), "state": state, "kw": kw})
        return [{"task_id": item["task_id"], "Level": 1, "kind": "text",
                 "gold": "42", "pred": "42" if i < correct else "",
                 "correct": i < correct, "in_tok": 1, "out_tok": 1,
                 "question": item["Question"]}
                for i, item in enumerate(items)]


def _stub_propose(ops):
    def propose(state, failed, **kw):
        propose.seen_failed = failed
        return {"ops": list(ops), "raw": json.dumps(ops), "model": "stub",
                "usage": {}, "parse_error": None, "n_failures_shown": len(failed)}
    propose.seen_failed = None
    return propose


@pytest.fixture
def centralized_lib(tmp_path):
    _write_library(tmp_path / "cur")
    return tmp_path


@pytest.fixture
def decentralized_lib(tmp_path):
    _write_library(tmp_path / "cur" / "agent_1")
    _write_library(tmp_path / "cur" / "agent_2")
    return tmp_path


# ── the iteration's control flow ───────────────────────────────────────


def test_only_failures_reach_the_optimizer(centralized_lib, monkeypatch):
    """The optimizer is shown the failed trajectories, not the whole batch."""
    args = _train_args(centralized_lib, "centralized")
    propose = _stub_propose([])
    monkeypatch.setattr(Centralized, "rollout", _StubRollout([4, 3]))
    monkeypatch.setattr(Centralized, "propose", staticmethod(propose))

    train_iteration(args)

    assert len(propose.seen_failed) == 2  # 6 train items, 4 scored correct
    assert all(not r["correct"] for r in propose.seen_failed)


def test_credit_ablation_hides_which_tasks_were_correct(centralized_lib, monkeypatch):
    """--ablation credit hands over every rollout, removing the credit signal."""
    args = _train_args(centralized_lib, "centralized", ablation="credit")
    propose = _stub_propose([])
    monkeypatch.setattr(Centralized, "rollout", _StubRollout([4, 3]))
    monkeypatch.setattr(Centralized, "propose", staticmethod(propose))

    train_iteration(args)

    assert len(propose.seen_failed) == 6


def test_gate_keeps_a_candidate_that_holds_up(centralized_lib, monkeypatch):
    args = _train_args(centralized_lib, "centralized")
    ops = [{"op": "induct", "slug": "gamma", "name": "skill-gamma",
            "description": "induced", "body": "Body of gamma."}]
    # 6/6 on train, then 3/3 on the validation slice: no regression.
    monkeypatch.setattr(Centralized, "rollout", _StubRollout([6, 3]))
    monkeypatch.setattr(Centralized, "propose", staticmethod(_stub_propose(ops)))

    meta = train_iteration(args)

    assert meta["decision"] == "accepted"
    assert "gamma" in load_lib(Path(args.out_skills))["skills"]


def test_gate_reverts_a_candidate_that_regresses(centralized_lib, monkeypatch):
    args = _train_args(centralized_lib, "centralized")
    ops = [{"op": "induct", "slug": "gamma", "name": "skill-gamma",
            "description": "induced", "body": "Body of gamma."}]
    # All 3 validation items were correct under K_i and none are under the
    # candidate, a drop well past the tolerance of 0.
    monkeypatch.setattr(Centralized, "rollout", _StubRollout([6, 0]))
    monkeypatch.setattr(Centralized, "propose", staticmethod(_stub_propose(ops)))

    meta = train_iteration(args)

    assert meta["decision"] == "rejected"
    # The candidate directory is restored from K_i, so the induced skill is gone.
    assert "gamma" not in load_lib(Path(args.out_skills))["skills"]


def test_rollback_ablation_commits_a_regressing_candidate(centralized_lib, monkeypatch):
    args = _train_args(centralized_lib, "centralized", ablation="rollback")
    ops = [{"op": "induct", "slug": "gamma", "name": "skill-gamma",
            "description": "induced", "body": "Body of gamma."}]
    monkeypatch.setattr(Centralized, "rollout", _StubRollout([6, 0]))
    monkeypatch.setattr(Centralized, "propose", staticmethod(_stub_propose(ops)))

    meta = train_iteration(args)

    assert meta["decision"] == "accepted_ablation"
    assert "gamma" in load_lib(Path(args.out_skills))["skills"]


def test_consolprune_ablation_drops_those_operators(centralized_lib, monkeypatch):
    args = _train_args(centralized_lib, "centralized", ablation="consolprune")
    ops = [
        {"op": "induct", "slug": "gamma", "name": "g", "description": "d",
         "body": "Body of gamma."},
        {"op": "prune", "slug": "alpha", "reason": "unused"},
    ]
    monkeypatch.setattr(Centralized, "rollout", _StubRollout([6, 3]))
    monkeypatch.setattr(Centralized, "propose", staticmethod(_stub_propose(ops)))

    meta = train_iteration(args)

    library = load_lib(Path(args.out_skills))["skills"]
    assert meta["n_ops_proposed"] == 1  # the prune was filtered out
    assert "gamma" in library
    assert "alpha" in library  # survived, because prune never ran


def test_decentralized_routes_ops_to_the_tagged_agent(decentralized_lib, monkeypatch):
    args = _train_args(decentralized_lib, "decentralized")
    ops = [
        {"op": "induct", "agent": "a", "slug": "researcher_only",
         "name": "r", "description": "d", "body": "Researcher body."},
        {"op": "induct", "agent": "b", "slug": "solver_only",
         "name": "s", "description": "d", "body": "Solver body."},
        {"op": "induct", "slug": "untagged", "name": "u", "description": "d",
         "body": "No agent tag."},
    ]
    monkeypatch.setattr(Decentralized, "rollout", _StubRollout([6, 3]))
    monkeypatch.setattr(Decentralized, "propose", staticmethod(_stub_propose(ops)))

    train_iteration(args)

    out = Path(args.out_skills)
    agent_1 = load_lib(out / "agent_1")["skills"]
    agent_2 = load_lib(out / "agent_2")["skills"]
    assert "researcher_only" in agent_1 and "researcher_only" not in agent_2
    assert "solver_only" in agent_2 and "solver_only" not in agent_1
    # An untagged op cannot be routed and must not land in either library.
    assert "untagged" not in agent_1 and "untagged" not in agent_2


def test_each_topology_gets_its_own_rollout_knobs(decentralized_lib, monkeypatch):
    args = _train_args(decentralized_lib, "decentralized",
                       rounds_a1=7, rounds_a2=2, budget_a1=6, budget_a2=1)
    rollout = _StubRollout([6, 3])
    monkeypatch.setattr(Decentralized, "rollout", rollout)
    monkeypatch.setattr(Decentralized, "propose", staticmethod(_stub_propose([])))

    train_iteration(args)

    kw = rollout.calls[0]["kw"]
    assert kw == {"max_tokens": 10, "rounds_a1": 7, "rounds_a2": 2,
                  "budget_a1": 6, "budget_a2": 1}
    # The centralized-only knobs must not leak into the two-agent rollout.
    assert "max_rounds" not in kw and "tool_budget" not in kw


def test_both_topologies_are_selectable():
    assert set(TOPOLOGIES) == {"centralized", "decentralized"}


# ── policy loading for the env-based evaluators ────────────────────────


def test_empty_skills_gives_empty_libraries():
    policies = env_eval.load_policies(env_eval.EMPTY, num_agents=2, task_type="qa")
    assert set(policies) == {"agent_1", "agent_2"}
    assert all(not p.skill_library.skills for p in policies.values())


def test_language_task_needs_a_role_but_locomo_must_not_have_one():
    """The two environments want opposite things from a seedless policy.

    The language env leaves a supplied policy's role alone, so it has to be
    written in; LOCOMO substitutes its retriever/reasoner prompts exactly when
    the role is blank, so writing one in would suppress them.
    """
    for_language = env_eval.load_policies(env_eval.EMPTY, task_type="qa")
    for_locomo = env_eval.load_policies(env_eval.EMPTY, fill_default_role=False)

    assert for_language["agent_1"].role.strip()
    assert for_locomo["agent_1"].role == ""


def test_loads_a_training_checkpoint(tmp_path):
    library = SkillLibrary(skills=[Skill(skill_id="s1", name="s1",
                                         description="d", body="b")])
    for agent in ("agent_1", "agent_2"):
        agent_dir = tmp_path / agent
        agent_dir.mkdir()
        (agent_dir / "role.md").write_text(f"role of {agent}")
        (agent_dir / "skill_library.json").write_text(json.dumps(library.to_dict()))

    policies = env_eval.load_policies(str(tmp_path))

    assert policies["agent_1"].role == "role of agent_1"
    assert [s.skill_id for s in policies["agent_2"].skill_library.skills] == ["s1"]


def test_loads_a_skill_md_directory(tmp_path):
    for agent in ("agent_1", "agent_2"):
        _write_library(tmp_path / agent, slugs=("alpha",))

    policies = env_eval.load_policies(str(tmp_path))

    assert policies["agent_1"].skill_library.skills


def test_a_missing_library_is_reported_not_silently_empty(tmp_path):
    with pytest.raises(SystemExit, match="No skill library"):
        env_eval.load_policies(str(tmp_path / "nope"))


def test_summarize_separates_errors_from_scores():
    summary = env_eval.summarize([
        {"reward": 1.0, "f1": 1.0, "in_tok": 10, "out_tok": 5},
        {"reward": 0.0, "f1": 0.0, "in_tok": 10, "out_tok": 5},
        {"error": "boom"},
    ])
    assert summary["n"] == 2
    assert summary["n_errors"] == 1
    assert summary["reward"] == pytest.approx(0.5)
    assert summary["in_tok"] == 20  # token cost counts the failed attempt too


# ── role evolution ─────────────────────────────────────────────────────


def test_role_is_static_unless_asked_for():
    """The default keeps the role fixed; the config knob is what opts in."""
    from maskills.config.base import LanguageTaskConfig

    assert LanguageTaskConfig().evolve_role is False


def test_gaia_drops_root_edits_when_the_role_is_static(centralized_lib, monkeypatch):
    """A refine aimed at _root_ is the GAIA spelling of a role edit."""
    args = _train_args(centralized_lib, "centralized")
    args.evolve_role = False
    ops = [
        {"op": "refine", "slug": "_root_", "name": "root", "description": "d",
         "body": "REWRITTEN ROOT"},
        {"op": "induct", "slug": "gamma", "name": "g", "description": "d",
         "body": "Body of gamma."},
    ]
    monkeypatch.setattr(Centralized, "rollout", _StubRollout([6, 3]))
    monkeypatch.setattr(Centralized, "propose", staticmethod(_stub_propose(ops)))

    meta = train_iteration(args)

    library = load_lib(Path(args.out_skills))
    assert meta["n_ops_proposed"] == 1  # the root edit was dropped
    assert library["root"]["body"].strip() == "Be careful."  # untouched
    assert "gamma" in library["skills"]  # the skill op still applied


def test_gaia_applies_root_edits_when_asked(centralized_lib, monkeypatch):
    args = _train_args(centralized_lib, "centralized")
    args.evolve_role = True
    ops = [{"op": "refine", "slug": "_root_", "name": "root",
            "description": "d", "body": "REWRITTEN ROOT"}]
    monkeypatch.setattr(Centralized, "rollout", _StubRollout([6, 3]))
    monkeypatch.setattr(Centralized, "propose", staticmethod(_stub_propose(ops)))

    train_iteration(args)

    assert load_lib(Path(args.out_skills))["root"]["body"].strip() == "REWRITTEN ROOT"


def test_an_insert_never_glues_onto_its_anchor():
    """Regression: `insert_after` used to run straight into the anchor text."""
    from maskills.core.optimizer import PolicyGradientOptimizer

    text = "- You can see the responses from Agent 1\n- YOUR output is FINAL"
    edited = PolicyGradientOptimizer._apply_edits(text, [
        {"op": "insert_after", "anchor": "from Agent 1",
         "new": "You are responsible for synthesis."},
    ])

    assert "Agent 1You are" not in edited
    assert "from Agent 1\nYou are responsible" in edited
    # The line that follows the insertion keeps its own separator.
    assert "synthesis.\n- YOUR output is FINAL" in edited


def test_an_insert_that_brings_its_own_separator_is_left_alone():
    from maskills.core.optimizer import PolicyGradientOptimizer

    edited = PolicyGradientOptimizer._apply_edits("head\ntail", [
        {"op": "insert_after", "anchor": "head", "new": "\nmiddle"},
    ])

    assert edited == "head\nmiddle\ntail"


def test_the_gate_treats_a_role_only_change_as_a_change():
    """Regression: comparing only the library discarded role-only edits.

    An iteration with evolve_role on can edit the role and leave the library
    untouched. The gate has to validate that candidate rather than conclude
    nothing happened and keep the base policy.
    """
    from maskills.core.policy import AgentPolicy
    from maskills.trainer.skill_evolution import SkillEvolutionTrainer

    library = SkillLibrary(skills=[Skill(skill_id="s", name="s",
                                         description="d", body="b")])
    base = {"agent_1": AgentPolicy(role="old role", skill_library=library.copy())}
    candidate = {"agent_1": AgentPolicy(role="new role", skill_library=library.copy())}

    class _Env:
        def sample_tasks(self, n, split="val"):
            return [{"task_id": "t1"}]

    trainer = SkillEvolutionTrainer.__new__(SkillEvolutionTrainer)
    trainer.config = type("C", (), {"n_val": 1})()
    trainer.env = _Env()
    trainer.delta = 0.0
    # A candidate that scores no worse than the base must be committed.
    trainer._val_score = lambda policies, tasks: 1.0

    accepted, gate = trainer._validation_gate(base, candidate)

    assert gate["per_agent"]["agent_1"]["changed"] is True
    assert accepted["agent_1"].role == "new role"


# ── the GAIA no-skills floor ───────────────────────────────────────────


def test_empty_gaia_library_is_protocol_only_not_blank():
    """The floor has to carry the wire format, or it measures the wrong thing.

    An agent handed a blank prompt would not know that a tool call is written
    ``SEARCH: <query>`` or that the answer arrives on a ``FINAL ANSWER:`` line,
    so it would fail on protocol rather than on the task.
    """
    from maskills.experiments.gaia import EMPTY, Centralized, Decentralized

    central = Centralized.load(EMPTY)
    assert central["skills"] == {}                      # no task knowledge
    assert "FINAL ANSWER:" in central["root"]["body"]   # but the contract is there
    assert "SEARCH:" in central["root"]["body"]

    researcher, solver = Decentralized.load(EMPTY)
    assert researcher["skills"] == {} and solver["skills"] == {}
    assert "HANDOFF_TO_SOLVER" in researcher["root"]["body"]
    assert "FINAL ANSWER:" not in researcher["root"]["body"].split("Do not emit")[0]
    assert "FINAL ANSWER:" in solver["root"]["body"]


@pytest.mark.parametrize("topology", ["centralized", "decentralized"])
def test_training_from_empty_materializes_the_floor(tmp_path, topology):
    """`--init-skills empty` has to write a real library for iteration 1."""
    from maskills.experiments.gaia import EMPTY, TOPOLOGIES

    topo = TOPOLOGIES[topology]
    out = tmp_path / "iter_1"
    topo.snapshot(EMPTY, out)

    # It reloads as a valid library, so the operators have something to edit.
    state = topo.load(out)
    bundles = [state] if topology == "centralized" else list(state)
    assert all(b["root"]["body"].strip() for b in bundles)
    assert all(b["skills"] == {} for b in bundles)


def test_a_run_from_empty_ends_with_the_induced_skill_on_disk(tmp_path, monkeypatch):
    from maskills.experiments.gaia import EMPTY, Centralized

    args = _train_args(tmp_path, "centralized", init_skills=EMPTY)
    ops = [{"op": "induct", "slug": "gamma", "name": "g", "description": "d",
            "body": "Body of gamma."}]
    monkeypatch.setattr(Centralized, "rollout", _StubRollout([6, 3]))
    monkeypatch.setattr(Centralized, "propose", staticmethod(_stub_propose(ops)))

    meta = train_iteration(args)

    library = load_lib(Path(args.out_skills))
    assert meta["decision"] == "accepted"
    assert "gamma" in library["skills"]
    assert "FINAL ANSWER:" in library["root"]["body"]  # the floor survived


def test_gaia_env_without_a_library_still_gets_the_protocol():
    """Regression: it used to hand the agent an empty system prompt."""
    from maskills.envs.gaia import protocol

    assert "HANDOFF_TO_SOLVER" in protocol.prompt_for("agent_1")
    assert "FINAL ANSWER:" in protocol.prompt_for("agent_2")
    # An unknown agent name falls back to the single-agent stack.
    assert "SEARCH:" in protocol.prompt_for("whoever")
