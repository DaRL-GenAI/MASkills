"""Tests for SkillEvolutionTrainer logic: operators, rollback, scheduling."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maskills.config.base import LanguageTaskConfig  # noqa: E402
from maskills.core.base import Trajectory  # noqa: E402
from maskills.core.policy import AgentPolicy  # noqa: E402
from maskills.core.skill_credit import (  # noqa: E402
    SkillCredit,
    TrajectorySkillCredit,
)
from maskills.core.skill_operators import SkillGradient  # noqa: E402
from maskills.core.skills import Skill, SkillLibrary  # noqa: E402
from maskills.store.local import LocalStore  # noqa: E402
from maskills.trainer.skill_evolution import OperatorSchedule, SkillEvolutionTrainer  # noqa: E402


# ── fakes ────────────────────────────────────────────────────────────────
class GateEnv:
    """Env whose trajectory reward = fraction of agents owning a `good` skill."""

    def sample_tasks(self, n, split="train"):
        return [{"id": i} for i in range(n)]

    def collect_trajectory(self, policies, task):
        scores = [1.0 if p.skill_library.get("good") else 0.0 for p in policies.values()]
        return Trajectory(task=task, steps=[], reward=sum(scores) / len(scores))


class FakeOptimizer:
    def __init__(self):
        self.calls = []
        self.groups = []
        self.prune = []

    def aggregate_skill_gradients(self, library, credits_by_skill):
        return {}

    def refine_skill(self, skill, grad):
        self.calls.append("refine")
        r = skill.copy()
        r.body = (skill.body + " [refined]").strip()
        r.provenance = "refined"
        return r

    def detect_redundancy_groups(self, library, gradients):
        self.calls.append("detect")
        return self.groups

    def consolidate_skills(self, library, group):
        self.calls.append("consolidate")
        return Skill(name="Macro", description="merged", body="merged body")

    def select_skills_to_prune(self, library, gradients):
        self.calls.append("prune")
        return self.prune

    def induce_skill(self, hard_evidence, library, residual_summary):
        self.calls.append("induce")
        return Skill(name="Induced", description="new", body="new body")


def _make_trainer(tmp, n_val=0):
    config = LanguageTaskConfig(
        exp_name="test", num_agents=2, experiment_dir=tmp,
        skill_evolution=True, n_val=n_val, skill_eval_delta=0.05,
    )
    trainer = SkillEvolutionTrainer(
        config=config,
        env=GateEnv(),
        critic=object(),          # unused in these unit tests
        optimizer=FakeOptimizer(),
        store=LocalStore(tmp),
    )
    return trainer


def _policy(role, *skill_specs):
    lib = SkillLibrary()
    for name, body in skill_specs:
        lib.add(Skill(name=name, description=f"d {name}", body=body))
    return AgentPolicy(role=role, skill_library=lib)


# ── OperatorSchedule ─────────────────────────────────────────────────────
def test_operator_schedule_cadence():
    s = OperatorSchedule(refine_every=1, induct_every=2, consolidate_every=3, prune_every=0)
    assert s.active(1) == {"refine": True, "induct": False, "consolidate": False, "prune": False}
    assert s.active(6)["induct"] is True and s.active(6)["consolidate"] is True
    assert s.active(6)["prune"] is False  # 0 disables


# ── _evolve_library ──────────────────────────────────────────────────────
def test_evolve_refine_only():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        policy = _policy("R", ("Alpha", "alpha body"))
        grads = {"alpha": SkillGradient(skill_id="alpha", verdict="refine",
                                        suggested_edit="x")}
        schedule = {"refine": True, "induct": False, "consolidate": False, "prune": False}
        lib, log = trainer._evolve_library("agent_1", policy, grads, "", [], schedule)
        assert log["refined"] == 1
        assert "[refined]" in lib.get("alpha").body
        assert trainer.optimizer.calls == ["refine"]


def test_evolve_induct_adds_skill():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        policy = _policy("R", ("Alpha", "alpha body"))
        schedule = {"refine": False, "induct": True, "consolidate": False, "prune": False}
        lib, log = trainer._evolve_library(
            "agent_1", policy, {}, "missing planning", ["hard case 1"], schedule
        )
        assert log["induced"] == 1
        assert len(lib) == 2
        assert lib.get("induced") is not None


def test_evolve_prune_removes_skill():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        trainer.optimizer.prune = ["beta"]
        policy = _policy("R", ("Alpha", "a"), ("Beta", "b"))
        schedule = {"refine": False, "induct": False, "consolidate": False, "prune": True}
        lib, log = trainer._evolve_library("agent_1", policy, {}, "", [], schedule)
        assert log["pruned"] == 1
        assert lib.get("beta") is None and lib.get("alpha") is not None


def test_evolve_consolidate_merges():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        trainer.optimizer.groups = [["alpha", "beta"]]
        policy = _policy("R", ("Alpha", "a"), ("Beta", "b"))
        schedule = {"refine": False, "induct": False, "consolidate": True, "prune": False}
        lib, log = trainer._evolve_library("agent_1", policy, {}, "", [], schedule)
        assert log["consolidated"] == 1
        assert lib.get("alpha") is None and lib.get("beta") is None
        assert any(s.provenance == "consolidated" for s in lib)


def test_evolve_respects_disabled_operators():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        policy = _policy("R", ("Alpha", "a"))
        schedule = {"refine": False, "induct": False, "consolidate": False, "prune": False}
        lib, log = trainer._evolve_library("agent_1", policy, {}, "x", ["hc"], schedule)
        assert log == {"refined": 0, "induced": 0, "consolidated": 0, "pruned": 0}
        assert trainer.optimizer.calls == []


# ── _validation_gate (rollback) ──────────────────────────────────────────
def test_validation_gate_accepts_and_rolls_back():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, n_val=4)
        base = {
            "agent_1": _policy("R1", ("neutral", "n")),         # no `good` skill
            "agent_2": _policy("R2", ("good", "g")),            # has `good`
        }
        candidate = {
            "agent_1": _policy("R1", ("neutral", "n"), ("good", "g")),  # improves
            "agent_2": _policy("R2"),                                   # regresses
        }
        accepted, gate = trainer._validation_gate(base, candidate)
        assert gate["status"] == "active"
        # agent_1 candidate raises val reward -> accepted
        assert accepted["agent_1"].skill_library.get("good") is not None
        assert gate["per_agent"]["agent_1"]["accepted"] is True
        # agent_2 candidate drops val reward -> rolled back to base
        assert accepted["agent_2"].skill_library.get("good") is not None
        assert gate["per_agent"]["agent_2"]["accepted"] is False


def test_validation_gate_disabled_without_val_set():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, n_val=0)
        base = {"agent_1": _policy("R", ("a", "b"))}
        candidate = {"agent_1": _policy("R", ("c", "d"))}
        accepted, gate = trainer._validation_gate(base, candidate)
        assert gate["status"] == "disabled"
        assert accepted["agent_1"] is candidate["agent_1"]  # committed unconditionally


def test_validation_gate_skips_unchanged_agent():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, n_val=4)
        base = {
            "agent_1": _policy("R1", ("good", "g")),
            "agent_2": _policy("R2", ("good", "g")),
        }
        candidate = {a: p.copy() for a, p in base.items()}  # identical
        accepted, gate = trainer._validation_gate(base, candidate)
        for a in ("agent_1", "agent_2"):
            assert gate["per_agent"][a]["changed"] is False
            assert accepted[a] is base[a]


# ── _update_skill_stats ──────────────────────────────────────────────────
def test_update_skill_stats():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        policies = {"agent_1": _policy("R", ("s1", "body"))}
        traj = Trajectory(task={}, steps=[], reward=1.0)
        traj.skill_trace = {"agent_1": [
            {"step": 0, "skills": ["s1"]}, {"step": 1, "skills": ["s1"]},
        ]}
        credits = [TrajectorySkillCredit(
            skill_credits=[SkillCredit(agent="agent_1", skill_id="s1", utility_delta=0.5)],
            reward=1.0,
        )]
        trainer._update_skill_stats(policies, [traj], credits)
        skill = policies["agent_1"].skill_library.get("s1")
        assert skill.invocations == 2          # two trace entries
        assert skill.utility == 0.2            # 0.6*0 + 0.4*0.5


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            import traceback
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
