"""Tests for the ξ skill-invocation trace (MASkills P1.3, full-injection mode)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maskills.core.base import Trajectory  # noqa: E402
from maskills.core.policy import AgentPolicy  # noqa: E402
from maskills.core.skills import (  # noqa: E402
    Skill,
    SkillLibrary,
    build_skill_trace,
    detect_invoked_skills,
)


def _library():
    lib = SkillLibrary()
    lib.add(Skill(name="Evidence Aggregation", description="d", body="b"))
    lib.add(Skill(name="Counterfactual Check", description="d", body="b"))
    return lib


def test_detect_by_name():
    lib = _library()
    hits = detect_invoked_skills(
        "I applied Evidence Aggregation to combine the quotes.", lib
    )
    assert hits == ["evidence-aggregation"]


def test_detect_by_id():
    lib = _library()
    hits = detect_invoked_skills("used skill counterfactual-check here", lib)
    assert hits == ["counterfactual-check"]


def test_detect_none_and_no_substring_false_positive():
    lib = SkillLibrary()
    lib.add(Skill(name="Reason", description="d", body="b"))
    assert detect_invoked_skills("nothing relevant here", lib) == []
    # 'Reason' must not match inside 'reasoning'
    assert detect_invoked_skills("careful reasoning matters", lib) == []
    # but a whole-word mention does match
    assert detect_invoked_skills("I will Reason about it", lib) == ["reason"]


def test_detect_empty_inputs():
    assert detect_invoked_skills("", _library()) == []
    assert detect_invoked_skills("text", SkillLibrary()) == []


def test_build_skill_trace_strict_detection():
    """With attribute_all_when_unreferenced=False, ξ is pure name detection."""
    lib1 = SkillLibrary()
    lib1.add(Skill(name="Retrieve", description="d", body="b"))
    lib2 = SkillLibrary()
    lib2.add(Skill(name="Summarize", description="d", body="b"))
    policies = {
        "agent_1": AgentPolicy(role="R1", skill_library=lib1),
        "agent_2": AgentPolicy(role="R2", skill_library=lib2),
    }
    steps = [
        {"agent": "agent_1", "output": "I will Retrieve the documents."},
        {"agent": "agent_2", "output": "Now Summarize them."},
        {"agent": "agent_2", "output": "No skill used this turn."},
    ]
    trace = build_skill_trace(steps, policies, attribute_all_when_unreferenced=False)
    assert trace["agent_1"] == [{"step": 0, "skills": ["retrieve"]}]
    assert trace["agent_2"] == [
        {"step": 1, "skills": ["summarize"]},
        {"step": 2, "skills": []},  # strict ∅ case
    ]


def test_build_skill_trace_attributes_all_when_unreferenced():
    """Default full-injection mode: an unreferenced step credits all skills."""
    lib = SkillLibrary()
    lib.add(Skill(name="grep-retrieval", description="d", body="b"))
    lib.add(Skill(name="evidence-pack", description="d", body="b"))
    policies = {"agent_1": AgentPolicy(role="R", skill_library=lib)}
    steps = [{"agent": "agent_1", "output": "yes"}]  # names no skill
    trace = build_skill_trace(steps, policies)
    assert trace["agent_1"] == [{"step": 0, "skills": ["grep-retrieval", "evidence-pack"]}]


def test_build_skill_trace_skips_unknown_agent():
    policies = {"agent_1": AgentPolicy(role="R", skills="")}
    steps = [{"agent": "ghost", "output": "x"}, {"agent": "agent_1", "output": "y"}]
    trace = build_skill_trace(steps, policies)
    assert "ghost" not in trace
    assert "agent_1" in trace


def test_trajectory_has_skill_trace_field():
    t = Trajectory(task={}, steps=[], reward=0.0)
    assert t.skill_trace == {}
    t.skill_trace = {"agent_1": [{"step": 0, "skills": ["s"]}]}
    assert t.skill_trace["agent_1"][0]["skills"] == ["s"]


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
