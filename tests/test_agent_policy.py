"""Tests for AgentPolicy backed by a discrete SkillLibrary (MASkills P1.2)."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maskills.core.policy import AgentPolicy  # noqa: E402
from maskills.core.skills import Skill, SkillLibrary  # noqa: E402
from maskills.store.local import LocalStore  # noqa: E402


def test_legacy_skills_string_constructor():
    p = AgentPolicy(role="R", skills="learned strategy")
    assert len(p.skill_library) == 1
    assert p.skills == "learned strategy"  # property collapses to raw body
    assert p.combined == "R\n\n## Skills\nlearned strategy"


def test_empty_skills_combined_is_role_only():
    p = AgentPolicy(role="R")
    assert len(p.skill_library) == 0
    assert p.skills == ""
    assert p.combined == "R"


def test_skills_setter_replaces_library():
    p = AgentPolicy(role="R", skills="old")
    p.skills = "new strategy"
    assert len(p.skill_library) == 1
    assert p.skills == "new strategy"


def test_explicit_multi_skill_library():
    lib = SkillLibrary()
    lib.add(Skill(name="Alpha", description="a", body="body-a"))
    lib.add(Skill(name="Beta", description="b", body="body-b"))
    p = AgentPolicy(role="R", skill_library=lib)
    # Two skills => combined_body separates with ### headers
    assert "### Alpha" in p.skills and "### Beta" in p.skills
    assert "body-a" in p.combined and "body-b" in p.combined


def test_single_skill_renders_without_header():
    """A 1-skill library must collapse identically to the legacy blob."""
    lib = SkillLibrary()
    lib.add(Skill(name="Solo", description="s", body="raw body text"))
    p = AgentPolicy(role="R", skill_library=lib)
    assert p.skills == "raw body text"  # no '### Solo' header


def test_to_from_dict_roundtrip_preserves_library():
    lib = SkillLibrary()
    lib.add(Skill(name="A", description="a", body="ba", utility=3.0, invocations=5))
    lib.add(Skill(name="B", description="b", body="bb", provenance="induced"))
    p = AgentPolicy(role="R", skill_library=lib)
    restored = AgentPolicy.from_dict(p.to_dict())
    assert restored.skill_library.ids() == p.skill_library.ids()
    assert restored.skill_library.get("a").utility == 3.0
    assert restored.skill_library.get("b").provenance == "induced"
    assert restored == p


def test_from_dict_legacy_format():
    legacy = {"role": "R", "skills": "flat skills"}
    p = AgentPolicy.from_dict(legacy)
    assert p.skills == "flat skills"
    p2 = AgentPolicy.from_dict({"policy": "old-style"})
    assert p2.role == "old-style" and p2.skills == ""


def test_checkpoint_roundtrip_preserves_discrete_library():
    """save_checkpoint -> load_checkpoint must keep multiple distinct skills."""
    lib = SkillLibrary()
    lib.add(Skill(name="Retrieve", description="r", body="how to retrieve"))
    lib.add(Skill(name="Reason", description="z", body="how to reason",
                  provenance="induced", utility=1.2))
    with tempfile.TemporaryDirectory() as tmp:
        store = LocalStore(tmp)
        run_id = store.create_run("test_run", config=None)
        policies = {"agent_1": AgentPolicy(role="R1", skill_library=lib)}
        store.save_checkpoint(run_id, 1, policies, meta={})
        loaded, _ = store.load_checkpoint(run_id, 1)
        lp = loaded["agent_1"]
        assert len(lp.skill_library) == 2
        assert set(lp.skill_library.ids()) == {"retrieve", "reason"}
        assert lp.skill_library.get("reason").provenance == "induced"
        assert lp.skill_library.get("reason").utility == 1.2


def test_checkpoint_loads_pre_maskills_format():
    """A checkpoint dir with only role.md + skills.md still loads."""
    with tempfile.TemporaryDirectory() as tmp:
        store = LocalStore(tmp)
        run_id = store.create_run("legacy_run", config=None)
        ckpt = store._ckpt_dir(run_id, 1)
        agent_dir = ckpt / "agent_1"
        agent_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / "role.md").write_text("LegacyRole")
        (agent_dir / "skills.md").write_text("legacy skill blob")
        loaded, _ = store.load_checkpoint(run_id, 1)
        lp = loaded["agent_1"]
        assert lp.role == "LegacyRole"
        assert lp.skills == "legacy skill blob"
        assert len(lp.skill_library) == 1


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
