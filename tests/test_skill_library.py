"""Tests for the discrete SkillLibrary data model (MASkills P1.1)."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maskills.core.skills import (  # noqa: E402
    Skill,
    SkillLibrary,
    parse_skill_md,
)


def test_skill_defaults_and_id():
    s = Skill(name="Evidence Aggregation", description="Combine cited quotes.")
    assert s.skill_id == "evidence-aggregation"  # slugified from name
    assert s.provenance == "human"
    assert s.utility == 0.0 and s.invocations == 0


def test_skill_md_roundtrip_with_extensions():
    s = Skill(
        name="My Skill",
        description="Does a thing.",
        body="Step 1. Do it.",
        skill_id="my-skill",
        provenance="induced",
        utility=1.5,
        invocations=7,
        metadata={"version": "2"},
    )
    text = s.format()
    parsed = parse_skill_md(text)
    assert parsed.skill_id == "my-skill"
    assert parsed.provenance == "induced"
    assert parsed.utility == 1.5
    assert parsed.invocations == 7
    assert parsed.metadata.get("version") == "2"
    assert parsed.body == "Step 1. Do it."


def test_legacy_skill_md_parses_with_defaults():
    """A SKILL.md with no MASkills fields still parses (back-compat)."""
    legacy = "---\nname: Old Skill\ndescription: Legacy.\n---\n\nBody here.\n"
    s = parse_skill_md(legacy)
    assert s.skill_id == "old-skill"
    assert s.provenance == "human"
    assert s.utility == 0.0


def test_library_add_dedup_and_get():
    lib = SkillLibrary()
    a = lib.add(Skill(name="Dup", description="d"))
    b = lib.add(Skill(name="Dup", description="d"))
    assert a.skill_id == "dup"
    assert b.skill_id == "dup-2"  # collision-free suffix
    assert lib.get("dup") is a
    assert len(lib) == 2


def test_library_remove_and_replace():
    lib = SkillLibrary()
    lib.add(Skill(name="A", description="a", body="old"))
    refined = Skill(name="A2", description="a2", body="new", skill_id="a")
    assert lib.replace(refined) is True
    assert lib.get("a").body == "new"
    assert lib.replace(Skill(name="X", description="x", skill_id="missing")) is False
    assert lib.remove("a") is True
    assert lib.remove("a") is False


def test_library_merge_consolidation():
    lib = SkillLibrary()
    lib.add(Skill(name="A", description="a", body="ba"))
    lib.add(Skill(name="B", description="b", body="bb"))
    lib.add(Skill(name="C", description="c", body="bc"))
    macro = Skill(name="AB", description="merged", body="merged body", skill_id="ab")
    inserted = lib.merge(["a", "b"], macro)
    assert inserted is not None
    assert inserted.provenance == "consolidated"
    assert lib.get("a") is None and lib.get("b") is None
    assert lib.get("c") is not None and lib.get("ab") is not None
    assert lib.merge(["nonexistent"], macro.copy()) is None  # nothing matched


def test_library_dict_roundtrip():
    lib = SkillLibrary()
    lib.add(Skill(name="A", description="a", body="ba", utility=2.0, invocations=3))
    lib.add(Skill(name="B", description="b", body="bb", provenance="induced"))
    restored = SkillLibrary.from_dict(lib.to_dict())
    assert restored.ids() == lib.ids()
    assert restored.get("a").utility == 2.0
    assert restored.get("b").provenance == "induced"


def test_library_skill_md_dir_roundtrip():
    lib = SkillLibrary()
    lib.add(Skill(name="A", description="a", body="ba"))
    lib.add(Skill(name="B", description="b", body="bb", provenance="refined"))
    with tempfile.TemporaryDirectory() as tmp:
        lib.to_skill_md_dir(tmp)
        assert (Path(tmp) / "a" / "SKILL.md").exists()
        restored = SkillLibrary.from_skill_md_dir(tmp)
        assert set(restored.ids()) == {"a", "b"}
        assert restored.get("b").provenance == "refined"


def test_legacy_body_bridge():
    empty = SkillLibrary.from_legacy_body("")
    assert len(empty) == 0
    lib = SkillLibrary.from_legacy_body("Some learned strategy text.")
    assert len(lib) == 1
    assert lib.skills[0].body == "Some learned strategy text."


def test_render_metadata_vs_full():
    lib = SkillLibrary()
    lib.add(Skill(name="A", description="desc-a", body="full body a"))
    meta = lib.render_metadata()
    full = lib.render_full()
    assert "desc-a" in meta and "full body a" not in meta  # progressive disclosure
    assert "full body a" in full
    assert lib.get_body("a") == "full body a"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
