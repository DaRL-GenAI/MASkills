"""Tests for skill-level credit + evolution operators (MASkills P2/P3)."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maskills.core.skill_credit import (  # noqa: E402
    SkillCredit,
    SkillCreditCritic,
    _extract_json,
)
from maskills.core.skill_operators import (  # noqa: E402
    SkillEvolutionOptimizer,
    SkillGradient,
    _strip_fences,
)
from maskills.core.skills import Skill, SkillLibrary  # noqa: E402


# ── mock optimizer: canned LLM responses, no API client ──────────────────
class MockOptimizer(SkillEvolutionOptimizer):
    def __init__(self, responses):
        self._responses = list(responses)
        self.tool_library = ""
        self.prune_utility_threshold = -0.3
        self.logger = logging.getLogger("mock")

    def _llm_call(self, prompt, max_tokens=400):
        return self._responses.pop(0)


def _lib(*specs):
    lib = SkillLibrary()
    for name, body in specs:
        lib.add(Skill(name=name, description=f"desc {name}", body=body))
    return lib


# ── helpers ──────────────────────────────────────────────────────────────
def test_extract_json_fenced_and_bare():
    assert _extract_json('```json\n{"a": 1}\n```')["a"] == 1
    assert _extract_json('noise {"b": 2} trailing')["b"] == 2


def test_strip_fences():
    assert _strip_fences("```md\nhello\n```") == "hello"
    assert _strip_fences("plain") == "plain"


# ── aggregation (P2) ─────────────────────────────────────────────────────
def test_gradient_from_single_credit():
    opt = MockOptimizer([])
    lib = _lib(("A", "body a"))
    credits = {"a": [SkillCredit(agent="agent_1", skill_id="a",
                                 contribution="helped", suggested_edit="tighten step 2",
                                 utility_delta=0.5)]}
    grads = opt.aggregate_skill_gradients(lib, credits)
    assert grads["a"].verdict == "refine"  # helped + suggested_edit => refine
    assert grads["a"].n_credits == 1
    assert grads["a"].utility_score == 0.5


def test_aggregate_multi_credit_calls_llm():
    opt = MockOptimizer(['```json\n{"verdict":"refine","summary":"useful but noisy",'
                         '"suggested_edit":"add guard","redundant_with":[],'
                         '"utility_score":0.3}\n```'])
    lib = _lib(("A", "body a"))
    credits = {"a": [
        SkillCredit(agent="a1", skill_id="a", contribution="helped", utility_delta=0.4),
        SkillCredit(agent="a1", skill_id="a", contribution="neutral", utility_delta=0.2),
    ]}
    grads = opt.aggregate_skill_gradients(lib, credits)
    assert grads["a"].verdict == "refine"
    assert grads["a"].n_credits == 2
    assert grads["a"].suggested_edit == "add guard"


# ── Refinement (Eq.13) ───────────────────────────────────────────────────
def test_refine_skill_applies_anchor_edit():
    opt = MockOptimizer(['```json\n[{"op":"replace","anchor":"step 2",'
                         '"old":"step 2","new":"step two"}]\n```'])
    skill = Skill(name="A", description="d", body="do step 1 then step 2")
    grad = SkillGradient(skill_id=skill.skill_id, verdict="refine",
                         summary="s", suggested_edit="rename step 2")
    refined = opt.refine_skill(skill, grad)
    assert refined is not None
    assert "step two" in refined.body
    assert refined.provenance == "refined"
    assert refined.skill_id == skill.skill_id  # stable id


def test_refine_skill_noop_returns_none():
    opt = MockOptimizer(['```json\n[]\n```'])
    skill = Skill(name="A", description="d", body="unchanged body")
    grad = SkillGradient(skill_id=skill.skill_id, verdict="refine", suggested_edit="x")
    assert opt.refine_skill(skill, grad) is None


# ── Induction (Eq.14) ────────────────────────────────────────────────────
def test_induce_skill_parses_skill_md():
    md = ("---\nname: Counterfactual Check\ndescription: Compare candidates.\n---\n"
          "Always compare at least two options before committing.")
    opt = MockOptimizer([md])
    lib = _lib(("A", "body a"))
    new = opt.induce_skill(["reward=0.1; residual=committed too early"], lib, "")
    assert new is not None
    assert new.provenance == "induced"
    assert new.name == "Counterfactual Check"


def test_induce_skill_none_response():
    opt = MockOptimizer(["NONE"])
    assert opt.induce_skill(["hard case"], _lib(("A", "b")), "") is None


# ── Consolidation (Eq.15) ────────────────────────────────────────────────
def test_detect_redundancy_groups():
    lib = _lib(("A", "ba"), ("B", "bb"), ("C", "bc"))
    grads = {
        "a": SkillGradient(skill_id="a", redundant_with=["b"]),
        "b": SkillGradient(skill_id="b", redundant_with=[]),
        "c": SkillGradient(skill_id="c", redundant_with=[]),
    }
    groups = SkillEvolutionOptimizer.detect_redundancy_groups(lib, grads)
    assert groups == [["a", "b"]]


def test_consolidate_skills_merges():
    md = ("---\nname: Merged\ndescription: Unified.\n---\nUnified procedure.")
    opt = MockOptimizer([md])
    lib = _lib(("A", "ba"), ("B", "bb"))
    macro = opt.consolidate_skills(lib, ["a", "b"])
    assert macro is not None
    assert macro.provenance == "consolidated"
    assert macro.name == "Merged"


# ── Pruning (Eq.16) ──────────────────────────────────────────────────────
def test_select_skills_to_prune_low_utility():
    opt = MockOptimizer([])
    lib = _lib(("A", "ba"), ("B", "bb"))
    grads = {
        "a": SkillGradient(skill_id="a", verdict="low_utility", utility_score=-0.8),
        "b": SkillGradient(skill_id="b", verdict="keep", utility_score=0.5),
    }
    assert opt.select_skills_to_prune(lib, grads) == ["a"]


def test_pruning_never_empties_library():
    opt = MockOptimizer([])
    lib = _lib(("A", "ba"))
    grads = {"a": SkillGradient(skill_id="a", verdict="low_utility", utility_score=-0.9)}
    # only skill -> guard keeps it
    assert opt.select_skills_to_prune(lib, grads) == []


# ── critic parsing ───────────────────────────────────────────────────────
def test_skill_credit_parse():
    critic = object.__new__(SkillCreditCritic)  # skip __init__ (no API client)
    raw = ('```json\n{"skills":[{"agent":"agent_1","skill_id":"retrieve",'
           '"contribution":"helped","evidence":"found the doc",'
           '"suggested_edit":"","redundant_with":[],"conflict_with":[],'
           '"utility_delta":0.7}],'
           '"residual":{"agent_1":{"summary":"no planning skill",'
           '"needs_new_skill":true,"missing_capability":"planning"}}}\n```')
    policies = {"agent_1": None, "agent_2": None}
    result = critic._parse(raw, policies, reward=1.0)
    assert len(result.skill_credits) == 1
    c = result.skill_credits[0]
    assert c.skill_id == "retrieve" and c.contribution == "helped"
    assert c.utility_delta == 0.7
    assert result.residuals["agent_1"].needs_new_skill is True


def test_skill_credit_parse_drops_unknown_agent():
    critic = object.__new__(SkillCreditCritic)
    raw = ('{"skills":[{"agent":"ghost","skill_id":"x","contribution":"helped"}],'
           '"residual":{}}')
    result = critic._parse(raw, {"agent_1": None}, reward=0.0)
    assert result.skill_credits == []


def test_skill_credit_parse_normalizes_agent_label():
    """Critic echoes 'Agent 1' / 'Agent 2'; must map onto agent_1 / agent_2."""
    critic = object.__new__(SkillCreditCritic)
    raw = ('{"skills":[{"agent":"Agent 1","skill_id":"retrieve","contribution":"helped"}],'
           '"residual":{"Agent 2":{"summary":"missing planning",'
           '"needs_new_skill":true,"missing_capability":"planning"}}}')
    result = critic._parse(raw, {"agent_1": None, "agent_2": None}, reward=0.0)
    assert len(result.skill_credits) == 1
    assert result.skill_credits[0].agent == "agent_1"   # 'Agent 1' -> 'agent_1'
    assert "agent_2" in result.residuals                # 'Agent 2' -> 'agent_2'
    assert result.residuals["agent_2"].needs_new_skill is True


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
