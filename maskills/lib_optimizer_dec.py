"""Decentralized MASkills optimizer.

The actor system has TWO separate skill libraries:
  - K_a (agent_1 / Researcher — owns SEARCH/BROWSE)
  - K_b (agent_2 / Solver — owns COMPUTE, emits FINAL ANSWER)

The optimizer (gpt-5.1) sees both libs + failed trajectories (which
contain both agents' turns + the HANDOFF dossier + final pred), and
outputs ops tagged with "agent": "a" | "b". Same op kinds as single-
agent (induct/refine/consolidate/prune).
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List

from openai import OpenAI

OPTIMIZER_MODEL = os.environ.get("MASKILLS_OPTIMIZER_MODEL", "openai/gpt-5.1")


SYSTEM_PROMPT = """\
You are the OPTIMIZER for a MASkills DECENTRALIZED training loop on \
the GAIA benchmark. The system has TWO agents working sequentially:

- **agent_1 (Researcher)**: tools = SEARCH, BROWSE. Reads the question \
+ inline attachment, gathers verbatim web evidence, emits a structured \
HANDOFF dossier containing: QUESTION_INTERPRETATION, KEY_FACTS_FOUND \
(with URLs), EVIDENCE_QUOTES (verbatim with URLs), \
SUGGESTED_COMPUTATION, SUGGESTED_ANSWER, UNCERTAINTY_NOTES. agent_1 \
NEVER emits FINAL ANSWER.

- **agent_2 (Solver)**: tools = COMPUTE only. Reads question + inline \
attachment + agent_1's HANDOFF. Audits the dossier, runs COMPUTE if \
needed, emits FINAL ANSWER. Can emit `REQUEST_MORE: <topic>` to bounce \
back to agent_1 (once per question).

Each agent has its own skill library:
- K_a = agent_1's skill library (root SKILL.md + sub-skills)
- K_b = agent_2's skill library (root SKILL.md + sub-skills)

You are given:
1. The CURRENT K_a and K_b.
2. A list of FAILED trajectories from running the actor system on \
training questions. Each trajectory shows both agents' turns, tool \
calls, the HANDOFF dossier, and the final pred.

Your job: propose a JSON array of operations that improve the actor \
system. Each op MUST be tagged with "agent": "a" or "agent": "b" \
indicating which library it targets.

Diagnose where each failure originates:
- BAD DOSSIER (wrong evidence / single source / wrong quotes) → fix in K_a
- BAD HANDOFF FORMAT (missing fields, ambiguous SUGGESTED_ANSWER) → fix in K_a
- BAD COMPUTE / WRONG INTERPRETATION OF DOSSIER → fix in K_b
- BAD FORMATTING / LITERAL "null" / "[unavailable]" → fix in K_b
- BAD AUDIT (blindly adopted SUGGESTED_ANSWER) → fix in K_b
- PROTOCOL BREAKDOWN (fake tool tokens, fabricated OBS) → fix in K_a or K_b

Operations:
- "induct" → ADD a sub-skill (with "agent" tag).
- "refine" → REPLACE an existing skill's body (slug + "agent" tag; \
use "_root_" for the agent's root SKILL.md). MUST include the FULL new body.
- "consolidate" → merge several sub-skills (in the same agent) into one.
- "prune" → DELETE a sub-skill (in the same agent).

Output STRICT JSON array. Each op has:

{
  "op":          "induct" | "refine" | "consolidate" | "prune",
  "agent":       "a" | "b",                  // REQUIRED
  "slug":        "<dir name>",               // for induct/refine/prune
  "name":        "<frontmatter name>",       // for induct/refine
  "description": "<one line>",               // for induct/refine
  "body":        "<full markdown body>",     // for induct/refine
  // for consolidate: "slugs_in" (list), "slug_new", "name", "description", "body"
  // for prune: "slug", "reason"
}

Be conservative — 1-3 ops total is usually best. Prefer REFINE over \
INDUCT (smaller deltas). Skill bodies must be load-bearing: include \
concrete task_ids and counter-examples from the trajectories, NOT \
generic advice. The validation gate will revert your changes if val \
score drops, so target real, repeated failure patterns."""


def render_libs_for_optimizer(bundle_a: Dict, bundle_b: Dict,
                                max_body_chars: int = 4000) -> str:
    parts = ["# CURRENT SKILL LIBRARIES (K_a and K_b)"]
    for tag, bundle in [("K_a (agent_1 Researcher)", bundle_a),
                         ("K_b (agent_2 Solver)", bundle_b)]:
        parts.append(f"\n## {tag}")
        root = bundle["root"]
        parts.append(
            f"\n### root SKILL.md  (slug _root_)\n"
            f"name: {root['name']}\n"
            f"description: {root['description']}\n"
            f"body:\n```markdown\n{root['body'][:max_body_chars]}\n```"
        )
        for slug, sk in bundle["skills"].items():
            parts.append(
                f"\n### sub-skill: {slug}\n"
                f"name: {sk['name']}\n"
                f"description: {sk['description']}\n"
                f"body:\n```markdown\n{sk['body'][:max_body_chars]}\n```"
            )
    return "\n".join(parts)


def render_failed_trajectories(failed: List[Dict],
                                 max_traj: int = 25,
                                 max_chars_each: int = 2500) -> str:
    parts = ["# FAILED TRAJECTORIES from running the 2-agent system on train"]
    for i, t in enumerate(failed[:max_traj], 1):
        tools = []
        for tl in (t.get("tool_log") or [])[:12]:
            payload = (tl.get("payload") or "")[:100]
            tools.append(f"  [{tl.get('agent','?'):8s} r{tl.get('round','?')}] "
                          f"{tl.get('kind','?'):16s} {payload}")
        tool_block = "\n".join(tools) if tools else "  (no tools)"
        # Handoff excerpt
        ho = (t.get("handoff") or "")[:1200]
        # Agent_2 final turn excerpt
        raw = t.get("raw_turns") or []
        a2_last = ""
        for r in reversed(raw):
            if isinstance(r, dict) and r.get("agent") == "agent_2":
                a2_last = (r.get("content") or "")[:800]
                break
        parts.append(
            f"\n## trajectory {i} — task_id={t.get('task_id','?')[:8]} "
            f"L{t.get('Level','?')} kind={t.get('kind','?')}\n"
            f"question: {t.get('question', t.get('Question',''))[:400]}\n"
            f"gold:     {t.get('gold', '')!r}\n"
            f"pred:     {t.get('pred', '')!r}\n"
            f"tool_log:\n{tool_block}\n"
            f"HANDOFF (excerpt):\n{ho}\n"
            f"agent_2 final turn (excerpt):\n{a2_last}\n"
        )
        excerpt = parts[-1]
        if len(excerpt) > max_chars_each:
            parts[-1] = excerpt[:max_chars_each] + "\n[…truncated…]"
    return "\n".join(parts)


def render_prior_rejected(prior: List[Dict]) -> str:
    if not prior:
        return ""
    parts = ["\n# PRIOR REJECTED OPS (failed val gate; do NOT re-propose close variants)\n"]
    for rec in prior:
        iter_n = rec.get("iter", "?")
        for op in rec.get("ops", []):
            slug = op.get("slug") or op.get("slug_new") or "?"
            kind = op.get("op", "?")
            agent = op.get("agent", "?")
            desc = (op.get("description") or "")[:200]
            parts.append(
                f"- iter {iter_n}: {kind}(agent={agent}, {slug}) — {desc} [REJECTED]")
    return "\n".join(parts) + "\n\n"


def propose_ops_dec(bundle_a: Dict, bundle_b: Dict,
                     failed_traj: List[Dict],
                     model: str = OPTIMIZER_MODEL,
                     max_ops: int = 4,
                     prior_rejected: List[Dict] = None,
                     temperature: float = 0.5) -> Dict:
    client = OpenAI()
    user_msg = (
        render_libs_for_optimizer(bundle_a, bundle_b) + "\n\n" +
        render_failed_trajectories(failed_traj) + "\n\n" +
        render_prior_rejected(prior_rejected or []) +
        f"Propose AT MOST {max_ops} operations. Prefer REFINE over INDUCT. "
        "Each op MUST have 'agent': 'a' or 'b'. "
        "Return STRICT JSON only — top-level array, no fences, no preamble."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
        max_tokens=8000,
    )
    raw = (resp.choices[0].message.content or "").strip()
    s = raw
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    ops, parse_err = [], None
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict) and "ops" in parsed:
            ops = parsed["ops"]
        elif isinstance(parsed, list):
            ops = parsed
        else:
            parse_err = f"unexpected top-level type {type(parsed).__name__}"
    except Exception as e:  # noqa: BLE001
        parse_err = f"{type(e).__name__}: {e}"
    return {
        "ops": ops,
        "raw": raw,
        "parse_error": parse_err,
        "usage": {
            "in_tok": resp.usage.prompt_tokens if resp.usage else 0,
            "out_tok": resp.usage.completion_tokens if resp.usage else 0,
        },
        "model": model,
        "n_failures_shown": min(len(failed_traj), 25),
    }
