"""MASkills optimizer.

Given (current skill library K_i, list of failed trajectories from
running K_i on training data), prompt gpt-5.1 (or any chat model) to
propose a structured JSON list of operations:

- induct   — add a new sub-skill
- refine   — edit an existing sub-skill or root
- consolidate — merge several similar sub-skills into one
- prune    — remove a sub-skill that is ineffective / redundant

Output format (one JSON array; strict):

[
  {
    "op": "induct",
    "slug": "magnitude_sanity",
    "name": "gaia-magnitude-sanity",
    "description": "<one-line, no newlines>",
    "body": "<markdown body>"
  },
  {
    "op": "refine",
    "slug": "answer_format",      // or "_root_" for the root SKILL.md
    "name": "<new name or same>",
    "description": "<new desc or same>",
    "body": "<FULL new body>"
  },
  {
    "op": "consolidate",
    "slugs_in": ["web_search", "web_browse_deep"],
    "slug_new": "web_research",
    "name": "gaia-web-research",
    "description": "...",
    "body": "..."
  },
  {
    "op": "prune",
    "slug": "<slug>",
    "reason": "<why>"
  }
]
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List

from openai import OpenAI

# Default optimizer routing — gpt-5.1 via OpenRouter
OPTIMIZER_MODEL = os.environ.get("MASKILLS_OPTIMIZER_MODEL", "openai/gpt-5.1")


SYSTEM_PROMPT = """\
You are the OPTIMIZER for a MASkills (multi-agent skills) training loop \
on the GAIA benchmark.

You are given:
1. The CURRENT skill library K_i (root SKILL.md + N sub-skills). Each \
skill is a Markdown document with `name`, `description`, and a body \
that an LLM-based actor reads at run time.
2. A list of FAILED trajectories from running the actor (gpt-4o) with \
K_i on training questions. Each trajectory is a question + gold + \
predicted answer + the actor's reasoning + tool calls + outputs.

Your job: propose a structured JSON array of operations that will \
IMPROVE the actor's score on the training distribution WITHOUT \
HURTING already-correct cases. Be conservative — most iterations \
should propose 1-4 ops. Operations:

- "induct" — ADD a new sub-skill that covers a failure pattern not \
addressed by any existing skill. Use only when the failure cluster is \
clearly new (e.g., the actor consistently fabricates tool output, but \
no skill currently says "don't fabricate"). Include concrete examples \
from the trajectories in the body — these examples are the skill's \
most valuable content.

- "refine" — EDIT an existing skill (or `_root_`) by REPLACING its \
body. Use when an existing skill is missing a specific rule, example, \
or anti-pattern that the failures show. The new body must be the \
FULL new body of that skill, not a diff.

- "consolidate" — merge 2+ overlapping skills into one. Use when \
multiple skills repeat each other and the redundancy bloats the \
system prompt without helping.

- "prune" — DELETE a skill that is never load-bearing across the \
failures shown. Use sparingly — most skills earn their keep.

Output STRICT JSON ONLY (no surrounding prose, no ```json fences). \
The top-level value must be a JSON array of operation objects per \
the schema given below. If you cannot find any worthwhile operation, \
output `[]`.

Schema for each op:

{
  "op": "induct" | "refine" | "consolidate" | "prune",
  // for induct:
  "slug":        "<kebab_or_snake_dir_name>",
  "name":        "<frontmatter name, kebab-case>",
  "description": "<one line — used to decide relevance in future tasks>",
  "body":        "<markdown body, can be multi-paragraph>",
  // for refine: slug + name + description + body (full replacement)
  // for consolidate:
  "slugs_in":    ["<slug1>", "<slug2>"],
  "slug_new":    "<new slug>",
  "name":        "...",
  "description": "...",
  "body":        "...",
  // for prune:
  "slug":        "<slug>",
  "reason":      "<why>"
}

Quality bar for skill bodies:
- Include concrete failed-task examples with task_id and the wrong-vs-right pair.
- Include anti-patterns the actor exhibited.
- State the rule in 1 sentence at the top, then expand with examples.
- No generic platitudes ("be careful", "think step by step") — those don't change behavior.

Validation will run the new K_{i+1} on a small held-out subset. If \
score drops, your ops will be REVERTED. So be conservative."""


def render_skill_lib_for_optimizer(bundle: Dict, max_body_chars: int = 5000) -> str:
    """Render the current skill library K_i in a compact form for the optimizer."""
    parts = ["# CURRENT SKILL LIBRARY (K_i)"]
    root = bundle["root"]
    parts.append(
        f"\n## root SKILL.md\n"
        f"name: {root['name']}\n"
        f"description: {root['description']}\n"
        f"body:\n```markdown\n{root['body'][:max_body_chars]}\n```"
    )
    for slug, sk in bundle["skills"].items():
        parts.append(
            f"\n## sub-skill: {slug}\n"
            f"name: {sk['name']}\n"
            f"description: {sk['description']}\n"
            f"body:\n```markdown\n{sk['body'][:max_body_chars]}\n```"
        )
    return "\n".join(parts)


def render_failures_for_optimizer(failed_traj: List[Dict],
                                    max_traj: int = 30,
                                    max_turns_chars: int = 1500) -> str:
    """Compress failed trajectories into a digest for the optimizer."""
    parts = ["# FAILED TRAJECTORIES from running K_i on train"]
    for i, t in enumerate(failed_traj[:max_traj], 1):
        # Tool log brief
        tools = []
        for tl in (t.get("tool_log") or [])[:12]:
            payload = (tl.get("payload") or "")[:100]
            tools.append(f"  [{tl.get('kind','?')}] {payload}")
        tool_block = "\n".join(tools) if tools else "  (no tools)"
        # Reasoning excerpt (first + last turn, truncated)
        raw = t.get("raw_turns") or []
        if isinstance(raw, list) and raw:
            first = (raw[0] if isinstance(raw[0], str) else
                     (raw[0].get("content") or ""))[:600]
            last = (raw[-1] if isinstance(raw[-1], str) else
                    (raw[-1].get("content") or ""))[:600]
            traj_excerpt = (
                f"  first-turn excerpt: {first}\n"
                f"  …\n"
                f"  last-turn excerpt:  {last}"
            )
        else:
            traj_excerpt = "  (no raw turns)"
        traj_excerpt = traj_excerpt[:max_turns_chars]
        parts.append(
            f"\n## trajectory {i} — task_id={t.get('task_id','?')[:8]} "
            f"L{t.get('Level','?')} kind={t.get('kind','?')}\n"
            f"question: {t.get('question', t.get('Question',''))[:400]}\n"
            f"gold:     {t.get('gold', '')!r}\n"
            f"pred:     {t.get('pred', '')!r}\n"
            f"tools:\n{tool_block}\n"
            f"trajectory:\n{traj_excerpt}\n"
        )
    return "\n".join(parts)


def render_prior_rejected_for_optimizer(prior_rejected: List[Dict]) -> str:
    """Show ops from previous iters that FAILED the validation gate, so
    the optimizer tries different angles instead of re-proposing them."""
    if not prior_rejected:
        return ""
    parts = ["\n# PRIOR REJECTED OPS (failed validation gate; do NOT re-propose these or close variants)\n"]
    for i, rec in enumerate(prior_rejected, 1):
        iter_n = rec.get("iter", "?")
        for op in rec.get("ops", []):
            slug = op.get("slug") or op.get("slug_new") or "?"
            kind = op.get("op", "?")
            desc = (op.get("description") or "")[:200]
            parts.append(
                f"- iter {iter_n}: {kind}({slug}) — {desc} "
                f"[REJECTED: val score dropped]")
    return "\n".join(parts) + "\n\n"


def propose_ops(bundle: Dict, failed_traj: List[Dict],
                 model: str = OPTIMIZER_MODEL,
                 max_ops: int = 6,
                 prior_rejected: List[Dict] = None,
                 temperature: float = 0.5) -> Dict:
    """Call the optimizer LLM, return {ops: [...], raw: <model text>, usage}.

    prior_rejected: list of dicts {iter: N, ops: [...]} that failed val
                    on prior iters. Helps the optimizer avoid repeats.
    """
    client = OpenAI()
    prior_section = render_prior_rejected_for_optimizer(prior_rejected or [])
    user_msg = (
        render_skill_lib_for_optimizer(bundle) + "\n\n" +
        render_failures_for_optimizer(failed_traj) + "\n\n" +
        prior_section +
        f"Propose AT MOST {max_ops} operations (fewer is better — 1-2 "
        f"is often best so the validation gate can isolate the effect). "
        "Prefer REFINE ops over INDUCT (smaller deltas, less risk). "
        "Return STRICT JSON only — a top-level array of op objects. "
        "No fences, no preamble, no postscript."
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

    # Strip ```json fences if model included them despite the instruction.
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)

    ops: List[Dict] = []
    parse_err = None
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict) and "ops" in parsed:
            ops = parsed["ops"]
        elif isinstance(parsed, list):
            ops = parsed
        else:
            parse_err = f"unexpected top-level type: {type(parsed).__name__}"
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
        "n_failures_shown": min(len(failed_traj), 30),
    }
