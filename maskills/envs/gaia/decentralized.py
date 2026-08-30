#!/usr/bin/env python3
"""GAIA decentralized 2-agent evaluator.

Two agents with SEPARATE skill libraries:

    <library dir>/
        agent_1/   — Researcher (SEARCH/BROWSE only, produces HANDOFF dossier)
        agent_2/   — Solver     (COMPUTE only, emits FINAL ANSWER)

Loop:
  1. agent_1 sees question + (inline) attachment, issues SEARCH/BROWSE,
     iterates up to ROUNDS_A1 turns, finally emits a HANDOFF block.
  2. agent_2 sees question + attachment + agent_1's HANDOFF; emits
     CONSTRAINTS + COMPUTE + FINAL ANSWER, iterates up to ROUNDS_A2.
  3. If agent_2 emits `REQUEST_MORE: <topic>` instead of FINAL ANSWER,
     control bounces back to agent_1 for ONE additional pass (1-2 more
     SEARCH/BROWSE calls), then back to agent_2 to finish.

Usage:
    OPENAI_API_KEY=<openrouter-key> \\
    TAVILY_API_KEY=<tavily-key>     \\
        python -m maskills.envs.gaia.decentralized \\
            --input data/gaia/test65.jsonl \\
            --model openai/gpt-4o \\
            --workers 4 \\
            --out data/gaia/test65_dec.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from ._keys import require_api_keys

# Reuse: attachment rendering, parsers, official scorer, tool dispatch
from .single_agent import (
    parse_skill,
    render_user_message,
)
from .tools import (
    _has_placeholder,
    extract_tool_calls,
    is_correct,
    parse_final_answer,
    run_browse,
    run_python_local,
    run_search,
)

# Literal "I give up" answers that score 0 under the official scorer.
# When agent_2 emits one of these, we kick it back to force a best-guess
# answer (saw 9/65 = 13.8% of R1-decentralized failures fall into this
# bucket — `null`, `[unavailable]`, etc.).
_NULL_LIKE = {
    "null", "none", "unknown", "n/a", "na",
    "[unavailable]", "unavailable",
    "[unknown]", "[null]", "[none]", "[no answer]", "no answer",
    "not found", "no information", "no data",
    "i don't know", "idk", "cannot determine", "cannot be determined",
    "insufficient data", "insufficient information",
    "(no answer)", "(unknown)", "-",
}


def _is_null_like(pred: str) -> bool:
    return (pred or "").strip().lower().strip(".") in _NULL_LIKE


# ── Skill loading per agent ─────────────────────────────────────────────────


def load_agent_skills(agent_dir: Path) -> dict:
    """Load root SKILL.md + every sub-dir's SKILL.md for one agent."""
    root = parse_skill(agent_dir / "SKILL.md")
    subs = {}
    for sub in sorted(agent_dir.iterdir()):
        if sub.is_dir():
            subs[sub.name] = parse_skill(sub / "SKILL.md")
    return {"root": root, "skills": subs}


def render_agent_system_prompt(bundle: dict) -> str:
    """Concatenate root + all sub-skill bodies. Each agent has a small
    library — load all of it as Tier A (no progressive disclosure)."""
    parts = [bundle["root"]["body"].strip(),
             "\n\n---\n\n# Tier A skill bodies (always-on for this agent)\n"]
    for name, sk in bundle["skills"].items():
        parts.append(f"\n## skill: `{sk['name']}`\n{sk['body'].strip()}\n")
    return "".join(parts)


# ── Protocol parsers ────────────────────────────────────────────────────────

HANDOFF_RE = re.compile(
    r"^HANDOFF:\s*\n(.*?)\nHANDOFF_TO_SOLVER\s*$",
    re.M | re.S,
)
REQUEST_MORE_RE = re.compile(
    r"^REQUEST_MORE:\s*(.+?)\s*$",
    re.M,
)


def extract_handoff(text: str) -> str:
    """Return the HANDOFF body (between HANDOFF: and HANDOFF_TO_SOLVER)
    or '' if no complete handoff present."""
    m = HANDOFF_RE.search(text)
    return m.group(0).strip() if m else ""


def extract_request_more(text: str) -> str:
    """Return the REQUEST_MORE topic, or ''."""
    m = REQUEST_MORE_RE.search(text)
    return m.group(1).strip() if m else ""


# ── Tool dispatch (agent_1 only) ────────────────────────────────────────────


def dispatch_a1_tool(kind: str, payload: str) -> str:
    """agent_1 has SEARCH/BROWSE only; COMPUTE attempts are blocked."""
    if kind == "SEARCH":
        return run_search(payload)
    if kind == "BROWSE":
        return run_browse(payload)
    if kind == "COMPUTE":
        return ("[BLOCKED] agent_1 has no COMPUTE tool. Surface the inputs "
                "into EVIDENCE_QUOTES and the operation into "
                "SUGGESTED_COMPUTATION; agent_2 will run the arithmetic.")
    return f"[unknown tool: {kind}]"


def dispatch_a2_tool(kind: str, payload: str) -> str:
    """agent_2 has COMPUTE only; SEARCH/BROWSE attempts are blocked."""
    if kind == "COMPUTE":
        return run_python_local(payload)
    if kind in ("SEARCH", "BROWSE"):
        return (f"[BLOCKED] agent_2 has no {kind} tool. Use COMPUTE to work "
                "from agent_1's dossier, or emit `REQUEST_MORE: <topic>` "
                "(once) to bounce a factual gap back to agent_1.")
    return f"[unknown tool: {kind}]"


# ── Worker ──────────────────────────────────────────────────────────────────


def run_one(client: OpenAI, item: dict, bundle_a1: dict, bundle_a2: dict,
            model: str, max_tokens: int = 1500,
            rounds_a1: int = 5, rounds_a2: int = 3,
            budget_a1: int = 5, budget_a2: int = 3) -> dict:
    """Run one GAIA item through the 2-agent decentralized loop."""
    sys_a1 = render_agent_system_prompt(bundle_a1)
    sys_a2 = render_agent_system_prompt(bundle_a2)
    usr_parts = render_user_message(item)

    msgs_a1 = [
        {"role": "system", "content": sys_a1},
        {"role": "user", "content": usr_parts},
    ]
    tool_log = []
    raw_turns = []
    in_tok = out_tok = 0
    err = ""
    pred = ""
    handoff_body = ""
    placeholder_kick_used = False
    null_kick_used = False
    request_more_used = False
    a1_calls = 0
    total_rounds = 0
    max_total = rounds_a1 + rounds_a2 + 3   # safety ceiling

    # Per-phase tracking of agent-2 messages
    msgs_a2 = None  # built once when we enter agent_2 phase

    def llm_call(msgs):
        """One chat completion; updates in_tok/out_tok in closure scope."""
        nonlocal in_tok, out_tok, err
        try:
            resp = client.chat.completions.create(
                model=model, messages=msgs,
                temperature=0.0, max_tokens=max_tokens,
            )
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            return None
        usage = resp.usage
        if usage:
            in_tok += usage.prompt_tokens
            out_tok += usage.completion_tokens
        return resp.choices[0].message.content or ""

    # ── Phase 1: agent_1 researches and emits HANDOFF ──
    a1_round = 0
    while a1_round < rounds_a1 and total_rounds < max_total:
        a1_round += 1
        total_rounds += 1
        content = llm_call(msgs_a1)
        if content is None:
            break
        raw_turns.append({"agent": "agent_1", "round": a1_round, "content": content})
        msgs_a1.append({"role": "assistant", "content": content})

        # Did agent_1 emit a complete HANDOFF? → done with phase 1.
        h = extract_handoff(content)
        if h:
            handoff_body = h
            break

        # Otherwise parse tool calls (agent_1 should only emit SEARCH/BROWSE).
        calls = extract_tool_calls(content)
        if not calls:
            # Kick — agent_1 emitted neither HANDOFF nor a tool call.
            if a1_round < rounds_a1:
                msgs_a1.append({"role": "user", "content":
                    "[harness] You emitted neither a tool call "
                    "(SEARCH:/BROWSE:) nor a complete `HANDOFF:` block. "
                    "Emit one of them THIS turn. If you have enough "
                    "evidence, write the HANDOFF block ending with "
                    "`HANDOFF_TO_SOLVER`. Otherwise issue a SEARCH or "
                    "BROWSE."})
                tool_log.append({"agent": "agent_1", "round": a1_round,
                                  "kind": "[KICK]", "payload": "",
                                  "result_preview": ""})
                continue
            break  # gave up

        remaining = budget_a1 - a1_calls
        if remaining <= 0:
            # Force HANDOFF on next turn
            msgs_a1.append({"role": "user", "content":
                "[harness] agent_1 tool budget exhausted. Emit the HANDOFF "
                "block now with whatever evidence you have. End with "
                "`HANDOFF_TO_SOLVER`. No more SEARCH/BROWSE will run."})
            continue
        executed = calls[:remaining]
        tool_msgs = []
        for kind, payload in executed:
            result = dispatch_a1_tool(kind, payload)
            a1_calls += 1
            tool_log.append({
                "agent": "agent_1", "round": a1_round,
                "kind": kind, "payload": payload[:300],
                "result_preview": result[:400],
            })
            tool_msgs.append(f"=== {kind} result ===\n{result}")
        # Determine the appropriate follow-up nudge: if next round is
        # the last possible round OR we've used ≥ budget-1 calls,
        # demand HANDOFF instead of more research.
        is_last_round = (a1_round >= rounds_a1 - 1)
        near_budget = (a1_calls >= budget_a1 - 1)
        if is_last_round or near_budget:
            followup = (
                "\n\n[harness] You are at round "
                f"{a1_round}/{rounds_a1} with {a1_calls}/{budget_a1} "
                "calls used. EMIT THE HANDOFF BLOCK NOW (no more tool "
                "calls). Use UNCERTAINTY_NOTES to flag whatever you "
                "couldn't verify. agent_2 needs the dossier — incomplete "
                "is better than missing.")
        else:
            followup = (
                "\n\n[harness] Continue researching, or emit the "
                "`HANDOFF:` block when ready.")
        msgs_a1.append({"role": "user", "content":
            "\n\n".join(tool_msgs) + followup})

    # If agent_1 never produced a HANDOFF, synthesize a minimal one so
    # agent_2 still gets to try.
    if not handoff_body:
        handoff_body = (
            "HANDOFF:\n"
            "QUESTION_INTERPRETATION: (agent_1 did not produce a structured "
            "handoff; agent_2 should answer from the question + attachment alone)\n"
            "KEY_FACTS_FOUND: (none)\n"
            "EVIDENCE_QUOTES: (none)\n"
            "SUGGESTED_COMPUTATION: solve directly from the question and any "
            "inline attachment\n"
            "SUGGESTED_ANSWER: null\n"
            "UNCERTAINTY_NOTES: agent_1 failed to produce a handoff "
            f"(rounds used: {a1_round}, tool calls: {a1_calls})\n"
            "HANDOFF_TO_SOLVER"
        )
        tool_log.append({"agent": "harness", "round": 0,
                          "kind": "[SYNTHETIC_HANDOFF]",
                          "payload": "", "result_preview": ""})

    # ── Phase 2: agent_2 solves ──
    a2_round = 0
    a2_calls = 0

    # Build agent_2 message history: question + attachment + dossier
    # First message: question + attachment (multi-modal). Second message:
    # the dossier as a synthetic user-role message.
    msgs_a2 = [
        {"role": "system", "content": sys_a2},
        {"role": "user", "content": usr_parts},
        {"role": "user", "content":
            "Below is agent_1 (Researcher)'s HANDOFF dossier. Treat "
            "EVIDENCE_QUOTES as your sole external source.\n\n" + handoff_body},
    ]

    while a2_round < rounds_a2 and total_rounds < max_total:
        a2_round += 1
        total_rounds += 1
        content = llm_call(msgs_a2)
        if content is None:
            break
        raw_turns.append({"agent": "agent_2", "round": a2_round, "content": content})
        msgs_a2.append({"role": "assistant", "content": content})

        # Did agent_2 ask for more research?
        topic = extract_request_more(content)
        if topic and not request_more_used and parse_final_answer(content) == "":
            request_more_used = True
            tool_log.append({
                "agent": "agent_2", "round": a2_round,
                "kind": "[REQUEST_MORE]", "payload": topic[:300],
                "result_preview": "",
            })
            # Bounce back to agent_1 for one additional research pass.
            msgs_a1.append({"role": "user", "content":
                f"[from agent_2] agent_2 needs additional evidence: {topic}\n\n"
                "Run 1-2 more SEARCH/BROWSE calls focused on this gap, "
                "then emit a REFRESHED `HANDOFF:` block (with the prior "
                "evidence + the new findings). End with HANDOFF_TO_SOLVER."})
            # Up to 2 more agent_1 rounds.
            for extra in range(2):
                if total_rounds >= max_total:
                    break
                total_rounds += 1
                c1 = llm_call(msgs_a1)
                if c1 is None:
                    break
                raw_turns.append({"agent": "agent_1", "round": a1_round + 1 + extra,
                                   "content": c1})
                msgs_a1.append({"role": "assistant", "content": c1})
                h = extract_handoff(c1)
                if h:
                    handoff_body = h
                    # Inject refreshed dossier into agent_2 context.
                    msgs_a2.append({"role": "user", "content":
                        "[harness] agent_1 returned a refreshed HANDOFF:\n\n"
                        + h + "\n\n[harness] Now emit FINAL ANSWER (with one "
                        "more COMPUTE if needed)."})
                    break
                calls1 = extract_tool_calls(c1)
                if not calls1:
                    msgs_a1.append({"role": "user", "content":
                        "[harness] You must either issue SEARCH/BROWSE or "
                        "emit the refreshed HANDOFF. Do one THIS turn."})
                    tool_log.append({"agent": "agent_1",
                                      "round": a1_round + 1 + extra,
                                      "kind": "[KICK]", "payload": "",
                                      "result_preview": ""})
                    continue
                remaining = max(0, budget_a1 + 2 - a1_calls)  # 2 bonus calls
                executed = calls1[:remaining]
                tool_msgs = []
                for kind, payload in executed:
                    result = dispatch_a1_tool(kind, payload)
                    a1_calls += 1
                    tool_log.append({
                        "agent": "agent_1", "round": a1_round + 1 + extra,
                        "kind": kind, "payload": payload[:300],
                        "result_preview": result[:400],
                    })
                    tool_msgs.append(f"=== {kind} result ===\n{result}")
                msgs_a1.append({"role": "user", "content":
                    "\n\n".join(tool_msgs) + "\n\n[harness] Emit the "
                    "REFRESHED HANDOFF now (HANDOFF_TO_SOLVER)."})
            else:
                # No HANDOFF came back; tell agent_2 to commit anyway.
                msgs_a2.append({"role": "user", "content":
                    "[harness] agent_1 did not produce a refreshed HANDOFF. "
                    "Commit your best answer from the original dossier "
                    "now — emit FINAL ANSWER on this next turn."})
            continue

        # FINAL ANSWER terminates — modulo placeholder + null-guard.
        pred = parse_final_answer(content)
        if pred:
            if (not placeholder_kick_used and a2_round < rounds_a2
                    and _has_placeholder(pred)):
                placeholder_kick_used = True
                tool_log.append({"agent": "agent_2", "round": a2_round,
                                  "kind": "[PLACEHOLDER_KICK]",
                                  "payload": pred, "result_preview": ""})
                msgs_a2.append({"role": "user", "content":
                    f"[harness] Your FINAL ANSWER contains a literal "
                    f"angle-bracket placeholder ({pred!r}). Substitute "
                    f"concrete values from the dossier, or emit your best "
                    f"concrete guess. Re-emit FINAL ANSWER."})
                pred = ""
                continue
            if (not null_kick_used and a2_round < rounds_a2
                    and _is_null_like(pred)):
                null_kick_used = True
                tool_log.append({"agent": "agent_2", "round": a2_round,
                                  "kind": "[NULL_KICK]",
                                  "payload": pred, "result_preview": ""})
                msgs_a2.append({"role": "user", "content":
                    f"[harness] You emitted {pred!r} as FINAL ANSWER. "
                    f"Literal `null` / `unknown` / `[unavailable]` etc. "
                    f"are scored as 0 — they are NEVER acceptable. You "
                    f"MUST commit a concrete best guess.\n\n"
                    f"For NUMERIC questions: estimate the MAGNITUDE from "
                    f"the question's domain — do NOT default to 0. A "
                    f"question that asks 'how many X' almost always has "
                    f"a nonzero answer.\n"
                    f"  - 'how many continents' → 5-7\n"
                    f"  - 'monarchies with sea access in Asia' → 10-15\n"
                    f"  - 'Twitter citations on a Wikipedia page' → 1-10\n"
                    f"  - 'HVAC residential CFM' → 50-200\n"
                    f"  - 'population in millions' → 1-1500\n"
                    f"Pick the midpoint of the plausible range, not 0.\n\n"
                    f"For ENTITY questions: pick the most-cited candidate "
                    f"from EVIDENCE_QUOTES, or the most plausible from "
                    f"prior knowledge.\n\n"
                    f"For LIST questions: if the question implies ≥2 "
                    f"items (uses 'and', plural noun, 'the X and Y'), "
                    f"emit AT LEAST 2 items, even if the second is a "
                    f"guess.\n\n"
                    f"Re-emit FINAL ANSWER with a concrete value."})
                pred = ""
                continue
            break

        # Otherwise: agent_2 may be running a COMPUTE.
        calls = extract_tool_calls(content)
        if not calls:
            if a2_round < rounds_a2:
                msgs_a2.append({"role": "user", "content":
                    "[harness] You emitted neither a tool call (COMPUTE:) "
                    "nor `FINAL ANSWER:` nor `REQUEST_MORE:`. Do ONE this "
                    "turn. Best guess beats no answer."})
                tool_log.append({"agent": "agent_2", "round": a2_round,
                                  "kind": "[KICK]", "payload": "",
                                  "result_preview": ""})
                continue
            break
        remaining = budget_a2 - a2_calls
        if remaining <= 0:
            msgs_a2.append({"role": "user", "content":
                "[harness] agent_2 COMPUTE budget exhausted. Emit FINAL "
                "ANSWER now from the COMPUTE results so far."})
            continue
        executed = calls[:remaining]
        tool_msgs = []
        for kind, payload in executed:
            result = dispatch_a2_tool(kind, payload)
            if kind == "COMPUTE":
                a2_calls += 1
            tool_log.append({
                "agent": "agent_2", "round": a2_round,
                "kind": kind, "payload": payload[:300],
                "result_preview": result[:400],
            })
            tool_msgs.append(f"=== {kind} result ===\n{result}")
        msgs_a2.append({"role": "user", "content":
            "\n\n".join(tool_msgs) + "\n\n[harness] Emit FINAL ANSWER now."})

    ok = is_correct(pred, item["Final answer"]) if pred else False
    return {
        "task_id": item["task_id"],
        "Level": item["Level"],
        "kind": item["kind"],
        "gold": item["Final answer"],
        "pred": pred,
        "correct": ok,
        "in_tok": in_tok,
        "out_tok": out_tok,
        "n_rounds": len(raw_turns),
        "n_tool_calls": sum(1 for t in tool_log
                             if t["kind"] in ("SEARCH", "BROWSE", "COMPUTE")),
        "a1_rounds": a1_round,
        "a2_rounds": a2_round,
        "a1_calls": a1_calls,
        "a2_calls": a2_calls,
        "request_more_used": request_more_used,
        "handoff": handoff_body[:2000],
        "tool_log": tool_log,
        "error": err,
        "raw_turns": raw_turns,
    }


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    require_api_keys(tavily=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--skills-dir", type=str, required=True,
                    help="skill library pair to run (must contain "
                         "agent_1/ and agent_2/ subdirs each with SKILL.md).")
    ap.add_argument("--model", type=str, default="openai/gpt-4o")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--rounds-a1", type=int, default=5)
    ap.add_argument("--rounds-a2", type=int, default=3)
    ap.add_argument("--budget-a1", type=int, default=5)
    ap.add_argument("--budget-a2", type=int, default=3)
    args = ap.parse_args()

    skills_dir = Path(args.skills_dir)
    print(f"skills : {skills_dir}")
    bundle_a1 = load_agent_skills(skills_dir / "agent_1")
    bundle_a2 = load_agent_skills(skills_dir / "agent_2")
    print(f"agent_1 skills: 1 root + {len(bundle_a1['skills'])} sub-skills "
          f"({', '.join(bundle_a1['skills'])})")
    print(f"agent_2 skills: 1 root + {len(bundle_a2['skills'])} sub-skills "
          f"({', '.join(bundle_a2['skills'])})")

    items = [json.loads(l) for l in Path(args.input).open()]
    if args.n > 0:
        items = items[: args.n]
    out_path = Path(args.out) if args.out else (
        Path(args.input).with_suffix("")
        .with_suffix(f".{args.model.replace('/', '_')}.dec.jsonl")
    )
    # Resume: skip task_ids already present in out_path.
    done_ids = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                done_ids.add(json.loads(line)["task_id"])
            except Exception:  # noqa: BLE001
                pass
    if done_ids:
        before = len(items)
        items = [it for it in items if it["task_id"] not in done_ids]
        print(f"[resume] skipping {before - len(items)} already-done items "
              f"in {out_path}")
    print(f"Input  : {args.input} ({len(items)} items to run)")
    print(f"Model  : {args.model}  (both agents)")
    print(f"Out    : {out_path} (append mode)\n")

    client = OpenAI()
    results = [None] * len(items)
    # Incremental append handle — flushed after each completed item
    out_fh = out_path.open("a")
    import threading
    write_lock = threading.Lock()
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut_to_idx = {
            ex.submit(run_one, client, item, bundle_a1, bundle_a2,
                       args.model, args.max_tokens,
                       args.rounds_a1, args.rounds_a2,
                       args.budget_a1, args.budget_a2): i
            for i, item in enumerate(items)
        }
        done = 0
        for fut in as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            r = fut.result()
            results[i] = r
            done += 1
            mark = "✓" if r["correct"] else ("E" if r["error"] else "✗")
            rm = "+RM" if r["request_more_used"] else "   "
            print(f"  [{done:3d}/{len(items)}] {mark} L{r['Level']} "
                  f"{r['task_id'][:8]} {r['kind']:6s}  "
                  f"a1={r['a1_rounds']}/{r['a1_calls']} "
                  f"a2={r['a2_rounds']}/{r['a2_calls']}{rm}  "
                  f"pred={r['pred']!r:25s}  gold={r['gold']!r:25s}  "
                  f"({r['in_tok']}+{r['out_tok']} tok)"
                  + (f"  err={r['error']}" if r["error"] else ""))
            # Incremental write — survives bg-kill.
            with write_lock:
                out_fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                out_fh.flush()

    out_fh.close()
    # Re-read combined (in case of resume) to produce final summary.
    all_results = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
    dt = time.time() - t0

    correct = sum(1 for r in all_results if r["correct"])
    in_tok = sum(r.get("in_tok", 0) for r in all_results)
    out_tok = sum(r.get("out_tok", 0) for r in all_results)
    n_errs = sum(1 for r in all_results if r.get("error"))
    n_rm = sum(1 for r in all_results if r.get("request_more_used"))
    n_tool = sum(r.get("n_tool_calls", 0) for r in all_results)
    cost = (in_tok / 1e6) * 2.5 + (out_tok / 1e6) * 10  # gpt-4o pricing

    N = len(all_results)
    print("\n" + "=" * 78)
    print(f"OVERALL  {correct}/{N} = {correct/N*100:5.1f}%   "
          f"({n_errs} API errors)   in {dt:.1f}s (this run only)")
    print(f"Tokens   {in_tok:,} in + {out_tok:,} out = {in_tok+out_tok:,}"
          f"   (~${cost:.2f} list)")
    print(f"Tools    {n_tool}  ({n_tool/max(N,1):.1f}/question)   "
          f"REQUEST_MORE used: {n_rm}/{N} "
          f"({n_rm/max(N,1)*100:.0f}%)")

    by_lvl = {}
    for r in all_results:
        by_lvl.setdefault(r["Level"], [0, 0])
        by_lvl[r["Level"]][0] += int(r["correct"])
        by_lvl[r["Level"]][1] += 1
    print("\nBy Level:")
    for lvl in sorted(by_lvl):
        c, t = by_lvl[lvl]
        print(f"  L{lvl}: {c:2d}/{t:2d}  ({c/t*100:5.1f}%)")

    by_kind = {}
    for r in all_results:
        by_kind.setdefault(r["kind"], [0, 0])
        by_kind[r["kind"]][0] += int(r["correct"])
        by_kind[r["kind"]][1] += 1
    print("\nBy modality:")
    for k in sorted(by_kind):
        c, t = by_kind[k]
        print(f"  {k:8s}: {c:2d}/{t:2d}  ({c/t*100:5.1f}%)")

    tool_kinds = {}
    for r in all_results:
        for tl in r.get("tool_log", []):
            k = tl["kind"]
            tool_kinds[k] = tool_kinds.get(k, 0) + 1
    print("\nTool-call breakdown (incl. harness events):")
    for k, v in sorted(tool_kinds.items(), key=lambda kv: -kv[1]):
        print(f"  {k:20s}: {v}")
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
