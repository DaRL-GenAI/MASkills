#!/usr/bin/env python3
"""GAIA seed-skill evaluator WITH REAL TOOLS.

Multi-turn loop. Each turn the harness parses the model's assistant
message for:

    SEARCH: <query>
    BROWSE: <url>
    COMPUTE:
    ```python
    <code>
    ```

…executes them, and feeds results back as a `user`-role tool result.
Loops until the model emits ``FINAL ANSWER:`` or hits the tool budget.

Tools:
- SEARCH / BROWSE  → Tavily Search & Extract APIs   (TAVILY_API_KEY env)
- COMPUTE          → local subprocess Python, 10 s timeout, no network env
                      (swap ``run_python_local`` for a Riza call to upgrade)

Usage:
    OPENAI_API_KEY=<openrouter-key> \\
    TAVILY_API_KEY=<tavily-key>     \\
        python -m maskills.envs.gaia.tools   \\
            --input data/gaia/test65.jsonl \\
            --model openai/gpt-4o \\
            --workers 4 \\
            --out data/gaia/test65_gpt4o_tools.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from ._keys import require_api_keys
from .single_agent import (
    load_skills,
    render_system_prompt,
    render_user_message,
    select_relevant_b,
)
from .single_agent import (
    parse_final_answer as _raw_parse_final_answer,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "gaia"

# Use the OFFICIAL GAIA scorer for is_correct (downloaded from the HF Space
# gaia-benchmark/leaderboard). Falls back to a local copy if cache missing.
_GAIA_SCORER_DIR = str(
    Path.home()
    / ".cache/huggingface/hub/spaces--gaia-benchmark--leaderboard"
    / "snapshots/9f133d71362e77b3539f1514f31b9c101a545fec"
)
try:
    sys.path.insert(0, _GAIA_SCORER_DIR)
    from scorer import question_scorer as _gaia_scorer  # noqa: E402

    def is_correct(pred: str, gold: str) -> bool:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            return bool(_gaia_scorer(pred if pred else "None", gold))
except Exception:  # noqa: BLE001
    # Fallback: our home-grown scorer (close to official, modulo whitespace
    # stripping). Only used if the HF Space cache is missing.
    from .single_agent import is_correct  # noqa: F811


# ── FINAL ANSWER cleanup (harness-side enforcement of answer-format rules) ─

# Leading hedge / prose patterns to strip from the answer value.
_HEDGE_PREFIXES = re.compile(
    r"^\s*(?:"
    r"\[\s*unverified\s*\]"
    r"|approximately"
    r"|approx\.?"
    r"|about"
    r"|roughly"
    r"|around"
    r"|circa"
    r"|est\.?"
    r"|~"
    r"|the\s+answer\s+is"
    r"|it\s+is"
    r"|answer\s*[:\-]"
    r"|i\s+think"
    r"|my\s+best\s+guess\s+is"
    r"|my\s+guess\s+is"
    r"|best\s+guess[:\-]?"
    r")\s*[:\-]?\s*",
    re.IGNORECASE,
)
# Strip a trailing parenthetical only if it contains a clear English
# explanation word (rounded, approximately, see source, etc.). This
# avoids eating math notation like "(A ∨ ¬B)" or "(x+1)".
_EXPLAIN_WORDS = re.compile(
    r"\b(rounded|approximately|approx|estimated|because|based|"
    r"according|see|from|circa|around|nearest|source|note|citation|"
    r"per\s+\w+|verified|unverified|"
    # Added in H1: hedge phrases that escaped earlier strip.
    # Real case `7a4a336d`: pred="Mario Kart Stadium (track not "
    # "confirmed, world record time unavailable)" — needed
    # `unavailable` / `confirmed` / `unknown` recognition.
    r"unavailable|confirmed|missing|unknown|not\s+\w+|"
    r"could\s+not|cannot\s+\w+|insufficient|unable|"
    r"best\s+\w+|likely|possibly|maybe|presumed|inferred|"
    r"no\s+\w+|undisclosed|undetermined|"
    r"agent\s*_?\s*\d|dossier|handoff)\b",
    re.IGNORECASE,
)


def _strip_trailing_explanation_paren(s: str) -> str:
    m = re.search(r"\s*\(([^()]+)\)\s*$", s)
    if m and _EXPLAIN_WORDS.search(m.group(1)):
        return s[: m.start()].rstrip()
    return s


def _clean_pred(s: str) -> str:
    """Apply mechanical answer-format cleanup the model often skips."""
    s = s.strip()
    # Strip surrounding quotes (model loves "quoting" final answers).
    if len(s) >= 2 and s[0] == s[-1] and s[0] in {'"', "'", "`"}:
        s = s[1:-1].strip()
    # Repeatedly strip hedge / prose prefixes (often stacked).
    for _ in range(3):
        new = _HEDGE_PREFIXES.sub("", s, count=1).strip()
        if new == s:
            break
        s = new
    # Trailing parenthetical (only if it's an English explanation).
    s = _strip_trailing_explanation_paren(s)
    # Trailing period / sentence ender (unless gold IS a sentence — we can't
    # tell here, so default to strip; the eval normalizer also strips).
    s = s.rstrip(".").strip()
    # Strip outer quotes after prefix stripping uncovers them.
    if len(s) >= 2 and s[0] == s[-1] and s[0] in {'"', "'"}:
        s = s[1:-1].strip()
    return s


def parse_final_answer(text: str) -> str:
    raw = _raw_parse_final_answer(text)
    return _clean_pred(raw) if raw else raw


# ── Placeholder guard: pred contains literal <...> template? ──────────────

# A "<word>" or "<short phrase>" inside the answer string indicates the
# model emitted an unfilled template. Real case `a0c07678` answered
# "<last name before>, <last name after>" and scored 0. We require the
# bracketed token to contain only letters/spaces/underscores/hyphens and
# to be short (<=40 chars) — avoids false positives on real text like
# "Smith<Jones" (unlikely) or math notation like "x<y" (no space inside).
_PLACEHOLDER_RE = re.compile(r"<[A-Za-z][A-Za-z0-9 _\-]{0,40}>")


def _has_placeholder(s: str) -> bool:
    return bool(_PLACEHOLDER_RE.search(s or ""))


# ── Heuristic: does this pred look like a number? ──────────────────────────


def _looks_numeric(s: str) -> bool:
    if not s:
        return False
    t = s.replace(",", "").replace("$", "").strip().rstrip("%")
    try:
        float(t)
        return True
    except ValueError:
        return False

# ── Tool implementations ───────────────────────────────────────────────────


TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"


def _tavily_post(url: str, payload: dict, timeout: int = 30) -> dict:
    payload = dict(payload)
    payload["api_key"] = os.environ["TAVILY_API_KEY"]
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def run_search_tavily(query: str, max_results: int = 5) -> str:
    """Tavily search. Returns top hits as title + URL + snippet."""
    try:
        data = _tavily_post(TAVILY_SEARCH_URL, {
            "query": query[:400],
            "max_results": max_results,
            "search_depth": "basic",
        })
    except Exception as e:  # noqa: BLE001
        return f"[SEARCH ERROR] {type(e).__name__}: {e}"
    results = data.get("results", [])
    if not results:
        return f"[SEARCH] no results for {query!r}"
    out = [f"[SEARCH RESULTS for {query!r}]"]
    for i, r in enumerate(results[:max_results], 1):
        title = r.get("title", "")[:120]
        url = r.get("url", "")
        snip = (r.get("content") or "")[:300].replace("\n", " ")
        out.append(f"{i}. {title}\n   {url}\n   {snip}")
    return "\n".join(out)


# Perplexity Sonar via OpenRouter — grounded web answer with citations.
_sonar_client = None


def _get_sonar_client():
    global _sonar_client
    if _sonar_client is None:
        _sonar_client = OpenAI()  # same OpenRouter creds; model selects routing
    return _sonar_client


def run_search_sonar(query: str, model: str = "perplexity/sonar") -> str:
    """Route a SEARCH: query through Perplexity Sonar — grounded answer with
    citations, instead of raw Tavily snippets. Returns text the orchestrator
    LLM (gpt-4o) can read directly."""
    try:
        client = _get_sonar_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "Answer the user's query factually. Cite sources "
                            "inline as [1], [2]. Keep the answer under 250 words."},
                {"role": "user", "content": query[:400]},
            ],
            temperature=0.0,
            max_tokens=500,
        )
        ans = (resp.choices[0].message.content or "").strip()
        # OpenRouter Sonar usually returns citations in annotations or
        # in the message.annotations; if present, append them.
        citations_lines = []
        try:
            ann = getattr(resp.choices[0].message, "annotations", None) or []
            for a in ann[:10]:
                u = (a.get("url_citation") or {}).get("url") if isinstance(a, dict) else getattr(getattr(a, "url_citation", None), "url", None)
                if u:
                    citations_lines.append(f"  - {u}")
        except Exception:  # noqa: BLE001
            pass
        cit = ("\nCitations:\n" + "\n".join(citations_lines)) if citations_lines else ""
        return f"[SONAR ANSWER for {query!r}]\n{ans}{cit}"
    except Exception as e:  # noqa: BLE001
        # Fallback to Tavily if sonar fails (rate limit, etc.)
        return f"[SONAR ERROR; falling back to Tavily] {type(e).__name__}: {e}\n" + run_search_tavily(query)


# Pick search backend via env var (default: sonar).
_SEARCH_BACKEND = os.environ.get("GAIA_SEARCH_BACKEND", "sonar").lower()


def run_search(query: str, **kw) -> str:
    if _SEARCH_BACKEND == "tavily":
        return run_search_tavily(query)
    return run_search_sonar(query)


def run_browse(url: str, max_chars: int = 6000) -> str:
    """Tavily extract. Returns the cleaned main text of a page."""
    try:
        data = _tavily_post(TAVILY_EXTRACT_URL, {
            "urls": [url],
            "extract_depth": "basic",
        })
    except Exception as e:  # noqa: BLE001
        return f"[BROWSE ERROR] {type(e).__name__}: {e}"
    results = data.get("results", [])
    if not results:
        fails = data.get("failed_results", [])
        return f"[BROWSE] no content for {url} (failed={fails})"
    content = results[0].get("raw_content") or results[0].get("content") or ""
    content = content[:max_chars]
    return f"[BROWSE {url}]\n{content}"


def run_python_local(code: str, timeout: int = 10) -> str:
    """Local-subprocess Python executor (Riza-compatible interface).

    For research use only — runs LLM-emitted code on the host. Mitigations:
    - 10 s wall-clock timeout
    - cwd = a fresh tempdir
    - PATH only; no HOME/credentials forwarded
    - No network is *not* enforced at the OS level here; for stronger
      isolation, swap this function for a Riza HTTP call.
    """
    with tempfile.TemporaryDirectory() as cwd:
        script_path = os.path.join(cwd, "snippet.py")
        Path(script_path).write_text(code)
        try:
            r = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True, timeout=timeout,
                cwd=cwd,
                env={"PATH": "/usr/bin:/bin:/usr/local/bin"},
            )
        except subprocess.TimeoutExpired:
            return f"[COMPUTE] timed out after {timeout}s"
        out = (r.stdout or "")[:4000]
        err = (r.stderr or "")[:2000]
        msg = f"[COMPUTE stdout]\n{out}" if out else "[COMPUTE] (no stdout)"
        if err:
            msg += f"\n[COMPUTE stderr]\n{err}"
        return msg


# ── Tool-call parser ───────────────────────────────────────────────────────

# Matches a fenced python block right after the COMPUTE: marker.
COMPUTE_RE = re.compile(
    r"^[ \t]*COMPUTE:[ \t]*\n+```(?:python|py)?\n(.*?)\n```",
    re.M | re.S,
)
SEARCH_RE = re.compile(r"^[ \t]*SEARCH:[ \t]*(.+?)[ \t]*$", re.M)
BROWSE_RE = re.compile(r"^[ \t]*BROWSE:[ \t]*(\S+)[ \t]*$", re.M)


def extract_tool_calls(text: str) -> list:
    """Return tool calls in their textual order: [(kind, payload), …]."""
    calls = []
    for m in COMPUTE_RE.finditer(text):
        calls.append((m.start(), "COMPUTE", m.group(1)))
    for m in SEARCH_RE.finditer(text):
        calls.append((m.start(), "SEARCH", m.group(1).strip()))
    for m in BROWSE_RE.finditer(text):
        calls.append((m.start(), "BROWSE", m.group(1).strip()))
    calls.sort()
    return [(k, p) for _, k, p in calls]


def dispatch_tool(kind: str, payload: str) -> str:
    if kind == "SEARCH":
        return run_search(payload)
    if kind == "BROWSE":
        return run_browse(payload)
    if kind == "COMPUTE":
        return run_python_local(payload)
    return f"[unknown tool: {kind}]"


# ── Multi-turn worker ──────────────────────────────────────────────────────


def run_one(client: OpenAI, item: dict, bundle: dict, model: str,
            max_tokens: int = 1500, max_rounds: int = 6,
            tool_budget: int = 6) -> dict:
    rel_b = select_relevant_b(item)
    sys_prompt = render_system_prompt(bundle, rel_b)
    usr_parts = render_user_message(item)

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": usr_parts},
    ]
    tool_log = []   # list of dicts: {round, kind, payload, result_preview}
    in_tok = out_tok = 0
    raw_turns = []
    err = ""
    pred = ""
    compute_force_used = False  # at most one COMPUTE-force retry
    placeholder_kick_used = False  # at most one placeholder-guard retry

    for rnd in range(1, max_rounds + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            break

        # Defensive: OpenRouter/Azure occasionally returns resp.choices=None
        # or empty, or messages with None content (e.g., on content filter).
        if not resp.choices:
            err = "NoChoicesReturned"
            break
        content = (resp.choices[0].message.content or "") if resp.choices[0].message else ""
        usage = resp.usage
        if usage:
            in_tok += usage.prompt_tokens
            out_tok += usage.completion_tokens
        raw_turns.append(content)
        messages.append({"role": "assistant", "content": content})

        # FINAL ANSWER terminates — unless we want to force a COMPUTE retry.
        pred = parse_final_answer(content)
        if pred:
            # PLACEHOLDER GUARD: angle-bracket template in answer.
            if (
                not placeholder_kick_used
                and rnd < max_rounds
                and _has_placeholder(pred)
            ):
                placeholder_kick_used = True
                tool_log.append({
                    "round": rnd, "kind": "[PLACEHOLDER_KICK]",
                    "payload": pred, "result_preview": "",
                })
                messages.append({"role": "user", "content":
                    f"[harness] Your FINAL ANSWER contains a literal "
                    f"angle-bracket placeholder ({pred!r}). Placeholders "
                    f"like '<name>' / '<X>' indicate an unsolved template, "
                    f"not a real answer. Either (a) finish the lookup / "
                    f"computation and substitute concrete values, or "
                    f"(b) emit your best concrete guess. Re-emit a fresh "
                    f"FINAL ANSWER line. Best guess beats placeholder."})
                pred = ""
                continue
            # COMPUTE-force triggers in any of these cases:
            #   (a) Numeric answer + no COMPUTE + had SEARCH/BROWSE
            #   (b) Pred is "unknown"/"none"/"i don't know" + had SEARCH/BROWSE
            #       (model gave up; force at least one COMPUTE attempt to
            #       extract from search results)
            #   (c) Pred is a list of length 1 but question implies multiple
            #       (heuristic: comma in question text + question contains
            #        "list" or "which X and Y" or "what …s")
            has_compute = any(t["kind"] == "COMPUTE" for t in tool_log)
            has_lookup = any(t["kind"] in ("SEARCH", "BROWSE") for t in tool_log)
            pred_lower = pred.strip().lower()
            gave_up = pred_lower in {
                "unknown", "none", "i don't know", "n/a", "na", "not found",
                "no answer", "no information", "no data"}
            should_force = (
                not compute_force_used
                and rnd < max_rounds
                and (
                    (_looks_numeric(pred) and not has_compute and has_lookup)
                    or (gave_up and has_lookup)
                )
            )
            if should_force:
                compute_force_used = True
                tool_log.append({
                    "round": rnd, "kind": "[COMPUTE_FORCE]",
                    "payload": pred, "result_preview": "",
                })
                if gave_up:
                    msg = (
                        f"[harness] You answered {pred!r} despite "
                        f"{sum(1 for t in tool_log if t['kind'] in ('SEARCH','BROWSE'))} "
                        "search/browse calls returning real content. "
                        "Re-read the search results in this conversation, "
                        "extract the relevant fact, and emit a fresh "
                        "FINAL ANSWER (use a COMPUTE: block if any "
                        "counting/arithmetic is needed). Best guess beats "
                        "'unknown'.")
                else:
                    msg = (
                        f"[harness] Your draft answer is a number ({pred!r}) "
                        "but you never used a `COMPUTE:` block. Mental-math "
                        "from search-result text is the #1 numeric failure "
                        "mode here. Re-derive this number with a `COMPUTE:` "
                        "Python block (paste the relevant excerpt into a "
                        "string, then `count`/`sum`/`len`/arithmetic + "
                        "`print(...)`), then re-emit `FINAL ANSWER: <value>`. "
                        "If your existing number is verified correct, still "
                        "emit COMPUTE confirming it.")
                messages.append({"role": "user", "content": msg})
                pred = ""
                continue
            break

        # Otherwise parse + execute tool calls (up to budget).
        calls = extract_tool_calls(content)
        if not calls:
            # Lazy-compute kick: one retry before giving up.
            if rnd < max_rounds and not getattr(run_one, "_kicked", False):
                messages.append({"role": "user", "content":
                    "[harness] You emitted neither a tool call "
                    "(SEARCH:/BROWSE:/COMPUTE:) nor `FINAL ANSWER:`. "
                    "You MUST do one of the two THIS turn. If your reasoning "
                    "is complete, emit `FINAL ANSWER: <answer>` (a best guess "
                    "if uncertain). Otherwise emit a tool call."})
                tool_log.append({"round": rnd, "kind": "[KICK]",
                                  "payload": "", "result_preview": ""})
                continue
            break  # model gave up without answer
        remaining = tool_budget - sum(1 for _ in tool_log)
        if remaining <= 0:
            # Force final-answer round
            messages.append({"role": "user", "content":
                "[harness] Tool budget exhausted. Emit FINAL ANSWER now."})
            continue
        executed = calls[:remaining]
        tool_msgs = []
        for kind, payload in executed:
            result = dispatch_tool(kind, payload)
            tool_log.append({
                "round": rnd,
                "kind": kind,
                "payload": payload[:300],
                "result_preview": result[:400],
            })
            tool_msgs.append(f"=== {kind} result ===\n{result}")
        messages.append({"role": "user", "content":
            "\n\n".join(tool_msgs) + "\n\n[harness] Continue, or emit FINAL ANSWER."})

    ok = is_correct(pred, item["Final answer"]) if pred else False
    return {
        "task_id": item["task_id"],
        "Level": item["Level"],
        "kind": item["kind"],
        "rel_B": rel_b,
        "gold": item["Final answer"],
        "pred": pred,
        "correct": ok,
        "in_tok": in_tok,
        "out_tok": out_tok,
        "n_rounds": len(raw_turns),
        "n_tool_calls": len(tool_log),
        "tool_log": tool_log,
        "error": err,
        "raw_turns": raw_turns,
    }


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    require_api_keys(tavily=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills-dir", type=str, required=True,
                    help="skill library to run: a directory with a root "
                         "SKILL.md and one sub-directory per sub-skill")
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--model", type=str, default="openai/gpt-4o")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--tool-budget", type=int, default=6)
    args = ap.parse_args()

    bundle = load_skills(Path(args.skills_dir))
    print(f"Skills: 1 root + {len(bundle['skills'])} sub-skills")
    print(f"Tools : SEARCH ({_SEARCH_BACKEND}) / BROWSE (Tavily) / COMPUTE (local subprocess)")

    items = [json.loads(l) for l in Path(args.input).open()]
    if args.n > 0:
        items = items[: args.n]
    out_path = Path(args.out) if args.out else (
        Path(args.input).with_suffix("")
        .with_suffix(f".{args.model.replace('/', '_')}.tools.jsonl")
    )
    print(f"Input : {args.input} ({len(items)} items)")
    print(f"Model : {args.model}")
    print(f"Out   : {out_path}\n")

    client = OpenAI()
    results = [None] * len(items)
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut_to_idx = {
            ex.submit(
                run_one, client, item, bundle, args.model,
                args.max_tokens, args.max_rounds, args.tool_budget,
            ): i
            for i, item in enumerate(items)
        }
        done = 0
        for fut in as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            r = fut.result()
            results[i] = r
            done += 1
            mark = "✓" if r["correct"] else ("E" if r["error"] else "✗")
            print(f"  [{done:3d}/{len(items)}] {mark} L{r['Level']} "
                  f"{r['task_id'][:8]} {r['kind']:6s}  "
                  f"rounds={r['n_rounds']} tools={r['n_tool_calls']}  "
                  f"pred={r['pred']!r:25s}  gold={r['gold']!r:25s}  "
                  f"({r['in_tok']}+{r['out_tok']} tok)"
                  + (f"  err={r['error']}" if r["error"] else ""))

    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results))
    dt = time.time() - t0

    correct = sum(1 for r in results if r["correct"])
    in_tok = sum(r["in_tok"] for r in results)
    out_tok = sum(r["out_tok"] for r in results)
    n_tools = sum(r["n_tool_calls"] for r in results)
    n_errs = sum(1 for r in results if r["error"])
    cost = (in_tok / 1e6) * 2.5 + (out_tok / 1e6) * 10  # gpt-4o pricing

    print("\n" + "=" * 78)
    print(f"OVERALL  {correct}/{len(items)} = {correct/len(items)*100:5.1f}%   "
          f"({n_errs} API errors)   in {dt:.1f}s")
    print(f"Tokens   {in_tok:,} in + {out_tok:,} out = {in_tok+out_tok:,}"
          f"   (~${cost:.2f} list)")
    print(f"Tool calls: {n_tools}  ({n_tools/len(items):.1f}/question)")

    by_lvl = {}
    for r in results:
        by_lvl.setdefault(r["Level"], [0, 0])
        by_lvl[r["Level"]][0] += int(r["correct"])
        by_lvl[r["Level"]][1] += 1
    print("\nBy Level:")
    for lvl in sorted(by_lvl):
        c, t = by_lvl[lvl]
        print(f"  L{lvl}: {c:2d}/{t:2d}  ({c/t*100:5.1f}%)")

    by_kind = {}
    for r in results:
        by_kind.setdefault(r["kind"], [0, 0])
        by_kind[r["kind"]][0] += int(r["correct"])
        by_kind[r["kind"]][1] += 1
    print("\nBy modality:")
    for k in sorted(by_kind):
        c, t = by_kind[k]
        print(f"  {k:8s}: {c:2d}/{t:2d}  ({c/t*100:5.1f}%)")

    # Tool usage breakdown
    tool_kinds = {"SEARCH": 0, "BROWSE": 0, "COMPUTE": 0}
    for r in results:
        for tl in r["tool_log"]:
            tool_kinds[tl["kind"]] = tool_kinds.get(tl["kind"], 0) + 1
    print("\nTool calls breakdown:")
    for k, v in tool_kinds.items():
        print(f"  {k:8s}: {v}")
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
