"""Environment-driven evaluation, shared by the three eval scripts.

Whatever the benchmark, evaluating a skill library is the same job: build the
environment for a topology, put a library into each agent's policy, roll out
the test pool in parallel, and summarize. This module owns that job; each
``scripts/eval_*.py`` supplies only the config its benchmark needs.

Rollouts are pinned to temperature 0 so a re-run of the same library reproduces
the same numbers, and results append to a JSONL keyed by task id, so an
interrupted sweep resumes instead of paying for the finished tasks again.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import maskills.llm.client as _llm_client
from maskills.core.policy import AgentPolicy, generate_default_agent_prompt
from maskills.core.skills import SkillLibrary

#: The agent topologies every environment implements.
TOPOLOGIES = ("decentralized", "centralized", "hybrid")

#: Passed as ``--skills`` to mean "no library at all" — the floor every
#: benchmark's table reports alongside the trained libraries.
EMPTY = "empty"


# ── deterministic rollouts ─────────────────────────────────────────────


def _chat_messages_temp0(self, messages, max_tokens=None):
    """``LLMClient.chat_messages_with_usage`` pinned to temperature 0.

    A content filter that returns no usable content is reported as an empty
    answer rather than an exception: the task then scores 0, which is the
    honest outcome, instead of aborting a sweep of hundreds of tasks.
    """
    max_tokens = max_tokens or self.config.max_tokens
    params = {"model": self.model, "messages": messages, "temperature": 0.0}
    if any(k in self.model.lower() for k in ("o1", "o3", "gpt-5")):
        params["max_completion_tokens"] = max_tokens
        params.pop("temperature", None)
    else:
        params["max_tokens"] = max_tokens

    try:
        response = _llm_client._create_with_retry(
            self._client.chat.completions.create, **params)
    except TypeError as exc:
        if "no usable content" in str(exc) or "content_filter" in str(exc):
            return "", {"input": 0, "output": 0}
        raise

    text = (response.choices[0].message.content or "").strip()
    if getattr(response, "usage", None):
        tokens = {"input": response.usage.prompt_tokens,
                  "output": response.usage.completion_tokens}
    else:
        joined = " ".join(m.get("content", "") for m in messages)
        tokens = {"input": len(joined.split()) * 2, "output": len(text.split()) * 2}
    return text, tokens


def use_deterministic_rollouts() -> None:
    """Pin every LLM call in this process to temperature 0."""
    _llm_client.LLMClient.chat_messages_with_usage = _chat_messages_temp0


# ── policies ───────────────────────────────────────────────────────────


def load_policies(skills: str, num_agents: int = 2,
                  task_type: str | None = None,
                  fill_default_role: bool = True) -> dict[str, AgentPolicy]:
    """Build one policy per agent from ``skills``.

    ``skills`` is either :data:`EMPTY`, a training checkpoint
    (``agent_N/role.md`` beside ``skill_library.json`` or a legacy
    ``skills.md``), or a directory of ``SKILL.md`` folders per agent.

    ``fill_default_role`` decides what an agent gets when the library carries
    no ``role.md``, and the two environments want opposite things. The language
    task does not fill a role in for a policy that already exists, so it needs
    the canonical protocol prompt written in here. LOCOMO substitutes its own
    semantic roles -- retriever and reasoner, from its ``prompts/`` directory --
    precisely when the role is blank, so writing one in would suppress them.
    """
    policies: dict[str, AgentPolicy] = {}
    root = None if skills in ("", EMPTY) else Path(skills)
    if root is not None and not root.is_dir():
        raise SystemExit(
            f"No skill library at {root}. Pass a directory holding agent_1/ and "
            f"agent_2/, or --skills {EMPTY} for the no-skills floor."
        )

    for idx in range(num_agents):
        agent = f"agent_{idx + 1}"
        role = generate_default_agent_prompt(idx, num_agents=num_agents,
                                             task_type=task_type) \
            if fill_default_role else ""
        library = SkillLibrary(skills=[])

        agent_dir = None if root is None else root / agent
        if agent_dir is not None and agent_dir.is_dir():
            role_file = agent_dir / "role.md"
            if role_file.exists():
                role = role_file.read_text()

            library_json = agent_dir / "skill_library.json"
            legacy = agent_dir / "skills.md"
            if library_json.exists():
                library = SkillLibrary.from_dict(json.loads(library_json.read_text()))
            elif legacy.exists():
                library = SkillLibrary.from_legacy_body(legacy.read_text())
            else:
                library = SkillLibrary.from_skill_md_dir(agent_dir)
            if not library.skills:
                print(f"  WARNING: no skills found under {agent_dir}", flush=True)
        elif root is not None:
            raise SystemExit(f"No {agent}/ directory under {root}.")

        policies[agent] = AgentPolicy(role=role, skill_library=library)
    return policies


def describe_policies(policies: dict[str, AgentPolicy]) -> str:
    counts = ", ".join(f"{a}={len(p.skill_library.skills)}"
                       for a, p in sorted(policies.items()))
    return counts if any(p.skill_library.skills for p in policies.values()) \
        else f"{counts} (empty libraries)"


# ── the rollout sweep ──────────────────────────────────────────────────


def _task_key(task: dict):
    return task.get("task_id") or task.get("question_id") or task.get("question")


def _row_for(task: dict, trajectory) -> dict:
    metadata = trajectory.metadata or {}
    qa = metadata.get("qa_metrics", {}) or {}
    return {
        "task_id": _task_key(task),
        "category": task.get("category"),
        "question": task.get("question"),
        "gold": str(task.get("ground_truth", "") or task.get("Final answer", "")),
        "pred": metadata.get("final_answer", ""),
        "reward": trajectory.reward,
        "f1": qa.get("f1"),
        "bleu": qa.get("bleu"),
        "em": qa.get("em"),
        "architecture": metadata.get("architecture"),
        "num_retrieve_calls": metadata.get("num_retrieve_calls", 0),
        "in_tok": sum(s.get("tokens", {}).get("input", 0) for s in trajectory.steps),
        "out_tok": sum(s.get("tokens", {}).get("output", 0) for s in trajectory.steps),
    }


def summarize(rows: list) -> dict:
    """Mean metrics over the rows that scored, with errors counted separately."""
    scored = [r for r in rows if "error" not in r]

    def mean(key):
        values = [r[key] for r in scored if r.get(key) is not None]
        return sum(values) / len(values) if values else 0.0

    return {
        "n": len(scored),
        "n_errors": len(rows) - len(scored),
        "reward": mean("reward"),
        "f1": mean("f1"),
        "bleu": mean("bleu"),
        "em": mean("em"),
        "in_tok": sum(r.get("in_tok", 0) for r in rows),
        "out_tok": sum(r.get("out_tok", 0) for r in rows),
    }


def run_eval(env, policies: dict, tasks: list, *, out_dir: Path, tag: str,
             workers: int = 24, max_retries: int = 4,
             report_only: bool = False, extra: dict | None = None) -> dict:
    """Roll ``tasks`` out through ``env`` and write rows + summary under ``tag``.

    Existing rows in ``<tag>_rows.jsonl`` are skipped, so re-running after an
    interruption only pays for what is missing.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"{tag}_rows.jsonl"
    summary_path = out_dir / f"{tag}_summary.json"

    done_ids = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("task_id") is not None:
                done_ids.add(row["task_id"])

    remaining = [t for t in tasks if _task_key(t) not in done_ids]
    print(f"  resuming: done={len(done_ids)} remaining={len(remaining)}")

    lock = threading.Lock()

    def append(row):
        with lock, rows_path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def run_one(task):
        last_exc = None
        for attempt in range(max_retries + 1):
            try:
                return _row_for(task, env.collect_trajectory(policies, task))
            except Exception as exc:  # noqa: BLE001 - retried, then recorded
                last_exc = exc
                time.sleep(min(60.0, 3.0 * (2 ** attempt)))
        return {"task_id": _task_key(task), "category": task.get("category"),
                "error": repr(last_exc)}

    if remaining and not report_only:
        progress = {"done": 0, "ok": 0, "fail": 0}
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for future in as_completed({ex.submit(run_one, t): t for t in remaining}):
                row = future.result()
                append(row)
                progress["done"] += 1
                progress["fail" if "error" in row else "ok"] += 1
                if progress["done"] % 25 == 0 or progress["done"] == len(remaining):
                    print(f"  ... {progress['done']}/{len(remaining)}  "
                          f"ok={progress['ok']} fail={progress['fail']}  "
                          f"({time.time() - t0:.0f}s)", flush=True)

    rows = []
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    by_category = defaultdict(list)
    for row in rows:
        if row.get("category") is not None:
            by_category[int(row["category"])].append(row)

    summary = {
        **(extra or {}),
        "overall": summarize(rows),
        "per_category": {c: summarize(by_category[c]) for c in sorted(by_category)},
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    overall = summary["overall"]
    print("\n=== RESULTS ===")
    print(f"{tag}")
    print(f"  n={overall['n']}  reward={overall['reward']:.4f}  "
          f"f1={overall['f1']:.4f}  bleu={overall['bleu']:.4f}  em={overall['em']:.4f}")
    if summary["per_category"]:
        print("  per category:")
        for category, metrics in summary["per_category"].items():
            print(f"    cat{category}: n={metrics['n']}  "
                  f"reward={metrics['reward']:.4f}  f1={metrics['f1']:.4f}")
    print(f"  rows   : {rows_path}")
    print(f"  summary: {summary_path}")
    return summary
