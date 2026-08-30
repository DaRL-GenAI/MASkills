"""Render GAIA SKILL.md folders as a single system prompt for an agent.

Layout expected (matches ``<library dir>/agent_X/``)::

    agent_X/
        SKILL.md                <- root, used as the agent's role
        <subskill>/SKILL.md     <- per-skill body, concatenated under Tier A
"""

from __future__ import annotations

import re
from pathlib import Path

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


def _parse(path: Path) -> dict:
    text = path.read_text()
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {"name": path.parent.name, "description": "", "body": text}
    fm, body = m.group(1), m.group(2)
    name_m = re.search(r"^name:\s*(.+)$", fm, re.M)
    desc_m = re.search(r"^description:\s*(.+)$", fm, re.M)
    return {
        "name": name_m.group(1).strip() if name_m else path.parent.name,
        "description": desc_m.group(1).strip() if desc_m else "",
        "body": body,
    }


def render_agent_prompt(agent_dir: str) -> str:
    """Return the concatenated root + sub-skill bodies for one agent."""
    root_path = Path(agent_dir)
    root_file = root_path / "SKILL.md"
    if not root_file.exists():
        raise FileNotFoundError(f"No SKILL.md in {root_path}")
    root = _parse(root_file)
    subs = []
    for sub in sorted(root_path.iterdir()):
        if sub.is_dir() and (sub / "SKILL.md").exists():
            subs.append(_parse(sub / "SKILL.md"))
    parts = [root["body"].strip()]
    if subs:
        parts.append("\n\n---\n\n# Tier A skill bodies (always-on for this agent)\n")
        for sk in subs:
            parts.append(f"\n## skill: `{sk['name']}`\n{sk['body'].strip()}\n")
    return "".join(parts)
