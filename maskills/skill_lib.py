"""Skill library I/O for MASkills.

A skill library is a directory:

    <lib_dir>/
        SKILL.md                # root identity
        <skill_name>/
            SKILL.md            # sub-skill
        …

Operations:
- load_lib(path) → {"root": skill_dict, "skills": {name: skill_dict}}
- save_skill(lib_dir, name, body, description, is_root=False)
- delete_skill(lib_dir, name)
- snapshot_lib(src, dst)         # copy whole directory
- apply_ops(lib_dir, ops)         # apply a list of operator outputs

Each skill_dict has: name, description, body, path.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Dict, List

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


def parse_skill(path: Path) -> Dict:
    text = path.read_text()
    m = FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError(f"no frontmatter in {path}")
    fm, body = m.group(1), m.group(2)
    name_m = re.search(r"^name:\s*(.+)$", fm, re.M)
    desc_m = re.search(r"^description:\s*(.+)$", fm, re.M)
    if not name_m or not desc_m:
        raise ValueError(f"missing name/description in {path}")
    return {
        "name": name_m.group(1).strip(),
        "description": desc_m.group(1).strip(),
        "body": body,
        "path": str(path),
    }


def load_lib(lib_dir: Path) -> Dict:
    root = parse_skill(lib_dir / "SKILL.md")
    skills = {}
    for sub in sorted(lib_dir.iterdir()):
        if sub.is_dir():
            sk = parse_skill(sub / "SKILL.md")
            skills[sub.name] = sk
    return {"root": root, "skills": skills, "path": str(lib_dir)}


def write_skill_file(path: Path, name: str, description: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Single-line description for the frontmatter (collapse newlines).
    desc_safe = " ".join(description.strip().split())
    content = f"---\nname: {name}\ndescription: {desc_safe}\n---\n\n{body.rstrip()}\n"
    path.write_text(content)


def save_sub_skill(lib_dir: Path, slug: str, name: str,
                    description: str, body: str) -> Path:
    """Create or overwrite a sub-skill at <lib_dir>/<slug>/SKILL.md."""
    p = lib_dir / slug / "SKILL.md"
    write_skill_file(p, name, description, body)
    return p


def delete_sub_skill(lib_dir: Path, slug: str) -> bool:
    d = lib_dir / slug
    if d.is_dir():
        shutil.rmtree(d)
        return True
    return False


def snapshot_lib(src: Path, dst: Path) -> None:
    """Recursive copy of a skill lib directory."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


# ── Apply an op (a dict produced by the optimizer) ────────────────────────


def apply_ops(lib_dir: Path, ops: List[Dict]) -> List[Dict]:
    """Apply a list of optimizer ops to a lib in place. Returns a list of
    {op, status, detail} per op, recording success/failure (so we can log)."""
    log = []
    for op in ops:
        kind = op.get("op")
        try:
            if kind == "induct":
                slug = op["slug"]
                save_sub_skill(
                    lib_dir, slug, op["name"], op["description"], op["body"])
                log.append({"op": "induct", "slug": slug, "status": "ok"})
            elif kind == "refine":
                slug = op["slug"]
                p = lib_dir / slug / "SKILL.md"
                if not p.exists() and slug == "_root_":
                    p = lib_dir / "SKILL.md"
                if not p.exists():
                    log.append({"op": "refine", "slug": slug,
                                 "status": "skip", "detail": "skill not found"})
                    continue
                # Overwrite full body — optimizer must return the FULL new body.
                cur = parse_skill(p)
                new_desc = op.get("description", cur["description"])
                new_name = op.get("name", cur["name"])
                new_body = op.get("body", cur["body"])
                write_skill_file(p, new_name, new_desc, new_body)
                log.append({"op": "refine", "slug": slug, "status": "ok"})
            elif kind == "consolidate":
                # Merge multiple skills into a new one; delete originals.
                slug_new = op["slug_new"]
                for src in op.get("slugs_in", []):
                    delete_sub_skill(lib_dir, src)
                save_sub_skill(
                    lib_dir, slug_new, op["name"], op["description"], op["body"])
                log.append({"op": "consolidate", "slug": slug_new,
                            "merged_from": op.get("slugs_in", []),
                            "status": "ok"})
            elif kind == "prune":
                slug = op["slug"]
                ok = delete_sub_skill(lib_dir, slug)
                log.append({"op": "prune", "slug": slug,
                            "status": "ok" if ok else "miss",
                            "detail": op.get("reason", "")})
            else:
                log.append({"op": kind, "status": "unknown_op",
                            "detail": str(op)[:200]})
        except Exception as e:  # noqa: BLE001
            log.append({"op": kind, "status": "error",
                        "detail": f"{type(e).__name__}: {e}",
                        "op_raw": str(op)[:200]})
    return log
