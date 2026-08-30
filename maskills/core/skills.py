"""Skill library in Anthropic Agent Skills / SKILL.md format.

A skill is a folder containing a ``SKILL.md`` file with YAML frontmatter
(``name``, ``description``, plus optional fields) followed by a markdown
body of instructions.  See :func:`load_skills_dir` for the layout.

This module handles both sides of the MASkills skill pipeline:

* **Invocation** — human-authored skills under a ``skills_dir`` are loaded
  at env init and rendered into every agent's system prompt as a fixed,
  non-trainable "Skill Library" block.
* **Auto-generation** — each agent's trainable ``skills`` body is dumped
  after every checkpoint save to ``<run>/skills_autogen/iter_<i>/<agent>/SKILL.md``
  so learned skills are browsable and reusable in the same format.

Both paths share the same ``Skill`` dataclass and the same ``SKILL.md``
serialization, so a policy learned in one run can be dropped into another
run's ``skills_dir`` unchanged.

**MASkills extension.**  A single agent owns a :class:`SkillLibrary` — a set
of *discrete* skills ``K_i = {k^(1), k^(2), ...}`` rather than one opaque
markdown blob.  Each :class:`Skill` carries a stable ``skill_id`` plus
running optimization bookkeeping (``utility``, ``invocations``,
``provenance``) so the four skill-evolution operators (refinement,
induction, consolidation, pruning) can act on individual skills.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<front>.*?)\n---\s*\n(?P<body>.*)\Z",
    re.DOTALL,
)

# Frontmatter keys that map onto first-class :class:`Skill` fields rather
# than the opaque ``metadata`` dict.
_RESERVED_KEYS = ("name", "description", "skill_id", "provenance", "utility", "invocations")

# Valid values for ``Skill.provenance`` — how the skill entered the library.
PROVENANCE_VALUES = ("human", "induced", "consolidated", "refined")


def _slugify(text: str) -> str:
    """Lowercase kebab-case identifier derived from arbitrary text."""
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s or "skill"


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


@dataclass
class Skill:
    """A single SKILL.md entry — one discrete skill ``k`` in a library.

    The first three fields plus ``metadata`` round-trip through ``SKILL.md``
    frontmatter.  ``skill_id`` / ``provenance`` / ``utility`` / ``invocations``
    are MASkills bookkeeping: ``skill_id`` is a *stable* identifier (the
    skill's ``name`` may be rewritten by refinement, the id never changes),
    while ``utility`` and ``invocations`` accumulate skill-level credit and
    drive the pruning operator.
    """

    name: str
    description: str
    body: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)
    source_path: Optional[Path] = None
    # ── MASkills bookkeeping ─────────────────────────────────────────────
    skill_id: str = ""
    provenance: str = "human"
    utility: float = 0.0
    invocations: int = 0

    def __post_init__(self):
        if not self.skill_id:
            self.skill_id = _slugify(self.name)
        if self.provenance not in PROVENANCE_VALUES:
            self.provenance = "human"

    def format(self) -> str:
        """Serialize back to SKILL.md text.

        ``skill_id`` is always written so the stable identity round-trips;
        ``provenance`` / ``utility`` / ``invocations`` are written only when
        non-default, keeping hand-authored skill files uncluttered.
        """
        lines = ["---", f"name: {self.name}", f"description: {self.description}"]
        lines.append(f"skill_id: {self.skill_id}")
        if self.provenance != "human":
            lines.append(f"provenance: {self.provenance}")
        for k, v in self.metadata.items():
            if k in _RESERVED_KEYS:
                continue
            lines.append(f"{k}: {v}")
        if self.utility != 0.0:
            lines.append(f"utility: {self.utility:g}")
        if self.invocations != 0:
            lines.append(f"invocations: {self.invocations}")
        lines.append("---")
        lines.append("")
        body = self.body.strip()
        if body:
            lines.append(body)
            lines.append("")
        return "\n".join(lines)

    def metadata_line(self) -> str:
        """One-line ``name — description`` summary for prompt injection."""
        return f"- `{self.name}` (id: {self.skill_id}) — {self.description}"

    def copy(self) -> "Skill":
        """Deep-ish copy (metadata dict copied; ``body`` is an immutable str)."""
        return Skill(
            name=self.name,
            description=self.description,
            body=self.body,
            metadata=dict(self.metadata),
            source_path=self.source_path,
            skill_id=self.skill_id,
            provenance=self.provenance,
            utility=self.utility,
            invocations=self.invocations,
        )

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "body": self.body,
            "metadata": dict(self.metadata),
            "skill_id": self.skill_id,
            "provenance": self.provenance,
            "utility": self.utility,
            "invocations": self.invocations,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Skill":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            body=data.get("body", ""),
            metadata=dict(data.get("metadata", {})),
            skill_id=data.get("skill_id", ""),
            provenance=data.get("provenance", "human"),
            utility=_to_float(data.get("utility", 0.0)),
            invocations=_to_int(data.get("invocations", 0)),
        )


def parse_skill_md(text: str, source_path: Optional[Path] = None) -> Skill:
    """Parse a SKILL.md string into a :class:`Skill`.

    Frontmatter is a minimal YAML-ish ``key: value`` block (one pair per
    line).  We intentionally avoid a YAML dependency — the spec only
    requires ``name`` and ``description``, and extended keys are treated
    as opaque strings.  ``skill_id`` / ``provenance`` / ``utility`` /
    ``invocations`` are recognized and lifted onto first-class fields.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError(
            "SKILL.md missing YAML frontmatter"
            + (f" (at {source_path})" if source_path else "")
        )
    front_text = match.group("front")
    body = match.group("body").strip()

    metadata: Dict[str, str] = {}
    for raw_line in front_text.splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.strip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        metadata[key.strip()] = value.strip()

    name = metadata.pop("name", "").strip()
    description = metadata.pop("description", "").strip()
    skill_id = metadata.pop("skill_id", "").strip()
    provenance = metadata.pop("provenance", "human").strip() or "human"
    utility = _to_float(metadata.pop("utility", "0"))
    invocations = _to_int(metadata.pop("invocations", "0"))
    if not name:
        raise ValueError(
            "SKILL.md frontmatter missing required 'name'"
            + (f" (at {source_path})" if source_path else "")
        )
    if not description:
        raise ValueError(
            "SKILL.md frontmatter missing required 'description'"
            + (f" (at {source_path})" if source_path else "")
        )
    return Skill(
        name=name,
        description=description,
        body=body,
        metadata=metadata,
        source_path=source_path,
        skill_id=skill_id,
        provenance=provenance,
        utility=utility,
        invocations=invocations,
    )


def load_skills_dir(skills_dir: str | Path) -> List[Skill]:
    """Discover and parse every ``SKILL.md`` under ``skills_dir``.

    The directory layout is::

        <lib_dir>/
            my-skill/
                SKILL.md
                ...
            another-skill/
                SKILL.md

    A top-level ``SKILL.md`` directly under ``skills_dir`` is also loaded.
    Skills are returned sorted by ``name`` for deterministic prompt order.
    """
    root = Path(skills_dir).expanduser()
    if not root.exists():
        return []
    if root.is_file() and root.name == "SKILL.md":
        return [parse_skill_md(root.read_text(), source_path=root)]
    if not root.is_dir():
        return []

    skills: List[Skill] = []
    for path in sorted(root.rglob("SKILL.md")):
        try:
            skill = parse_skill_md(path.read_text(), source_path=path)
        except (OSError, ValueError):
            continue
        skills.append(skill)
    skills.sort(key=lambda s: s.name)
    return skills


def render_skill_library(
    skills: List[Skill],
    heading: str = "Skill Library",
) -> str:
    """Render a list of skills as a markdown block for prompt injection.

    The block is self-describing (header + list of skill names with
    descriptions) and then inlines each skill's body under an ``### <name>``
    subheading.  Agents reference skills by name in their responses.
    """
    if not skills:
        return ""
    lines = [f"## {heading}"]
    lines.append(
        "The following skills are available to you. Apply them whenever "
        "relevant to the task; reference a skill by its name."
    )
    lines.append("")
    lines.append("**Available skills:**")
    for skill in skills:
        lines.append(f"- `{skill.name}` — {skill.description}")
    lines.append("")
    for skill in skills:
        lines.append(f"### {skill.name}")
        lines.append(f"_{skill.description}_")
        lines.append("")
        if skill.body.strip():
            lines.append(skill.body.strip())
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


@dataclass
class SkillLibrary:
    """An agent's evolving set of discrete skills ``K_i = {k^(1), k^(2), ...}``.

    This is the learnable component of an agent's policy under MASkills.
    Skills are keyed by their stable ``skill_id``; ``add`` rejects duplicate
    ids (auto-suffixing instead) so the four evolution operators can mutate
    the library without id collisions.
    """

    skills: List[Skill] = field(default_factory=list)

    # ── container protocol ───────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.skills)

    def __iter__(self):
        return iter(self.skills)

    def __bool__(self) -> bool:
        return bool(self.skills)

    def ids(self) -> List[str]:
        return [s.skill_id for s in self.skills]

    def get(self, skill_id: str) -> Optional[Skill]:
        for s in self.skills:
            if s.skill_id == skill_id:
                return s
        return None

    # ── mutation ─────────────────────────────────────────────────────────
    def _unique_id(self, skill_id: str) -> str:
        """Return ``skill_id`` if free, else append a numeric suffix."""
        existing = set(self.ids())
        if skill_id not in existing:
            return skill_id
        n = 2
        while f"{skill_id}-{n}" in existing:
            n += 1
        return f"{skill_id}-{n}"

    def add(self, skill: Skill) -> Skill:
        """Append ``skill``, assigning a collision-free ``skill_id``."""
        skill.skill_id = self._unique_id(skill.skill_id or _slugify(skill.name))
        self.skills.append(skill)
        return skill

    def remove(self, skill_id: str) -> bool:
        """Drop the skill with ``skill_id``; return True if one was removed."""
        before = len(self.skills)
        self.skills = [s for s in self.skills if s.skill_id != skill_id]
        return len(self.skills) < before

    def replace(self, skill: Skill) -> bool:
        """Replace an existing skill in place by ``skill_id``.

        Used by the refinement operator: the id is preserved, body/name/
        description are swapped.  Returns False if the id is not present.
        """
        for i, s in enumerate(self.skills):
            if s.skill_id == skill.skill_id:
                self.skills[i] = skill
                return True
        return False

    def merge(self, skill_ids: List[str], consolidated: Skill) -> Optional[Skill]:
        """Consolidation primitive: drop ``skill_ids``, insert ``consolidated``.

        The LLM ``Merge`` logic that *builds* the consolidated skill lives in
        the consolidation operator (P3); this only performs the library
        bookkeeping.  Returns the inserted skill, or None if no source id
        matched (nothing to consolidate).
        """
        present = [sid for sid in skill_ids if self.get(sid) is not None]
        if not present:
            return None
        for sid in present:
            self.remove(sid)
        consolidated.provenance = "consolidated"
        return self.add(consolidated)

    # ── rendering ────────────────────────────────────────────────────────
    def render_metadata(self, heading: str = "Skill Library") -> str:
        """Render only the skill *index* (name + description per skill).

        This is the lightweight block injected into every agent prompt:
        the agent sees what skills exist and picks one to invoke, and only
        then is the full body disclosed (see :meth:`render_full`).
        """
        if not self.skills:
            return ""
        lines = [f"## {heading}"]
        lines.append(
            "The following skills are available. Select a skill by its id "
            "when it is relevant to the task; its full instructions are "
            "disclosed on invocation."
        )
        lines.append("")
        for skill in self.skills:
            lines.append(skill.metadata_line())
        return "\n".join(lines).rstrip() + "\n"

    def render_full(self, heading: str = "Skill Library") -> str:
        """Render the full library (every skill body inlined)."""
        return render_skill_library(list(self.skills), heading=heading)

    def get_body(self, skill_id: str) -> str:
        """Progressive disclosure: full ``SKILL.md`` body for one skill."""
        skill = self.get(skill_id)
        return skill.body if skill else ""

    def combined_body(self) -> str:
        """Concatenate every skill body into one markdown blob.

        Bridge for legacy code paths (and ``AgentPolicy.skills``) that still
        expect skills as a single string.  A single skill renders as its raw
        body (identical to the pre-MASkills single-blob representation); two
        or more are separated by ``### <name>`` subheadings.
        """
        bodies = [(s.name, s.body.strip()) for s in self.skills if s.body.strip()]
        if not bodies:
            return ""
        if len(bodies) == 1:
            return bodies[0][1]
        return "\n\n".join(f"### {name}\n{body}" for name, body in bodies)

    # ── serialization ────────────────────────────────────────────────────
    def to_dict(self) -> Dict:
        return {"skills": [s.to_dict() for s in self.skills]}

    @classmethod
    def from_dict(cls, data: Dict) -> "SkillLibrary":
        return cls(skills=[Skill.from_dict(d) for d in data.get("skills", [])])

    def to_skill_md_dir(self, root: str | Path) -> Path:
        """Write each skill to ``<root>/<skill_id>/SKILL.md``."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        for skill in self.skills:
            skill_dir = root / skill.skill_id
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(skill.format())
        return root

    @classmethod
    def from_skill_md_dir(cls, root: str | Path) -> "SkillLibrary":
        """Load a library from a directory of ``<skill>/SKILL.md`` folders."""
        return cls(skills=load_skills_dir(root))

    # ── legacy bridge ────────────────────────────────────────────────────
    @classmethod
    def from_legacy_body(
        cls,
        body: str,
        name: str = "agent-skills",
        description: str = "Agent skill set.",
    ) -> "SkillLibrary":
        """Wrap a single opaque skills blob as a one-skill library.

        Lets pre-MASkills checkpoints (which stored ``skills`` as one string)
        load into the discrete-library representation unchanged.
        """
        if not body.strip():
            return cls(skills=[])
        return cls(skills=[Skill(name=name, description=description, body=body.strip())])

    def copy(self) -> "SkillLibrary":
        return SkillLibrary(skills=[s.copy() for s in self.skills])


def policy_skills_to_skill_md(
    skills_body: str,
    name: str,
    description: str,
) -> str:
    """Wrap a trainable skills body (raw markdown) in SKILL.md frontmatter."""
    return Skill(name=name, description=description, body=skills_body).format()


# ── Skill-invocation trace (ξ_i) ─────────────────────────────────────────
#
# MASkills §4.1 records, per agent, which skill was invoked at each step
# (the trace ξ_i).  Under the "full-injection" execution mode the whole
# library is always in the prompt, so there is no explicit tool call to
# observe; instead we *approximate* ξ_i by detecting which skills the agent
# referenced (by name or id) in its output.  This is a deliberately
# lightweight proxy — it cannot prove a skill was *not* used — but it gives
# the credit assigner (P2) a per-skill attribution signal to work with.


def detect_invoked_skills(text: str, library: "SkillLibrary") -> List[str]:
    """Return the ``skill_id``s of skills referenced in ``text``.

    A skill counts as referenced if its ``name`` or ``skill_id`` appears in
    ``text`` as a whole token (case-insensitive, not inside a longer word).
    """
    if not text or not library:
        return []
    lowered = text.lower()
    hits: List[str] = []
    for skill in library:
        for token in (skill.name, skill.skill_id):
            tok = (token or "").strip().lower()
            if not tok:
                continue
            if re.search(r"(?<![\w-])" + re.escape(tok) + r"(?![\w-])", lowered):
                hits.append(skill.skill_id)
                break
    return hits


def build_skill_trace(
    steps: List[Dict],
    policies: Dict,
    attribute_all_when_unreferenced: bool = True,
) -> Dict[str, List[Dict]]:
    """Build the per-agent skill-invocation trace ξ for a trajectory.

    Args:
        steps: trajectory ``steps`` (each a dict with ``agent``/``agent_id``
            and ``output``/``action``).
        policies: ``agent_name -> AgentPolicy`` (duck-typed: needs a
            ``skill_library`` attribute).
        attribute_all_when_unreferenced: in the full-injection execution mode
            the whole library is always present in the agent's prompt, so an
            agent that names no skill has still been *exposed* to all of them.
            When True (default), a step with no explicitly referenced skill is
            attributed to **every** skill in the agent's library — every
            injected skill is a candidate contributor.  When False the step
            keeps an empty list (the strict ∅ case of ``c_t^i ∈ K_i ∪ {∅}``),
            which is only meaningful under explicit tool-call skill selection.

    Returns:
        ``{agent_name: [{"step": idx, "skills": [skill_id, ...]}, ...]}``.
    """
    trace: Dict[str, List[Dict]] = {}
    for idx, step in enumerate(steps):
        agent = step.get("agent") or step.get("agent_id")
        if not agent:
            continue
        policy = policies.get(agent)
        library = getattr(policy, "skill_library", None)
        if library is None:
            continue
        text = step.get("output") or step.get("action") or ""
        invoked = detect_invoked_skills(text, library)
        if not invoked and attribute_all_when_unreferenced:
            invoked = library.ids()
        trace.setdefault(agent, []).append({"step": idx, "skills": invoked})
    return trace
