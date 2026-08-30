"""Generic tool-dispatcher for language-task agents.

Originally only ``<search>...</search>`` (wiki search) was supported.  This
module now also dispatches:

* ``<grep>PATTERN ||| TEXT</grep>`` — extract lines matching a regex
  pattern from a chunk of text (typically the body of a previous search
  result).  Uses Python's ``re`` engine, line-oriented like Unix grep.
* ``<sympy>EXPR</sympy>`` — evaluate a SymPy expression (or short snippet)
  with the SymPy namespace pre-imported.  Returns the printed value.

The env still emits and parses tool tags by regex.  The first recognised
tag in a response is executed; other tags in the same turn are ignored.
"""

from __future__ import annotations

import contextlib
import io
import re
from typing import List, Optional, Tuple

# ── Tag parsing ────────────────────────────────────────────────────────

_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE)
_TOOL_RE = re.compile(
    r"<(search|grep|sympy)>(.*?)</\1>",
    re.DOTALL | re.IGNORECASE,
)
TOOL_NAMES = ("search", "grep", "sympy")


def parse_search_query(text: str) -> Optional[str]:
    """Return the first <search>...</search> query in text, or None.

    Kept for backward compatibility with callers that only care about
    the wiki-search tag.
    """
    match = _SEARCH_RE.search(text or "")
    if not match:
        return None
    query = match.group(1).strip()
    return query or None


def parse_tool_call(text: str) -> Optional[Tuple[str, str]]:
    """Return ``(tool_name, payload)`` for the first known tag in text.

    Tool names are normalised to lowercase.  Payload is stripped.
    """
    match = _TOOL_RE.search(text or "")
    if not match:
        return None
    name = match.group(1).lower()
    payload = match.group(2).strip()
    if not payload:
        return None
    return name, payload


# ── search_wiki ────────────────────────────────────────────────────────

#: Imported on first use rather than at module import: the MediaWiki client
#: pulls in ``requests``, which only the ``language`` extra installs, and a
#: run that never searches should not need it.
_search_wiki = None


def _load_search_wiki():
    """Return ``search_wiki()``, importing the MediaWiki client on first use."""
    try:
        from .search_wiki import search_wiki
    except ImportError as exc:
        raise ImportError(
            f"The wiki search tool needs the 'language' extra: {exc}. "
            "Install it with: pip install 'maskills[language]'"
        ) from exc
    return search_wiki


def format_results(hits: List[dict], max_hits: int = 5) -> str:
    """Render search hits into a compact, agent-readable block."""
    if not hits:
        return "No results."
    lines = []
    for i, h in enumerate(hits[:max_hits], 1):
        title = h.get("title", "")
        extract = (h.get("extract") or h.get("snippet") or "").strip()
        lines.append(f"[{i}] {title}\n{extract}")
    return "\n\n".join(lines)


def run_search(query: str, limit: int = 5) -> str:
    """Execute a wiki search and return a formatted result block."""
    global _search_wiki
    if _search_wiki is None:
        _search_wiki = _load_search_wiki()
    try:
        hits = _search_wiki(query, limit=limit, intro_only=True, max_chars=800)
    except Exception as e:  # network / API failure
        return f"Search error: {type(e).__name__}: {e}"
    return format_results(hits, max_hits=limit)


# ── grep ───────────────────────────────────────────────────────────────

_GREP_SEP = "|||"


def _split_grep_payload(payload: str, default_text: str) -> Tuple[str, str]:
    """Parse grep payload as ``PATTERN ||| TEXT``.

    If the separator is missing, ``default_text`` (typically the most
    recent search result) is used as the haystack and the whole payload
    is treated as the pattern.
    """
    if _GREP_SEP in payload:
        pattern, _, text = payload.partition(_GREP_SEP)
        return pattern.strip(), text.strip()
    return payload.strip(), default_text


def run_grep(payload: str, fallback_text: str = "", max_lines: int = 20) -> str:
    """Filter ``fallback_text`` by lines matching ``pattern``.

    Returns up to ``max_lines`` matching lines, with line numbers, or a
    short notice if there is no match.
    """
    pattern, text = _split_grep_payload(payload, fallback_text)
    if not pattern:
        return "Grep error: empty pattern."
    if not text:
        return (
            "Grep error: no text to search. "
            "Run a <search>...</search> first, or pass text via "
            "<grep>PATTERN ||| TEXT</grep>."
        )
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Grep error: invalid regex: {e}"

    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        if regex.search(line):
            hits.append(f"{i}: {line.strip()}")
            if len(hits) >= max_lines:
                break
    if not hits:
        return f"No lines match /{pattern}/."
    return "\n".join(hits)


# ── sympy ──────────────────────────────────────────────────────────────

_SYMPY_NS = None


def _build_sympy_namespace():
    """Lazy-import sympy; return a namespace dict for ``eval``/``exec``.

    We disable Python builtins to keep the sandbox tight.  Anything the
    agent needs from the standard library has to come through sympy.
    """
    global _SYMPY_NS
    if _SYMPY_NS is None:
        import sympy  # type: ignore
        from sympy import abc as sympy_abc  # type: ignore
        ns = {name: getattr(sympy, name) for name in dir(sympy) if not name.startswith("_")}
        # Pre-populate single-letter symbols (a..z and Greek) so an agent
        # can write ``solve(x**2 - 4, x)`` without first defining ``x``.
        for name in dir(sympy_abc):
            if name.startswith("_"):
                continue
            ns.setdefault(name, getattr(sympy_abc, name))
        ns["sympy"] = sympy
        _SYMPY_NS = ns
    return dict(_SYMPY_NS)  # fresh copy per call


def run_sympy(code: str, max_chars: int = 1500) -> str:
    """Evaluate a SymPy expression (or short snippet) and return the result.

    Single expressions are ``eval``'d and the value's ``str()`` is returned.
    Multi-line snippets are ``exec``'d with stdout captured.  Errors are
    returned as a string so the caller can recover.
    """
    code = (code or "").strip()
    if not code:
        return "Sympy error: empty input."
    try:
        ns = _build_sympy_namespace()
    except ImportError as e:
        return f"Sympy error: SymPy not installed ({e})."

    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        """Allow ``from sympy[.x] import ...`` and ``import sympy[.x]``."""
        if name == "sympy" or name.startswith("sympy."):
            import importlib
            return importlib.import_module(name)
        raise ImportError(
            f"Only sympy modules can be imported in <sympy>; got '{name}'."
        )

    safe_builtins = {
        "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
        "range": range, "int": int, "float": float, "str": str,
        "list": list, "tuple": tuple, "dict": dict, "set": set,
        "True": True, "False": False, "None": None, "print": print,
        "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
        "sorted": sorted, "reversed": reversed, "any": any, "all": all,
        "round": round, "divmod": divmod, "pow": pow,
        "__import__": _safe_import,
    }
    globs = {"__builtins__": safe_builtins, **ns}

    # Try expression eval first; fall back to exec for assignments / prints.
    try:
        result = eval(code, globs, {})
        out = str(result)
    except SyntaxError:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, globs, {})
        except Exception as e:
            return f"Sympy error: {type(e).__name__}: {e}"
        out = buf.getvalue().strip() or "(no output)"
    except Exception as e:
        return f"Sympy error: {type(e).__name__}: {e}"

    if len(out) > max_chars:
        out = out[:max_chars] + "\n... (truncated)"
    return out


# ── unified dispatcher ─────────────────────────────────────────────────

def run_tool(
    name: str,
    payload: str,
    *,
    search_limit: int = 5,
    last_search_result: str = "",
) -> str:
    """Dispatch ``name`` to its handler and return a string result."""
    if name == "search":
        return run_search(payload, limit=search_limit)
    if name == "grep":
        return run_grep(payload, fallback_text=last_search_result)
    if name == "sympy":
        return run_sympy(payload)
    return f"Unknown tool: {name}"
