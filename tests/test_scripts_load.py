"""Every entry point must still load after a refactor.

Two sets are covered. The drivers in ``scripts/`` are the reproduction entry
points for the paper and they import each other as well as the package
(``maskills_eval`` reuses ``maskills_train_iter``'s rollout, which in turn
drives ``maskills.envs.gaia.tools``). The GAIA and wiki-search modules inside
the package double as standalone CLIs and are imported lazily by their
environments, so nothing else would catch a broken import until an hour into a
training run.

A script with a CLI is checked with ``--help``, which runs every import and
exits before doing any work. A script without one is imported, which is only
safe because its module-level work lives in ``main()``.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = sorted((REPO_ROOT / "scripts").glob("*.py"))

#: The scripts refuse to start without these; the values are never used because
#: no script reaches a network call during --help or import.
DUMMY_ENV = {"OPENAI_API_KEY": "dummy", "TAVILY_API_KEY": "dummy"}


def _optional_dependency(exc: BaseException) -> str | None:
    """Name the missing extra, when that is all that went wrong."""
    if isinstance(exc, ModuleNotFoundError) and exc.name:
        root = exc.name.split(".")[0]
        if root not in {"maskills"}:
            return root
    return None


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_loads(script: pathlib.Path):
    env = {**os.environ, **DUMMY_ENV}

    if "add_argument" in script.read_text():
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True, text=True, cwd=REPO_ROOT, env=env,
        )
        if "ModuleNotFoundError" in result.stderr:
            missing = result.stderr.strip().rsplit("No module named ", 1)[-1]
            pytest.skip(f"optional dependency not installed: {missing}")
        assert "Traceback" not in result.stderr, result.stderr[-2000:]
        assert result.returncode == 0, result.stderr[-2000:]
        return

    # No CLI to exercise, so import it instead.
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    os.environ.update(DUMMY_ENV)
    spec = importlib.util.spec_from_file_location(f"_check_{script.stem}", script)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except BaseException as exc:  # noqa: BLE001 - report whatever import raised
        missing = _optional_dependency(exc)
        if missing:
            pytest.skip(f"optional dependency not installed: {missing}")
        raise
    finally:
        sys.path.remove(str(REPO_ROOT / "scripts"))


#: Package modules that are only ever imported lazily, by the environment that
#: needs them, and that also run standalone as ``python -m <module>``.
LAZY_MODULES = [
    "maskills.envs.gaia.single_agent",
    "maskills.envs.gaia.tools",
    "maskills.envs.gaia.decentralized",
    "maskills.envs.language.search_wiki",
]


@pytest.mark.parametrize("module", LAZY_MODULES)
def test_lazy_module_loads(module: str):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True, text=True, cwd=REPO_ROOT, env={**os.environ, **DUMMY_ENV},
    )
    if "ModuleNotFoundError" in result.stderr:
        missing = result.stderr.strip().rsplit("No module named ", 1)[-1]
        pytest.skip(f"optional dependency not installed: {missing}")
    assert "Traceback" not in result.stderr, result.stderr[-2000:]
    assert result.returncode == 0, result.stderr[-2000:]


def test_gaia_modules_do_not_exit_on_import_without_keys(monkeypatch):
    """Importing the GAIA modules must not kill the process.

    They used to check for OPENAI_API_KEY / TAVILY_API_KEY at module scope,
    which was fine while they lived in scripts/ and is a landmine now that the
    environment imports them. The check moved into ``require_api_keys()``.
    """
    import importlib

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    for name in LAZY_MODULES:
        importlib.reload(importlib.import_module(name))

    from maskills.envs.gaia._keys import require_api_keys

    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        require_api_keys()
