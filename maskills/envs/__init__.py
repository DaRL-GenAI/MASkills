"""Environment registry and plugin system.

Environments register themselves with the ``@register_env(...)`` decorator at
import time. The three benchmarks the paper uses — ``language`` (HotpotQA /
MATH / HumanEval), ``locomo`` and ``gaia`` — are imported lazily on the first
:func:`make_env` or :func:`list_envs` call, so a task whose optional
dependencies are missing degrades to a useful error instead of vanishing from
the registry.
"""

from __future__ import annotations

from ..core.base import BaseEnvironment

_ENV_REGISTRY: dict[str, type[BaseEnvironment]] = {}

#: Built-in environments -> the optional extra providing their dependencies.
_BUILTINS = {"language": "language", "locomo": "language", "gaia": "gaia"}

#: name -> why it could not be imported, for a useful make_env() error.
_UNAVAILABLE: dict[str, str] = {}

_BUILTINS_LOADED = False


def register_env(name: str):
    """Decorator registering an environment class under ``name``."""

    def decorator(cls):
        _ENV_REGISTRY[name] = cls
        return cls

    return decorator


def _ensure_builtins_loaded():
    """Import the built-in environments so their decorators run."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True

    import importlib
    import logging

    logger = logging.getLogger(__name__)
    for name in _BUILTINS:
        try:
            importlib.import_module(f".{name}.env", __package__)
        except ImportError as exc:
            logger.debug("Environment %r unavailable: %s", name, exc)
            _UNAVAILABLE[name] = str(exc)


def make_env(name: str, config) -> BaseEnvironment:
    """Create a registered environment by name."""
    _ensure_builtins_loaded()

    if name not in _ENV_REGISTRY:
        if name in _UNAVAILABLE:
            extra = _BUILTINS.get(name)
            hint = f" Install its dependencies with: pip install 'maskills[{extra}]'" if extra else ""
            raise ValueError(
                f"Environment {name!r} is built in but could not be imported: "
                f"{_UNAVAILABLE[name]}.{hint}"
            )
        raise ValueError(f"Unknown environment: {name}. Available: {list(_ENV_REGISTRY)}")
    return _ENV_REGISTRY[name](config)


def list_envs() -> list[str]:
    """List the environments currently registered."""
    _ensure_builtins_loaded()
    return list(_ENV_REGISTRY.keys())
