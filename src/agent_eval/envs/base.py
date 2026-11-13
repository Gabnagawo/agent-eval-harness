"""The Environment interface and the registry that maps a ``kind`` to its class.

An environment is a deterministic, in-memory fake of some system (a file tree, an issue
tracker, ...). It owns a slice of state, exposes tools that read and change it, and can dump
that state as plain JSON for the success checker. No network, no clock, no global randomness:
the only entropy allowed is ``self.rng``, seeded from the task.
"""

from __future__ import annotations

import copy
import importlib
import pkgutil
import random
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from agent_eval.models import ToolResult, ToolSpec


class ToolFailure(Exception):  # noqa: N818 - reads better at the raise site than "ToolFailureError"
    """Raise inside a ``tool_<name>`` method to return ``ToolResult(ok=False, error=...)``."""


class Environment(ABC):
    """Base class for mock environments.

    Subclasses set ``kind``, implement :meth:`load`, :meth:`tools` and :meth:`snapshot`, and
    define one ``tool_<tool name>(self, **arguments)`` method per tool. Arguments reach the
    method already validated against the tool's JSON Schema.
    """

    kind: ClassVar[str]

    def __init__(self, state: dict[str, Any] | None = None, seed: int = 0) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.load(copy.deepcopy(state or {}))

    @abstractmethod
    def load(self, state: dict[str, Any]) -> None:
        """Set the initial state from the task's ``envs.<kind>`` mapping (already a deep copy)."""

    @abstractmethod
    def tools(self) -> list[ToolSpec]:
        """Every tool of this environment. Names are globally unique: prefix them with ``kind``."""

    @abstractmethod
    def snapshot(self) -> dict[str, Any]:
        """The current state as JSON-serialisable data. Same calls, same seed: same snapshot."""

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Run one tool. Domain errors come back as ``ok=False``; this never raises for them."""
        handler = getattr(self, f"tool_{name}", None)
        if handler is None:
            return ToolResult(ok=False, error=f"unknown tool '{name}'")
        try:
            return ToolResult(ok=True, result=handler(**arguments))
        except ToolFailure as exc:
            return ToolResult(ok=False, error=str(exc))


REGISTRY: dict[str, type[Environment]] = {}


def register[E: type[Environment]](cls: E) -> E:
    """Class decorator: make the environment available to tasks under ``cls.kind``."""
    kind = getattr(cls, "kind", None)
    if not kind:
        raise TypeError(f"{cls.__name__} must define a 'kind'")
    if kind in REGISTRY and REGISTRY[kind] is not cls:
        raise ValueError(f"environment kind '{kind}' is already registered")
    REGISTRY[kind] = cls
    return cls


def load_builtin() -> None:
    """Import every module in ``agent_eval.envs`` so their ``@register`` decorators run."""
    import agent_eval.envs as package

    for module in pkgutil.iter_modules(package.__path__):
        if module.name != "base":
            importlib.import_module(f"{package.__name__}.{module.name}")


def get_environment(kind: str) -> type[Environment]:
    if kind not in REGISTRY:
        load_builtin()
    try:
        return REGISTRY[kind]
    except KeyError:
        known = ", ".join(sorted(REGISTRY)) or "none"
        raise KeyError(f"unknown environment kind '{kind}' (known: {known})") from None
