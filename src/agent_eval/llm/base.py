"""Provider protocol and shared result types."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

Message = dict[str, Any]
"""A chat message in Ollama/OpenAI shape: ``{"role": ..., "content": ...}`` plus optional keys."""


class LLMError(RuntimeError):
    """Raised when a provider cannot produce a completion (after retries)."""


@dataclass(frozen=True)
class ToolCall:
    """A function call requested by the model."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Completion:
    """Text (and tool calls) returned by a provider, plus the numbers we report."""

    text: str
    latency_s: float
    tool_calls: tuple[ToolCall, ...] = ()
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMProvider(Protocol):
    """Anything that turns a list of chat messages into a completion."""

    name: str
    model: str

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        """Return one completion. ``tools`` enables function calling, ``json_schema`` JSON output."""
        ...


def complete(
    provider: LLMProvider,
    system: str,
    user: str,
    *,
    json_schema: dict[str, Any] | None = None,
) -> Completion:
    """Single-turn helper: one system prompt, one user message."""
    messages: list[Message] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return provider.chat(messages, json_schema=json_schema)


def strip_reasoning(text: str) -> str:
    """Drop ``<think>...</think>`` blocks that reasoning models may emit."""
    return _THINK.sub("", text).strip()


def split_system(messages: list[Message]) -> tuple[str, list[Message]]:
    """Separate system messages (joined) from the conversation, for APIs that want them apart."""
    system = "\n\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
    rest = [m for m in messages if m.get("role") != "system"]
    return system, rest
