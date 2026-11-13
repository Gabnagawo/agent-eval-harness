"""Provider protocol and shared result types."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

Message = dict[str, Any]
"""A chat message in Ollama shape. Assistant turns may carry ``tool_calls``
(``[{"id", "function": {"name", "arguments": {...}}}]``); tool results have role ``tool`` with
``tool_name`` and ``tool_call_id``. Hosted providers translate this shape to their own."""


class LLMError(RuntimeError):
    """Raised when a provider cannot produce a completion (after retries)."""


@dataclass(frozen=True)
class ToolCall:
    """A function call requested by the model."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = ""


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


def assistant_message(completion: Completion) -> Message:
    """The assistant turn to append to the conversation after ``completion``."""
    message: Message = {"role": "assistant", "content": completion.text}
    if completion.tool_calls:
        message["tool_calls"] = [
            {"id": call.id, "function": {"name": call.name, "arguments": call.arguments}}
            for call in completion.tool_calls
        ]
    return message


def tool_message(call: ToolCall, output: str) -> Message:
    """The tool-result turn that answers ``call``."""
    return {"role": "tool", "content": output, "tool_name": call.name, "tool_call_id": call.id}


def strip_reasoning(text: str) -> str:
    """Drop ``<think>...</think>`` blocks that reasoning models may emit."""
    return _THINK.sub("", text).strip()


def split_system(messages: list[Message]) -> tuple[str, list[Message]]:
    """Separate system messages (joined) from the conversation, for APIs that want them apart."""
    system = "\n\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
    rest = [m for m in messages if m.get("role") != "system"]
    return system, rest
