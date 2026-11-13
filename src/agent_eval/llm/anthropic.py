"""Anthropic provider. The SDK is an optional extra: ``uv sync --extra anthropic``."""

from __future__ import annotations

import json
import time
from typing import Any

from agent_eval.llm.base import Completion, LLMError, Message, ToolCall, split_system


def to_anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI/Ollama ``tools`` entries as Messages API tool definitions."""
    return [
        {
            "name": tool["function"]["name"],
            "description": tool["function"].get("description", ""),
            "input_schema": tool["function"].get("parameters") or {"type": "object"},
        }
        for tool in tools
    ]


def to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Translate the internal conversation (without system turns) to Messages API turns.

    Tool calls become ``tool_use`` blocks; the tool results that follow are grouped into
    one user turn of ``tool_result`` blocks, as the API requires.
    """
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        content = str(message.get("content", ""))
        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id", ""),
                "content": content,
            }
            previous = converted[-1] if converted else None
            if (
                previous
                and previous["role"] == "user"
                and isinstance(previous["content"], list)
                and previous["content"][-1]["type"] == "tool_result"
            ):
                previous["content"].append(block)
            else:
                converted.append({"role": "user", "content": [block]})
        elif role == "assistant" and message.get("tool_calls"):
            blocks: list[dict[str, Any]] = [{"type": "text", "text": content}] if content else []
            blocks += [
                {
                    "type": "tool_use",
                    "id": call.get("id", ""),
                    "name": call["function"]["name"],
                    "input": call["function"].get("arguments") or {},
                }
                for call in message["tool_calls"]
            ]
            converted.append({"role": "assistant", "content": blocks})
        else:
            converted.append({"role": role, "content": content})
    return converted


class AnthropicProvider:
    """Claude via the official SDK, imported lazily so the base install stays small.

    Sampling parameters are left at the API defaults.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str,
        api_key: str | None,
        *,
        timeout: float = 180.0,
        max_retries: int = 2,
        max_tokens: int = 1024,
    ) -> None:
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on installed extras
            raise LLMError("Install the extra: uv sync --extra anthropic") from exc
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(
            api_key=api_key, timeout=timeout, max_retries=max_retries
        )

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        system, rest = split_system(messages)
        if json_schema is not None:
            system += f"\n\nReturn only JSON matching this schema:\n{json.dumps(json_schema)}"
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = to_anthropic_tools(tools)
        started = time.perf_counter()
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system.strip(),
                messages=to_anthropic_messages(rest),
                **kwargs,
            )
        except Exception as exc:  # the SDK raises many error types
            raise LLMError(f"Anthropic call failed: {exc}") from exc
        text = "".join(
            getattr(block, "text", "")
            for block in message.content
            if getattr(block, "type", "text") == "text"
        )
        calls = tuple(
            ToolCall(name=block.name, arguments=dict(block.input or {}), id=block.id)
            for block in message.content
            if getattr(block, "type", "") == "tool_use"
        )
        return Completion(
            text=text.strip(),
            latency_s=time.perf_counter() - started,
            tool_calls=calls,
            prompt_tokens=message.usage.input_tokens,
            completion_tokens=message.usage.output_tokens,
        )
