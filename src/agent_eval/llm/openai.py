"""OpenAI-compatible provider. The SDK is an optional extra: ``uv sync --extra openai``.

Works with api.openai.com and with any server that speaks the Chat Completions API
(set ``OPENAI_BASE_URL``).
"""

from __future__ import annotations

import json
import time
from typing import Any

from agent_eval.llm.base import Completion, LLMError, Message, ToolCall


def to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Translate the internal (Ollama-shaped) conversation to Chat Completions messages."""
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        content = str(message.get("content", ""))
        if role == "tool":
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": message.get("tool_call_id", ""),
                    "content": content,
                }
            )
        elif role == "assistant" and message.get("tool_calls"):
            converted.append(
                {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [
                        {
                            "id": call.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": call["function"]["name"],
                                "arguments": json.dumps(call["function"].get("arguments") or {}),
                            },
                        }
                        for call in message["tool_calls"]
                    ],
                }
            )
        else:
            converted.append({"role": role, "content": content})
    return converted


def _parse_tool_calls(raw_calls: Any) -> tuple[ToolCall, ...]:
    calls: list[ToolCall] = []
    for index, raw in enumerate(raw_calls or [], start=1):
        text = raw.function.arguments or "{}"
        try:
            arguments = json.loads(text)
        except json.JSONDecodeError:
            arguments = {"_raw": text}
        if not isinstance(arguments, dict):
            arguments = {"_raw": arguments}
        calls.append(
            ToolCall(
                name=raw.function.name,
                arguments=arguments,
                id=getattr(raw, "id", "") or f"call_{index}",
            )
        )
    return tuple(calls)


class OpenAIProvider:
    """Chat Completions via the official SDK, imported lazily."""

    name = "openai"

    def __init__(
        self,
        model: str,
        api_key: str | None,
        *,
        base_url: str | None = None,
        timeout: float = 180.0,
        max_retries: int = 2,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> None:
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set")
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - depends on installed extras
            raise LLMError("Install the extra: uv sync --extra openai") from exc
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = openai.OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries
        )

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": json_schema},
            }
        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=to_openai_messages(messages),
                **kwargs,
            )
        except Exception as exc:  # the SDK raises many error types
            raise LLMError(f"OpenAI call failed: {exc}") from exc
        usage = response.usage
        message = response.choices[0].message
        return Completion(
            text=(message.content or "").strip(),
            latency_s=time.perf_counter() - started,
            tool_calls=_parse_tool_calls(getattr(message, "tool_calls", None)),
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )
