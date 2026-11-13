"""Deterministic provider for tests and offline demos."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent_eval.llm.base import Completion, LLMError, Message

Responder = Callable[[list[Message], list[dict[str, Any]] | None], Completion | str]


class FakeProvider:
    """Returns canned responses in order (or from a callable) and records every request."""

    name = "fake"

    def __init__(
        self,
        responses: list[Completion | str] | None = None,
        *,
        responder: Responder | None = None,
        model: str = "fake-model",
        fail_times: int = 0,
    ) -> None:
        self.model = model
        self._responses = list(responses or [])
        self._responder = responder
        self._fail_times = fail_times
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        self.calls.append(
            {"messages": [dict(m) for m in messages], "tools": tools, "json_schema": json_schema}
        )
        if self._fail_times > 0:
            self._fail_times -= 1
            raise LLMError("fake failure")
        if self._responder is not None:
            result = self._responder(messages, tools)
        elif self._responses:
            result = self._responses.pop(0)
        else:
            raise LLMError("FakeProvider has no more responses")
        if isinstance(result, Completion):
            return result
        return Completion(text=result, latency_s=0.0)
