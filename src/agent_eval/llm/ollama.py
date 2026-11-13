"""Ollama chat provider (default). Talks to ``/api/chat`` over HTTP with httpx.

Two server faults seen on a long-lived llama-server runner are handled here, because a
plain retry on the same runner repeats them:

* a ``format`` (JSON schema) request answered with HTTP 200 and empty content;
* a request aborted with HTTP 500 "prediction aborted, token repeat limit reached".

Recovery ladder for one call: the request again, with the runner reloaded after the first
fault; then, for schema requests, JSON mode with the schema in the prompt; then an
unconstrained request with a repeat penalty. ``unload_every`` unloads the model after every
N answered calls, so a long run works on freshly loaded runners.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx

from agent_eval.llm.base import Completion, LLMError, Message, ToolCall, strip_reasoning
from agent_eval.llm.retry import call_with_retries

# One request at a time per process: the Ollama server may be shared with other workloads.
_OLLAMA_LOCK = threading.Lock()


# Text of the HTTP 500 body when llama-server stops a prediction that repeats one token.
REPEAT_ABORT = "token repeat limit reached"
FALLBACK_REPEAT_PENALTY = 1.15
_SCHEMA_INSTRUCTION = (
    "\n\nAnswer with one JSON object only, no prose and no code fence. "
    "It must match this JSON schema:\n"
)


class RetryableHTTPError(RuntimeError):
    """A 429/5xx response worth retrying."""


class RunnerFaultError(RuntimeError):
    """The runner aborted the prediction; the same request on the same runner fails again."""


def check_response(response: httpx.Response, service: str) -> None:
    """Raise :class:`RetryableHTTPError` for 429/5xx and :class:`LLMError` for other 4xx."""
    if response.status_code >= 500 and REPEAT_ABORT in response.text:
        raise RunnerFaultError(f"HTTP {response.status_code}: {response.text[:200]}")
    if response.status_code == 429 or response.status_code >= 500:
        raise RetryableHTTPError(f"HTTP {response.status_code}: {response.text[:200]}")
    if response.status_code >= 400:
        raise LLMError(f"{service} returned HTTP {response.status_code}: {response.text[:200]}")


def _parse_tool_calls(message: dict[str, Any]) -> tuple[ToolCall, ...]:
    calls: list[ToolCall] = []
    for index, raw in enumerate(message.get("tool_calls") or [], start=1):
        function = raw.get("function") or {}
        name = str(function.get("name", "")).strip()
        if not name:
            continue
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"_raw": arguments}
        if not isinstance(arguments, dict):
            arguments = {"_raw": arguments}
        calls.append(
            ToolCall(name=name, arguments=arguments, id=str(raw.get("id") or f"call_{index}"))
        )
    return tuple(calls)


class OllamaProvider:
    """Chat completions from a local Ollama server, bounded by a timeout and retries."""

    name = "ollama"

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        *,
        num_ctx: int | None = 16384,
        timeout: float = 180.0,
        max_retries: int = 2,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        unload_every: int = 0,
        reload_on_fault: bool = True,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.unload_every = unload_every
        self.reload_on_fault = reload_on_fault
        self._client = client or httpx.Client()
        self._sleep = sleep
        self._answered = 0
        self.stats: dict[str, int] = {
            "requests": 0,
            "empty_completions": 0,
            "aborted_completions": 0,
            "runner_unloads": 0,
            "fallback_answers": 0,
        }

    def build_payload(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The exact JSON body sent to ``/api/chat``."""
        options: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": 0.0,
            "num_predict": self.max_tokens,
        }
        if self.num_ctx is not None:
            # Same value as every other client of the shared server, so the model never reloads.
            options["num_ctx"] = self.num_ctx
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "options": options,
        }
        if tools:
            payload["tools"] = tools
        if json_schema is not None:
            payload["format"] = json_schema
        return payload

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        payload = self.build_payload(messages, tools=tools, json_schema=json_schema)
        # (request, is_fallback): the request twice, then the fallbacks.
        steps: list[tuple[dict[str, Any], bool]] = [(payload, False), (payload, False)]
        if json_schema is not None:
            steps.append((_json_mode(payload, json_schema), True))
        steps.append((_relaxed(steps[-1][0]), True))

        started = time.perf_counter()
        problems: list[str] = []
        completion: Completion | None = None
        with _OLLAMA_LOCK:
            for index, (request, fallback) in enumerate(steps):
                try:
                    data = self._post(request)
                except RunnerFaultError as exc:
                    self.stats["aborted_completions"] += 1
                    problems.append(str(exc))
                else:
                    message = data.get("message") or {}
                    completion = Completion(
                        text=strip_reasoning(str(message.get("content", ""))),
                        latency_s=time.perf_counter() - started,
                        tool_calls=_parse_tool_calls(message),
                        prompt_tokens=data.get("prompt_eval_count"),
                        completion_tokens=data.get("eval_count"),
                    )
                    if completion.text or completion.tool_calls:
                        self.stats["fallback_answers"] += int(fallback)
                        break
                    self.stats["empty_completions"] += 1
                    problems.append("empty completion")
                if index == 0 and self.reload_on_fault:
                    self._unload()
            else:
                if completion is None or json_schema is not None:
                    raise LLMError(
                        f"Ollama gave no usable completion for {self.model}: {'; '.join(problems)}"
                    )
                # A chat reply that stayed empty is returned as it is: the caller decides.
            self._answered += 1
            if self.unload_every and self._answered % self.unload_every == 0:
                self._unload()
        return completion

    def _post(self, request: dict[str, Any]) -> dict[str, Any]:
        def _call() -> dict[str, Any]:
            self.stats["requests"] += 1
            response = self._client.post(
                f"{self.host}/api/chat", json=request, timeout=self.timeout
            )
            check_response(response, "Ollama")
            data: dict[str, Any] = response.json()
            return data

        try:
            return call_with_retries(
                _call,
                retries=self.max_retries,
                retry_on=(httpx.TransportError, RetryableHTTPError),
                sleep=self._sleep,
            )
        except (httpx.TransportError, RetryableHTTPError) as exc:
            raise LLMError(
                f"Ollama call failed after {self.max_retries + 1} attempts: {exc}"
            ) from exc

    def _unload(self) -> None:
        """Ask Ollama to unload the model now (``keep_alive: 0``); the next call reloads it."""
        try:
            self._client.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "keep_alive": 0},
                timeout=60.0,
            )
        except httpx.HTTPError:
            return  # best effort: the next request says more than this one could
        self.stats["runner_unloads"] += 1


def _copy(payload: dict[str, Any]) -> dict[str, Any]:
    copy = dict(payload)
    copy["options"] = dict(payload["options"])
    copy["messages"] = [dict(m) for m in payload["messages"]]
    return copy


def _json_mode(payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Same request with ``format: "json"`` and the schema written into the prompt."""
    request = _copy(payload)
    request["format"] = "json"
    instruction = _SCHEMA_INSTRUCTION + json.dumps(schema)
    for message in request["messages"]:
        if message.get("role") == "system":
            message["content"] = str(message.get("content", "")) + instruction
            break
    else:
        request["messages"].insert(0, {"role": "system", "content": instruction.strip()})
    return request


def _relaxed(payload: dict[str, Any]) -> dict[str, Any]:
    """Same request without a grammar and with a repeat penalty."""
    request = _copy(payload)
    request.pop("format", None)
    request["options"]["repeat_penalty"] = FALLBACK_REPEAT_PENALTY
    return request
