from __future__ import annotations

import json
import sys
import types
from typing import Any

import httpx
import pytest

from agent_eval.config import Settings
from agent_eval.llm import (
    Completion,
    LLMError,
    ToolCall,
    assistant_message,
    complete,
    get_provider,
    tool_message,
)
from agent_eval.llm.base import split_system, strip_reasoning
from agent_eval.llm.fake import FakeProvider
from agent_eval.llm.ollama import OllamaProvider
from agent_eval.llm.retry import call_with_retries
from agent_eval.models import ToolSpec


def _ollama(handler: Any, **kwargs: Any) -> OllamaProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaProvider("qwen3:8b", client=client, sleep=lambda _s: None, **kwargs)


def test_ollama_payload_uses_shared_num_ctx_and_qwen3_sampling() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "<think>x</think>hi"}})

    provider = _ollama(handler)
    result = complete(provider, "sys", "user", json_schema={"type": "object"})
    body = seen[0]
    assert result.text == "hi"
    assert body["options"]["num_ctx"] == 16384
    assert body["options"]["temperature"] == 0.7
    assert body["options"]["top_p"] == 0.8
    assert body["options"]["top_k"] == 20
    assert body["think"] is False
    assert body["format"] == {"type": "object"}
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert "tools" not in body


def test_ollama_num_ctx_can_be_omitted() -> None:
    provider = _ollama(lambda _r: httpx.Response(200, json={}), num_ctx=None)
    assert "num_ctx" not in provider.build_payload([])["options"]


def test_ollama_parses_tool_calls_including_string_arguments() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["tools"] == [{"type": "function"}]
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "send_email", "arguments": {"to": "a@b.c"}}},
                        {"function": {"name": "lookup_order", "arguments": '{"order_id": "A"}'}},
                        {"function": {"name": "x", "arguments": "not json"}},
                        {"function": {"name": "y", "arguments": [1, 2]}},
                        {"function": {"name": ""}},
                    ],
                },
                "prompt_eval_count": 10,
                "eval_count": 3,
            },
        )

    result = _ollama(handler).chat([], tools=[{"type": "function"}])
    names = [c.name for c in result.tool_calls]
    assert names == ["send_email", "lookup_order", "x", "y"]
    assert result.tool_calls[1].arguments == {"order_id": "A"}
    assert result.tool_calls[2].arguments == {"_raw": "not json"}
    assert result.tool_calls[3].arguments == {"_raw": [1, 2]}
    assert result.prompt_tokens == 10


def test_ollama_retries_5xx_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, text="busy")
        return httpx.Response(200, json={"message": {"content": "ok"}})

    assert _ollama(handler).chat([]).text == "ok"
    assert calls["n"] == 2


def test_ollama_gives_up_after_bounded_retries() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    with pytest.raises(LLMError, match="after 2 attempts"):
        _ollama(handler, max_retries=1).chat([])


def test_ollama_4xx_is_not_retried() -> None:
    with pytest.raises(LLMError, match="HTTP 404"):
        _ollama(lambda _r: httpx.Response(404, text="no model")).chat([])


def test_retry_helper_reraises_after_budget() -> None:
    sleeps: list[float] = []

    def boom() -> None:
        raise ValueError("x")

    with pytest.raises(ValueError, match="x"):
        call_with_retries(boom, retries=2, retry_on=(ValueError,), sleep=sleeps.append)
    assert sleeps == [1.0, 2.0]


def test_fake_provider_modes() -> None:
    fake = FakeProvider(["a", Completion(text="b", latency_s=0.1)], fail_times=1)
    with pytest.raises(LLMError):
        fake.chat([])
    assert fake.chat([]).text == "a"
    assert fake.chat([]).text == "b"
    with pytest.raises(LLMError, match="no more"):
        fake.chat([])
    assert len(fake.calls) == 4


def test_helpers() -> None:
    assert strip_reasoning("<think>\nplan\n</think> answer") == "answer"
    system, rest = split_system(
        [
            {"role": "system", "content": "a"},
            {"role": "user", "content": "u"},
            {"role": "system", "content": "b"},
        ]
    )
    assert system == "a\n\nb"
    assert rest == [{"role": "user", "content": "u"}]


def test_factory_defaults_to_ollama_qwen3() -> None:
    provider = get_provider(Settings())
    assert isinstance(provider, OllamaProvider)
    assert provider.model == "qwen3:8b"
    assert provider.num_ctx == 16384


def test_factory_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")
    monkeypatch.setenv("LLM_MODEL", "qwen3:4b")
    provider = get_provider()
    assert isinstance(provider, OllamaProvider)
    assert (provider.model, provider.num_ctx) == ("qwen3:4b", 8192)


def test_factory_openai_needs_model() -> None:
    with pytest.raises(LLMError, match="Set LLM_MODEL"):
        get_provider(Settings(llm_provider="openai"))


def test_factory_hosted_providers_need_keys() -> None:
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        get_provider(Settings(llm_provider="anthropic"))
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        get_provider(Settings(llm_provider="openai", llm_model="gpt-x"))


class _Usage:
    input_tokens = 5
    output_tokens = 2
    prompt_tokens = 5
    completion_tokens = 2


CONVERSATION: list[dict[str, Any]] = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "rename it"},
    {
        "role": "assistant",
        "content": "on it",
        "tool_calls": [
            {"id": "c1", "function": {"name": "kv_get", "arguments": {"key": "a"}}},
            {"id": "c2", "function": {"name": "kv_list", "arguments": {}}},
        ],
    },
    {"role": "tool", "content": "1", "tool_name": "kv_get", "tool_call_id": "c1"},
    {"role": "tool", "content": "[]", "tool_name": "kv_list", "tool_call_id": "c2"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "c3", "function": {"name": "kv_list"}}],
    },
    {"role": "tool", "content": "[]", "tool_name": "kv_list", "tool_call_id": "c3"},
]
TOOLS = [ToolSpec(name="kv_list", description="List keys.").as_function()]


def _install_fake_sdks(monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]) -> None:
    class Messages:
        def create(self, **kwargs: Any) -> Any:
            captured["anthropic"] = kwargs
            return types.SimpleNamespace(
                content=[
                    types.SimpleNamespace(type="text", text=" checking "),
                    types.SimpleNamespace(type="tool_use", id="tu1", name="kv_list", input={}),
                ],
                usage=_Usage(),
            )

    class Anthropic:
        def __init__(self, **kwargs: Any) -> None:
            captured["anthropic_init"] = kwargs
            self.messages = Messages()

    def _call(call_id: str, arguments: str) -> Any:
        function = types.SimpleNamespace(name="kv_get", arguments=arguments)
        return types.SimpleNamespace(id=call_id, function=function)

    class Completions:
        def create(self, **kwargs: Any) -> Any:
            captured["openai"] = kwargs
            calls = [_call("x1", '{"key": "a"}'), _call("", "not json"), _call("x3", "[1]")]
            message = types.SimpleNamespace(content="ok ", tool_calls=calls)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=message)], usage=_Usage()
            )

    class OpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured["openai_init"] = kwargs
            self.chat = types.SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=Anthropic))
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=OpenAI))


def test_anthropic_provider_translates_tools_and_conversation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    _install_fake_sdks(monkeypatch, captured)
    provider = get_provider(Settings(llm_provider="anthropic", anthropic_api_key="k"))
    assert provider.model == "claude-sonnet-5-5"
    result = provider.chat(CONVERSATION, tools=TOOLS, json_schema={"type": "object"})
    assert result.text == "checking"
    assert result.tool_calls == (ToolCall(name="kv_list", arguments={}, id="tu1"),)
    assert (result.prompt_tokens, result.completion_tokens) == (5, 2)
    sent = captured["anthropic"]
    assert sent["system"].startswith("sys")
    assert "Return only JSON" in sent["system"]
    assert sent["tools"] == [
        {
            "name": "kv_list",
            "description": "List keys.",
            "input_schema": TOOLS[0]["function"]["parameters"],
        }
    ]
    assert "temperature" not in sent
    assert [m["role"] for m in sent["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assistant = sent["messages"][1]["content"]
    assert assistant[0] == {"type": "text", "text": "on it"}
    assert assistant[1] == {"type": "tool_use", "id": "c1", "name": "kv_get", "input": {"key": "a"}}
    # The two results of one assistant turn are grouped into one user turn.
    assert [b["tool_use_id"] for b in sent["messages"][2]["content"]] == ["c1", "c2"]
    assert sent["messages"][3]["content"] == [
        {"type": "tool_use", "id": "c3", "name": "kv_list", "input": {}}
    ]


def test_openai_provider_translates_tools_and_conversation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    _install_fake_sdks(monkeypatch, captured)
    settings = Settings(
        llm_provider="openai", llm_model="m", openai_api_key="k", openai_base_url="http://x/v1"
    )
    provider = get_provider(settings)
    assert captured["openai_init"]["base_url"] == "http://x/v1"
    result = provider.chat(CONVERSATION, tools=TOOLS, json_schema={"type": "object"})
    assert result.text == "ok"
    assert [c.id for c in result.tool_calls] == ["x1", "call_2", "x3"]
    assert result.tool_calls[0].arguments == {"key": "a"}
    assert result.tool_calls[1].arguments == {"_raw": "not json"}
    assert result.tool_calls[2].arguments == {"_raw": [1]}
    sent = captured["openai"]
    assert sent["tools"] == TOOLS
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["messages"][0] == {"role": "system", "content": "sys"}
    assert sent["messages"][2]["tool_calls"][0] == {
        "id": "c1",
        "type": "function",
        "function": {"name": "kv_get", "arguments": '{"key": "a"}'},
    }
    assert sent["messages"][3] == {"role": "tool", "tool_call_id": "c1", "content": "1"}
    assert sent["messages"][5]["content"] is None
    assert provider.chat([{"role": "user", "content": "hi"}]).text == "ok"
    assert "tools" not in captured["openai"]


def test_hosted_providers_wrap_sdk_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _install_fake_sdks(monkeypatch, captured)
    anthropic = get_provider(Settings(llm_provider="anthropic", anthropic_api_key="k"))
    openai = get_provider(Settings(llm_provider="openai", llm_model="m", openai_api_key="k"))

    def fail(**_kwargs: Any) -> Any:
        raise RuntimeError("sdk down")

    anthropic._client.messages.create = fail  # type: ignore[attr-defined]
    openai._client.chat.completions.create = fail  # type: ignore[attr-defined]
    with pytest.raises(LLMError, match="Anthropic call failed"):
        anthropic.chat([{"role": "user", "content": "x"}])
    with pytest.raises(LLMError, match="OpenAI call failed"):
        openai.chat([{"role": "user", "content": "x"}])


def test_message_helpers() -> None:
    call = ToolCall(name="kv_get", arguments={"key": "a"}, id="c1")
    with_calls = assistant_message(Completion(text="", latency_s=0.0, tool_calls=(call,)))
    assert with_calls["tool_calls"] == [
        {"id": "c1", "function": {"name": "kv_get", "arguments": {"key": "a"}}}
    ]
    assert "tool_calls" not in assistant_message(Completion(text="done", latency_s=0.0))
    assert tool_message(call, "1") == {
        "role": "tool",
        "content": "1",
        "tool_name": "kv_get",
        "tool_call_id": "c1",
    }


# -- Ollama runner faults and unloading ----------------------------------------------------

ABORT = "prediction aborted, token repeat limit reached"
VERDICT = '{"verdict": "pass"}'


class FaultyServer:
    """Fake Ollama: a runner that turns faulty until it is unloaded (``keep_alive: 0``)."""

    def __init__(self, *, faulty: bool = True, fault: str = "empty", heal_on_unload: bool = True):
        self.faulty = faulty
        self.fault = fault
        self.heal_on_unload = heal_on_unload
        self.chats: list[dict[str, Any]] = []
        self.unloads = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/api/generate":
            assert body == {"model": "qwen3:8b", "keep_alive": 0}
            self.unloads += 1
            self.faulty = self.faulty and not self.heal_on_unload
            return httpx.Response(200, json={"done": True})
        self.chats.append(body)
        if self.faulty and self.fault == "abort":
            return httpx.Response(500, json={"error": ABORT})
        if self.faulty and self.fault == "empty" and isinstance(body.get("format"), dict):
            return httpx.Response(200, json={"message": {"content": ""}})
        if self.faulty and self.fault == "always-empty":
            return httpx.Response(200, json={"message": {"content": ""}})
        return httpx.Response(200, json={"message": {"content": VERDICT}})


SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string"}}}


def test_ollama_reloads_the_runner_after_an_empty_schema_completion() -> None:
    server = FaultyServer()
    provider = _ollama(server)
    assert complete(provider, "sys", "user", json_schema=SCHEMA).text == VERDICT
    assert server.unloads == 1
    assert [c.get("format") for c in server.chats] == [SCHEMA, SCHEMA]
    assert provider.stats == {
        "requests": 2,
        "empty_completions": 1,
        "aborted_completions": 0,
        "runner_unloads": 1,
        "fallback_answers": 0,
    }


def test_ollama_repeat_abort_is_not_retried_unchanged_on_the_same_runner() -> None:
    server = FaultyServer(fault="abort")
    provider = _ollama(server)
    assert provider.chat([{"role": "user", "content": "hi"}]).text == VERDICT
    # One aborted request (no blind 5xx retries), one unload, one good request.
    assert len(server.chats) == 2
    assert server.unloads == 1
    assert provider.stats["aborted_completions"] == 1


def test_ollama_falls_back_to_json_mode_with_the_schema_in_the_prompt() -> None:
    server = FaultyServer(heal_on_unload=False)
    provider = _ollama(server)
    assert complete(provider, "sys", "user", json_schema=SCHEMA).text == VERDICT
    assert [c.get("format") for c in server.chats] == [SCHEMA, SCHEMA, "json"]
    system = server.chats[2]["messages"][0]["content"]
    assert system.startswith("sys")
    assert json.dumps(SCHEMA) in system
    assert server.chats[0]["messages"][0]["content"] == "sys"  # the original is not mutated
    assert provider.stats["fallback_answers"] == 1
    assert provider.stats["empty_completions"] == 2


def test_ollama_last_step_drops_the_grammar_and_adds_a_repeat_penalty() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={})
        seen.append(body)
        if "format" in body:
            return httpx.Response(500, json={"error": ABORT})
        return httpx.Response(200, json={"message": {"content": VERDICT}})

    provider = _ollama(handler)
    # No system message: the schema instruction becomes one.
    result = provider.chat([{"role": "user", "content": "u"}], json_schema=SCHEMA)
    assert result.text == VERDICT
    assert [c.get("format") for c in seen] == [SCHEMA, SCHEMA, "json", None]
    assert seen[3]["options"]["repeat_penalty"] == 1.15
    assert "repeat_penalty" not in seen[0]["options"]
    assert seen[3]["messages"][0]["role"] == "system"
    assert provider.stats["aborted_completions"] == 3


def test_ollama_gives_up_with_every_problem_listed() -> None:
    server = FaultyServer(fault="abort", heal_on_unload=False)
    provider = _ollama(server, reload_on_fault=False)
    with pytest.raises(LLMError, match=r"no usable completion.*token repeat limit"):
        complete(provider, "sys", "user", json_schema=SCHEMA)
    assert len(server.chats) == 4
    assert server.unloads == 0  # reload_on_fault is off

    empty = FaultyServer(fault="always-empty", heal_on_unload=False)
    with pytest.raises(LLMError, match="empty completion"):
        complete(_ollama(empty), "sys", "user", json_schema=SCHEMA)


def test_ollama_returns_an_empty_chat_reply_when_every_attempt_is_empty() -> None:
    # Without a schema an empty reply is a possible answer: the caller decides what it means.
    server = FaultyServer(fault="always-empty", heal_on_unload=False)
    provider = _ollama(server)
    assert provider.chat([{"role": "user", "content": "hi"}]).text == ""
    assert len(server.chats) == 3
    assert provider.stats["empty_completions"] == 3


def test_ollama_unloads_every_n_answered_calls() -> None:
    server = FaultyServer(faulty=False)
    provider = _ollama(server, unload_every=2)
    for _ in range(5):
        provider.chat([{"role": "user", "content": "hi"}])
    assert server.unloads == 2
    assert provider.stats["runner_unloads"] == 2

    off = FaultyServer(faulty=False)
    quiet = _ollama(off)
    for _ in range(3):
        quiet.chat([{"role": "user", "content": "hi"}])
    assert off.unloads == 0  # off by default


def test_ollama_unload_is_best_effort() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/generate":
            raise httpx.ConnectError("down")
        return httpx.Response(200, json={"message": {"content": "ok"}})

    provider = _ollama(handler, unload_every=1)
    assert provider.chat([]).text == "ok"
    assert provider.stats["runner_unloads"] == 0


def test_factory_passes_the_low_ram_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_UNLOAD_EVERY", "4")
    monkeypatch.setenv("OLLAMA_RELOAD_ON_FAULT", "false")
    provider = get_provider(Settings(_env_file=None))  # type: ignore[call-arg]
    assert isinstance(provider, OllamaProvider)
    assert provider.unload_every == 4
    assert provider.reload_on_fault is False
    default = OllamaProvider("qwen3:8b")
    assert default.unload_every == 0
    assert default.reload_on_fault is True
