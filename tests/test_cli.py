from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import agent_eval.cli as cli
from agent_eval.llm.base import Completion, LLMError, Message, ToolCall
from agent_eval.llm.fake import FakeProvider
from agent_eval.models import Comparison, RunReport
from agent_eval.testing import sample_comparison, sample_report

EXAMPLES = str(Path(__file__).parent.parent / "examples" / "tasks")
runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run every CLI test in an empty directory: no .env, no ./tasks, no ./out."""
    monkeypatch.chdir(tmp_path)
    for name in ("LLM_PROVIDER", "LLM_MODEL", "OLLAMA_UNLOAD_EVERY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def html_renderer(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, bool]]:
    """Install a stand-in ``agent_eval.report.html`` and record how it is called."""
    seen: list[tuple[int, bool]] = []

    def render_html(reports: list[RunReport], comparison: Comparison | None = None) -> str:
        seen.append((len(reports), comparison is not None))
        return "<!doctype html><title>stub</title>"

    module = types.SimpleNamespace(render_html=render_html)
    monkeypatch.setitem(sys.modules, "agent_eval.report.html", module)
    return seen


def test_list() -> None:
    result = runner.invoke(cli.app, ["list", "--tasks", EXAMPLES])
    assert result.exit_code == 0
    assert "| kv-001-rotate-token | kv | tool_error, destructive | 4 | 8 |" in result.output
    assert "1 tasks in 1 families" in result.output


def test_list_errors() -> None:
    missing = runner.invoke(cli.app, ["list", "--tasks", "nowhere"])
    assert missing.exit_code == 2
    assert "no such file" in missing.output
    empty = runner.invoke(cli.app, ["list", "--tasks", EXAMPLES, "--family", "other"])
    assert empty.exit_code == 2
    assert "no task matches" in empty.output


def test_run_oracle_writes_reports(html_renderer: list[tuple[int, bool]]) -> None:
    result = runner.invoke(
        cli.app, ["run", "--agent", "oracle", "--tasks", EXAMPLES, "-k", "2", "--out", "o"]
    )
    assert result.exit_code == 0, result.output
    assert "[2/2] PASS kv-001-rotate-token trial 2: 4 steps (optimal 4)" in result.output
    assert "| Success rate | 100.0% |" in result.output
    assert "| kv | 1 | 100.0% | 100.0% | 0.0% | 100.0% | 1.00 |" in result.output
    assert "| tool_error | 1 |" in result.output
    report = RunReport.model_validate_json(Path("o/oracle.json").read_text(encoding="utf-8"))
    assert (report.run.agent, report.run.model, report.run.k) == ("oracle", "none", 2)
    assert report.run.options == {}
    assert len(Path("o/oracle.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert Path("o/oracle.html").read_text(encoding="utf-8").startswith("<!doctype html>")
    assert html_renderer == [(1, False)]


def test_run_resume_reuses_the_checkpoint_and_a_fresh_run_discards_it() -> None:
    args = ["run", "--agent", "oracle", "--tasks", EXAMPLES, "--out", "o"]
    assert runner.invoke(cli.app, args).exit_code == 0
    checkpoint = Path("o/oracle.jsonl")
    first = checkpoint.read_text(encoding="utf-8")
    assert runner.invoke(cli.app, [*args, "--resume"]).exit_code == 0
    assert checkpoint.read_text(encoding="utf-8") == first
    checkpoint.write_text(first + first, encoding="utf-8")
    assert runner.invoke(cli.app, args).exit_code == 0
    assert len(checkpoint.read_text(encoding="utf-8").splitlines()) == 1


def test_run_noop_fails_only_with_a_threshold() -> None:
    args = ["run", "--agent", "noop", "--tasks", EXAMPLES, "--out", "o"]
    plain = runner.invoke(cli.app, args)
    assert plain.exit_code == 0
    assert "[1/1] FAIL" in plain.output
    assert "| Valid tool calls | n/a |" in plain.output
    assert "HTML report: " in plain.output
    gated = runner.invoke(cli.app, [*args, "--fail-under", "0.5"])
    assert gated.exit_code == 1
    assert "below 50.0%" in gated.output


def test_run_rejects_unknown_agents_and_tasks() -> None:
    bad_agent = runner.invoke(cli.app, ["run", "--agent", "nope", "--tasks", EXAMPLES])
    assert bad_agent.exit_code == 2
    assert "unknown agent" in bad_agent.output
    bad_task = runner.invoke(
        cli.app, ["run", "--agent", "oracle", "--tasks", EXAMPLES, "--task", "nope"]
    )
    assert bad_task.exit_code == 2
    assert "unknown task id" in bad_task.output


def _scripted_model(messages: list[Message], tools: list[dict[str, Any]] | None) -> Completion:
    """A fake model: delete a production key, pass a bad argument, then answer."""
    turn = sum(1 for m in messages if m["role"] == "assistant")
    script = [
        ("kv_delete", {"key": "prod_backup"}),
        ("kv_get", {"key": 1}),
    ]
    if turn < len(script):
        name, args = script[turn]
        return Completion(
            text="",
            latency_s=0.0,
            tool_calls=(ToolCall(name=name, arguments=args, id=f"c{turn}"),),
            prompt_tokens=7,
            completion_tokens=3,
        )
    return Completion(text="eu-west-1", latency_s=0.0, prompt_tokens=7, completion_tokens=3)


def test_run_model_backed_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_get_provider(settings: Any) -> FakeProvider:
        seen["unload_every"] = settings.ollama_unload_every
        provider = FakeProvider(responder=_scripted_model, model="fake")
        provider.stats = {"requests": 3}  # type: ignore[attr-defined]
        return provider

    monkeypatch.setattr(cli, "get_provider", fake_get_provider)
    result = runner.invoke(
        cli.app,
        ["run", "--agent", "baseline", "--tasks", EXAMPLES, "--out", "o", "--unload-every", "5"],
    )
    assert result.exit_code == 0, result.output
    assert "FAIL kv-001-rotate-token trial 1: 2 steps (optimal 4)" in result.output
    assert "forbidden invalid=1" in result.output
    assert seen["unload_every"] == 5
    report = RunReport.model_validate_json(Path("o/baseline.json").read_text(encoding="utf-8"))
    assert (report.run.provider, report.run.model) == ("ollama", "qwen3:8b")
    assert report.run.options["unload_every"] == 5
    assert report.run.client_stats == {"requests": 3}
    assert report.summary.forbidden_action_rate == 1.0
    assert report.summary.prompt_tokens == 21


def test_run_reports_a_stop_reason_and_provider_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "get_provider", lambda _s: FakeProvider([]))
    result = runner.invoke(
        cli.app, ["run", "--agent", "guarded", "--tasks", EXAMPLES, "--out", "o"]
    )
    assert result.exit_code == 0
    assert "[llm_error]" in result.output

    def no_key(_settings: Any) -> FakeProvider:
        raise LLMError("ANTHROPIC_API_KEY is not set")

    monkeypatch.setattr(cli, "get_provider", no_key)
    failed = runner.invoke(cli.app, ["run", "--agent", "baseline", "--tasks", EXAMPLES])
    assert failed.exit_code == 2
    assert "ANTHROPIC_API_KEY is not set" in failed.output


def _write_samples() -> None:
    a, b, _ = sample_comparison()
    Path("a.json").write_text(a.model_dump_json(), encoding="utf-8")
    Path("b.json").write_text(b.model_dump_json(), encoding="utf-8")


def test_compare(html_renderer: list[tuple[int, bool]]) -> None:
    _write_samples()
    result = runner.invoke(cli.app, ["compare", "a.json", "b.json", "--out", "c", "--html"])
    assert result.exit_code == 0, result.output
    assert "| Metric | A: sloppy | B: oracle | Delta (B - A) |" in result.output
    assert "| Success rate | 25.0% | 100.0% | +75.0 pp |" in result.output
    assert "| Forbidden-action rate | 25.0% | 0.0% | -25.0 pp |" in result.output
    assert "| beta | 0.0% | 100.0% | 0.0% | 100.0% | 50.0% | 0.0% |" in result.output
    assert "Tasks improved in B: 3 | regressed: 0 | unchanged: 1" in result.output
    assert "- improved: beta-001-clean-tmp (0.0% -> 100.0%)" in result.output
    comparison = json.loads(Path("c/compare.json").read_text(encoding="utf-8"))
    assert comparison["shared_tasks"] == 4
    assert Path("c/compare.md").read_text(encoding="utf-8").startswith("A: sloppy")
    assert Path("c/compare.html").exists()
    assert html_renderer == [(2, True)]


def test_compare_shows_undefined_deltas_and_bad_input() -> None:
    _write_samples()
    noop = sample_report("oracle").model_copy(deep=True)
    noop.tasks = []
    Path("empty.json").write_text(noop.model_dump_json(), encoding="utf-8")
    result = runner.invoke(cli.app, ["compare", "a.json", "empty.json"])
    assert result.exit_code == 0
    assert "| Valid tool calls | n/a | n/a | n/a |" in result.output
    assert not Path("out/compare.html").exists()
    Path("bad.json").write_text("{}", encoding="utf-8")
    bad = runner.invoke(cli.app, ["compare", "a.json", "bad.json"])
    assert bad.exit_code == 2
    assert "not a run report" in bad.output
    missing = runner.invoke(cli.app, ["compare", "a.json", "missing.json"])
    assert missing.exit_code == 2


def test_report(html_renderer: list[tuple[int, bool]]) -> None:
    _write_samples()
    one = runner.invoke(cli.app, ["report", "a.json", "--out", "r/one.html"])
    assert one.exit_code == 0, one.output
    assert Path("r/one.html").exists()
    two = runner.invoke(cli.app, ["report", "a.json", "b.json"])
    assert two.exit_code == 0
    assert Path("out/report.html").exists()
    assert html_renderer == [(1, False), (2, True)]
    many = runner.invoke(cli.app, ["report", "a.json", "b.json", "a.json"])
    assert many.exit_code == 2


def test_report_without_a_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    _write_samples()
    monkeypatch.setitem(sys.modules, "agent_eval.report.html", None)
    result = runner.invoke(cli.app, ["report", "a.json"])
    assert result.exit_code == 2
    assert "renderer is not available" in result.output
