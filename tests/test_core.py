from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from agent_eval.agents import AgentResult, NoopAgent, OracleAgent, ScriptedAgent
from agent_eval.checks import CheckError, evaluate
from agent_eval.envs import REGISTRY, Environment, get_environment, register
from agent_eval.llm.base import LLMError
from agent_eval.metrics import aggregate
from agent_eval.models import Check, ForbiddenAction, ReferenceCall, TaskSpec, ToolSpec
from agent_eval.report import compare
from agent_eval.runner import run, run_trial
from agent_eval.tasks import TaskLoadError, default_tasks_dir, load_task, load_tasks
from agent_eval.testing import (
    replay_noop,
    replay_reference,
    sample_comparison,
    sample_report,
    sample_tasks,
)
from agent_eval.world import TaskConfigError, World, matches_forbidden

EXAMPLES = Path(__file__).parent.parent / "examples" / "tasks"


def _task(**overrides: Any) -> TaskSpec:
    base: dict[str, Any] = {
        "id": "t-1",
        "family": "kv",
        "title": "t",
        "prompt": "p",
        "envs": {"kv": {"items": {"a": "1"}}},
        "max_steps": 3,
        "optimal_steps": 0,
        "success": [{"path": "kv.items.a", "op": "exists"}],
    }
    return TaskSpec.model_validate(base | overrides)


# ------------------------------------------------------------------ checks


@pytest.mark.parametrize(
    ("check", "data", "expected"),
    [
        ({"op": "exists", "path": "a"}, {"a": 0}, True),
        ({"op": "exists", "path": "b"}, {"a": 0}, False),
        ({"op": "absent", "path": "b"}, {"a": 0}, True),
        ({"op": "equals", "path": "a", "value": "X", "ignore_case": True}, {"a": "x"}, True),
        ({"op": "equals", "path": "a", "value": "X"}, {"a": "x"}, False),
        ({"op": "not_equals", "path": "a", "value": 2}, {"a": 1}, True),
        ({"op": "contains", "path": "a", "value": "ell"}, {"a": "hello"}, True),
        ({"op": "contains", "path": "a", "value": "b"}, {"a": ["a", "b"]}, True),
        ({"op": "contains", "path": "a", "value": "k"}, {"a": {"k": 1}}, True),
        ({"op": "contains", "path": "a", "value": "k"}, {"a": 5}, False),
        ({"op": "contains", "path": "zz", "value": "k"}, {"a": 5}, False),
        ({"op": "not_contains", "path": "a", "value": "z"}, {"a": "hello"}, True),
        ({"op": "matches", "path": "a", "value": r"^\d+ items$"}, {"a": "12 items"}, True),
        ({"op": "matches", "path": "zz", "value": "x"}, {}, False),
        ({"op": "length", "path": "a", "value": 2}, {"a": [1, 2]}, True),
        ({"op": "length", "path": "a", "value": 2}, {"a": 7}, False),
        ({"op": "in", "path": "a", "value": ["open", "closed"]}, {"a": "open"}, True),
        ({"op": "gte", "path": "a", "value": 3}, {"a": 3}, True),
        ({"op": "lte", "path": "a", "value": 3}, {"a": 4}, False),
        ({"op": "gte", "path": "zz", "value": 3}, {}, False),
        ({"op": "contains", "value": "PARIS", "ignore_case": True}, "It is Paris.", True),
        (
            {"op": "equals", "path": "items[?id=='b'].n | [0]", "value": 2},
            {"items": [{"id": "a", "n": 1}, {"id": "b", "n": 2}]},
            True,
        ),
    ],
)
def test_check_ops(check: dict[str, Any], data: Any, expected: bool) -> None:
    assert evaluate(Check.model_validate(check), data).passed is expected


@pytest.mark.parametrize(
    "check",
    [
        {"op": "equals", "path": "a..b", "value": 1},
        {"op": "matches", "path": "a", "value": "("},
        {"op": "in", "path": "a", "value": "x"},
        {"op": "gte", "path": "a", "value": "x"},
        {"op": "gte", "path": "b", "value": 1},
    ],
)
def test_check_errors(check: dict[str, Any]) -> None:
    with pytest.raises(CheckError):
        evaluate(Check.model_validate(check), {"a": 1, "b": True})


# ------------------------------------------------------------------ task spec


@pytest.mark.parametrize(
    "overrides",
    [
        {"success": []},
        {"optimal_steps": 9},
        {"optimal_steps": 1},
        {"success": [{"op": "exists"}]},
        {"answer": [{"op": "exists", "path": "x"}], "reference_answer": "x"},
        {"answer": [{"op": "contains", "value": "x"}]},
        {"unknown_key": 1},
        {"id": "Bad Id"},
    ],
)
def test_task_spec_rejects_inconsistent_tasks(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _task(**overrides)


def test_tool_spec_as_function() -> None:
    spec = ToolSpec(name="kv_list", description="List keys.")
    assert spec.as_function()["function"]["name"] == "kv_list"
    assert spec.as_function()["function"]["parameters"]["type"] == "object"


# ------------------------------------------------------------------ environments and world


def test_registry_and_unknown_kind() -> None:
    assert get_environment("kv").kind == "kv"
    with pytest.raises(KeyError, match="unknown environment kind"):
        get_environment("nope")


def test_register_rejects_bad_classes() -> None:
    class NoKind(Environment):
        def load(self, state: dict[str, Any]) -> None: ...
        def tools(self) -> list[ToolSpec]:
            return []

        def snapshot(self) -> dict[str, Any]:
            return {}

    with pytest.raises(TypeError):
        register(NoKind)

    class Clash(NoKind):
        kind = "kv"

    with pytest.raises(ValueError, match="already registered"):
        register(Clash)
    assert register(REGISTRY["kv"]) is REGISTRY["kv"]
    assert NoKind.call(Clash({}), "missing", {}).ok is False


def test_environment_does_not_mutate_the_task_state() -> None:
    task = _task()
    world = World(task)
    world.call("kv_set", {"key": "a", "value": "2"})
    assert task.envs["kv"]["items"]["a"] == "1"
    assert world.snapshot() == {"kv": {"items": {"a": "2"}}}


def test_world_records_every_kind_of_step() -> None:
    task = _task(
        faults=[{"tool": "kv_get", "calls": [1], "error": "503"}],
        forbidden=[{"tool": "kv_delete", "args": {"key": "a*"}, "reason": "keep a"}],
        max_steps=10,
    )
    world = World(task)
    unknown = world.call("kv_nope", {})
    invalid = world.call("kv_get", {"key": 5})
    faulted = world.call("kv_get", {"key": "a"})
    ok = world.call("kv_get", {"key": "a"})
    missing = world.call("kv_get", {"key": "zz"})
    forbidden = world.call("kv_delete", {"key": "a"})
    blocked = world.record_blocked("kv_delete", {"key": "a"}, "needs confirmation")
    blocked_unknown = world.record_blocked("kv_nope", {}, "unknown")

    assert (unknown.known_tool, unknown.valid_args, unknown.executed) == (False, False, False)
    assert invalid.valid_args is False
    assert "key" in (invalid.validation_error or "")
    assert (faulted.fault, faulted.executed, faulted.ok) == (True, False, False)
    assert json.loads(faulted.output) == {"ok": False, "error": "503"}
    assert json.loads(ok.output) == {"ok": True, "result": "1"}
    assert (missing.executed, missing.ok) == (True, False)
    assert (forbidden.forbidden, forbidden.forbidden_reason) == (True, "keep a")
    assert (blocked.blocked, blocked.executed, blocked.valid_args) == (True, False, True)
    assert "blocked by guardrail" in blocked.output
    assert blocked_unknown.known_tool is False
    assert [s.index for s in world.steps] == list(range(1, 9))
    assert world.steps_left == 2
    assert world.spec("kv_get") is not None
    assert world.spec("kv_nope") is None


def test_world_limits_tools_and_validates_references() -> None:
    world = World(_task(tools=["kv_get"]))
    assert [s.name for s in world.tool_specs()] == ["kv_get"]
    assert world.call("kv_set", {"key": "a", "value": "2"}).known_tool is False
    for field in ("tools", "expected_tools"):
        with pytest.raises(TaskConfigError, match=field):
            World(_task(**{field: ["kv_nope"]}))
    with pytest.raises(TaskConfigError, match="forbidden"):
        World(_task(forbidden=[{"tool": "kv_nope"}]))


def test_matches_forbidden() -> None:
    rule = ForbiddenAction(tool="t", args={"path": "/prod/*", "force": True})
    assert matches_forbidden(rule, "t", {"path": "/prod/db", "force": True})
    assert not matches_forbidden(rule, "t", {"path": "/prod/db", "force": False})
    assert not matches_forbidden(rule, "t", {"path": "/dev/db", "force": True})
    assert not matches_forbidden(rule, "t", {"path": "/prod/db"})
    assert not matches_forbidden(rule, "other", {"path": "/prod/db", "force": True})
    assert matches_forbidden(ForbiddenAction(tool="t"), "t", {"anything": 1})


# ------------------------------------------------------------------ loader


def test_example_task_loads_and_its_reference_solution_passes() -> None:
    task = load_task(EXAMPLES / "kv-001-rotate-token.yaml")
    result = replay_reference(task)
    assert result.metrics.success is True
    assert result.metrics.step_ratio == 1.0
    assert result.metrics.tool_precision == 1.0
    assert result.metrics.tool_recall == 1.0
    assert replay_noop(task).metrics.success is False


def test_load_tasks_filters_and_reports_problems(tmp_path: Path) -> None:
    assert [t.id for t in load_tasks(EXAMPLES)] == ["kv-001-rotate-token"]
    assert load_tasks(EXAMPLES, families=["other"]) == []
    assert len(load_tasks(EXAMPLES / "kv-001-rotate-token.yaml", ids=["kv-001-rotate-token"])) == 1
    with pytest.raises(TaskLoadError, match="unknown task id"):
        load_tasks(EXAMPLES, ids=["nope"])
    with pytest.raises(TaskLoadError, match="no such file"):
        load_tasks(tmp_path / "missing")

    text = (EXAMPLES / "kv-001-rotate-token.yaml").read_text(encoding="utf-8")
    (tmp_path / "a.yaml").write_text(text, encoding="utf-8")
    (tmp_path / "b.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(TaskLoadError, match="duplicate task id"):
        load_tasks(tmp_path)
    (tmp_path / "b.yaml").write_text(text.replace("kv:\n    items", "nope:\n    items"), "utf-8")
    with pytest.raises(TaskLoadError, match=r"b\.yaml"):
        load_tasks(tmp_path)
    (tmp_path / "b.yaml").write_text("id: [unclosed", encoding="utf-8")
    with pytest.raises(TaskLoadError, match=r"b\.yaml"):
        load_tasks(tmp_path)


def test_default_tasks_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert default_tasks_dir().name == "_tasks"
    (tmp_path / "tasks").mkdir()
    assert default_tasks_dir() == Path("tasks")


# ------------------------------------------------------------------ runner, metrics, report


def test_oracle_scores_everything_and_noop_fails_state_tasks() -> None:
    tasks = sample_tasks()
    oracle = run(OracleAgent(), tasks, k=2)
    assert oracle.summary.success_rate == 1.0
    assert oracle.summary.pass_k == 1.0
    assert oracle.summary.mean_step_ratio == 1.0
    noop = run(NoopAgent(), tasks, k=1)
    assert noop.summary.success_rate == 0.0
    assert noop.summary.valid_call_rate is None


def test_sample_report_numbers() -> None:
    report = sample_report()
    by_id = {t.task_id: t for t in report.tasks}
    first = by_id["alpha-001-update-owner"].trials[0].metrics
    assert (first.success, first.steps, first.invalid_calls, first.step_ratio) == (True, 2, 1, 2.0)
    assert first.valid_call_rate == 0.5
    fault = by_id["alpha-002-read-env"].trials[0]
    assert fault.metrics.goal_met is False
    assert fault.trajectory.steps[0].fault is True
    unsafe = by_id["beta-001-clean-tmp"].trials[0].metrics
    assert (unsafe.goal_met, unsafe.safe, unsafe.success) == (False, False, False)
    assert unsafe.forbidden_actions == 1
    looped = by_id["beta-002-rename-key"].trials[0]
    assert looped.trajectory.stop_reason == "max_steps"
    assert looped.metrics.steps == 3
    assert (looped.metrics.tool_precision, looped.metrics.tool_recall) == (0.0, 0.0)
    assert report.summary.tasks == 4
    assert report.summary.trials == 8
    assert report.summary.success_rate == 0.25
    assert report.summary.pass_k == 0.25
    assert report.summary.forbidden_action_rate == 0.25
    assert set(report.families) == {"alpha", "beta"}
    assert set(report.traps) == {"tool_error", "destructive"}
    assert report.families["beta"].success_rate == 0.0
    assert type(report).model_validate_json(report.model_dump_json()) == report


def test_compare() -> None:
    a, b, comparison = sample_comparison()
    assert comparison.shared_tasks == 4
    summary = {row.metric: row for row in comparison.summary}
    assert summary["success_rate"].delta == 0.75
    assert summary["forbidden_action_rate"].delta == -0.25
    assert [d.task_id for d in comparison.regressed] == []
    assert len(comparison.improved) == 3
    assert comparison.unchanged == 1
    assert set(comparison.families) == {"alpha", "beta"}
    reverse = compare(b, a)
    assert len(reverse.regressed) == 3
    noop = run(NoopAgent(), sample_tasks()[:1])
    undefined = {row.metric: row for row in compare(noop, a).summary}
    assert undefined["valid_call_rate"].delta is None


def test_aggregate_of_nothing() -> None:
    empty = aggregate({})
    assert (empty.tasks, empty.trials, empty.valid_call_rate) == (0, 0, None)


class _Crashing:
    name = "crashing"

    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.usage = AgentResult(llm_calls=2, prompt_tokens=10, completion_tokens=5, plan="p")

    def run(self, task: TaskSpec, world: World) -> AgentResult:
        world.call("kv_get", {"key": "owner"})
        raise self.exc


def test_agent_crashes_become_failed_trials() -> None:
    task = sample_tasks()[0]
    llm = run_trial(_Crashing(LLMError("no completion")), task)
    assert llm.trajectory.stop_reason == "llm_error"
    assert llm.trajectory.error == "LLMError: no completion"
    assert (llm.trajectory.llm_calls, llm.trajectory.prompt_tokens) == (2, 10)
    assert llm.metrics.steps == 1
    assert llm.metrics.success is False
    other = run_trial(_Crashing(RuntimeError("boom")), task)
    assert other.trajectory.stop_reason == "agent_error"


def test_checkpoint_resumes_and_ignores_other_agents(tmp_path: Path) -> None:
    tasks = sample_tasks()[:2]
    path = tmp_path / "out" / "run.jsonl"
    seen: list[str] = []
    first = run(
        OracleAgent(), tasks, k=2, checkpoint=path, progress=lambda t, _r: seen.append(t.id)
    )
    assert len(seen) == 4
    assert len(path.read_text(encoding="utf-8").splitlines()) == 4
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"agent": "oracle", "task_id": "alpha-001-upd')  # interrupted write

    calls: list[str] = []

    class Counting(OracleAgent):
        def run(self, task: TaskSpec, world: World) -> AgentResult:
            calls.append(task.id)
            return super().run(task, world)

    resumed = run(Counting(), tasks, k=2, checkpoint=path, client_stats=lambda: {"requests": 0})
    assert calls == []
    assert resumed.summary == first.summary
    assert resumed.run.client_stats == {"requests": 0}
    other = ScriptedAgent({}, name="other")
    assert run(other, tasks, k=1, checkpoint=path).summary.success_rate == 0.0


def test_scripted_agent_reference_call_defaults() -> None:
    assert ReferenceCall(tool="kv_list").args == {}
