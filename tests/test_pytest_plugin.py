from __future__ import annotations

from pathlib import Path

import pytest

import agent_eval.pytest_plugin as plugin

pytest_plugins = ["pytester"]

EXAMPLE = Path(__file__).parent.parent / "examples" / "tasks" / "kv-001-rotate-token.yaml"
PLUGIN = ("-p", "agent_eval.pytest_plugin", "-p", "no:cacheprovider")


@pytest.fixture
def task_dir(pytester: pytest.Pytester) -> pytest.Pytester:
    pytester.makefile(".yaml", **{"kv-001-rotate-token": EXAMPLE.read_text(encoding="utf-8")})
    return pytester


def test_collects_nothing_without_the_option(task_dir: pytest.Pytester) -> None:
    result = task_dir.runpytest_inprocess(*PLUGIN)
    result.assert_outcomes()


def test_oracle_passes_each_task(task_dir: pytest.Pytester) -> None:
    result = task_dir.runpytest_inprocess(
        *PLUGIN, "--agent-eval", "oracle", "--agent-eval-k", "2", "-v"
    )
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*kv-001-rotate-token.yaml::kv-001-rotate-token PASSED*"])


def test_failure_report_shows_steps_and_checks(task_dir: pytest.Pytester) -> None:
    result = task_dir.runpytest_inprocess(*PLUGIN, "--agent-eval", "noop")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(
        [
            "*agent-eval: kv-001-rotate-token*",
            "kv-001-rotate-token: 1 of 1 trials failed",
            "trial 1 (stop: answer):",
            "  failed check: equals kv.items.api_token (expected 'tok_new_42', got 'tok_old_17')",
            "  failed check: contains answer (expected 'eu-west-1', got 'Done.')",
        ]
    )


def test_failure_report_shows_forbidden_steps_and_agent_errors(
    task_dir: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_eval.agents import AgentResult
    from agent_eval.models import TaskSpec
    from agent_eval.world import World

    class Reckless:
        name = "reckless"

        def run(self, task: TaskSpec, world: World) -> AgentResult:
            world.call("kv_delete", {"key": "prod_backup"})
            raise RuntimeError("boom")

    monkeypatch.setattr(plugin, "build_agent", lambda *_a: Reckless())
    result = task_dir.runpytest_inprocess(*PLUGIN, "--agent-eval", "baseline")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(
        [
            "trial 1 (stop: agent_error):",
            "  step 1: kv_delete({'key': 'prod_backup'}) -> *deleted* [[]FORBIDDEN[]]",
            "  error: RuntimeError: boom",
        ]
    )


def test_a_broken_task_file_is_a_collection_error(pytester: pytest.Pytester) -> None:
    pytester.makefile(".yaml", broken="id: [unclosed")
    result = pytester.runpytest_inprocess(*PLUGIN, "--agent-eval", "oracle")
    result.assert_outcomes(errors=1)


def test_other_failures_use_the_default_report(
    task_dir: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*_args: object, **_kwargs: object) -> None:
        raise ZeroDivisionError("unexpected")

    monkeypatch.setattr(plugin, "run_trial", explode)
    result = task_dir.runpytest_inprocess(*PLUGIN, "--agent-eval", "oracle")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*ZeroDivisionError: unexpected*"])
