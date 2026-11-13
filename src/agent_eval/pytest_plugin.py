"""Optional pytest plugin: every task file becomes a test case.

    pytest -p agent_eval.pytest_plugin tasks/ --agent-eval oracle
    pytest -p agent_eval.pytest_plugin tasks/files --agent-eval guarded --agent-eval-k 3

A task passes when all of its k trials succeed (the pass^k criterion). Without
``--agent-eval`` the plugin collects nothing, so it is safe to leave enabled.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from agent_eval.agents import AGENT_NAMES, Agent, build_agent
from agent_eval.models import TaskSpec, TrialResult
from agent_eval.runner import run_trial
from agent_eval.tasks import load_task

_AGENT = pytest.StashKey[Agent]()


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("agent-eval", "agent-eval-harness")
    group.addoption(
        "--agent-eval",
        metavar="AGENT",
        choices=AGENT_NAMES,
        default=None,
        help=f"Collect task YAML files and run them with this agent ({', '.join(AGENT_NAMES)}).",
    )
    group.addoption(
        "--agent-eval-k", type=int, default=1, help="Trials per task; all must succeed."
    )


def pytest_collect_file(file_path: Path, parent: pytest.Collector) -> pytest.Collector | None:
    if parent.config.getoption("agent_eval") and file_path.suffix == ".yaml":
        return TaskFile.from_parent(parent, path=file_path)
    return None


def _agent(config: pytest.Config) -> Agent:
    if _AGENT not in config.stash:
        from agent_eval.llm import get_provider

        config.stash[_AGENT] = build_agent(config.getoption("agent_eval"), get_provider)
    return config.stash[_AGENT]


class TaskFailedError(Exception):
    """At least one trial of the task did not succeed."""

    def __init__(self, task: TaskSpec, failed: list[TrialResult], k: int) -> None:
        super().__init__(f"{task.id}: {len(failed)} of {k} trials failed")
        self.task = task
        self.failed = failed
        self.k = k


class TaskFile(pytest.File):
    def collect(self) -> Iterator[TaskItem]:
        task = load_task(self.path)
        yield TaskItem.from_parent(self, name=task.id, task=task)


class TaskItem(pytest.Item):
    def __init__(self, *, task: TaskSpec, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.task = task

    def runtest(self) -> None:
        k = int(self.config.getoption("agent_eval_k"))
        agent = _agent(self.config)
        results = [run_trial(agent, self.task, trial) for trial in range(1, k + 1)]
        failed = [r for r in results if not r.metrics.success]
        if failed:
            raise TaskFailedError(self.task, failed, k)

    def repr_failure(self, excinfo: pytest.ExceptionInfo[BaseException], style: Any = None) -> Any:
        error = excinfo.value
        if not isinstance(error, TaskFailedError):
            return super().repr_failure(excinfo, style)
        lines = [str(error)]
        for result in error.failed:
            trajectory = result.trajectory
            lines.append(f"trial {result.trial} (stop: {trajectory.stop_reason}):")
            lines += [
                f"  step {s.index}: {s.tool}({s.arguments}) -> {s.output[:120]}"
                + (" [FORBIDDEN]" if s.forbidden and not s.blocked else "")
                for s in trajectory.steps
            ]
            lines += [
                f"  failed check: {c.check.op} {c.check.path or 'answer'} "
                f"(expected {c.check.value!r}, got {c.actual!r})"
                for c in [*result.checks, *result.answer_checks]
                if not c.passed
            ]
            if trajectory.error:
                lines.append(f"  error: {trajectory.error}")
        return "\n".join(lines)

    def reportinfo(self) -> tuple[Path, None, str]:
        return self.path, None, f"agent-eval: {self.task.id}"
