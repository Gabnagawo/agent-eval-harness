"""Verification suite for all tasks in the agent-eval-harness benchmark.

Tests that every YAML task:
1. Loads cleanly via load_tasks / load_task.
2. Adheres to folder and naming conventions (family == parent folder, id starts with family).
3. Replays reference solution with 100% success, step_ratio == 1.0, tool_precision == 1.0, and tool_recall == 1.0.
4. Fails under a noop agent (cannot succeed without taking action).
5. Satisfies suite-level count, family, and trap distribution constraints.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from agent_eval.tasks import load_task, load_tasks
from agent_eval.testing import replay_noop, replay_reference

TASKS_DIR = Path(__file__).resolve().parents[2] / "tasks"
ALL_TASK_PATHS = sorted(TASKS_DIR.rglob("*.yaml"))
ALL_TASK_IDS = [p.stem for p in ALL_TASK_PATHS]


@pytest.mark.parametrize("path", ALL_TASK_PATHS, ids=ALL_TASK_IDS)
def test_task_replay_and_conventions(path: Path) -> None:
    task = load_task(path)

    # Naming and folder conventions
    expected_family = path.parent.name
    assert task.family == expected_family, (
        f"Task family '{task.family}' does not match folder '{expected_family}'"
    )
    assert task.id.startswith(f"{task.family}-"), (
        f"Task id '{task.id}' does not start with '{task.family}-'"
    )

    # Reference solution replay must pass with perfect metrics
    ref_result = replay_reference(task)
    assert ref_result.metrics.success is True, f"Reference replay failed for {task.id}"
    assert ref_result.metrics.step_ratio == 1.0, (
        f"step_ratio is {ref_result.metrics.step_ratio}, expected 1.0"
    )
    assert ref_result.metrics.tool_precision == 1.0, (
        f"tool_precision is {ref_result.metrics.tool_precision}, expected 1.0"
    )
    assert ref_result.metrics.tool_recall == 1.0, (
        f"tool_recall is {ref_result.metrics.tool_recall}, expected 1.0"
    )

    # Noop agent must fail
    noop_result = replay_noop(task)
    assert noop_result.metrics.success is False, f"Replay noop unexpectedly succeeded for {task.id}"


def test_task_suite_counts_and_distributions() -> None:
    tasks = load_tasks(TASKS_DIR)
    assert len(tasks) == 32, f"Expected exactly 32 tasks in suite, got {len(tasks)}"

    # Family distribution
    family_counts = Counter(t.family for t in tasks)
    assert family_counts["files"] == 7, f"Expected 7 files tasks, got {family_counts['files']}"
    assert family_counts["tracker"] == 7, (
        f"Expected 7 tracker tasks, got {family_counts['tracker']}"
    )
    assert family_counts["calendar"] == 5, (
        f"Expected 5 calendar tasks, got {family_counts['calendar']}"
    )
    assert family_counts["api"] == 7, f"Expected 7 api tasks, got {family_counts['api']}"
    assert family_counts["triage"] == 6, f"Expected 6 triage tasks, got {family_counts['triage']}"

    # Trap distribution
    trap_counts = Counter(tr for t in tasks for tr in t.traps)
    assert trap_counts["tool_error"] >= 4, f"tool_error trap count {trap_counts['tool_error']} < 4"
    assert trap_counts["prompt_injection"] >= 4, (
        f"prompt_injection trap count {trap_counts['prompt_injection']} < 4"
    )
    assert trap_counts["destructive"] >= 4, (
        f"destructive trap count {trap_counts['destructive']} < 4"
    )
    assert trap_counts["ambiguous"] >= 3, f"ambiguous trap count {trap_counts['ambiguous']} < 3"

    # Control group (tasks without traps)
    control_tasks = [t for t in tasks if not t.traps]
    assert 8 <= len(control_tasks) <= 16, (
        f"Expected approximately 10-14 control tasks without traps, got {len(control_tasks)}"
    )

    # Optimal steps difficulty range
    steps = [t.optimal_steps for t in tasks]
    assert min(steps) >= 1
    assert max(steps) <= 8
