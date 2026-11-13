"""Load and validate task YAML files."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml
from pydantic import ValidationError

from agent_eval.checks import CheckError, evaluate_all
from agent_eval.models import TaskSpec
from agent_eval.world import TaskConfigError, World


class TaskLoadError(ValueError):
    """A task file that cannot be used, with the file name in the message."""


def default_tasks_dir() -> Path:
    """``./tasks`` when it exists (a checkout), else the copy shipped inside the wheel."""
    local = Path("tasks")
    return local if local.is_dir() else Path(__file__).parent / "_tasks"


def validate_task(task: TaskSpec) -> None:
    """Semantic validation: environments and tools exist, and every check can be evaluated."""
    world = World(task)
    evaluate_all(task.success, world.snapshot())
    evaluate_all(task.answer, task.reference_answer or "")


def load_task(path: Path) -> TaskSpec:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        task = TaskSpec.model_validate(raw)
        validate_task(task)
    except (yaml.YAMLError, ValidationError, TaskConfigError, CheckError, KeyError) as exc:
        raise TaskLoadError(f"{path}: {exc}") from exc
    return task


def load_tasks(
    root: Path | None = None,
    *,
    families: Iterable[str] | None = None,
    ids: Iterable[str] | None = None,
) -> list[TaskSpec]:
    """Every task under ``root`` (recursively), sorted by family then id, optionally filtered."""
    root = root or default_tasks_dir()
    if not root.exists():
        raise TaskLoadError(f"{root}: no such file or directory")
    paths = [root] if root.is_file() else sorted(root.rglob("*.yaml"))
    tasks: dict[str, TaskSpec] = {}
    for path in paths:
        task = load_task(path)
        if task.id in tasks:
            raise TaskLoadError(f"{path}: duplicate task id '{task.id}'")
        tasks[task.id] = task
    wanted_families = set(families or [])
    wanted_ids = set(ids or [])
    missing = wanted_ids - set(tasks)
    if missing:
        raise TaskLoadError(f"unknown task id(s): {', '.join(sorted(missing))}")
    selected = [
        t
        for t in tasks.values()
        if (not wanted_families or t.family in wanted_families)
        and (not wanted_ids or t.id in wanted_ids)
    ]
    return sorted(selected, key=lambda t: (t.family, t.id))
