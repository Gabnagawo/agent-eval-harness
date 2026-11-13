"""The world an agent acts in during one trial: the task's environments behind one tool API.

The world is the single choke point for tool calls. It validates arguments against the tool
schema, applies scripted faults, flags forbidden actions, dispatches to the owning
environment and records a :class:`~agent_eval.models.Step` for every attempt.
"""

from __future__ import annotations

import fnmatch
import json
from typing import Any

from jsonschema import Draft202012Validator

from agent_eval.envs.base import Environment, get_environment
from agent_eval.models import ForbiddenAction, Step, TaskSpec, ToolResult, ToolSpec


class TaskConfigError(ValueError):
    """The task references tools or environments that do not exist."""


def render_result(result: ToolResult) -> str:
    """The text an agent sees for a tool result."""
    payload: dict[str, Any] = {"ok": result.ok}
    if result.ok:
        payload["result"] = result.result
    else:
        payload["error"] = result.error
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def matches_forbidden(rule: ForbiddenAction, tool: str, arguments: dict[str, Any]) -> bool:
    if rule.tool != tool:
        return False
    for key, pattern in rule.args.items():
        if key not in arguments:
            return False
        actual = arguments[key]
        if isinstance(pattern, str) and isinstance(actual, str):
            if not fnmatch.fnmatchcase(actual, pattern):
                return False
        elif actual != pattern:
            return False
    return True


class World:
    """Environments + tool catalogue + step recorder for one trial."""

    def __init__(self, task: TaskSpec) -> None:
        self.task = task
        self.envs: dict[str, Environment] = {
            kind: get_environment(kind)(state, seed=task.seed) for kind, state in task.envs.items()
        }
        self._owner: dict[str, Environment] = {}
        self._specs: dict[str, ToolSpec] = {}
        for env in self.envs.values():
            for spec in env.tools():
                if spec.name in self._specs:
                    raise TaskConfigError(f"tool '{spec.name}' is defined by two environments")
                self._specs[spec.name] = spec
                self._owner[spec.name] = env
        if task.tools is not None:
            self._require_known(task.tools, "tools")
            self._specs = {name: self._specs[name] for name in task.tools}
        self._require_known(task.expected_tools, "expected_tools")
        self._require_known([f.tool for f in task.faults], "faults")
        self._require_known([f.tool for f in task.forbidden], "forbidden")
        self._require_known([c.tool for c in task.reference_solution], "reference_solution")
        self._validators = {
            name: Draft202012Validator(spec.parameters) for name, spec in self._specs.items()
        }
        self._calls_per_tool: dict[str, int] = {}
        self.steps: list[Step] = []

    def _require_known(self, names: list[str], where: str) -> None:
        unknown = sorted(set(names) - set(self._specs))
        if unknown:
            raise TaskConfigError(
                f"task '{self.task.id}': {where} names unknown tools: {', '.join(unknown)}"
            )

    # ------------------------------------------------------------------ catalogue

    def tool_specs(self) -> list[ToolSpec]:
        """The tools this task exposes to the agent, in a stable order."""
        return list(self._specs.values())

    def spec(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def validation_error(self, name: str, arguments: dict[str, Any]) -> str | None:
        """Why ``arguments`` do not fit the tool's schema, or ``None`` if they do."""
        errors = sorted(self._validators[name].iter_errors(arguments), key=lambda e: list(e.path))
        if not errors:
            return None
        first = errors[0]
        where = ".".join(str(p) for p in first.path)
        return f"{where}: {first.message}" if where else first.message

    @property
    def steps_left(self) -> int:
        return self.task.max_steps - len(self.steps)

    # ------------------------------------------------------------------ calls

    def call(self, name: str, arguments: dict[str, Any]) -> Step:
        """Attempt one tool call and record it. Never raises for anything the agent did."""
        step = Step(index=len(self.steps) + 1, tool=name, arguments=arguments)
        self.steps.append(step)
        if name not in self._specs:
            step.known_tool = False
            step.valid_args = False
            return self._finish(step, ToolResult(ok=False, error=f"unknown tool '{name}'"))
        problem = self.validation_error(name, arguments)
        if problem is not None:
            step.valid_args = False
            step.validation_error = problem
            return self._finish(
                step, ToolResult(ok=False, error=f"invalid arguments for {name}: {problem}")
            )
        for rule in self.task.forbidden:
            if matches_forbidden(rule, name, arguments):
                step.forbidden = True
                step.forbidden_reason = rule.reason or f"{name} is forbidden in this task"
                break
        count = self._calls_per_tool.get(name, 0) + 1
        self._calls_per_tool[name] = count
        for fault in self.task.faults:
            if fault.tool == name and count in fault.calls:
                step.fault = True
                return self._finish(step, ToolResult(ok=False, error=fault.error))
        step.executed = True
        return self._finish(step, self._owner[name].call(name, arguments))

    def record_blocked(self, name: str, arguments: dict[str, Any], reason: str) -> Step:
        """Record a call an agent's guardrail refused to send. The world state is untouched."""
        step = Step(index=len(self.steps) + 1, tool=name, arguments=arguments)
        self.steps.append(step)
        step.known_tool = name in self._specs
        if step.known_tool:
            step.validation_error = self.validation_error(name, arguments)
        step.valid_args = step.known_tool and step.validation_error is None
        step.blocked = True
        step.block_reason = reason
        return self._finish(step, ToolResult(ok=False, error=f"blocked by guardrail: {reason}"))

    @staticmethod
    def _finish(step: Step, result: ToolResult) -> Step:
        step.ok = result.ok
        step.output = render_result(result)
        return step

    def snapshot(self) -> dict[str, Any]:
        """``{"<env kind>": <env snapshot>}``: the data ``success`` checks run against."""
        return {kind: env.snapshot() for kind, env in self.envs.items()}
