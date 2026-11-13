"""Agents without a model: they validate the harness and the task suite, and feed the tests.

* :class:`OracleAgent` replays each task's reference solution. It must score 100%.
* :class:`NoopAgent` answers immediately and touches nothing. It must fail every task that
  requires a state change.
* :class:`ScriptedAgent` replays calls you give it, per task.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from agent_eval.agents.base import AgentResult
from agent_eval.models import ReferenceCall, TaskSpec
from agent_eval.world import World


def _replay(world: World, calls: Sequence[ReferenceCall]) -> bool:
    """Send ``calls`` in order. Returns ``False`` when the step budget ran out first."""
    for call in calls:
        if world.steps_left <= 0:
            return False
        world.call(call.tool, dict(call.args))
    return True


class OracleAgent:
    name = "oracle"

    def run(self, task: TaskSpec, world: World) -> AgentResult:
        _replay(world, task.reference_solution)
        return AgentResult(final_answer=task.reference_answer or "Done.")


class NoopAgent:
    name = "noop"

    def run(self, task: TaskSpec, world: World) -> AgentResult:
        return AgentResult(final_answer="Done.")


class ScriptedAgent:
    """Replays ``scripts[task.id]`` and answers ``answers[task.id]`` (default ``"Done."``)."""

    def __init__(
        self,
        scripts: Mapping[str, Sequence[ReferenceCall]],
        answers: Mapping[str, str] | None = None,
        *,
        name: str = "scripted",
    ) -> None:
        self.name = name
        self._scripts = scripts
        self._answers = answers or {}

    def run(self, task: TaskSpec, world: World) -> AgentResult:
        finished = _replay(world, self._scripts.get(task.id, ()))
        if not finished:
            return AgentResult(stop_reason="max_steps")
        return AgentResult(final_answer=self._answers.get(task.id, "Done."))
