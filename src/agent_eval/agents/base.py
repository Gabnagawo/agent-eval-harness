"""The Agent interface: anything that can work on a task through a :class:`World`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agent_eval.models import StopReason, TaskSpec
from agent_eval.world import World


@dataclass
class AgentResult:
    """What an agent hands back when it stops. Tool calls are recorded by the world."""

    final_answer: str | None = None
    stop_reason: StopReason = "answer"
    error: str | None = None
    plan: str | None = None
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Agent(Protocol):
    """An agent under test.

    A fair agent uses only ``task.prompt``, ``world.tool_specs()``, ``world.call()`` and
    ``world.steps_left``. It must stop when ``world.steps_left`` reaches zero. Checks,
    forbidden rules and the reference solution are for the harness, not for the agent.
    """

    name: str

    def run(self, task: TaskSpec, world: World) -> AgentResult: ...
