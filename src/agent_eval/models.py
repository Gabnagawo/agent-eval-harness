"""Core contracts: tasks, tools, trajectories, metrics and the report schema.

Everything that crosses a module boundary (task YAML, tool calls, the JSON report) is a
pydantic model defined here. ``docs/CONTRACTS.md`` is the prose version of this file.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1"

CheckOp = Literal[
    "exists",
    "absent",
    "equals",
    "not_equals",
    "contains",
    "not_contains",
    "matches",
    "length",
    "gte",
    "lte",
    "in",
]
Trap = Literal["ambiguous", "tool_error", "prompt_injection", "destructive"]
StopReason = Literal["answer", "max_steps", "llm_error", "agent_error"]


class _Strict(BaseModel):
    """Base for models read from YAML: a typo in a key is an error, not a silent default."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- tools


class ToolSpec(_Strict):
    """A tool an environment exposes. ``parameters`` is a JSON Schema of type ``object``."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}, "additionalProperties": False}
    )
    destructive: bool = False

    def as_function(self) -> dict[str, Any]:
        """The OpenAI/Ollama ``tools`` entry for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolResult(_Strict):
    """What a tool call returns. Domain errors are data (``ok=False``), never exceptions."""

    ok: bool
    result: Any = None
    error: str | None = None


# --------------------------------------------------------------------------- tasks


class Check(_Strict):
    """One deterministic assertion.

    In ``success`` the ``path`` is a JMESPath expression over the final world snapshot
    (``{"<env kind>": <env snapshot>, ...}``). In ``answer`` the ``path`` is omitted and the
    check applies to the agent's final answer text.
    """

    op: CheckOp
    path: str | None = None
    value: Any = None
    ignore_case: bool = False
    description: str = ""


class ForbiddenAction(_Strict):
    """A tool call the agent must not make.

    ``args`` narrows the match: string patterns use shell wildcards (``/prod/*``), any other
    value must be equal. With no ``args`` every call to ``tool`` is forbidden.
    """

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class Fault(_Strict):
    """A scripted tool failure: the Nth call(s) to ``tool`` (1-based) return ``error``.

    The call does not reach the environment, so a retry of the same call succeeds.
    """

    tool: str
    calls: list[int] = Field(min_length=1)
    error: str


class ReferenceCall(_Strict):
    """One tool call of a task's reference solution."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class TaskSpec(_Strict):
    """A reproducible task. One YAML file per task under ``tasks/<family>/``."""

    id: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    family: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    title: str
    prompt: str
    envs: dict[str, dict[str, Any]] = Field(min_length=1)
    seed: int = 0
    faults: list[Fault] = Field(default_factory=list)
    tools: list[str] | None = None
    max_steps: int = Field(ge=1)
    optimal_steps: int = Field(ge=0)
    expected_tools: list[str] = Field(default_factory=list)
    success: list[Check] = Field(default_factory=list)
    answer: list[Check] = Field(default_factory=list)
    forbidden: list[ForbiddenAction] = Field(default_factory=list)
    traps: list[Trap] = Field(default_factory=list)
    reference_solution: list[ReferenceCall] = Field(default_factory=list)
    reference_answer: str | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> TaskSpec:
        if not self.success and not self.answer:
            raise ValueError("a task needs at least one 'success' or 'answer' check")
        if self.optimal_steps > self.max_steps:
            raise ValueError("optimal_steps cannot exceed max_steps")
        if len(self.reference_solution) != self.optimal_steps:
            raise ValueError(
                f"optimal_steps is {self.optimal_steps} but reference_solution has "
                f"{len(self.reference_solution)} calls"
            )
        if any(c.path is None for c in self.success):
            raise ValueError("every 'success' check needs a 'path'")
        if any(c.path is not None for c in self.answer):
            raise ValueError("'answer' checks apply to the final answer and take no 'path'")
        if self.answer and self.reference_answer is None:
            raise ValueError("a task with 'answer' checks needs a 'reference_answer'")
        return self


# --------------------------------------------------------------------------- trajectories


class Step(BaseModel):
    """One tool call the agent attempted, and what happened to it."""

    index: int
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    known_tool: bool = True
    valid_args: bool = True
    validation_error: str | None = None
    blocked: bool = False
    """Stopped by the agent's guardrail layer before reaching the world."""
    block_reason: str | None = None
    fault: bool = False
    """Answered by a scripted fault instead of the environment."""
    executed: bool = False
    """The environment ran the call (it may still have returned ``ok=False``)."""
    forbidden: bool = False
    forbidden_reason: str | None = None
    ok: bool = False
    output: str = ""
    """Exactly what the agent saw as the tool result."""


class Trajectory(BaseModel):
    """Everything one agent did in one trial of one task."""

    task_id: str
    agent: str
    trial: int
    steps: list[Step] = Field(default_factory=list)
    plan: str | None = None
    final_answer: str | None = None
    stop_reason: StopReason = "answer"
    error: str | None = None
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


# --------------------------------------------------------------------------- metrics


class CheckResult(BaseModel):
    check: Check
    passed: bool
    actual: Any = None


class TrialMetrics(BaseModel):
    """Numbers for one trial. Rates that are undefined for the trial are ``None``."""

    success: bool
    """``goal_met and safe``."""
    goal_met: bool
    """Every ``success`` and ``answer`` check passed."""
    safe: bool
    """No forbidden action reached the world."""
    steps: int
    valid_calls: int
    invalid_calls: int
    valid_call_rate: float | None
    tool_precision: float | None
    tool_recall: float | None
    step_ratio: float | None
    """``steps / optimal_steps`` for a successful trial with ``optimal_steps > 0``."""
    forbidden_actions: int
    blocked_calls: int
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class TrialResult(BaseModel):
    trial: int
    metrics: TrialMetrics
    checks: list[CheckResult] = Field(default_factory=list)
    answer_checks: list[CheckResult] = Field(default_factory=list)
    trajectory: Trajectory


class Aggregate(BaseModel):
    """Metrics over a set of trials (one task, one family, one trap, or the whole run)."""

    tasks: int
    trials: int
    success_rate: float
    pass_k: float
    """pass^k: share of tasks whose k trials ALL succeeded."""
    goal_met_rate: float
    forbidden_action_rate: float
    """Share of trials with at least one forbidden action."""
    valid_call_rate: float | None
    """Micro-average: valid calls / all calls."""
    tool_precision: float | None
    tool_recall: float | None
    mean_steps: float
    mean_step_ratio: float | None
    blocked_calls: int
    mean_llm_calls: float
    prompt_tokens: int
    completion_tokens: int
    mean_latency_s: float
    total_latency_s: float


class TaskResult(BaseModel):
    task_id: str
    family: str
    title: str
    prompt: str
    traps: list[Trap] = Field(default_factory=list)
    optimal_steps: int
    max_steps: int
    expected_tools: list[str] = Field(default_factory=list)
    aggregate: Aggregate
    trials: list[TrialResult]


class RunInfo(BaseModel):
    agent: str
    provider: str
    model: str
    k: int
    started_at: str
    """UTC, ISO 8601."""
    duration_s: float
    harness_version: str
    options: dict[str, Any] = Field(default_factory=dict)
    client_stats: dict[str, int] = Field(default_factory=dict)


class RunReport(BaseModel):
    """The JSON report of one run: one agent, a set of tasks, k trials each."""

    schema_version: str = SCHEMA_VERSION
    run: RunInfo
    summary: Aggregate
    families: dict[str, Aggregate]
    traps: dict[str, Aggregate]
    tasks: list[TaskResult]


class MetricDelta(BaseModel):
    metric: str
    a: float | None
    b: float | None
    delta: float | None
    """``b - a``; ``None`` when either side is undefined."""


class TaskDelta(BaseModel):
    task_id: str
    family: str
    a_success_rate: float
    b_success_rate: float


class Comparison(BaseModel):
    """A vs. B over the tasks both runs share."""

    schema_version: str = SCHEMA_VERSION
    a: RunInfo
    b: RunInfo
    shared_tasks: int
    summary: list[MetricDelta]
    families: dict[str, list[MetricDelta]]
    improved: list[TaskDelta]
    regressed: list[TaskDelta]
    unchanged: int
