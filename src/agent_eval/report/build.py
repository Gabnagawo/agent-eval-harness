"""Build the JSON report of a run, and the A vs. B comparison of two reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from agent_eval.metrics import aggregate
from agent_eval.models import (
    Aggregate,
    Comparison,
    MetricDelta,
    RunInfo,
    RunReport,
    TaskDelta,
    TaskResult,
    TaskSpec,
    TrialMetrics,
    TrialResult,
)

COMPARED_METRICS = (
    "success_rate",
    "pass_k",
    "goal_met_rate",
    "forbidden_action_rate",
    "valid_call_rate",
    "tool_precision",
    "tool_recall",
    "mean_steps",
    "mean_step_ratio",
    "mean_llm_calls",
    "mean_latency_s",
    "prompt_tokens",
    "completion_tokens",
)


def build_report(
    info: RunInfo, tasks: Sequence[TaskSpec], results: Mapping[str, Sequence[TrialResult]]
) -> RunReport:
    by_task: dict[str, list[TrialMetrics]] = {
        task.id: [r.metrics for r in results.get(task.id, [])] for task in tasks
    }

    def subset(keep: Sequence[TaskSpec]) -> Aggregate:
        return aggregate({t.id: by_task[t.id] for t in keep})

    families = sorted({t.family for t in tasks})
    traps = sorted({trap for t in tasks for trap in t.traps})
    return RunReport(
        run=info,
        summary=subset(tasks),
        families={f: subset([t for t in tasks if t.family == f]) for f in families},
        traps={trap: subset([t for t in tasks if trap in t.traps]) for trap in traps},
        tasks=[
            TaskResult(
                task_id=task.id,
                family=task.family,
                title=task.title,
                prompt=task.prompt,
                traps=task.traps,
                optimal_steps=task.optimal_steps,
                max_steps=task.max_steps,
                expected_tools=task.expected_tools,
                aggregate=subset([task]),
                trials=list(results.get(task.id, [])),
            )
            for task in tasks
        ],
    )


def _deltas(a: Aggregate, b: Aggregate) -> list[MetricDelta]:
    rows: list[MetricDelta] = []
    for metric in COMPARED_METRICS:
        left, right = getattr(a, metric), getattr(b, metric)
        delta = None if left is None or right is None else right - left
        rows.append(MetricDelta(metric=metric, a=left, b=right, delta=delta))
    return rows


def compare(a: RunReport, b: RunReport) -> Comparison:
    """Compare two runs on the tasks they share. Deltas are ``b - a``."""
    a_tasks = {t.task_id: t for t in a.tasks}
    b_tasks = {t.task_id: t for t in b.tasks}
    shared = [task_id for task_id in a_tasks if task_id in b_tasks]

    def agg(tasks: Mapping[str, TaskResult], ids: Sequence[str]) -> Aggregate:
        return aggregate({i: [r.metrics for r in tasks[i].trials] for i in ids})

    families = sorted({a_tasks[i].family for i in shared})
    improved: list[TaskDelta] = []
    regressed: list[TaskDelta] = []
    for task_id in shared:
        left = a_tasks[task_id].aggregate.success_rate
        right = b_tasks[task_id].aggregate.success_rate
        if left == right:
            continue
        row = TaskDelta(
            task_id=task_id,
            family=a_tasks[task_id].family,
            a_success_rate=left,
            b_success_rate=right,
        )
        (improved if right > left else regressed).append(row)
    return Comparison(
        a=a.run,
        b=b.run,
        shared_tasks=len(shared),
        summary=_deltas(agg(a_tasks, shared), agg(b_tasks, shared)),
        families={
            f: _deltas(
                agg(a_tasks, [i for i in shared if a_tasks[i].family == f]),
                agg(b_tasks, [i for i in shared if a_tasks[i].family == f]),
            )
            for f in families
        },
        improved=improved,
        regressed=regressed,
        unchanged=len(shared) - len(improved) - len(regressed),
    )
