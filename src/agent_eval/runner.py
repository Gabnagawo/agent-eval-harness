"""Run agents on tasks: one trial, or a whole run with k trials per task and a checkpoint."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_eval import __version__
from agent_eval.agents.base import Agent, AgentResult
from agent_eval.llm.base import LLMError
from agent_eval.metrics import score_trial
from agent_eval.models import RunInfo, RunReport, StopReason, TaskSpec, Trajectory, TrialResult
from agent_eval.report.build import build_report
from agent_eval.world import World

ProgressFn = Callable[[TaskSpec, TrialResult], None]


def run_trial(agent: Agent, task: TaskSpec, trial: int = 1) -> TrialResult:
    """One agent, one task, one fresh world. Agent crashes become a failed trial, not a crash."""
    world = World(task)
    started = time.perf_counter()
    try:
        result = agent.run(task, world)
    except Exception as exc:
        # An agent that crashes mid-task is a result worth reporting, with what it did so far.
        reason: StopReason = "llm_error" if isinstance(exc, LLMError) else "agent_error"
        result = AgentResult(stop_reason=reason, error=f"{type(exc).__name__}: {exc}")
        usage = getattr(agent, "usage", None)
        if isinstance(usage, AgentResult):
            result.llm_calls = usage.llm_calls
            result.prompt_tokens = usage.prompt_tokens
            result.completion_tokens = usage.completion_tokens
            result.plan = usage.plan
    trajectory = Trajectory(
        task_id=task.id,
        agent=agent.name,
        trial=trial,
        steps=world.steps,
        plan=result.plan,
        final_answer=result.final_answer,
        stop_reason=result.stop_reason,
        error=result.error,
        llm_calls=result.llm_calls,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        latency_s=time.perf_counter() - started,
    )
    return score_trial(task, world, trajectory)


def _load_checkpoint(path: Path, agent: str) -> dict[tuple[str, int], TrialResult]:
    done: dict[tuple[str, int], TrialResult] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            result = TrialResult.model_validate(record["result"])
        except (ValueError, KeyError):
            continue  # a line cut short by an interrupted run
        if record.get("agent") == agent:
            done[(record["task_id"], result.trial)] = result
    return done


def run(
    agent: Agent,
    tasks: Sequence[TaskSpec],
    *,
    k: int = 1,
    provider: str = "none",
    model: str = "none",
    options: dict[str, Any] | None = None,
    checkpoint: Path | None = None,
    progress: ProgressFn | None = None,
    client_stats: Callable[[], dict[str, int]] | None = None,
) -> RunReport:
    """k trials of every task. With ``checkpoint``, finished trials are saved and reused."""
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    started = time.perf_counter()
    done = _load_checkpoint(checkpoint, agent.name) if checkpoint else {}
    if checkpoint:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[TrialResult]] = {}
    for task in tasks:
        results[task.id] = []
        for trial in range(1, k + 1):
            result = done.get((task.id, trial))
            if result is None:
                result = run_trial(agent, task, trial)
                if checkpoint:
                    record = {
                        "agent": agent.name,
                        "task_id": task.id,
                        "result": result.model_dump(mode="json"),
                    }
                    with checkpoint.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            results[task.id].append(result)
            if progress:
                progress(task, result)
    info = RunInfo(
        agent=agent.name,
        provider=provider,
        model=model,
        k=k,
        started_at=started_at,
        duration_s=time.perf_counter() - started,
        harness_version=__version__,
        options=options or {},
        client_stats=client_stats() if client_stats else {},
    )
    return build_report(info, tasks, results)
