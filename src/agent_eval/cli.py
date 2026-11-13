"""Command line: ``agent-eval list | run | compare | report``."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from agent_eval.agents import AGENT_NAMES, MODEL_FREE, build_agent
from agent_eval.config import Settings
from agent_eval.llm import LLMError, LLMProvider, get_provider
from agent_eval.models import Aggregate, Comparison, RunReport, TaskSpec, TrialResult
from agent_eval.report import compare as compare_reports
from agent_eval.runner import run as run_tasks
from agent_eval.tasks import TaskLoadError, load_tasks

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Evaluate tool-using AI agents on reproducible tasks.",
)

TasksOpt = Annotated[
    Path | None, typer.Option("--tasks", help="Task file or directory (default: ./tasks).")
]
FamilyOpt = Annotated[list[str] | None, typer.Option("--family", help="Only this family.")]


def _fail(message: str) -> typer.Exit:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    return typer.Exit(2)


def _load(tasks: Path | None, families: list[str] | None, ids: list[str] | None) -> list[TaskSpec]:
    try:
        selected = load_tasks(tasks, families=families, ids=ids)
    except TaskLoadError as exc:
        raise _fail(str(exc)) from exc
    if not selected:
        raise _fail("no task matches the selection")
    return selected


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


SUMMARY_ROWS: tuple[tuple[str, str, bool], ...] = (
    ("success_rate", "Success rate", True),
    ("pass_k", "pass^k", True),
    ("goal_met_rate", "Goal met", True),
    ("forbidden_action_rate", "Forbidden-action rate", True),
    ("valid_call_rate", "Valid tool calls", True),
    ("tool_precision", "Tool precision", True),
    ("tool_recall", "Tool recall", True),
    ("mean_steps", "Mean steps", False),
    ("mean_step_ratio", "Steps / optimal (successes)", False),
    ("mean_llm_calls", "Mean LLM calls", False),
    ("mean_latency_s", "Mean latency (s)", False),
    ("prompt_tokens", "Prompt tokens", False),
    ("completion_tokens", "Completion tokens", False),
)


def _fmt(value: float | None, percent: bool) -> str:
    if percent:
        return _pct(value)
    if isinstance(value, int):
        return str(value)
    return _num(value)


def summary_table(report: RunReport) -> str:
    """Markdown summary of one run: overall, then one row per family and per trap."""
    run = report.run
    lines = [
        f"agent: {run.agent} | model: {run.model} ({run.provider}) | "
        f"tasks: {report.summary.tasks} | k: {run.k} | trials: {report.summary.trials} | "
        f"duration: {run.duration_s:.0f}s",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    lines += [
        f"| {label} | {_fmt(getattr(report.summary, key), percent)} |"
        for key, label, percent in SUMMARY_ROWS
    ]

    def breakdown(title: str, groups: dict[str, Aggregate]) -> list[str]:
        if not groups:
            return []
        rows = [
            "",
            f"| {title} | Tasks | Success | pass^k | Forbidden | Valid calls | Steps/optimal |",
            "|---|---|---|---|---|---|---|",
        ]
        rows += [
            f"| {name} | {agg.tasks} | {_pct(agg.success_rate)} | {_pct(agg.pass_k)} | "
            f"{_pct(agg.forbidden_action_rate)} | {_pct(agg.valid_call_rate)} | "
            f"{_num(agg.mean_step_ratio)} |"
            for name, agg in groups.items()
        ]
        return rows

    lines += breakdown("Family", report.families)
    lines += breakdown("Trap", report.traps)
    return "\n".join(lines)


def comparison_table(comparison: Comparison) -> str:
    """Markdown table of A vs. B."""
    labels = {key: (label, percent) for key, label, percent in SUMMARY_ROWS}
    a, b = comparison.a, comparison.b
    lines = [
        f"A: {a.agent} ({a.model}) | B: {b.agent} ({b.model}) | "
        f"shared tasks: {comparison.shared_tasks} | k: {a.k} vs {b.k}",
        "",
        f"| Metric | A: {a.agent} | B: {b.agent} | Delta (B - A) |",
        "|---|---|---|---|",
    ]
    for row in comparison.summary:
        label, percent = labels[row.metric]
        if row.delta is None:
            delta = "n/a"
        elif percent:
            delta = f"{row.delta * 100:+.1f} pp"
        else:
            delta = f"{row.delta:+.2f}"
        lines.append(f"| {label} | {_fmt(row.a, percent)} | {_fmt(row.b, percent)} | {delta} |")
    lines += [
        "",
        "| Family | Success A | Success B | pass^k A | pass^k B | Forbidden A | Forbidden B |",
        "|---|---|---|---|---|---|---|",
    ]
    for family, rows in comparison.families.items():
        by = {row.metric: row for row in rows}
        lines.append(
            f"| {family} | {_pct(by['success_rate'].a)} | {_pct(by['success_rate'].b)} | "
            f"{_pct(by['pass_k'].a)} | {_pct(by['pass_k'].b)} | "
            f"{_pct(by['forbidden_action_rate'].a)} | {_pct(by['forbidden_action_rate'].b)} |"
        )
    lines += [
        "",
        f"Tasks improved in B: {len(comparison.improved)} | regressed: "
        f"{len(comparison.regressed)} | unchanged: {comparison.unchanged}",
    ]
    for title, rows_ in (("improved", comparison.improved), ("regressed", comparison.regressed)):
        lines += [
            f"- {title}: {r.task_id} ({_pct(r.a_success_rate)} -> {_pct(r.b_success_rate)})"
            for r in rows_
        ]
    return "\n".join(lines)


def _write_html(path: Path, reports: list[RunReport], comparison: Comparison | None) -> bool:
    try:
        renderer = importlib.import_module("agent_eval.report.html")
    except ImportError:
        return False
    path.write_text(renderer.render_html(reports, comparison), encoding="utf-8")
    return True


def _read_report(path: Path) -> RunReport:
    try:
        return RunReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise _fail(f"{path}: not a run report ({str(exc).splitlines()[0]})") from exc


@app.command("list")
def list_tasks(tasks: TasksOpt = None, family: FamilyOpt = None) -> None:
    """List the tasks of the suite."""
    selected = _load(tasks, family, None)
    typer.echo("| Id | Family | Traps | Optimal | Max | Title |")
    typer.echo("|---|---|---|---|---|---|")
    for task in selected:
        traps = ", ".join(task.traps) or "-"
        typer.echo(
            f"| {task.id} | {task.family} | {traps} | {task.optimal_steps} | "
            f"{task.max_steps} | {task.title} |"
        )
    typer.echo(f"\n{len(selected)} tasks in {len({t.family for t in selected})} families")


@app.command()
def run(
    agent: Annotated[str, typer.Option("--agent", help=f"One of: {', '.join(AGENT_NAMES)}.")],
    tasks: TasksOpt = None,
    family: FamilyOpt = None,
    task: Annotated[list[str] | None, typer.Option("--task", help="Only this task id.")] = None,
    k: Annotated[int, typer.Option("-k", "--trials", min=1, help="Trials per task.")] = 1,
    out: Annotated[Path, typer.Option("--out", help="Output directory.")] = Path("out"),
    unload_every: Annotated[
        int | None,
        typer.Option(help="Ollama: unload the model every N calls (low-RAM machines)."),
    ] = None,
    resume: Annotated[
        bool, typer.Option("--resume", help="Reuse the trials already in the checkpoint.")
    ] = False,
    fail_under: Annotated[
        float | None,
        typer.Option(help="Exit 1 if the success rate is below this (0-1). Oracle: 1.0."),
    ] = None,
) -> None:
    """Run one agent on the tasks, k trials each, and write the JSON and HTML reports."""
    if agent not in AGENT_NAMES:
        raise _fail(f"unknown agent '{agent}' (choose from: {', '.join(AGENT_NAMES)})")
    selected = _load(tasks, family, task)
    settings = Settings()
    if unload_every is not None:
        settings.ollama_unload_every = unload_every
    provider: LLMProvider | None = None

    def provider_factory() -> LLMProvider:
        nonlocal provider
        provider = get_provider(settings)
        return provider

    try:
        chosen = build_agent(agent, provider_factory)
    except LLMError as exc:
        raise _fail(str(exc)) from exc

    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / f"{agent}.jsonl"
    if not resume and checkpoint.exists():
        checkpoint.unlink()

    total = len(selected) * k
    done = 0

    def progress(spec: TaskSpec, result: TrialResult) -> None:
        nonlocal done
        done += 1
        m = result.metrics
        verdict = "PASS" if m.success else "FAIL"
        flags = "".join(
            [
                " forbidden" if m.forbidden_actions else "",
                f" blocked={m.blocked_calls}" if m.blocked_calls else "",
                f" invalid={m.invalid_calls}" if m.invalid_calls else "",
                f" [{result.trajectory.stop_reason}]"
                if result.trajectory.stop_reason != "answer"
                else "",
            ]
        )
        typer.echo(
            f"[{done}/{total}] {verdict} {spec.id} trial {result.trial}: "
            f"{m.steps} steps (optimal {spec.optimal_steps}), {m.latency_s:.1f}s{flags}"
        )

    model_free = agent in MODEL_FREE
    report = run_tasks(
        chosen,
        selected,
        k=k,
        provider="none" if model_free else settings.llm_provider,
        model="none" if model_free else (settings.resolved_model() or "unknown"),
        options={}
        if model_free
        else {
            "temperature": settings.llm_temperature,
            "top_p": settings.llm_top_p,
            "top_k": settings.llm_top_k,
            "max_tokens": settings.llm_max_tokens,
            "num_ctx": settings.ollama_num_ctx,
            "unload_every": settings.ollama_unload_every,
        },
        checkpoint=checkpoint,
        progress=progress,
        client_stats=lambda: dict(getattr(provider, "stats", {})),
    )
    json_path = out / f"{agent}.json"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    typer.echo("")
    typer.echo(summary_table(report))
    typer.echo(f"\nJSON report: {json_path}")
    html_path = out / f"{agent}.html"
    if _write_html(html_path, [report], None):
        typer.echo(f"HTML report: {html_path}")

    threshold = 1.0 if fail_under is None and agent == "oracle" else fail_under
    if threshold is not None and report.summary.success_rate < threshold:
        typer.secho(
            f"success rate {_pct(report.summary.success_rate)} is below {_pct(threshold)}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)


@app.command()
def compare(
    a: Annotated[Path, typer.Argument(help="JSON report of run A.")],
    b: Annotated[Path, typer.Argument(help="JSON report of run B.")],
    out: Annotated[Path, typer.Option("--out", help="Output directory.")] = Path("out"),
    html: Annotated[bool, typer.Option("--html", help="Also write compare.html.")] = False,
) -> None:
    """Compare two runs (A vs. B) on the tasks they share."""
    report_a, report_b = _read_report(a), _read_report(b)
    comparison = compare_reports(report_a, report_b)
    table = comparison_table(comparison)
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(comparison.model_dump_json(indent=2), encoding="utf-8")
    (out / "compare.md").write_text(table + "\n", encoding="utf-8")
    typer.echo(table)
    typer.echo(f"\nJSON: {out / 'compare.json'}")
    if html and _write_html(out / "compare.html", [report_a, report_b], comparison):
        typer.echo(f"HTML: {out / 'compare.html'}")


@app.command()
def report(
    reports: Annotated[list[Path], typer.Argument(help="One JSON report, or two for A vs. B.")],
    out: Annotated[Path, typer.Option("--out", help="HTML file to write.")] = Path(
        "out/report.html"
    ),
) -> None:
    """Render the HTML report from one JSON report (or two, side by side)."""
    if len(reports) > 2:
        raise _fail("give one report, or two for a comparison")
    loaded = [_read_report(path) for path in reports]
    comparison = compare_reports(loaded[0], loaded[1]) if len(loaded) == 2 else None
    out.parent.mkdir(parents=True, exist_ok=True)
    if not _write_html(out, loaded, comparison):
        raise _fail("the HTML renderer is not available in this install")
    typer.echo(f"HTML report: {out}")
    typer.echo(json.dumps({"reports": len(loaded), "comparison": comparison is not None}))
