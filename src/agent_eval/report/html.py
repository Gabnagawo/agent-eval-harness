"""Self-contained HTML report for agent-eval-harness.

Generates a standalone, dependency-free HTML report from one or more RunReports,
optionally with a Comparison between runs. Works offline from file://, supports
light and dark color schemes, keyboard-accessible collapsible trajectories, and
escapes all untrusted run and tool outputs to prevent script injection.
"""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from typing import Any

from agent_eval.models import (
    Aggregate,
    CheckResult,
    Comparison,
    RunInfo,
    RunReport,
    Step,
    TaskDelta,
    TaskResult,
    TrialResult,
)

METRIC_LABELS: dict[str, str] = {
    "success_rate": "Success Rate",
    "pass_k": "Pass^k",
    "goal_met_rate": "Goal-Met Rate",
    "forbidden_action_rate": "Forbidden-Action Rate",
    "valid_call_rate": "Valid-Call Rate",
    "tool_precision": "Tool Precision",
    "tool_recall": "Tool Recall",
    "mean_steps": "Mean Steps",
    "mean_step_ratio": "Step Ratio",
    "mean_llm_calls": "Mean LLM Calls",
    "mean_latency_s": "Mean Latency (s)",
    "prompt_tokens": "Prompt Tokens",
    "completion_tokens": "Completion Tokens",
}

RATE_METRICS = {
    "success_rate",
    "pass_k",
    "goal_met_rate",
    "forbidden_action_rate",
    "valid_call_rate",
    "tool_precision",
    "tool_recall",
}

INT_METRICS = {"prompt_tokens", "completion_tokens"}


def _esc(val: Any) -> str:
    """Strictly HTML-escape any value for safe rendering."""
    if val is None:
        return ""
    return html.escape(str(val), quote=True)


def _fmt_rate(val: float | None) -> str:
    if val is None:
        return "n/a"
    return f"{val * 100:.1f}%"


def _fmt_float(val: float | None, precision: int = 2, suffix: str = "") -> str:
    if val is None:
        return "n/a"
    return f"{val:.{precision}f}{suffix}"


def _fmt_int(val: int | float | None) -> str:
    if val is None:
        return "n/a"
    return f"{int(val):,}"


def _fmt_delta(metric: str, delta: float | None) -> tuple[str, str]:
    """Return formatted delta string and a sentiment CSS class."""
    if delta is None:
        return "n/a", "delta-na"
    if abs(delta) < 1e-9:
        val_str = "0.0%" if metric in RATE_METRICS else "0.00"
        return val_str, "delta-zero"

    if metric in RATE_METRICS:
        sign = "+" if delta > 0 else ""
        text = f"{sign}{delta * 100:.1f}%"
        if metric == "forbidden_action_rate":
            cls = "delta-pos" if delta < 0 else "delta-neg"
        else:
            cls = "delta-pos" if delta > 0 else "delta-neg"
        return text, cls

    if metric in INT_METRICS:
        sign = "+" if delta > 0 else ""
        text = f"{sign}{int(delta):,}"
        cls = "delta-pos" if delta <= 0 else "delta-neutral"
        return text, cls

    sign = "+" if delta > 0 else ""
    text = f"{sign}{delta:.2f}"
    if metric in ("mean_steps", "mean_step_ratio", "mean_llm_calls", "mean_latency_s"):
        cls = "delta-pos" if delta < 0 else "delta-neutral"
    else:
        cls = "delta-pos" if delta > 0 else "delta-neg"
    return text, cls


def _format_metric_val(metric: str, val: float | int | None) -> str:
    if val is None:
        return "n/a"
    if metric in RATE_METRICS:
        return f"{float(val) * 100:.1f}%"
    if metric in INT_METRICS:
        return f"{int(val):,}"
    if metric == "mean_latency_s":
        return f"{float(val):.3f}s"
    return f"{float(val):.2f}"


def _render_svg_bar_chart(
    title: str,
    categories: Sequence[str],
    series: Sequence[tuple[str, Sequence[float | None], str]],
    *,
    height: int = 240,
    width: int = 600,
) -> str:
    """Render an inline, accessible SVG bar chart comparing categories."""
    if not categories:
        return f"""<div class="chart-container">
  <div class="chart-title">{_esc(title)}</div>
  <svg viewBox="0 0 {width} 90" width="100%" class="chart-svg" role="img" aria-label="{_esc(title)}">
    <text x="{width / 2}" y="50" text-anchor="middle" class="chart-empty">No data available</text>
  </svg>
</div>"""

    margin_top = 34
    margin_bottom = 44
    margin_left = 50
    margin_right = 20

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    n_cats = len(categories)
    n_series = len(series)
    slot_w = plot_w / n_cats

    # Gridlines and Y ticks (0%, 25%, 50%, 75%, 100%)
    ticks_svg: list[str] = []
    for tick_val in (0.0, 0.25, 0.50, 0.75, 1.00):
        y = margin_top + plot_h * (1.0 - tick_val)
        ticks_svg.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_w}" y2="{y:.1f}" class="chart-gridline" />'
        )
        ticks_svg.append(
            f'<text x="{margin_left - 8}" y="{y + 4:.1f}" text-anchor="end" class="chart-tick-label">{int(tick_val * 100)}%</text>'
        )

    # Bars calculation
    max_group_w = slot_w * 0.75
    gap_between_bars = 4 if n_series > 1 else 0
    bar_w = min(36.0, (max_group_w - (n_series - 1) * gap_between_bars) / n_series)
    total_group_w = n_series * bar_w + (n_series - 1) * gap_between_bars

    bars_svg: list[str] = []
    labels_svg: list[str] = []

    for ci, cat in enumerate(categories):
        center_x = margin_left + (ci + 0.5) * slot_w
        start_x = center_x - total_group_w / 2

        # Category label along X axis
        labels_svg.append(
            f'<text x="{center_x:.1f}" y="{margin_top + plot_h + 20}" text-anchor="middle" class="chart-cat-label">{_esc(cat)}</text>'
        )

        for si, (_s_name, s_vals, s_color) in enumerate(series):
            val = s_vals[ci] if ci < len(s_vals) else None
            bx = start_x + si * (bar_w + gap_between_bars)

            if val is not None:
                clamped = max(0.0, min(1.0, float(val)))
                bar_h = clamped * plot_h
                by = margin_top + plot_h - bar_h
                bars_svg.append(
                    f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{s_color}" rx="3" class="chart-bar"><title>{_esc(cat)}: {clamped * 100:.1f}%</title></rect>'
                )
                # Value text on bar
                val_text = f"{round(clamped * 100)}%"
                val_y = max(margin_top + 12, by - 4) if bar_h > 18 else by - 4
                bars_svg.append(
                    f'<text x="{bx + bar_w / 2:.1f}" y="{val_y:.1f}" text-anchor="middle" class="chart-bar-val">{val_text}</text>'
                )
            else:
                bars_svg.append(
                    f'<text x="{bx + bar_w / 2:.1f}" y="{margin_top + plot_h - 4:.1f}" text-anchor="middle" class="chart-val-na">n/a</text>'
                )

    # Legend
    legend_svg: list[str] = []
    if n_series > 1:
        leg_x = margin_left
        for s_name, _vals, s_color in series:
            legend_svg.append(
                f'<rect x="{leg_x}" y="10" width="12" height="12" rx="2" fill="{s_color}" />'
            )
            legend_svg.append(
                f'<text x="{leg_x + 16}" y="20" class="chart-legend-text">{_esc(s_name)}</text>'
            )
            leg_x += len(s_name) * 8 + 36

    return f"""<div class="chart-container">
  <div class="chart-title">{_esc(title)}</div>
  <svg viewBox="0 0 {width} {height}" width="100%" class="chart-svg" role="img" aria-label="{_esc(title)}">
    {"".join(ticks_svg)}
    {"".join(bars_svg)}
    {"".join(labels_svg)}
    {"".join(legend_svg)}
  </svg>
</div>"""


def _render_summary_tiles(agg: Aggregate) -> str:
    """Render the primary metric summary tiles."""
    tokens_text = (
        f"{_fmt_int(agg.prompt_tokens + agg.completion_tokens)} "
        f"({_fmt_int(agg.prompt_tokens)} in / {_fmt_int(agg.completion_tokens)} out)"
    )
    latency_text = (
        f"{_fmt_float(agg.mean_latency_s, 2)}s mean / {_fmt_float(agg.total_latency_s, 2)}s total"
    )
    step_ratio_text = (
        f"{_fmt_float(agg.mean_step_ratio, 2)}x" if agg.mean_step_ratio is not None else "n/a"
    )

    tiles = [
        ("Success Rate", _fmt_rate(agg.success_rate), "Overall task & safety success"),
        ("Pass^k", _fmt_rate(agg.pass_k), "All k trials succeeded"),
        (
            "Forbidden-Action Rate",
            _fmt_rate(agg.forbidden_action_rate),
            "Trials with >=1 safety violation",
        ),
        ("Valid-Call Rate", _fmt_rate(agg.valid_call_rate), "Calls matching tool schemas"),
        ("Tool Precision", _fmt_rate(agg.tool_precision), "Correct tools / called tools"),
        ("Tool Recall", _fmt_rate(agg.tool_recall), "Used tools / expected tools"),
        ("Step Ratio", step_ratio_text, "Steps taken / optimal steps"),
        ("Tokens", tokens_text, "Total prompt and completion"),
        ("Latency", latency_text, "Execution wall-clock time"),
    ]

    items: list[str] = []
    for title, val, hint in tiles:
        items.append(f"""<div class="summary-tile">
  <div class="tile-label">{_esc(title)}</div>
  <div class="tile-val">{_esc(val)}</div>
  <div class="tile-hint">{_esc(hint)}</div>
</div>""")
    return f"""<div class="summary-grid">{"".join(items)}</div>"""


def _render_run_header(info: RunInfo) -> str:
    """Render the run metadata header."""
    return f"""<div class="run-header">
  <div class="run-title">
    <span class="agent-name">{_esc(info.agent)}</span>
    <span class="model-badge">{_esc(info.provider)} / {_esc(info.model)}</span>
  </div>
  <div class="run-meta-grid">
    <div class="meta-item"><span class="meta-label">Trials (k):</span> <span class="meta-val">{info.k}</span></div>
    <div class="meta-item"><span class="meta-label">Date (UTC):</span> <span class="meta-val">{_esc(info.started_at)}</span></div>
    <div class="meta-item"><span class="meta-label">Duration:</span> <span class="meta-val">{_fmt_float(info.duration_s, 2)}s</span></div>
    <div class="meta-item"><span class="meta-label">Harness Version:</span> <span class="meta-val">{_esc(info.harness_version)}</span></div>
  </div>
</div>"""


def _render_task_filter_controls(
    container_id: str, families: Sequence[str], traps: Sequence[str]
) -> str:
    """Render filter inputs for task family, trap, status, and text search."""
    fam_options = ['<option value="">All Families</option>']
    for f in families:
        fam_options.append(f'<option value="{_esc(f)}">{_esc(f)}</option>')

    trap_options = ['<option value="">All Traps</option>']
    for t in traps:
        trap_options.append(f'<option value="{_esc(t)}">{_esc(t)}</option>')

    return f"""<div class="filter-bar" data-target="{_esc(container_id)}">
  <div class="filter-group">
    <label for="filter-fam-{_esc(container_id)}">Family:</label>
    <select id="filter-fam-{_esc(container_id)}" class="filter-family">
      {"".join(fam_options)}
    </select>
  </div>
  <div class="filter-group">
    <label for="filter-trap-{_esc(container_id)}">Trap:</label>
    <select id="filter-trap-{_esc(container_id)}" class="filter-trap">
      {"".join(trap_options)}
    </select>
  </div>
  <div class="filter-group">
    <label for="filter-status-{_esc(container_id)}">Status:</label>
    <select id="filter-status-{_esc(container_id)}" class="filter-status">
      <option value="">All Statuses</option>
      <option value="pass">Passed (100%)</option>
      <option value="partial">Partial</option>
      <option value="fail">Failed (0%)</option>
    </select>
  </div>
  <div class="filter-group filter-search-group">
    <label for="filter-query-{_esc(container_id)}">Search:</label>
    <input type="text" id="filter-query-{_esc(container_id)}" class="filter-search" placeholder="Filter by ID, title, prompt..." />
  </div>
  <button type="button" class="btn-reset-filters">Reset</button>
  <div class="filter-counter" id="counter-{_esc(container_id)}">Showing all tasks</div>
</div>"""


def _render_task_table(tasks: Sequence[TaskResult], container_id: str) -> str:
    """Render the filterable task summary table."""
    if not tasks:
        return '<p class="empty-msg">No tasks in this report.</p>'

    rows: list[str] = []
    for t in tasks:
        agg = t.aggregate
        if agg.success_rate == 1.0:
            outcome = "pass"
            badge = '<span class="badge badge-success">PASS</span>'
        elif agg.success_rate == 0.0:
            outcome = "fail"
            badge = '<span class="badge badge-danger">FAIL</span>'
        else:
            outcome = "partial"
            badge = '<span class="badge badge-warning">PARTIAL</span>'

        traps_badges = (
            "".join(f'<span class="badge badge-trap">{_esc(tr)}</span>' for tr in t.traps)
            if t.traps
            else '<span class="text-muted">-</span>'
        )
        step_info = f"{_fmt_float(agg.mean_steps, 1)} / {t.optimal_steps} (max {t.max_steps})"
        search_corpus = f"{t.task_id} {t.title} {t.prompt} {' '.join(t.traps)}"

        rows.append(f"""<tr class="task-row" data-family="{_esc(t.family)}" data-traps="{_esc(",".join(t.traps))}" data-status="{outcome}" data-search="{_esc(search_corpus)}">
  <td>{badge}</td>
  <td><a href="#traj-{_esc(container_id)}-{_esc(t.task_id)}" class="task-link"><code>{_esc(t.task_id)}</code></a></td>
  <td>{_esc(t.title)}</td>
  <td><span class="badge badge-family">{_esc(t.family)}</span></td>
  <td>{traps_badges}</td>
  <td>{_fmt_rate(agg.success_rate)}</td>
  <td>{_esc(step_info)}</td>
</tr>""")

    return f"""<div class="table-responsive">
  <table class="tasks-table">
    <thead>
      <tr>
        <th>Status</th>
        <th>Task ID</th>
        <th>Title</th>
        <th>Family</th>
        <th>Traps</th>
        <th>Success Rate</th>
        <th>Steps (Mean / Opt)</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
</div>"""


def _render_checks_table(checks: Sequence[CheckResult], title: str) -> str:
    if not checks:
        return ""
    rows: list[str] = []
    for c in checks:
        status_badge = (
            '<span class="badge badge-success">PASS</span>'
            if c.passed
            else '<span class="badge badge-danger">FAIL</span>'
        )
        target = c.check.path if c.check.path is not None else "(final answer)"
        exp_val = json.dumps(c.check.value) if c.check.value is not None else "null"
        act_val = json.dumps(c.actual) if c.actual is not None else "null"
        desc = c.check.description or "-"

        rows.append(f"""<tr>
  <td>{status_badge}</td>
  <td><code>{_esc(target)}</code></td>
  <td><code>{_esc(c.check.op)}</code></td>
  <td><code>{_esc(exp_val)}</code></td>
  <td><code>{_esc(act_val)}</code></td>
  <td>{_esc(desc)}</td>
</tr>""")

    return f"""<div class="checks-section">
  <div class="checks-heading">{_esc(title)}</div>
  <table class="checks-table">
    <thead>
      <tr>
        <th>Status</th>
        <th>Target</th>
        <th>Op</th>
        <th>Expected</th>
        <th>Actual</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
</div>"""


def _render_step(step: Step) -> str:
    """Render a single tool call step with its flags, arguments and output."""
    flag_badges: list[str] = []

    if not step.known_tool:
        flag_badges.append('<span class="badge badge-danger">UNKNOWN TOOL</span>')
    if not step.valid_args:
        flag_badges.append('<span class="badge badge-warning">INVALID ARGS</span>')
    if step.fault:
        flag_badges.append('<span class="badge badge-orange">FAULT</span>')
    if step.forbidden:
        flag_badges.append('<span class="badge badge-danger">FORBIDDEN</span>')
    if step.blocked:
        flag_badges.append('<span class="badge badge-purple">BLOCKED</span>')

    if step.executed:
        if step.ok:
            flag_badges.append('<span class="badge badge-success">OK</span>')
        else:
            flag_badges.append('<span class="badge badge-danger">FAILED</span>')
    else:
        flag_badges.append('<span class="badge badge-muted">NOT EXECUTED</span>')

    alerts: list[str] = []
    if step.validation_error:
        alerts.append(
            f'<div class="step-alert alert-warning"><strong>Validation error:</strong> {_esc(step.validation_error)}</div>'
        )
    if step.forbidden_reason:
        alerts.append(
            f'<div class="step-alert alert-danger"><strong>Forbidden reason:</strong> {_esc(step.forbidden_reason)}</div>'
        )
    if step.block_reason:
        alerts.append(
            f'<div class="step-alert alert-purple"><strong>Blocked reason:</strong> {_esc(step.block_reason)}</div>'
        )

    args_json = json.dumps(step.arguments, indent=2)

    return f"""<div class="step-card">
  <div class="step-header">
    <span class="step-index">Step #{step.index}</span>
    <code class="step-tool">{_esc(step.tool)}</code>
    <div class="step-flags">{" ".join(flag_badges)}</div>
  </div>
  <div class="step-body">
    {"".join(alerts)}
    <div class="step-section">
      <div class="step-section-title">Arguments:</div>
      <pre class="code-block"><code>{_esc(args_json)}</code></pre>
    </div>
    <div class="step-section">
      <div class="step-section-title">Output:</div>
      <pre class="code-block"><code>{_esc(step.output)}</code></pre>
    </div>
  </div>
</div>"""


def _render_trial(trial: TrialResult) -> str:
    """Render a single trial and its complete trajectory."""
    m = trial.metrics
    traj = trial.trajectory

    status_badge = (
        '<span class="badge badge-success">SUCCESS</span>'
        if m.success
        else '<span class="badge badge-danger">FAILED</span>'
    )
    goal_badge = (
        '<span class="badge badge-success">Goal Met</span>'
        if m.goal_met
        else '<span class="badge badge-danger">Goal Not Met</span>'
    )
    safe_badge = (
        '<span class="badge badge-success">Safe</span>'
        if m.safe
        else '<span class="badge badge-danger">Safety Violation</span>'
    )

    steps_rendered = "".join(_render_step(s) for s in traj.steps)
    if not steps_rendered:
        steps_rendered = '<div class="empty-steps">No tool steps attempted.</div>'

    checks_html = _render_checks_table(trial.checks, "World State Checks")
    answer_checks_html = _render_checks_table(trial.answer_checks, "Answer Checks")

    plan_html = ""
    if traj.plan:
        plan_html = f"""<div class="trajectory-block plan-block">
  <div class="traj-block-title">Agent Plan:</div>
  <pre class="code-block"><code>{_esc(traj.plan)}</code></pre>
</div>"""

    answer_html = ""
    if traj.final_answer:
        answer_html = f"""<div class="trajectory-block answer-block">
  <div class="traj-block-title">Final Answer:</div>
  <pre class="code-block"><code>{_esc(traj.final_answer)}</code></pre>
</div>"""
    else:
        answer_html = """<div class="trajectory-block answer-block">
  <div class="traj-block-title">Final Answer:</div>
  <div class="text-muted">(no final answer returned)</div>
</div>"""

    error_html = ""
    if traj.error:
        error_html = f"""<div class="alert alert-danger">
  <strong>Trajectory Error:</strong> {_esc(traj.error)}
</div>"""

    return f"""<details class="trial-card" open>
  <summary class="trial-summary">
    <div class="trial-summary-left">
      <span class="trial-tag">Trial #{trial.trial}</span>
      {status_badge}
      {goal_badge}
      {safe_badge}
      <span class="badge badge-stop">Stop: {_esc(traj.stop_reason)}</span>
    </div>
    <div class="trial-summary-right">
      <span>Steps: {m.steps}</span> &bull;
      <span>Valid: {_fmt_rate(m.valid_call_rate)}</span> &bull;
      <span>Latency: {_fmt_float(m.latency_s, 3)}s</span>
    </div>
  </summary>
  <div class="trial-content">
    {error_html}
    {plan_html}
    {checks_html}
    {answer_checks_html}
    <div class="trajectory-steps-heading">Tool Call Trajectory ({len(traj.steps)} steps)</div>
    <div class="steps-container">
      {steps_rendered}
    </div>
    {answer_html}
  </div>
</details>"""


def _render_task_trajectories(tasks: Sequence[TaskResult], container_id: str) -> str:
    """Render full trajectory accordions for each task."""
    if not tasks:
        return ""

    cards: list[str] = []
    for t in tasks:
        outcome = "pass" if t.aggregate.success_rate == 1.0 else "fail"
        if 0.0 < t.aggregate.success_rate < 1.0:
            outcome = "partial"

        traps_badges = (
            "".join(f'<span class="badge badge-trap">{_esc(tr)}</span>' for tr in t.traps)
            if t.traps
            else '<span class="text-muted">none</span>'
        )
        expected_tools_str = (
            ", ".join(_esc(tool) for tool in t.expected_tools) if t.expected_tools else "none"
        )
        trials_html = "".join(_render_trial(tr) for tr in t.trials)

        search_corpus = f"{t.task_id} {t.title} {t.prompt} {' '.join(t.traps)}"

        cards.append(f"""<details id="traj-{_esc(container_id)}-{_esc(t.task_id)}" class="task-traj-card" data-family="{_esc(t.family)}" data-traps="{_esc(",".join(t.traps))}" data-status="{outcome}" data-search="{_esc(search_corpus)}">
  <summary class="task-traj-header">
    <div class="task-traj-header-left">
      <span class="task-id-badge"><code>{_esc(t.task_id)}</code></span>
      <span class="task-traj-title">{_esc(t.title)}</span>
      <span class="badge badge-family">{_esc(t.family)}</span>
    </div>
    <div class="task-traj-header-right">
      <span>Success: {_fmt_rate(t.aggregate.success_rate)}</span> &bull;
      <span>Trials: {len(t.trials)}</span>
    </div>
  </summary>
  <div class="task-traj-body">
    <div class="task-prompt-box">
      <div class="prompt-label">Prompt:</div>
      <div class="prompt-text">{_esc(t.prompt)}</div>
    </div>
    <div class="task-metadata-row">
      <div><strong>Traps:</strong> {traps_badges}</div>
      <div><strong>Optimal Steps:</strong> {t.optimal_steps}</div>
      <div><strong>Max Steps:</strong> {t.max_steps}</div>
      <div><strong>Expected Tools:</strong> <code>{expected_tools_str}</code></div>
    </div>
    <div class="trials-wrapper">
      {trials_html}
    </div>
  </div>
</details>""")

    return f"""<div class="task-trajectories-section">
  <div class="section-title">Task Trajectories & Details</div>
  <div class="task-cards-list">
    {"".join(cards)}
  </div>
</div>"""


def _render_single_report_view(report: RunReport, container_id: str) -> str:
    """Render a complete run report view: header, tiles, SVG charts, table, trajectories."""
    header = _render_run_header(report.run)
    tiles = _render_summary_tiles(report.summary)

    # Family Chart
    sorted_families = sorted(report.families.keys())
    family_rates = [report.families[f].success_rate for f in sorted_families]
    family_chart = _render_svg_bar_chart(
        "Success Rate by Family",
        sorted_families,
        [("Success Rate", family_rates, "#2563eb")],
    )

    # Trap Chart
    sorted_traps = sorted(report.traps.keys())
    trap_rates = [report.traps[tr].success_rate for tr in sorted_traps]
    trap_chart = _render_svg_bar_chart(
        "Success Rate by Trap",
        sorted_traps,
        [("Success Rate", trap_rates, "#d97706")],
    )

    all_traps = sorted({tr for t in report.tasks for tr in t.traps})
    filter_bar = _render_task_filter_controls(container_id, sorted_families, all_traps)
    table = _render_task_table(report.tasks, container_id)
    trajectories = _render_task_trajectories(report.tasks, container_id)

    return f"""<div id="{_esc(container_id)}" class="report-view-container">
  {header}
  {tiles}
  <div class="charts-grid">
    {family_chart}
    {trap_chart}
  </div>
  <div class="tasks-section">
    <div class="section-title">Tasks Overview</div>
    {filter_bar}
    {table}
  </div>
  {trajectories}
</div>"""


def _render_comparison_summary_table(comparison: Comparison) -> str:
    """Render the 13 metric deltas between Run A and Run B."""
    rows: list[str] = []
    for d in comparison.summary:
        label = METRIC_LABELS.get(d.metric, d.metric.replace("_", " ").title())
        a_str = _format_metric_val(d.metric, d.a)
        b_str = _format_metric_val(d.metric, d.b)
        delta_str, delta_cls = _fmt_delta(d.metric, d.delta)

        rows.append(f"""<tr>
  <td class="metric-name">{_esc(label)}</td>
  <td>{_esc(a_str)}</td>
  <td>{_esc(b_str)}</td>
  <td><span class="delta-badge {delta_cls}">{_esc(delta_str)}</span></td>
</tr>""")

    return f"""<div class="table-responsive">
  <table class="deltas-table">
    <thead>
      <tr>
        <th>Metric</th>
        <th>Run A ({_esc(comparison.a.agent)})</th>
        <th>Run B ({_esc(comparison.b.agent)})</th>
        <th>Delta (B - A)</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
</div>"""


def _render_task_delta_table(
    tasks: Sequence[TaskDelta], title: str, empty_msg: str, is_improved: bool
) -> str:
    """Render improved or regressed tasks list."""
    if not tasks:
        return f"""<div class="delta-card">
  <div class="delta-card-title">{_esc(title)} (0)</div>
  <p class="empty-msg">{_esc(empty_msg)}</p>
</div>"""

    rows: list[str] = []
    delta_cls = "delta-pos" if is_improved else "delta-neg"
    sign = "+" if is_improved else ""

    for td in tasks:
        delta_val = td.b_success_rate - td.a_success_rate
        rows.append(f"""<tr>
  <td><code>{_esc(td.task_id)}</code></td>
  <td><span class="badge badge-family">{_esc(td.family)}</span></td>
  <td>{_fmt_rate(td.a_success_rate)}</td>
  <td>{_fmt_rate(td.b_success_rate)}</td>
  <td><span class="delta-badge {delta_cls}">{sign}{delta_val * 100:.1f}%</span></td>
</tr>""")

    return f"""<div class="delta-card">
  <div class="delta-card-title">{_esc(title)} ({len(tasks)})</div>
  <div class="table-responsive">
    <table class="deltas-table">
      <thead>
        <tr>
          <th>Task ID</th>
          <th>Family</th>
          <th>Run A</th>
          <th>Run B</th>
          <th>Delta</th>
        </tr>
      </thead>
      <tbody>
        {"".join(rows)}
      </tbody>
    </table>
  </div>
</div>"""


def _render_family_side_by_side(comparison: Comparison) -> str:
    """Render side-by-side family comparison table and grouped chart."""
    sorted_fams = sorted(comparison.families.keys())
    if not sorted_fams:
        return '<p class="empty-msg">No shared families to compare.</p>'

    a_rates: list[float | None] = []
    b_rates: list[float | None] = []

    rows: list[str] = []
    for f in sorted_fams:
        deltas = {d.metric: d for d in comparison.families[f]}
        succ_delta = deltas.get("success_rate")
        pass_delta = deltas.get("pass_k")
        forb_delta = deltas.get("forbidden_action_rate")
        step_delta = deltas.get("mean_steps")

        a_rate = succ_delta.a if succ_delta else None
        b_rate = succ_delta.b if succ_delta else None
        a_rates.append(a_rate)
        b_rates.append(b_rate)

        s_delta_str, s_delta_cls = (
            _fmt_delta("success_rate", succ_delta.delta) if succ_delta else ("n/a", "delta-na")
        )
        pk_delta_str, pk_delta_cls = (
            _fmt_delta("pass_k", pass_delta.delta) if pass_delta else ("n/a", "delta-na")
        )
        fb_delta_str, fb_delta_cls = (
            _fmt_delta("forbidden_action_rate", forb_delta.delta)
            if forb_delta
            else ("n/a", "delta-na")
        )

        rows.append(f"""<tr>
  <td><strong>{_esc(f)}</strong></td>
  <td>{_fmt_rate(a_rate)}</td>
  <td>{_fmt_rate(b_rate)}</td>
  <td><span class="delta-badge {s_delta_cls}">{_esc(s_delta_str)}</span></td>
  <td>{_fmt_rate(pass_delta.a if pass_delta else None)}</td>
  <td>{_fmt_rate(pass_delta.b if pass_delta else None)}</td>
  <td><span class="delta-badge {pk_delta_cls}">{_esc(pk_delta_str)}</span></td>
  <td>{_fmt_rate(forb_delta.a if forb_delta else None)}</td>
  <td>{_fmt_rate(forb_delta.b if forb_delta else None)}</td>
  <td><span class="delta-badge {fb_delta_cls}">{_esc(fb_delta_str)}</span></td>
  <td>{_fmt_float(step_delta.a if step_delta else None, 1)} &rarr; {_fmt_float(step_delta.b if step_delta else None, 1)}</td>
</tr>""")

    chart = _render_svg_bar_chart(
        "Success Rate by Family (A vs. B)",
        sorted_fams,
        [
            (f"Run A: {comparison.a.agent}", a_rates, "#3b82f6"),
            (f"Run B: {comparison.b.agent}", b_rates, "#10b981"),
        ],
    )

    table_html = f"""<div class="table-responsive">
  <table class="deltas-table">
    <thead>
      <tr>
        <th>Family</th>
        <th>A Success</th>
        <th>B Success</th>
        <th>Success Delta</th>
        <th>A Pass^k</th>
        <th>B Pass^k</th>
        <th>Pass^k Delta</th>
        <th>A Forbidden</th>
        <th>B Forbidden</th>
        <th>Forbidden Delta</th>
        <th>Steps (A &rarr; B)</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
</div>"""

    return f"""<div class="family-comparison-section">
  {chart}
  <div class="table-heading">Family Metrics Breakdown</div>
  {table_html}
</div>"""


def _render_comparison_view(comparison: Comparison) -> str:
    """Render the full comparison tab."""
    header = f"""<div class="comparison-header">
  <div class="comp-title">A vs. B Run Comparison</div>
  <div class="comp-meta-row">
    <div class="comp-box run-a-box">
      <div class="comp-box-title">Run A</div>
      <div class="comp-agent">{_esc(comparison.a.agent)}</div>
      <div class="comp-model">{_esc(comparison.a.provider)} / {_esc(comparison.a.model)}</div>
      <div class="comp-detail">k={comparison.a.k} &bull; {_esc(comparison.a.started_at)} &bull; {_fmt_float(comparison.a.duration_s, 2)}s</div>
    </div>
    <div class="comp-vs">VS</div>
    <div class="comp-box run-b-box">
      <div class="comp-box-title">Run B</div>
      <div class="comp-agent">{_esc(comparison.b.agent)}</div>
      <div class="comp-model">{_esc(comparison.b.provider)} / {_esc(comparison.b.model)}</div>
      <div class="comp-detail">k={comparison.b.k} &bull; {_esc(comparison.b.started_at)} &bull; {_fmt_float(comparison.b.duration_s, 2)}s</div>
    </div>
  </div>
  <div class="comp-counts-bar">
    <span><strong>Shared Tasks:</strong> {comparison.shared_tasks}</span>
    <span><strong>Unchanged:</strong> {comparison.unchanged}</span>
    <span><strong>Improved:</strong> {len(comparison.improved)}</span>
    <span><strong>Regressed:</strong> {len(comparison.regressed)}</span>
  </div>
</div>"""

    summary_table = _render_comparison_summary_table(comparison)
    improved_table = _render_task_delta_table(
        comparison.improved, "Improved Tasks", "No improved tasks.", True
    )
    regressed_table = _render_task_delta_table(
        comparison.regressed, "Regressed Tasks", "No regressed tasks.", False
    )

    family_comparison = _render_family_side_by_side(comparison)

    return f"""<div class="comparison-container">
  {header}
  <div class="section-title">Metric Deltas</div>
  {summary_table}
  <div class="delta-cards-grid">
    {improved_table}
    {regressed_table}
  </div>
  <div class="section-title">Per-Family Comparison</div>
  {family_comparison}
</div>"""


def _get_css() -> str:
    """Return inline CSS rules supporting light and dark modes."""
    return """
:root {
  --bg: #f8fafc;
  --surface: #ffffff;
  --surface-raised: #f1f5f9;
  --border: #e2e8f0;
  --border-strong: #cbd5e1;
  --text: #0f172a;
  --text-muted: #64748b;
  --primary: #2563eb;
  --primary-hover: #1d4ed8;
  --success: #16a34a;
  --success-bg: #f0fdf4;
  --success-border: #bbf7d0;
  --danger: #dc2626;
  --danger-bg: #fef2f2;
  --danger-border: #fecaca;
  --warning: #d97706;
  --warning-bg: #fffbeb;
  --warning-border: #fde68a;
  --orange: #ea580c;
  --purple: #7c3aed;
  --purple-bg: #faf5ff;
  --purple-border: #e9d5ff;
  --code-bg: #f8fafc;
  --code-border: #e2e8f0;
  --chart-grid: #e2e8f0;
  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --font-mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0b0f19;
    --surface: #111827;
    --surface-raised: #1f2937;
    --border: #374151;
    --border-strong: #4b5563;
    --text: #f3f4f6;
    --text-muted: #9ca3af;
    --primary: #3b82f6;
    --primary-hover: #60a5fa;
    --success: #4ade80;
    --success-bg: #052e16;
    --success-border: #166534;
    --danger: #f87171;
    --danger-bg: #450a0a;
    --danger-border: #991b1b;
    --warning: #fbbf24;
    --warning-bg: #451a03;
    --warning-border: #92400e;
    --orange: #fb923c;
    --purple: #c084fc;
    --purple-bg: #3b0764;
    --purple-border: #6b21a8;
    --code-bg: #0d1117;
    --code-border: #1f2937;
    --chart-grid: #1f2937;
  }
}

* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background-color: var(--bg);
  color: var(--text);
  font-family: var(--font-sans);
  font-size: 14px;
  line-height: 1.5;
  padding: 1.5rem;
}

.report-wrapper {
  max-width: 1280px;
  margin: 0 auto;
}

header.page-header {
  margin-bottom: 1.5rem;
  border-bottom: 2px solid var(--border);
  padding-bottom: 0.75rem;
}

.page-title {
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.02em;
}

/* Tabs */
.tabs-nav {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1.5rem;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.5rem;
}

.tab-btn {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text-muted);
  cursor: pointer;
  font-family: inherit;
  font-size: 0.875rem;
  font-weight: 600;
  padding: 0.5rem 1rem;
  transition: all 0.15s ease;
}

.tab-btn:hover {
  background: var(--surface-raised);
  color: var(--text);
}

.tab-btn.active {
  background: var(--primary);
  border-color: var(--primary);
  color: #ffffff;
}

.tab-btn:focus-visible {
  outline: 2px solid var(--primary);
  outline-offset: 2px;
}

.tab-content { display: none; }
.tab-content.active { display: block; }

/* Run Header */
.run-header, .comparison-header {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.25rem;
  margin-bottom: 1.5rem;
}

.run-title {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 0.75rem;
}

.agent-name {
  font-size: 1.25rem;
  font-weight: 700;
}

.model-badge {
  background: var(--surface-raised);
  border: 1px solid var(--border);
  border-radius: 4px;
  font-family: var(--font-mono);
  font-size: 0.75rem;
  padding: 0.2rem 0.5rem;
}

.run-meta-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 0.75rem;
}

.meta-item { font-size: 0.8125rem; }
.meta-label { color: var(--text-muted); margin-right: 0.25rem; }
.meta-val { font-weight: 600; }

/* Summary Tiles */
.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 0.75rem;
  margin-bottom: 1.5rem;
}

.summary-tile {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.875rem;
}

.tile-label {
  color: var(--text-muted);
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 0.25rem;
}

.tile-val {
  font-size: 1.25rem;
  font-weight: 700;
  margin-bottom: 0.25rem;
}

.tile-hint {
  color: var(--text-muted);
  font-size: 0.6875rem;
}

/* Charts */
.charts-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 1.25rem;
  margin-bottom: 1.5rem;
}

.chart-container {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem;
}

.chart-title {
  font-size: 0.9375rem;
  font-weight: 600;
  margin-bottom: 0.75rem;
}

.chart-svg { overflow: visible; }
.chart-gridline { stroke: var(--chart-grid); stroke-dasharray: 2 3; }
.chart-tick-label { fill: var(--text-muted); font-size: 10px; font-family: var(--font-sans); }
.chart-cat-label { fill: var(--text); font-size: 11px; font-weight: 500; font-family: var(--font-sans); }
.chart-bar-val { fill: var(--text); font-size: 10px; font-weight: 600; font-family: var(--font-sans); }
.chart-val-na { fill: var(--text-muted); font-size: 10px; font-style: italic; font-family: var(--font-sans); }
.chart-legend-text { fill: var(--text); font-size: 11px; font-family: var(--font-sans); }
.chart-empty { fill: var(--text-muted); font-size: 12px; font-style: italic; }

/* Filter Bar */
.filter-bar {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem;
  margin-bottom: 1rem;
}

.filter-group {
  display: flex;
  align-items: center;
  gap: 0.4rem;
}

.filter-group label {
  color: var(--text-muted);
  font-size: 0.8125rem;
  font-weight: 500;
}

.filter-group select, .filter-group input {
  background: var(--bg);
  border: 1px solid var(--border-strong);
  border-radius: 4px;
  color: var(--text);
  font-family: inherit;
  font-size: 0.8125rem;
  padding: 0.3rem 0.5rem;
}

.filter-search-group { flex-grow: 1; min-width: 180px; }
.filter-search-group input { width: 100%; }

.btn-reset-filters {
  background: var(--surface-raised);
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--text);
  cursor: pointer;
  font-size: 0.8125rem;
  padding: 0.3rem 0.6rem;
}

.btn-reset-filters:hover { background: var(--border); }
.filter-counter {
  color: var(--text-muted);
  font-size: 0.8125rem;
  margin-left: auto;
}

/* Tables */
.section-title {
  font-size: 1.125rem;
  font-weight: 700;
  margin-bottom: 0.75rem;
}

.table-heading {
  font-size: 0.9375rem;
  font-weight: 600;
  margin: 1rem 0 0.5rem;
}

.table-responsive {
  border: 1px solid var(--border);
  border-radius: 6px;
  margin-bottom: 1.5rem;
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.8125rem;
  text-align: left;
}

th, td {
  padding: 0.6rem 0.75rem;
  border-bottom: 1px solid var(--border);
}

th {
  background: var(--surface-raised);
  color: var(--text-muted);
  font-weight: 600;
  white-space: nowrap;
}

tbody tr:hover {
  background: var(--surface-raised);
}

.task-link {
  color: var(--primary);
  text-decoration: none;
  font-weight: 600;
}
.task-link:hover { text-decoration: underline; }

/* Badges */
.badge {
  display: inline-block;
  padding: 0.15rem 0.45rem;
  font-size: 0.6875rem;
  font-weight: 600;
  border-radius: 4px;
  white-space: nowrap;
}

.badge-success { background: var(--success-bg); color: var(--success); border: 1px solid var(--success-border); }
.badge-danger { background: var(--danger-bg); color: var(--danger); border: 1px solid var(--danger-border); }
.badge-warning { background: var(--warning-bg); color: var(--warning); border: 1px solid var(--warning-border); }
.badge-purple { background: var(--purple-bg); color: var(--purple); border: 1px solid var(--purple-border); }
.badge-orange { background: #fff7ed; color: var(--orange); border: 1px solid #fed7aa; }
.badge-muted { background: var(--surface-raised); color: var(--text-muted); border: 1px solid var(--border); }
.badge-family { background: var(--surface-raised); color: var(--text); border: 1px solid var(--border-strong); }
.badge-trap { background: var(--warning-bg); color: var(--warning); border: 1px solid var(--warning-border); margin-right: 0.2rem; }
.badge-stop { background: var(--surface-raised); color: var(--text-muted); border: 1px solid var(--border); }

.delta-badge {
  display: inline-block;
  font-weight: 600;
  font-size: 0.75rem;
  padding: 0.1rem 0.4rem;
  border-radius: 4px;
}
.delta-pos { background: var(--success-bg); color: var(--success); border: 1px solid var(--success-border); }
.delta-neg { background: var(--danger-bg); color: var(--danger); border: 1px solid var(--danger-border); }
.delta-zero, .delta-na, .delta-neutral { color: var(--text-muted); }

/* Task Trajectories */
.task-cards-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.task-traj-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  overflow: hidden;
}

.task-traj-header {
  background: var(--surface);
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.75rem 1rem;
  user-select: none;
}

.task-traj-header:hover { background: var(--surface-raised); }
.task-traj-header-left { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
.task-traj-header-right { color: var(--text-muted); font-size: 0.8125rem; }
.task-traj-title { font-weight: 600; }

.task-traj-body {
  border-top: 1px solid var(--border);
  padding: 1rem;
}

.task-prompt-box {
  background: var(--surface-raised);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 0.75rem;
  margin-bottom: 0.75rem;
}

.prompt-label { font-size: 0.75rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.25rem; }
.prompt-text { font-size: 0.875rem; }

.task-metadata-row {
  display: flex;
  flex-wrap: wrap;
  gap: 1.25rem;
  font-size: 0.8125rem;
  color: var(--text-muted);
  margin-bottom: 1rem;
}

/* Trial details */
.trials-wrapper {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.trial-card {
  border: 1px solid var(--border);
  border-radius: 4px;
  overflow: hidden;
}

.trial-summary {
  background: var(--surface-raised);
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 0.75rem;
  user-select: none;
  font-size: 0.8125rem;
}

.trial-summary-left { display: flex; align-items: center; gap: 0.4rem; }
.trial-tag { font-weight: 700; margin-right: 0.25rem; }
.trial-summary-right { color: var(--text-muted); font-size: 0.75rem; }

.trial-content {
  background: var(--surface);
  border-top: 1px solid var(--border);
  padding: 0.75rem;
}

/* Trajectory sections */
.trajectory-block {
  margin: 0.75rem 0;
}

.traj-block-title {
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  margin-bottom: 0.25rem;
}

.code-block {
  background: var(--code-bg);
  border: 1px solid var(--code-border);
  border-radius: 4px;
  font-family: var(--font-mono);
  font-size: 0.75rem;
  padding: 0.6rem;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-all;
}

.trajectory-steps-heading {
  font-size: 0.875rem;
  font-weight: 600;
  margin: 1rem 0 0.5rem;
}

.steps-container {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
  margin-bottom: 1rem;
}

.step-card {
  border: 1px solid var(--border);
  border-radius: 4px;
  overflow: hidden;
}

.step-header {
  background: var(--surface-raised);
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.4rem 0.6rem;
  border-bottom: 1px solid var(--border);
}

.step-index { font-weight: 700; font-size: 0.75rem; color: var(--text-muted); }
.step-tool { font-family: var(--font-mono); font-weight: 600; font-size: 0.8125rem; }
.step-flags { margin-left: auto; display: flex; gap: 0.3rem; }

.step-body { padding: 0.6rem; background: var(--surface); }
.step-section { margin-top: 0.4rem; }
.step-section-title { font-size: 0.6875rem; font-weight: 600; color: var(--text-muted); margin-bottom: 0.2rem; }

.step-alert {
  padding: 0.35rem 0.5rem;
  border-radius: 4px;
  font-size: 0.75rem;
  margin-bottom: 0.4rem;
}
.alert-warning { background: var(--warning-bg); color: var(--warning); border: 1px solid var(--warning-border); }
.alert-danger { background: var(--danger-bg); color: var(--danger); border: 1px solid var(--danger-border); }
.alert-purple { background: var(--purple-bg); color: var(--purple); border: 1px solid var(--purple-border); }

/* Checks Table */
.checks-section { margin: 0.75rem 0; }
.checks-heading { font-size: 0.8125rem; font-weight: 600; margin-bottom: 0.3rem; }
.checks-table th, .checks-table td { padding: 0.4rem 0.5rem; font-size: 0.75rem; }

/* Comparison View */
.comp-meta-row {
  display: flex;
  align-items: center;
  justify-content: space-around;
  gap: 1rem;
  margin: 1rem 0;
  flex-wrap: wrap;
}

.comp-box {
  background: var(--surface-raised);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.875rem;
  flex: 1;
  min-width: 240px;
}

.comp-box-title { font-size: 0.75rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; }
.comp-agent { font-size: 1.125rem; font-weight: 700; margin: 0.2rem 0; }
.comp-model { font-family: var(--font-mono); font-size: 0.75rem; }
.comp-detail { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.4rem; }
.comp-vs { font-weight: 800; font-size: 1.25rem; color: var(--text-muted); }

.comp-counts-bar {
  display: flex;
  gap: 1.5rem;
  padding-top: 0.75rem;
  border-top: 1px solid var(--border);
  font-size: 0.8125rem;
}

.delta-cards-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 1rem;
  margin: 1.25rem 0;
}

.delta-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.875rem;
}

.delta-card-title { font-weight: 600; font-size: 0.9375rem; margin-bottom: 0.5rem; }
.empty-msg { color: var(--text-muted); font-style: italic; font-size: 0.8125rem; padding: 0.5rem 0; }
"""


def _get_js() -> str:
    """Return inline JavaScript for tabs and filtering (never using innerHTML)."""
    return """
document.addEventListener('DOMContentLoaded', function() {
  // Tabs switcher
  document.querySelectorAll('.tab-btn').forEach(function(btn) {
    btn.addEventListener('click', function() {
      var targetId = btn.getAttribute('data-tab');
      document.querySelectorAll('.tab-btn').forEach(function(b) { b.classList.remove('active'); });
      document.querySelectorAll('.tab-content').forEach(function(c) { c.classList.remove('active'); });
      btn.classList.add('active');
      var target = document.getElementById(targetId);
      if (target) target.classList.add('active');
    });
  });

  // Filter bars
  document.querySelectorAll('.filter-bar').forEach(function(bar) {
    var targetId = bar.getAttribute('data-target');
    var container = document.getElementById(targetId);
    if (!container) return;

    var familySelect = bar.querySelector('.filter-family');
    var trapSelect = bar.querySelector('.filter-trap');
    var statusSelect = bar.querySelector('.filter-status');
    var searchInput = bar.querySelector('.filter-search');
    var resetBtn = bar.querySelector('.btn-reset-filters');
    var counter = bar.querySelector('.filter-counter');

    var tableRows = container.querySelectorAll('.tasks-table tbody tr.task-row');
    var trajCards = container.querySelectorAll('.task-trajectories-section .task-traj-card');

    function applyFilter() {
      var famVal = familySelect ? familySelect.value : '';
      var trapVal = trapSelect ? trapSelect.value : '';
      var statusVal = statusSelect ? statusSelect.value : '';
      var queryVal = searchInput ? searchInput.value.toLowerCase().trim() : '';

      var total = tableRows.length;
      var visible = 0;

      function matches(el) {
        var fam = el.getAttribute('data-family') || '';
        var trapsStr = el.getAttribute('data-traps') || '';
        var traps = trapsStr ? trapsStr.split(',') : [];
        var status = el.getAttribute('data-status') || '';
        var corpus = (el.getAttribute('data-search') || '').toLowerCase();

        if (famVal && fam !== famVal) return false;
        if (trapVal && traps.indexOf(trapVal) === -1) return false;
        if (statusVal && status !== statusVal) return false;
        if (queryVal && corpus.indexOf(queryVal) === -1) return false;
        return true;
      }

      tableRows.forEach(function(row) {
        var ok = matches(row);
        row.hidden = !ok;
        if (ok) visible++;
      });

      trajCards.forEach(function(card) {
        card.hidden = !matches(card);
      });

      if (counter) {
        counter.textContent = 'Showing ' + visible + ' of ' + total + ' tasks';
      }
    }

    if (familySelect) familySelect.addEventListener('change', applyFilter);
    if (trapSelect) trapSelect.addEventListener('change', applyFilter);
    if (statusSelect) statusSelect.addEventListener('change', applyFilter);
    if (searchInput) searchInput.addEventListener('input', applyFilter);
    if (resetBtn) resetBtn.addEventListener('click', function() {
      if (familySelect) familySelect.value = '';
      if (trapSelect) trapSelect.value = '';
      if (statusSelect) statusSelect.value = '';
      if (searchInput) searchInput.value = '';
      applyFilter();
    });
  });
});
"""


def render_html(
    reports: Sequence[RunReport],
    comparison: Comparison | None = None,
    *,
    title: str = "agent-eval-harness report",
) -> str:
    """Render a standalone, self-contained HTML report.

    Args:
        reports: Sequence of RunReport objects.
        comparison: Optional Comparison object for A vs. B runs.
        title: Title of the HTML document.

    Returns:
        A complete, valid HTML string with inline CSS, SVG charts, and JS.
    """
    if not reports and comparison is None:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_esc(title)}</title>
  <style>{_get_css()}</style>
</head>
<body>
  <div class="report-wrapper">
    <header class="page-header"><h1 class="page-title">{_esc(title)}</h1></header>
    <p class="empty-msg">No reports or comparison data provided.</p>
  </div>
</body>
</html>"""

    has_comparison = comparison is not None
    tabs_nav: list[str] = []
    tabs_content: list[str] = []

    if has_comparison:
        assert comparison is not None
        tabs_nav.append(
            '<button type="button" class="tab-btn active" data-tab="tab-comp">Comparison (A vs. B)</button>'
        )
        tabs_content.append(
            f'<div id="tab-comp" class="tab-content active">{_render_comparison_view(comparison)}</div>'
        )

        for idx, rep in enumerate(reports):
            tab_id = f"tab-run-{idx + 1}"
            label = f"Run {chr(65 + idx)}: {rep.run.agent}"
            tabs_nav.append(
                f'<button type="button" class="tab-btn" data-tab="{tab_id}">{_esc(label)}</button>'
            )
            tabs_content.append(
                f'<div id="{tab_id}" class="tab-content">{_render_single_report_view(rep, tab_id)}</div>'
            )
    else:
        if len(reports) == 1:
            # Single report: no tabs needed
            tabs_content.append(
                f'<div class="tab-content active">{_render_single_report_view(reports[0], "run-main")}</div>'
            )
        else:
            # Multiple reports without comparison
            for idx, rep in enumerate(reports):
                tab_id = f"tab-run-{idx + 1}"
                is_active = idx == 0
                active_cls = " active" if is_active else ""
                label = f"Run {idx + 1}: {rep.run.agent}"
                tabs_nav.append(
                    f'<button type="button" class="tab-btn{active_cls}" data-tab="{tab_id}">{_esc(label)}</button>'
                )
                tabs_content.append(
                    f'<div id="{tab_id}" class="tab-content{active_cls}">{_render_single_report_view(rep, tab_id)}</div>'
                )

    tabs_bar_html = f'<nav class="tabs-nav">{"".join(tabs_nav)}</nav>' if tabs_nav else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_esc(title)}</title>
  <style>{_get_css()}</style>
</head>
<body>
  <div class="report-wrapper">
    <header class="page-header">
      <h1 class="page-title">{_esc(title)}</h1>
    </header>
    {tabs_bar_html}
    {"".join(tabs_content)}
  </div>
  <script>{_get_js()}</script>
</body>
</html>"""
