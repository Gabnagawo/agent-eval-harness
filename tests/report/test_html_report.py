"""Tests for the self-contained HTML report (agent_eval.report.html)."""

from __future__ import annotations

import re

from agent_eval.models import (
    Aggregate,
    Check,
    CheckResult,
    RunInfo,
    RunReport,
    Step,
    TaskResult,
    Trajectory,
    TrialMetrics,
    TrialResult,
)
from agent_eval.report.build import compare
from agent_eval.report.html import render_html
from agent_eval.testing import sample_comparison, sample_report


def test_render_single_report_sloppy() -> None:
    """A single run report renders headers, summary tiles, SVG charts, and trajectories."""
    report = sample_report("sloppy")
    html = render_html([report], title="Sloppy Run Evaluation")

    # Document & title
    assert "<!DOCTYPE html>" in html
    assert "<title>Sloppy Run Evaluation</title>" in html
    assert "Sloppy Run Evaluation</h1>" in html

    # Header metadata
    assert "sloppy" in html
    assert "scripted" in html
    assert 'k):</span> <span class="meta-val">2</span>' in html

    # Summary tiles
    assert "Success Rate" in html
    assert "Pass^k" in html
    assert "Forbidden-Action Rate" in html
    assert "Valid-Call Rate" in html
    assert "Tool Precision" in html
    assert "Tool Recall" in html
    assert "Step Ratio" in html
    assert "Tokens" in html
    assert "Latency" in html

    # Inline SVG charts
    assert "<svg" in html
    assert "Success Rate by Family" in html
    assert "Success Rate by Trap" in html
    assert 'class="chart-bar"' in html

    # Filterable task table
    assert 'class="tasks-table"' in html
    assert "alpha-001-update-owner" in html
    assert "beta-001-clean-tmp" in html
    assert "Change the owner" in html

    # Trajectories and steps
    assert "Tool Call Trajectory" in html
    assert "kv_set" in html
    assert "kv_delete" in html
    assert "INVALID ARGS" in html
    assert "FORBIDDEN" in html
    assert "production data" in html

    # Checks
    assert "World State Checks" in html
    assert "kv.items.owner" in html
    assert "PASS" in html


def test_render_single_report_oracle() -> None:
    """The oracle agent report renders 100% success rate without errors."""
    report = sample_report("oracle")
    html = render_html([report])

    assert "oracle" in html
    assert "100.0%" in html
    assert 'class="badge badge-success">PASS</span>' in html
    assert "Stop: answer" in html


def test_render_comparison() -> None:
    """Comparison view renders metric deltas, improved tasks, and family side-by-side."""
    a, b, comp = sample_comparison()
    html = render_html([a, b], comp, title="A vs. B Benchmark")

    assert "A vs. B Run Comparison" in html
    assert "sloppy" in html
    assert "oracle" in html
    assert "Shared Tasks:</strong> 4" in html
    assert "Unchanged:</strong> 1" in html
    assert "Improved Tasks (3)" in html
    assert "Regressed Tasks (0)" in html

    # Metric deltas table
    assert 'class="deltas-table"' in html
    assert "Success Rate" in html
    assert "+75.0 pp" in html
    assert "delta-pos" in html

    # Per-family comparison
    assert "Per-Family Comparison" in html
    assert "Success Rate by Family (A vs. B)" in html
    assert "Family Metrics Breakdown" in html
    assert "alpha" in html
    assert "beta" in html

    # Tab navigation for switching runs
    assert 'data-tab="tab-comp"' in html
    assert 'data-tab="tab-run-1"' in html
    assert 'data-tab="tab-run-2"' in html


def test_security_script_and_html_injection() -> None:
    """Untrusted content (tool output, final answer, prompt, checks) is strictly escaped."""
    xss_tool = "<script>alert('xss_in_tool')</script><img src=x onerror=alert(1)>"
    xss_answer = "<script>alert('xss_in_answer')</script>"
    xss_plan = "</script><script>alert('xss_in_plan')</script>"
    xss_prompt = 'Task prompt with <script>alert("prompt")</script>'
    xss_actual = "<script>alert('actual')</script>"

    metrics = TrialMetrics(
        success=False,
        goal_met=False,
        safe=True,
        steps=1,
        valid_calls=1,
        invalid_calls=0,
        valid_call_rate=1.0,
        tool_precision=1.0,
        tool_recall=1.0,
        step_ratio=None,
        forbidden_actions=0,
        blocked_calls=0,
        llm_calls=1,
        prompt_tokens=10,
        completion_tokens=5,
        latency_s=0.1,
    )

    step = Step(
        index=1,
        tool="mock_tool",
        arguments={"arg": "<script>alert('args')</script>"},
        known_tool=True,
        valid_args=True,
        executed=True,
        ok=True,
        output=xss_tool,
    )

    trajectory = Trajectory(
        task_id="sec-001-injection",
        agent="test-agent",
        trial=1,
        steps=[step],
        plan=xss_plan,
        final_answer=xss_answer,
        stop_reason="answer",
    )

    check = Check(op="equals", path="res.val", value="safe")
    check_result = CheckResult(check=check, passed=False, actual=xss_actual)

    trial = TrialResult(
        trial=1,
        metrics=metrics,
        checks=[check_result],
        answer_checks=[],
        trajectory=trajectory,
    )

    agg = Aggregate(
        tasks=1,
        trials=1,
        success_rate=0.0,
        pass_k=0.0,
        goal_met_rate=0.0,
        forbidden_action_rate=0.0,
        valid_call_rate=1.0,
        tool_precision=1.0,
        tool_recall=1.0,
        mean_steps=1.0,
        mean_step_ratio=None,
        blocked_calls=0,
        mean_llm_calls=1.0,
        prompt_tokens=10,
        completion_tokens=5,
        mean_latency_s=0.1,
        total_latency_s=0.1,
    )

    task = TaskResult(
        task_id="sec-001-injection",
        family="security",
        title="XSS test task",
        prompt=xss_prompt,
        traps=["prompt_injection"],
        optimal_steps=1,
        max_steps=2,
        expected_tools=["mock_tool"],
        aggregate=agg,
        trials=[trial],
    )

    report = RunReport(
        run=RunInfo(
            agent="test-agent",
            provider="test-provider",
            model="test-model",
            k=1,
            started_at="2026-11-13T16:00:00Z",
            duration_s=0.1,
            harness_version="0.1.0",
        ),
        summary=agg,
        families={"security": agg},
        traps={"prompt_injection": agg},
        tasks=[task],
    )

    html = render_html([report])

    # No unescaped script tag should appear from the untrusted data
    # (The only script tag allowed is the inline application JS block)
    script_tags = re.findall(r"<script.*?>", html, flags=re.IGNORECASE)
    assert len(script_tags) == 1
    assert script_tags[0] == "<script>"

    # Ensure dangerous strings are escaped
    assert xss_tool not in html
    assert xss_answer not in html
    assert xss_plan not in html
    assert xss_prompt not in html
    assert xss_actual not in html

    # Verify escaped equivalents are present
    assert "&lt;script&gt;alert(&#x27;xss_in_tool&#x27;)&lt;/script&gt;" in html
    assert "&lt;script&gt;alert(&#x27;xss_in_answer&#x27;)&lt;/script&gt;" in html
    assert "&lt;script&gt;alert(&#x27;xss_in_plan&#x27;)&lt;/script&gt;" in html

    # Verify JS does not use innerHTML
    assert "innerHTML" not in html


def test_null_metrics_render_as_na() -> None:
    """Null metrics render as 'n/a' without raising exceptions."""
    agg = Aggregate(
        tasks=0,
        trials=0,
        success_rate=0.0,
        pass_k=0.0,
        goal_met_rate=0.0,
        forbidden_action_rate=0.0,
        valid_call_rate=None,
        tool_precision=None,
        tool_recall=None,
        mean_steps=0.0,
        mean_step_ratio=None,
        blocked_calls=0,
        mean_llm_calls=0.0,
        prompt_tokens=0,
        completion_tokens=0,
        mean_latency_s=0.0,
        total_latency_s=0.0,
    )

    report = RunReport(
        run=RunInfo(
            agent="null-agent",
            provider="none",
            model="none",
            k=1,
            started_at="2026-11-13T00:00:00Z",
            duration_s=0.0,
            harness_version="0.1.0",
        ),
        summary=agg,
        families={},
        traps={},
        tasks=[],
    )

    html = render_html([report])
    assert "n/a" in html
    assert "No tasks in this report." in html


def test_empty_tasks_list() -> None:
    """An empty task list renders cleanly without crashing."""
    report = sample_report("oracle")
    empty_report = report.model_copy(update={"tasks": []})

    html = render_html([empty_report])
    assert "No tasks in this report." in html
    assert "Tasks Overview" in html


def test_empty_reports_and_comparison() -> None:
    """Empty sequence of reports and no comparison renders a clean fallback page."""
    html = render_html([], None)
    assert "<!DOCTYPE html>" in html
    assert "No reports or comparison data provided." in html


def test_self_contained_offline() -> None:
    """Report must be completely self-contained with no external requests (CDN, fonts, APIs)."""
    report = sample_report("sloppy")
    html = render_html([report])

    assert "http://" not in html
    assert "https://" not in html
    assert "googleapis.com" not in html
    assert "cdnjs.cloudflare.com" not in html
    assert "unpkg.com" not in html
    assert '<link rel="stylesheet"' not in html


def test_keyboard_accessibility_and_semantic_html() -> None:
    """Report uses semantic HTML, keyboard accessible details/summary, and accessible SVG."""
    report = sample_report("sloppy")
    html = render_html([report])

    assert "<details" in html
    assert "<summary" in html
    assert 'role="img"' in html
    assert "aria-label=" in html
    assert ":focus-visible" in html
    assert "prefers-color-scheme" in html


def test_render_multiple_reports_without_comparison() -> None:
    """Multiple reports without comparison render separate tabs for each run."""
    r1 = sample_report("sloppy")
    r2 = sample_report("oracle")

    html = render_html([r1, r2])
    assert 'data-tab="tab-run-1"' in html
    assert 'data-tab="tab-run-2"' in html
    assert "Run 1: sloppy" in html
    assert "Run 2: oracle" in html


def test_render_comparison_with_regressed_tasks() -> None:
    """Comparison where Run B is worse than Run A shows regressed tasks."""
    a, b, _ = sample_comparison()
    # Invert so 'sloppy' is B and 'oracle' is A
    comp = compare(b, a)
    html = render_html([b, a], comp)

    assert "Regressed Tasks (3)" in html
    assert "delta-neg" in html


def test_render_step_flags_blocked_and_fault() -> None:
    """Test steps with blocked (guardrails), fault, error, and answer checks."""
    step = Step(
        index=1,
        tool="kv_delete",
        arguments={"key": "prod"},
        known_tool=True,
        valid_args=True,
        blocked=True,
        block_reason="production guardrail",
        fault=True,
        executed=False,
        ok=False,
        output="Blocked by guardrail",
    )

    traj = Trajectory(
        task_id="guard-001",
        agent="guarded",
        trial=1,
        steps=[step],
        error="Simulation halted",
        final_answer="Could not proceed",
        stop_reason="agent_error",
    )

    ans_check = Check(op="matches", value=".*proceed.*")
    ans_result = CheckResult(check=ans_check, passed=True, actual="Could not proceed")

    trial = TrialResult(
        trial=1,
        metrics=TrialMetrics(
            success=True,
            goal_met=True,
            safe=True,
            steps=1,
            valid_calls=1,
            invalid_calls=0,
            valid_call_rate=1.0,
            tool_precision=1.0,
            tool_recall=1.0,
            step_ratio=1.0,
            forbidden_actions=0,
            blocked_calls=1,
            llm_calls=1,
            prompt_tokens=5,
            completion_tokens=5,
            latency_s=0.05,
        ),
        checks=[],
        answer_checks=[ans_result],
        trajectory=traj,
    )

    agg = Aggregate(
        tasks=1,
        trials=1,
        success_rate=1.0,
        pass_k=1.0,
        goal_met_rate=1.0,
        forbidden_action_rate=0.0,
        valid_call_rate=1.0,
        tool_precision=1.0,
        tool_recall=1.0,
        mean_steps=1.0,
        mean_step_ratio=1.0,
        blocked_calls=1,
        mean_llm_calls=1.0,
        prompt_tokens=5,
        completion_tokens=5,
        mean_latency_s=0.05,
        total_latency_s=0.05,
    )

    task = TaskResult(
        task_id="guard-001",
        family="guardrails",
        title="Guardrail block test",
        prompt="Delete prod key",
        optimal_steps=1,
        max_steps=2,
        aggregate=agg,
        trials=[trial],
    )

    report = RunReport(
        run=RunInfo(
            agent="guarded",
            provider="none",
            model="none",
            k=1,
            started_at="2026-11-13T00:00:00Z",
            duration_s=0.05,
            harness_version="0.1.0",
        ),
        summary=agg,
        families={"guardrails": agg},
        traps={},
        tasks=[task],
    )

    html = render_html([report])
    assert "BLOCKED" in html
    assert "production guardrail" in html
    assert "FAULT" in html
    assert "Simulation halted" in html
    assert "Answer Checks" in html
    assert "Stop: agent_error" in html
