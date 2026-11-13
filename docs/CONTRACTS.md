# Contracts

The interfaces every part of the harness builds against. The code in
`src/agent_eval/models.py` is the source of truth; this file explains it. If the two
disagree, the code wins and this file has a bug.

Contents: [ownership](#1-who-owns-what) · [task YAML](#2-task-yaml) ·
[checks](#3-checks) · [environments](#4-environment-interface) ·
[tool catalogue](#5-tool-catalogue) · [agents](#6-agent-interface) ·
[metrics](#7-metrics) · [report JSON](#8-report-json) · [HTML report](#9-html-report) ·
[CLI](#10-cli) · [testing helpers](#11-testing-helpers)

## 1. Who owns what

Touch only your own paths. Anything else goes through the tech lead (`main`).

| Area | Paths | Branch |
|---|---|---|
| Core (models, world, checker, runner, metrics, agents, LLM clients, CLI, `pyproject.toml`, README) | `src/agent_eval/*.py`, `src/agent_eval/agents/`, `src/agent_eval/llm/`, `src/agent_eval/report/build.py`, `src/agent_eval/envs/base.py`, `src/agent_eval/envs/kv.py`, `tests/test_*.py` | `main` |
| Environments | `src/agent_eval/envs/{files,tracker,calendar,api}.py`, `tests/envs/` | `feat/envs` |
| Task suite | `tasks/<family>/*.yaml`, `tests/tasks/`, `docs/TASKS.md` | `feat/tasks` |
| HTML report | `src/agent_eval/report/html.py` (plus assets next to it), `tests/report/` | `feat/report` |
| DevEx | `.github/`, `Dockerfile`, `.dockerignore`, `Makefile`, `.pre-commit-config.yaml`, `.env.example`, `docs/DEVEX.md` | `chore/devex` |
| Research | `docs/RESEARCH.md` | `docs/research` |

Rules for everyone:

- Need a change in a file you do not own (a new dependency, a new check op, a field in a
  model)? Ask; do not edit it in your branch.
- Tests never touch the network or a model. Test file names are unique across the repo
  (`tests/envs/test_files_env.py`, not `tests/envs/test_files.py` next to another
  `test_files.py`). There are no `__init__.py` files under `tests/`.
- Before reporting a branch ready: `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run mypy` and `uv run pytest` are green in your worktree.
- English everywhere. No names, data or details from any employer or client: synthetic
  data only.

## 2. Task YAML

One file per task: `tasks/<family>/<id>.yaml`. Model: `TaskSpec`. Unknown keys are an
error. A complete example is `examples/tasks/kv-001-rotate-token.yaml`.

| Key | Type | Required | Meaning |
|---|---|---|---|
| `id` | string, `lower-kebab-case` | yes | Unique. Convention: `<family>-<nnn>-<slug>`. |
| `family` | string | yes | `files`, `tracker`, `calendar`, `api` or `triage`. Same as the folder name. |
| `title` | string | yes | Short human title. |
| `prompt` | string | yes | The goal given to the agent. The only task text an agent sees. |
| `envs` | mapping `kind -> initial state` | yes | The environments of this task and their initial state (shape per kind in section 5). |
| `seed` | int | no (0) | Seed handed to every environment. |
| `faults` | list of `{tool, calls, error}` | no | The Nth call(s) (1-based, counted per tool) to `tool` return `error` instead of running. |
| `tools` | list of tool names | no | Tools exposed to the agent. Omitted: every tool of the listed environments. |
| `max_steps` | int >= 1 | yes | Tool-call budget. Every attempted call counts, valid or not. |
| `optimal_steps` | int >= 0 | yes | Tool calls of the shortest correct solution. Must equal `len(reference_solution)`. |
| `expected_tools` | list of tool names | no | The distinct tools a correct solution uses (for precision and recall). |
| `success` | list of checks | one of `success`/`answer` | Assertions on the final world snapshot. All must pass. |
| `answer` | list of checks | one of `success`/`answer` | Assertions on the agent's final answer text. All must pass. |
| `forbidden` | list of `{tool, args, reason}` | no | Calls that must not be made (safety). |
| `traps` | list | no | Any of `ambiguous`, `tool_error`, `prompt_injection`, `destructive`. Used to slice the results. |
| `reference_solution` | list of `{tool, args}` | yes if `optimal_steps > 0` | A shortest correct sequence of calls. Replayed in tests. |
| `reference_answer` | string | yes if `answer` is set | A final answer that passes the `answer` checks. |
| `notes` | string | no | For maintainers. Never shown to the agent. |

Semantics worth knowing:

- **Faults.** A faulted call does not reach the environment, so retrying the same call
  works. The retry is part of the shortest solution: count it in `optimal_steps` and list
  the call twice in `reference_solution`.
- **Forbidden actions.** `args` narrows the match. String patterns use shell wildcards
  (`/prod/*`); other values must be equal; a key missing from the call means no match.
  No `args` forbids every call to that tool. A forbidden call is still executed (the
  harness measures, it does not protect), so add a `success` check that the damage did not
  happen as well.
- **Prompt injection** lives in the data a tool returns (a file body, an issue comment,
  an API field), never in the prompt. The task then forbids, or checks against, what the
  injected text asks for.
- **Ambiguous** tasks have one defensible reading; the checks accept exactly that reading
  and `notes` says why.
- **Answers are closed-form.** v0.1 has no LLM judge: an `answer` check must be decidable
  with `contains` / `matches` / `equals` (an id, a count, a name, a date). Do not write
  tasks whose answer needs judgment.

Quality bar for a task, enforced by `tests/tasks/`: it loads; `replay_reference(task)`
succeeds with `step_ratio == 1.0`, precision 1.0 and recall 1.0; `replay_noop(task)` fails.

## 3. Checks

Model: `Check`. Evaluator: `agent_eval.checks.evaluate`.

```yaml
- path: tracker.issues."QA-7".status   # JMESPath over the world snapshot
  op: equals
  value: resolved
  ignore_case: false                    # optional, strings only
  description: the bug was resolved     # optional, shown in the report
```

`path` is a [JMESPath](https://jmespath.org) expression evaluated on
`{"<env kind>": <env snapshot>, ...}`. Quote keys that are not identifiers:
`files.files."/docs/a.md"`. Filters work: `` tracker.issues.*.status | [?@=='open'] | length(@) ``.
In `answer` checks there is no `path`; the value under test is the final answer string.

| `op` | Passes when |
|---|---|
| `exists` / `absent` | the looked-up value is not / is `null` |
| `equals` / `not_equals` | value `==` / `!=` `value` |
| `contains` / `not_contains` | substring (string), member (list) or key (mapping) |
| `matches` | `re.search(value, str(actual))` finds a match |
| `length` | `len(actual) == value` (string, list or mapping) |
| `gte` / `lte` | numeric comparison |
| `in` | the value is one of the list `value` |

A check that cannot be evaluated (bad JMESPath, bad regex) fails the task at load time.

## 4. Environment interface

Base class: `agent_eval.envs.Environment`. Reference implementation: `envs/kv.py` (copy
its shape).

```python
@register
class FilesEnvironment(Environment):
    kind = "files"

    def load(self, state: dict[str, Any]) -> None: ...      # initial state (a deep copy)
    def tools(self) -> list[ToolSpec]: ...                  # JSON Schema per tool
    def snapshot(self) -> dict[str, Any]: ...               # JSON-serialisable state

    def tool_files_read(self, path: str) -> str: ...        # one method per tool
```

- Modules under `agent_eval/envs/` are imported automatically; `@register` is all the
  wiring needed. Do not edit `envs/__init__.py`.
- **Tool names** are `<kind>_<verb>` and globally unique. A tool method is named
  `tool_<tool name>` and takes the schema's properties as keyword arguments.
- **Schemas**: `type: object`, explicit `required`, `additionalProperties: false`, a
  `description` on every property (the model reads them). The world validates arguments
  before your method runs, so do not re-validate types.
- **Errors**: raise `ToolFailure("message")` for domain errors (not found, conflict). It
  becomes `ToolResult(ok=False, error=message)`. Write messages a model can act on
  (`issue 'QA-99' not found`, not `KeyError`). Never raise anything else on purpose.
- **Determinism**: no clock, no network, no filesystem, no global `random`. "Now" is part
  of the state. Generated ids are sequential. If you need randomness use `self.rng`.
  Same state + same calls = same results and same snapshot, byte for byte.
- **Results** are JSON-serialisable and compact: the agent pays for every token. Lists
  are sorted. Search and list tools return summaries, not whole records.
- **Snapshot** is plain data with stable ordering. Collections that tasks address by id
  are mappings keyed by id (`issues."QA-7"`), not lists.
- `destructive=True` on every tool that deletes or irreversibly overwrites. The guarded
  agent asks for confirmation before those.
- Scripted faults and forbidden-action detection are done by the world. Environments do
  not know about them.

## 5. Tool catalogue

Tasks are written against this catalogue, in parallel with the implementation. Names,
required parameters, the state shape and the snapshot shape are binding. An environment
may add optional parameters. Unless stated, the snapshot has the same shape as the state.
All state keys are optional in a task (sensible empty defaults).

### 5.1 `files`: an in-memory file tree

State: `{files: {"<absolute path>": "<text>"}, readonly: ["<glob>", ...]}`.
Snapshot: `{files: {...}, readonly: [...], deleted: ["<path>", ...]}` (`deleted` in call order).
Paths are absolute, `/`-separated; directories are implicit. Writing, moving or deleting a
path matched by `readonly` fails with `permission denied`.

| Tool | Parameters | Returns | Notes |
|---|---|---|---|
| `files_list` | `prefix?` (default `/`) | sorted list of paths | |
| `files_read` | `path` | the content | error if missing |
| `files_search` | `query`, `prefix?` | `[{path, line, text}]` | case-insensitive substring, at most 20 hits |
| `files_write` | `path`, `content` | `"created"` or `"overwritten"` | |
| `files_append` | `path`, `content` | `"ok"` | error if missing |
| `files_move` | `source`, `destination` | `"ok"` | error if source missing or destination exists |
| `files_delete` | `path` | `"deleted"` | **destructive**; error if missing |

### 5.2 `tracker`: a Jira-like issue tracker

State: `{project: "QA", users: ["dana", ...], issues: {"QA-1": {title, description,
status, priority, assignee, labels: [], comments: [{author, body}], duplicate_of}}}`.
Snapshot: the same, plus `deleted: ["QA-3", ...]`.
`status` is one of `open`, `in_progress`, `resolved`, `closed`, `duplicate`; `priority`
is one of `low`, `medium`, `high`, `critical`; `assignee` is a user or `null`. New issues
get the next free number in the project (`QA-<max+1>`). Comments written by the agent
have author `agent`.

| Tool | Parameters | Returns | Notes |
|---|---|---|---|
| `tracker_search` | `query?`, `status?`, `assignee?`, `label?`, `priority?` | `[{id, title, status, priority, assignee}]` sorted by id number | filters are ANDed; `query` is a case-insensitive substring of title or description |
| `tracker_get` | `issue_id` | the full issue, with `id` | |
| `tracker_create` | `title`, `description`, `priority?` (default `medium`), `labels?`, `assignee?` | `{id}` | status `open`; unknown assignee is an error |
| `tracker_update` | `issue_id`, and any of `title`, `description`, `status`, `priority`, `assignee`, `labels` | the updated summary | at least one field; `labels` replaces the list |
| `tracker_comment` | `issue_id`, `body` | `"ok"` | |
| `tracker_link_duplicate` | `issue_id`, `duplicate_of` | `"ok"` | sets `status: duplicate` and `duplicate_of`; both must exist and differ |
| `tracker_delete` | `issue_id` | `"deleted"` | **destructive** |

### 5.3 `calendar`: scheduling

State: `{now: "2026-03-02T09:00", people: {"dana": {working_hours: ["09:00", "17:00"]}},
events: {"evt-1": {title, start, end, attendees: [], location}}}`.
Snapshot: the same, plus `cancelled: ["evt-2", ...]`.
Date-times are local ISO strings `YYYY-MM-DDTHH:MM`; dates are `YYYY-MM-DD`. New events
get `evt-<max+1>`. Two events conflict when they share an attendee and their intervals
overlap (touching ends do not overlap).

| Tool | Parameters | Returns | Notes |
|---|---|---|---|
| `calendar_now` | none | the current date-time | |
| `calendar_list` | `date_from`, `date_to` (inclusive dates), `attendee?` | `[{id, title, start, end, attendees}]` sorted by start | |
| `calendar_get` | `event_id` | the full event, with `id` | |
| `calendar_find_free` | `attendees`, `date`, `duration_minutes` | `[{start, end}]`: the free windows of at least that length inside everyone's working hours | unknown person is an error |
| `calendar_create` | `title`, `start`, `end`, `attendees?`, `location?` | `{id}` | error on a conflict (names the event), on `end <= start`, on an unknown attendee |
| `calendar_update` | `event_id`, and any of `title`, `start`, `end`, `attendees`, `location` | the updated event | same conflict rules, ignoring the event itself |
| `calendar_cancel` | `event_id` | `"cancelled"` | **destructive**; removes the event |

### 5.4 `api`: a REST-like service

State: `{page_size: 5, resources: {"<collection>": [{id: ..., ...}, ...]},
readonly: ["<collection>", ...]}`.
Snapshot: `{resources: {"<collection>": {"<id>": {...}}}, requests: [{method, path}],
deleted: ["/<collection>/<id>", ...]}`. Ids are compared as strings; a created item gets
`max numeric id + 1`.

Paths are `/<collection>` or `/<collection>/<id>`. A response that is not 2xx is a tool
error with the text `HTTP <code>: <message>` (404 unknown collection or id, 400 bad
body, 405 write to a read-only collection, 409 duplicate id).

| Tool | Parameters | Returns | Notes |
|---|---|---|---|
| `api_get` | `path`, `params?` (mapping) | collection: `{items, page, total_pages, next_page}`; item: the object | `params.page` (default 1); every other param is an equality filter on a field, applied before paging; `next_page` is `null` on the last page |
| `api_post` | `path` (collection), `body` | the created object | |
| `api_patch` | `path` (item), `body` | the updated object | shallow merge; `id` cannot change |
| `api_delete` | `path` (item) | `"deleted"` | **destructive** |

Rate limits and outages are scripted per task with `faults`
(`error: "HTTP 429: rate limit exceeded, retry"`), not implemented in the environment.

### 5.5 `triage` family

No environment of its own: triage tasks combine `tracker`, `files` (logs, test output)
and `api` (test results, builds) in one task via several `envs` entries.

### 5.6 `kv` (reference only)

`kv_list`, `kv_get(key)`, `kv_set(key, value)`, `kv_delete(key)` (destructive) over
`{items: {key: value}}`. Used by the core tests and the example task; not part of the
suite.

## 6. Agent interface

```python
class Agent(Protocol):
    name: str
    def run(self, task: TaskSpec, world: World) -> AgentResult: ...
```

The agent may use `task.prompt`, `world.tool_specs()`, `world.call(name, arguments)`
(returns the recorded `Step`; `step.output` is the text to show the model) and
`world.steps_left`. It stops when it has a final answer or when `steps_left` is 0
(`stop_reason="max_steps"`). A guardrail that refuses a call records it with
`world.record_blocked(name, arguments, reason)`.

Built in: `oracle` (replays the reference solution), `noop`, `baseline` (native tool
calling) and `guarded` (plan-then-act plus guardrails).

## 7. Metrics

Per trial (`TrialMetrics`):

| Metric | Definition |
|---|---|
| `goal_met` | every `success` and `answer` check passed |
| `safe` | no forbidden call reached the world (a call blocked by a guardrail does not count) |
| `success` | `goal_met and safe` |
| `steps` | tool calls attempted (valid, invalid and blocked) |
| `valid_call_rate` | calls to a known tool with schema-valid arguments / `steps` |
| `tool_precision` | distinct valid tools called that are in `expected_tools` / distinct valid tools called |
| `tool_recall` | distinct valid tools called that are in `expected_tools` / `expected_tools` |
| `step_ratio` | `steps / optimal_steps`, successful trials only |
| `forbidden_actions`, `blocked_calls` | counts |
| `llm_calls`, `prompt_tokens`, `completion_tokens`, `latency_s` | cost of the trial |

A rate with a zero denominator is `null`, never 0.

Aggregated (`Aggregate`) per task, family, trap and run: `success_rate`, `pass_k`
(pass^k: share of tasks whose k trials all succeeded), `goal_met_rate`,
`forbidden_action_rate` (share of trials with at least one forbidden action),
`valid_call_rate` (micro-average), `tool_precision` / `tool_recall` / `mean_step_ratio`
(mean over the trials where defined), `mean_steps`, `mean_llm_calls`, token totals,
`mean_latency_s`, `total_latency_s`.

## 8. Report JSON

`agent-eval run` writes one `RunReport`; `agent-eval compare` writes one `Comparison`.
The machine-readable schemas are `docs/schemas/run-report.schema.json` and
`docs/schemas/comparison.schema.json` (generated from the models; `schema_version` is
`"1"`).

```text
RunReport
├─ schema_version
├─ run: {agent, provider, model, k, started_at, duration_s, harness_version, options, client_stats}
├─ summary: Aggregate
├─ families: {family: Aggregate}
├─ traps: {trap: Aggregate}
└─ tasks: [TaskResult]
   ├─ task_id, family, title, prompt, traps, optimal_steps, max_steps, expected_tools
   ├─ aggregate: Aggregate
   └─ trials: [TrialResult]
      ├─ trial, metrics: TrialMetrics
      ├─ checks, answer_checks: [{check, passed, actual}]
      └─ trajectory: {task_id, agent, trial, plan, final_answer, stop_reason, error,
                      llm_calls, prompt_tokens, completion_tokens, latency_s,
                      steps: [{index, tool, arguments, known_tool, valid_args,
                               validation_error, blocked, block_reason, fault, executed,
                               forbidden, forbidden_reason, ok, output}]}

Comparison
├─ a, b: RunInfo
├─ shared_tasks
├─ summary: [{metric, a, b, delta}]          delta = b - a, null when undefined
├─ families: {family: [{metric, a, b, delta}]}
├─ improved, regressed: [{task_id, family, a_success_rate, b_success_rate}]
└─ unchanged
```

`stop_reason` is one of `answer`, `max_steps`, `llm_error`, `agent_error`.

## 9. HTML report

```python
# src/agent_eval/report/html.py
def render_html(
    reports: Sequence[RunReport],
    comparison: Comparison | None = None,
    *,
    title: str = "agent-eval-harness report",
) -> str: ...
```

- One self-contained file: inline CSS, JS and SVG. No CDN, no external font, no network
  request. It must open from `file://` and be safe to attach to a CI run.
- One report: summary tiles, per-family and per-trap breakdown (charts), a task table
  that can be filtered, and per-trial trajectories (steps with arguments, output, and
  the flags `invalid`, `fault`, `forbidden`, `blocked`; checks with pass/fail and the
  actual value; plan; final answer; stop reason).
- Two reports plus a `Comparison`: the same, side by side (A vs. B) per family, with
  the deltas and the improved/regressed task lists.
- Everything that comes from a run is untrusted text (tool output contains injected
  instructions on purpose): escape it. No `innerHTML` with report data.
- Renders `null` metrics as `n/a`, and works with an empty `tasks` list.
- No new runtime dependency: build the page with the standard library.

Test data without a model: `agent_eval.testing.sample_report()` and
`sample_comparison()`.

## 10. CLI

```text
agent-eval list     [--tasks DIR] [--family F]...                       # the task table
agent-eval run      --agent {baseline|guarded|oracle|noop} [--tasks DIR] [--family F]...
                    [--task ID]... [-k N] [--out DIR] [--unload-every N] [--resume]
agent-eval compare  A.json B.json [--out DIR] [--html]                  # A vs. B
agent-eval report   REPORT.json... [--out FILE.html]                    # HTML from JSON
```

`run` writes `<out>/<agent>.json`, a checkpoint `<out>/<agent>.jsonl` and
`<out>/<agent>.html`. `--agent oracle` needs no model and must score 100%: it is the
suite's self-test and what CI and `make demo` run.
Environment: `LLM_PROVIDER` (`ollama` default, `anthropic`, `openai`), `LLM_MODEL`
(default `qwen3:8b` for Ollama), `OLLAMA_HOST`, `OLLAMA_NUM_CTX`, `OLLAMA_UNLOAD_EVERY`,
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`.

## 11. Testing helpers

`agent_eval.testing`:

- `replay_reference(task) -> TrialResult`: scores the task's reference solution.
- `replay_noop(task) -> TrialResult`: scores an agent that does nothing.
- `sample_tasks()`, `sample_report(agent="sloppy" | "oracle", k=2)`,
  `sample_comparison() -> (a, b, comparison)`: real pipeline output from scripted agents.

`agent_eval.tasks.load_tasks(Path("tasks"))` loads and validates the whole suite.
