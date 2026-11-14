# Build report: agent-eval-harness v0.1.0

Date: 2026-11-14. Results measured at commit `bebef0d`; the commits after it change documentation only.

## What was built

A harness that runs a tool-using agent on YAML-defined tasks in deterministic mock environments and reports success, pass^k, tool-call correctness, trajectory efficiency, forbidden actions and cost.

- 32 tasks in 5 families (`api` 7, `calendar` 5, `files` 7, `tracker` 7, `triage` 6); 18 carry a trap (tool error, destructive action, prompt injection, ambiguity).
- 4 in-memory environments (files, tracker, calendar, REST-like API) behind one `World` choke point that validates arguments, applies scripted faults, flags forbidden calls and records every step.
- 2 agents under test on the same model: `baseline` (plain tool calling) and `guarded` (plan-then-act prompt, argument pre-validation, confirmation check before destructive tools). 2 model-free agents for self-testing: `oracle` and `noop`.
- Providers: Ollama (default), OpenAI-compatible and Anthropic as lazy optional extras.
- CLI: `agent-eval list | run | compare | report`, with checkpoint and `--resume`. A pytest plugin exposes each task as a test case.
- Outputs: JSON report (schema in `docs/schemas/`), self-contained HTML report, A vs. B comparison.

## Folder tree

```text
.github/workflows/ci.yml
docs/            CONTRACTS.md, RESEARCH.md, TASKS.md, DEVEX.md, REPO-META.md, BUILD-REPORT.md,
                 schemas/, sample-report.html, results/
examples/tasks/
src/agent_eval/  agents/, envs/, llm/, report/, cli.py, runner.py, world.py, metrics.py, ...
tasks/           api/, calendar/, files/, tracker/, triage/   (32 YAML files)
tests/           envs/, report/, tasks/, test_agents.py, test_cli.py, test_core.py, test_llm.py,
                 test_pytest_plugin.py
Dockerfile, Makefile, pyproject.toml, uv.lock, .env.example, .pre-commit-config.yaml
```

## Checks

Run on Windows 11, Python 3.12.10, on 2026-11-14.

| Check | Command | Result |
|---|---|---|
| Lint | `uv run ruff check .` | passed |
| Format | `uv run ruff format --check .` | passed |
| Types | `uv run mypy src` | no issues in 33 source files |
| Tests | `uv run pytest` | 209 passed, coverage 98.66% (threshold 80%) |
| Self-test, oracle | `uv run agent-eval run --agent oracle` | 32/32 tasks, 100.0% |
| Self-test, noop | `uv run agent-eval run --agent noop` | 0.0% |
| Confidential terms | `git grep -n -i -E` for the three employer names | no match |
| Secrets | `git grep` for common API key patterns | no key (the only matches are the CSS class `task-trajectories-section`) |

No test needs a live model.

## Real run

`qwen3:8b` through Ollama 0.40.16, temperature 0.7, 16k context, 16 GB laptop, 32 tasks, k = 3, 96 trials per agent.

```bash
uv run agent-eval run --agent baseline -k 3 --out out/full --unload-every 20
uv run agent-eval run --agent guarded  -k 3 --out out/full --unload-every 20
uv run agent-eval compare out/full/baseline.json out/full/guarded.json --out out/full --html
```

| Metric | A: baseline | B: guarded |
|---|---|---|
| Success rate | 75.0% | 77.1% |
| pass^k (k = 3) | 71.9% | 71.9% |
| Forbidden-action rate | 0.0% | 0.0% |
| Valid tool calls | 98.5% | 100.0% |
| Tool precision | 99.0% | 87.1% |
| Tool recall | 97.4% | 92.9% |
| Steps / optimal (successes) | 1.02 | 1.43 |
| Mean LLM calls per trial | 2.91 | 4.45 |
| Mean latency per trial (s) | 17.08 | 32.34 |
| Prompt tokens | 192,812 | 357,702 |
| Completion tokens | 9,781 | 23,808 |

B improved 5 tasks, regressed 6 and left 21 unchanged. The raw reports and logs are in `docs/results/`.

Run notes:

- The baseline run took 1,640 s in one session on 2026-11-13.
- The guarded run was interrupted after trial 21 on 2026-11-13, when the process was killed under memory pressure, and was completed on 2026-11-14 with `--resume` at the same commit. The resumed session took 2,550 s for 75 trials. `run.started_at`, `run.duration_s` and `run.client_stats` in `guarded.json` describe the resumed session only; the per-trial metrics cover all 96 trials.
- With the model loaded, the llama.cpp runner held about 7.8 GB and free memory dropped to about 450 MB at the lowest point observed. `--unload-every 20` unloaded the runner 19 times in the baseline run and 18 times in the resumed guarded session.
- Ollama client counters: baseline had 6 empty completions, 3 aborted completions and 3 fallback answers in 288 requests; the resumed guarded session had 2 aborted completions in 334 requests.

## Findings

- A and B are tied on success (+2.1 pp, two trials) and on pass^k.
- B costs about 1.9x the prompt tokens and 1.9x the latency per trial.
- The guardrails were not exercised: no forbidden call by either agent and zero blocked calls recorded for B. The A vs. B difference in this run comes from the prompt.
- B passed all 15 prompt-injection trials (A: 12) and more tool-error trials (88.9% vs. 66.7%), and did worse on the ambiguous tasks (33.3% vs. 66.7%) and on `triage` (55.6% vs. 72.2%).

## Known limitations

See the README. In short: one small model on one machine, one run per agent, 32 tasks with small slices, mock environments, closed-form answers only, unseeded sampling.

## Not done

- **LLM judge.** Not built. Every task has a closed-form answer, so nothing needs one in v0.1.
- **MCP adapter** (stretch goal). Not built; it is on the roadmap.
- **Hosted providers.** The OpenAI-compatible and Anthropic adapters are unit-tested against stubbed SDKs and were not run against the real APIs.
- **Docker.** The Dockerfile was written; the image was not built on this machine.
- **CI.** The workflow has not run, because the repository has no remote yet.
- **Guardrail effectiveness.** To be measured on a model that attempts a forbidden call.
- **Variance of the aggregate.** To be measured with repeated runs.
