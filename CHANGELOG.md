# Changelog

## v0.1.0 - 2026-11-14

- Core contracts: task schema, environment interface, world, checker, metrics, report schema.
- Mock environments: files, tracker, calendar and a REST-like API, deterministic and seeded.
- Task suite: 32 tasks in 5 families, 18 with a trap (tool error, destructive action, prompt injection, ambiguity).
- Agents: `baseline` and `guarded` (plan-then-act, argument pre-validation, confirmation before destructive tools), plus the model-free `oracle` and `noop` self-test agents.
- Providers: Ollama by default; OpenAI-compatible and Anthropic as optional extras.
- CLI: `agent-eval list | run | compare | report`, with checkpoint and `--resume`; pytest plugin.
- Reports: JSON, self-contained HTML with per-task trajectories, and A vs. B comparison.
- Real results for `baseline` vs. `guarded` on `qwen3:8b` (96 trials each), with raw output in `docs/results/`.
