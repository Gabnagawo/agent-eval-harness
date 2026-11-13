# Developer Experience & Workflows

This document describes developer tooling, CI/CD configuration, Docker workflows, and Windows-equivalent commands for `agent-eval-harness`.

---

## README Badges

Copy and paste this block directly under the title and value proposition in `README.md`:

```markdown
[![CI](https://github.com/Gabnagawo/agent-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/Gabnagawo/agent-eval-harness/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
```

---

## Command Reference: Makefile vs. Windows Powershell / Command Prompt

Because Windows does not include `make` natively, the table below provides the direct `uv` and PowerShell commands corresponding to each `Makefile` target:

| Target | Description | Makefile (Linux / macOS / CI) | Windows / PowerShell (`uv run`) |
|---|---|---|---|
| `install` | Install dependencies | `make install` | `uv sync` |
| `lint` | Check style with Ruff | `make lint` | `uv run ruff check .` <br> `uv run ruff format --check .` |
| `format` | Auto-format & fix code | `make format` | `uv run ruff format .` <br> `uv run ruff check --fix .` |
| `typecheck` | Static type checking | `make typecheck` | `uv run mypy` |
| `test` | Run tests with coverage | `make test` | `uv run pytest` |
| `demo` | Run model-free oracle self-test | `make demo` | `uv run agent-eval run --agent oracle --out out` |
| `run-baseline` | Evaluate baseline agent on Ollama | `make run-baseline` | `uv run agent-eval run --agent baseline -k 3 --unload-every 20 --out out` |
| `run-guarded` | Evaluate guarded agent on Ollama | `make run-guarded` | `uv run agent-eval run --agent guarded -k 3 --unload-every 20 --out out` |
| `compare` | Compare baseline vs. guarded | `make compare` | `uv run agent-eval compare out/baseline.json out/guarded.json --out out --html` |
| `clean` | Remove build, test, and run artifacts | `make clean` | PowerShell: <br> `Remove-Item -Recurse -Force out, .pytest_cache, .ruff_cache, .mypy_cache, .coverage, coverage.xml, htmlcov -ErrorAction SilentlyContinue` |

---

## Development Environment Setup

### 1. Prerequisites
- Python 3.12 (`python --version`)
- `uv` (Fast Python package manager)
- (Optional) Local Ollama instance running at `http://localhost:11434` with model `qwen3:8b`
- (Optional) Docker

### 2. Initial Setup
```bash
# Clone and enter repo
cd agent-eval-harness

# Create virtual environment and install dependencies
uv sync

# Copy environment template
cp .env.example .env
```

### 3. Pre-commit Hooks
The repository includes `.pre-commit-config.yaml` with Ruff linting, formatting, and file hygiene checks.
```bash
# Run against all files manually
uv run pre-commit run --all-files
```

---

## Continuous Integration (CI)

The GitHub Actions workflow is defined in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml):

- **Environment**: Ubuntu Latest, Python 3.12, Astral `setup-uv@v5` with caching.
- **Security**: Least-privilege `permissions: contents: read`.
- **Concurrency**: `cancel-in-progress: true` per branch to avoid redundant runs on fast pushes.
- **Steps**:
  1. `uv sync --locked`: Reproducible dependency installation from `uv.lock`.
  2. `uv run ruff check .`: Lint rules (`E`, `F`, `W`, `I`, `UP`, `B`, `SIM`, `C4`, `PT`, `RUF`, `N`, `PTH`).
  3. `uv run ruff format --check .`: Code formatting validation.
  4. `uv run mypy`: Strict static type checking.
  5. `uv run pytest`: Test suite with coverage enforcement (fail under 80%).
  6. `uv run agent-eval run --agent oracle --out out`: Model-free oracle benchmark self-test.
  7. `actions/upload-artifact@v4`: Uploads `out/` report and trajectory artifacts.

---

## Containerization (Docker)

The [`Dockerfile`](../Dockerfile) provides a hermetic execution environment based on `python:3.12-slim-bookworm` with a non-root user (`aeh`, UID 10001) and pinned `uv`:

### Build image:
```bash
docker build -t agent-eval-harness .
```

### Run default oracle self-test (no model or network needed):
```bash
docker run --rm agent-eval-harness
```

### Run evaluation against Ollama running on the host machine:
```bash
docker run --rm \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -v "$PWD/out:/app/out" \
  agent-eval-harness run --agent baseline -k 3 --unload-every 20 --out out
```
