# Linux/macOS/CI. On Windows run the `uv run ...` equivalents in docs/DEVEX.md.
.PHONY: install lint format typecheck test demo run-baseline run-guarded compare clean

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy

test:
	uv run pytest

demo:
	uv run agent-eval run --agent oracle --out out

run-baseline:
	uv run agent-eval run --agent baseline -k 3 --unload-every 20 --out out

run-guarded:
	uv run agent-eval run --agent guarded -k 3 --unload-every 20 --out out

compare:
	uv run agent-eval compare out/baseline.json out/guarded.json --out out --html

clean:
	rm -rf out .pytest_cache .ruff_cache .mypy_cache .coverage coverage.xml htmlcov dist build *.egg-info
