# agent-eval-harness: evaluate tool-using AI agents on reproducible tasks.
#
# Build:
#   docker build -t agent-eval-harness .
#
# Run oracle self-test (default command, no LLM required):
#   docker run --rm agent-eval-harness
#
# Run against local Ollama on host machine:
#   docker run --rm -e OLLAMA_HOST=http://host.docker.internal:11434 \
#     -v "$PWD/out:/app/out" agent-eval-harness run --agent baseline -k 3 --unload-every 20 --out out
FROM python:3.12-slim-bookworm

# Copy pinned uv binary from official astral-sh image (Hadolint DL3006 compliance)
COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /uvx /bin/

# Create non-root user (Hadolint DL3002 compliance)
RUN useradd --create-home --uid 10001 aeh

WORKDIR /app

# Configure python and uv environment
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies first for layer caching
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-dev --no-install-project

# Copy source code and tasks
COPY src ./src
COPY tasks ./tasks

# Install the application and assign ownership to non-root user
RUN uv sync --locked --no-dev && mkdir -p /app/out && chown -R aeh:aeh /app

# Switch to non-root user and set PATH
USER aeh
ENV PATH="/app/.venv/bin:$PATH"

# Default command: oracle self-test
ENTRYPOINT ["agent-eval"]
CMD ["run", "--agent", "oracle", "--out", "out"]
