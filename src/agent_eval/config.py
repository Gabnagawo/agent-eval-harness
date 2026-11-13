"""Runtime configuration read from environment variables (and an optional ``.env``)."""

from __future__ import annotations

from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["ollama", "anthropic", "openai"]

DEFAULT_MODELS: dict[str, str | None] = {
    "ollama": "qwen3:8b",
    "anthropic": "claude-sonnet-5-5",
    "openai": None,  # deliberately unset: pass LLM_MODEL
}


class Settings(BaseSettings):
    """Every knob that affects an LLM call."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: ProviderName = "ollama"
    llm_model: str | None = None
    ollama_host: str = "http://localhost:11434"
    # Sent on every Ollama chat request. Every client of a shared Ollama server should send the
    # same value: a different num_ctx makes Ollama reload the model and evict it for the others.
    ollama_num_ctx: int = 16384
    # Low-RAM machines: unload the model after every N answered calls (0 = never), and
    # reload the runner once when it returns an empty or aborted completion.
    ollama_unload_every: int = 0
    ollama_reload_on_fault: bool = True
    llm_timeout: float = 180.0
    llm_max_retries: int = 2
    llm_max_tokens: int = 1024
    # Qwen3 recommended sampling for non-thinking mode (not greedy decoding).
    llm_temperature: float = 0.7
    llm_top_p: float = 0.8
    llm_top_k: int = 20
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    # Any OpenAI-compatible endpoint (vLLM, LM Studio, a gateway). Unset: api.openai.com.
    openai_base_url: str | None = None

    def resolved_model(self) -> str | None:
        """The model to use: ``LLM_MODEL`` if set, else the provider default."""
        return self.llm_model or DEFAULT_MODELS[self.llm_provider]
