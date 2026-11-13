"""Build the configured provider from :class:`Settings`."""

from __future__ import annotations

from agent_eval.config import Settings
from agent_eval.llm.base import LLMError, LLMProvider
from agent_eval.llm.ollama import OllamaProvider


def get_provider(settings: Settings | None = None) -> LLMProvider:
    """Return the provider selected by ``LLM_PROVIDER`` (default: Ollama ``qwen3:8b``)."""
    settings = settings or Settings()
    model = settings.resolved_model()
    if model is None:
        raise LLMError(f"Set LLM_MODEL for provider '{settings.llm_provider}'")
    if settings.llm_provider == "anthropic":
        from agent_eval.llm.anthropic import AnthropicProvider

        key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
        return AnthropicProvider(
            model,
            key,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
            max_tokens=settings.llm_max_tokens,
        )
    if settings.llm_provider == "openai":
        from agent_eval.llm.openai import OpenAIProvider

        key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        return OpenAIProvider(
            model,
            key,
            base_url=settings.openai_base_url,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )
    return OllamaProvider(
        model,
        settings.ollama_host,
        num_ctx=settings.ollama_num_ctx,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        top_p=settings.llm_top_p,
        top_k=settings.llm_top_k,
        unload_every=settings.ollama_unload_every,
        reload_on_fault=settings.ollama_reload_on_fault,
    )
