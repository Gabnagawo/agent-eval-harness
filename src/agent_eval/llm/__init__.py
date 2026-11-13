"""LLM provider abstraction: Ollama by default, Anthropic and OpenAI as optional extras."""

from agent_eval.llm.base import (
    Completion,
    LLMError,
    LLMProvider,
    Message,
    ToolCall,
    assistant_message,
    complete,
    tool_message,
)
from agent_eval.llm.factory import get_provider

__all__ = [
    "Completion",
    "LLMError",
    "LLMProvider",
    "Message",
    "ToolCall",
    "assistant_message",
    "complete",
    "get_provider",
    "tool_message",
]
