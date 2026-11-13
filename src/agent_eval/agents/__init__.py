"""Agents under test, and the model-free agents that validate the harness itself."""

from __future__ import annotations

from collections.abc import Callable

from agent_eval.agents.base import Agent, AgentResult
from agent_eval.agents.scripted import NoopAgent, OracleAgent, ScriptedAgent
from agent_eval.agents.tool_calling import BaselineAgent, GuardedAgent
from agent_eval.llm.base import LLMProvider

MODEL_FREE = ("oracle", "noop")
AGENT_NAMES = ("baseline", "guarded", *MODEL_FREE)


def build_agent(name: str, provider_factory: Callable[[], LLMProvider] | None = None) -> Agent:
    """Build a built-in agent by name. Model-backed agents call ``provider_factory``."""
    if name == "oracle":
        return OracleAgent()
    if name == "noop":
        return NoopAgent()
    if name not in AGENT_NAMES:
        raise ValueError(f"unknown agent '{name}' (choose from: {', '.join(AGENT_NAMES)})")
    if provider_factory is None:
        raise ValueError(f"agent '{name}' needs an LLM provider")
    provider = provider_factory()
    return BaselineAgent(provider) if name == "baseline" else GuardedAgent(provider)


__all__ = [
    "AGENT_NAMES",
    "MODEL_FREE",
    "Agent",
    "AgentResult",
    "BaselineAgent",
    "GuardedAgent",
    "NoopAgent",
    "OracleAgent",
    "ScriptedAgent",
    "build_agent",
]
