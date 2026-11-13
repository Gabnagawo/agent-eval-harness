"""``kv``: a tiny key-value store. The reference environment: copy its shape for new ones.

State and snapshot::

    {"items": {"<key>": "<value>", ...}}
"""

from __future__ import annotations

from typing import Any

from agent_eval.envs.base import Environment, ToolFailure, register
from agent_eval.models import ToolSpec


def _params(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_KEY = {"type": "string", "minLength": 1, "description": "Key name."}


@register
class KVEnvironment(Environment):
    kind = "kv"

    def load(self, state: dict[str, Any]) -> None:
        self.items: dict[str, str] = {str(k): str(v) for k, v in state.get("items", {}).items()}

    def tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(name="kv_list", description="List every key, sorted."),
            ToolSpec(
                name="kv_get",
                description="Read the value stored under a key.",
                parameters=_params({"key": _KEY}, ["key"]),
            ),
            ToolSpec(
                name="kv_set",
                description="Create or overwrite a key.",
                parameters=_params(
                    {"key": _KEY, "value": {"type": "string", "description": "New value."}},
                    ["key", "value"],
                ),
            ),
            ToolSpec(
                name="kv_delete",
                description="Delete a key. This cannot be undone.",
                parameters=_params({"key": _KEY}, ["key"]),
                destructive=True,
            ),
        ]

    def snapshot(self) -> dict[str, Any]:
        return {"items": dict(sorted(self.items.items()))}

    def tool_kv_list(self) -> list[str]:
        return sorted(self.items)

    def tool_kv_get(self, key: str) -> str:
        if key not in self.items:
            raise ToolFailure(f"key '{key}' not found")
        return self.items[key]

    def tool_kv_set(self, key: str, value: str) -> str:
        self.items[key] = value
        return "ok"

    def tool_kv_delete(self, key: str) -> str:
        if key not in self.items:
            raise ToolFailure(f"key '{key}' not found")
        del self.items[key]
        return "deleted"
