"""``files``: an in-memory file tree.

State and snapshot::

    {
        "files": {"<absolute path>": "<text>"},
        "readonly": ["<glob>", ...],
        "deleted": ["<path>", ...]
    }
"""

from __future__ import annotations

import fnmatch
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


_PATH = {"type": "string", "minLength": 1, "description": "Absolute file path (starting with '/')."}
_CONTENT = {"type": "string", "description": "Text content of the file."}
_PREFIX = {
    "type": "string",
    "description": "Path prefix filter (defaults to '/').",
    "default": "/",
}


@register
class FilesEnvironment(Environment):
    kind = "files"

    def load(self, state: dict[str, Any]) -> None:
        self.files: dict[str, str] = {str(k): str(v) for k, v in state.get("files", {}).items()}
        self.readonly: list[str] = [str(g) for g in state.get("readonly", [])]
        self.deleted: list[str] = [str(p) for p in state.get("deleted", [])]

    def tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="files_list",
                description="List every file path matching a prefix, sorted.",
                parameters=_params({"prefix": _PREFIX}, []),
            ),
            ToolSpec(
                name="files_read",
                description="Read the entire text content of a file.",
                parameters=_params({"path": _PATH}, ["path"]),
            ),
            ToolSpec(
                name="files_search",
                description="Search for a case-insensitive substring across files, returning up to 20 hits.",
                parameters=_params(
                    {
                        "query": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Substring to search for.",
                        },
                        "prefix": _PREFIX,
                    },
                    ["query"],
                ),
            ),
            ToolSpec(
                name="files_write",
                description="Create or overwrite a file with text content.",
                parameters=_params({"path": _PATH, "content": _CONTENT}, ["path", "content"]),
            ),
            ToolSpec(
                name="files_append",
                description="Append text content to an existing file.",
                parameters=_params({"path": _PATH, "content": _CONTENT}, ["path", "content"]),
            ),
            ToolSpec(
                name="files_move",
                description="Move or rename an existing file to a new destination path.",
                parameters=_params(
                    {
                        "source": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Source path of existing file.",
                        },
                        "destination": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Destination path.",
                        },
                    },
                    ["source", "destination"],
                ),
            ),
            ToolSpec(
                name="files_delete",
                description="Delete a file permanently. This cannot be undone.",
                parameters=_params({"path": _PATH}, ["path"]),
                destructive=True,
            ),
        ]

    def snapshot(self) -> dict[str, Any]:
        return {
            "files": dict(sorted(self.files.items())),
            "readonly": list(self.readonly),
            "deleted": list(self.deleted),
        }

    def _is_readonly(self, path: str) -> bool:
        return any(fnmatch.fnmatchcase(path, pattern) for pattern in self.readonly)

    def tool_files_list(self, prefix: str = "/") -> list[str]:
        return sorted(p for p in self.files if p.startswith(prefix))

    def tool_files_read(self, path: str) -> str:
        if path not in self.files:
            raise ToolFailure(f"file '{path}' not found")
        return self.files[path]

    def tool_files_search(self, query: str, prefix: str = "/") -> list[dict[str, Any]]:
        if not query:
            raise ToolFailure("search query cannot be empty")
        hits: list[dict[str, Any]] = []
        lower_query = query.lower()
        for path in sorted(self.files):
            if not path.startswith(prefix):
                continue
            for line_no, line in enumerate(self.files[path].splitlines(), start=1):
                if lower_query in line.lower():
                    hits.append({"path": path, "line": line_no, "text": line})
                    if len(hits) >= 20:
                        return hits
        return hits

    def tool_files_write(self, path: str, content: str) -> str:
        if self._is_readonly(path):
            raise ToolFailure(f"permission denied: '{path}' is read-only")
        status = "overwritten" if path in self.files else "created"
        self.files[path] = content
        return status

    def tool_files_append(self, path: str, content: str) -> str:
        if self._is_readonly(path):
            raise ToolFailure(f"permission denied: '{path}' is read-only")
        if path not in self.files:
            raise ToolFailure(f"file '{path}' not found")
        self.files[path] += content
        return "ok"

    def tool_files_move(self, source: str, destination: str) -> str:
        if source not in self.files:
            raise ToolFailure(f"file '{source}' not found")
        if self._is_readonly(source):
            raise ToolFailure(f"permission denied: '{source}' is read-only")
        if destination in self.files:
            raise ToolFailure(f"destination '{destination}' already exists")
        if self._is_readonly(destination):
            raise ToolFailure(f"permission denied: '{destination}' is read-only")
        self.files[destination] = self.files.pop(source)
        return "ok"

    def tool_files_delete(self, path: str) -> str:
        if path not in self.files:
            raise ToolFailure(f"file '{path}' not found")
        if self._is_readonly(path):
            raise ToolFailure(f"permission denied: '{path}' is read-only")
        del self.files[path]
        self.deleted.append(path)
        return "deleted"
