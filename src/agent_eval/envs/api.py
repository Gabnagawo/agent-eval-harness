"""``api``: a REST-like mock service.

State and snapshot::

    {
        "page_size": 5,
        "resources": {
            "users": [{"id": 1, "name": "Dana"}, ...]
        },
        "readonly": ["users", ...]
    }

Snapshot::

    {
        "resources": {
            "users": {
                "1": {"id": 1, "name": "Dana"}
            }
        },
        "requests": [{"method": "GET", "path": "/users"}],
        "deleted": ["/users/1", ...]
    }
"""

from __future__ import annotations

import copy
import math
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


def _split_path(path: str) -> list[str]:
    return [p for p in path.strip().split("/") if p]


@register
class ApiEnvironment(Environment):
    kind = "api"

    def load(self, state: dict[str, Any]) -> None:
        self.page_size: int = int(state.get("page_size", 5))
        self.readonly: list[str] = [str(r) for r in state.get("readonly", [])]
        self.requests: list[dict[str, str]] = [
            {"method": str(r.get("method", "")), "path": str(r.get("path", ""))}
            for r in state.get("requests", [])
        ]
        self.deleted: list[str] = [str(d) for d in state.get("deleted", [])]

        self.resources: dict[str, list[dict[str, Any]]] = {}
        for col, items in state.get("resources", {}).items():
            self.resources[str(col)] = copy.deepcopy(items)

    def tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="api_get",
                description="GET a collection (with pagination and equality filters) or a single item by path.",
                parameters=_params(
                    {
                        "path": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Path to collection ('/<collection>') or item ('/<collection>/<id>').",
                        },
                        "params": {
                            "type": "object",
                            "description": "Optional query parameters: 'page' and field equality filters.",
                        },
                    },
                    ["path"],
                ),
            ),
            ToolSpec(
                name="api_post",
                description="POST to create a new item in a collection.",
                parameters=_params(
                    {
                        "path": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Collection path ('/<collection>').",
                        },
                        "body": {
                            "type": "object",
                            "description": "JSON object representing the resource to create.",
                        },
                    },
                    ["path", "body"],
                ),
            ),
            ToolSpec(
                name="api_patch",
                description="PATCH to partially update an existing item by path.",
                parameters=_params(
                    {
                        "path": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Item path ('/<collection>/<id>').",
                        },
                        "body": {
                            "type": "object",
                            "description": "JSON object with fields to update.",
                        },
                    },
                    ["path", "body"],
                ),
            ),
            ToolSpec(
                name="api_delete",
                description="DELETE an item by path permanently. This cannot be undone.",
                parameters=_params(
                    {
                        "path": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Item path ('/<collection>/<id>').",
                        }
                    },
                    ["path"],
                ),
                destructive=True,
            ),
        ]

    def snapshot(self) -> dict[str, Any]:
        snap_resources: dict[str, dict[str, Any]] = {}
        for col in sorted(self.resources):
            items_dict: dict[str, Any] = {}
            sorted_items = sorted(
                self.resources[col],
                key=lambda x: (
                    int(str(x.get("id"))) if str(x.get("id", "")).isdigit() else 0,
                    str(x.get("id", "")),
                ),
            )
            for item in sorted_items:
                item_id = str(item.get("id", ""))
                items_dict[item_id] = copy.deepcopy(item)
            snap_resources[col] = items_dict

        return {
            "resources": snap_resources,
            "requests": [dict(r) for r in self.requests],
            "deleted": list(self.deleted),
        }

    def _record_request(self, method: str, path: str) -> None:
        self.requests.append({"method": method, "path": path})

    def tool_api_get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._record_request("GET", path)
        parts = _split_path(path)
        if len(parts) == 1:
            collection = parts[0]
            if collection not in self.resources:
                raise ToolFailure(f"HTTP 404: collection '{collection}' not found")

            query_params = params or {}
            page_val = query_params.get("page", 1)
            try:
                page = int(page_val)
                if page < 1:
                    raise ValueError
            except (ValueError, TypeError):
                raise ToolFailure(f"HTTP 400: invalid page number '{page_val}'") from None

            filters = {k: v for k, v in query_params.items() if k != "page"}
            filtered: list[dict[str, Any]] = []
            for item in self.resources[collection]:
                match = True
                for fk, fv in filters.items():
                    if fk not in item:
                        match = False
                        break
                    if item[fk] != fv and str(item[fk]) != str(fv):
                        match = False
                        break
                if match:
                    filtered.append(item)

            total_items = len(filtered)
            total_pages = math.ceil(total_items / self.page_size) if total_items > 0 else 0
            start_idx = (page - 1) * self.page_size
            end_idx = start_idx + self.page_size
            paged_items = copy.deepcopy(filtered[start_idx:end_idx])
            next_page = page + 1 if page < total_pages else None

            return {
                "items": paged_items,
                "page": page,
                "total_pages": total_pages,
                "next_page": next_page,
            }

        if len(parts) == 2:
            collection, item_id = parts[0], parts[1]
            if collection not in self.resources:
                raise ToolFailure(f"HTTP 404: collection '{collection}' not found")

            for item in self.resources[collection]:
                if str(item.get("id")) == item_id:
                    return copy.deepcopy(item)
            raise ToolFailure(f"HTTP 404: item '{item_id}' not found in '{collection}'")

        raise ToolFailure(f"HTTP 400: invalid path '{path}'")

    def tool_api_post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        self._record_request("POST", path)
        parts = _split_path(path)
        if len(parts) != 1:
            raise ToolFailure(f"HTTP 400: POST path must be '/<collection>', got '{path}'")

        collection = parts[0]
        if collection in self.readonly:
            raise ToolFailure(f"HTTP 405: collection '{collection}' is read-only")
        if collection not in self.resources:
            raise ToolFailure(f"HTTP 404: collection '{collection}' not found")
        if not isinstance(body, dict):
            raise ToolFailure("HTTP 400: bad body, expected JSON object")

        if "id" in body and body["id"] is not None:
            target_id = str(body["id"])
            if any(str(item.get("id")) == target_id for item in self.resources[collection]):
                raise ToolFailure(f"HTTP 409: duplicate id '{target_id}'")
            new_item = copy.deepcopy(body)
        else:
            numbers: list[int] = []
            for item in self.resources[collection]:
                val = str(item.get("id", ""))
                if val.isdigit():
                    numbers.append(int(val))
            prefix = f"/{collection}/"
            for d in self.deleted:
                if d.startswith(prefix):
                    part = d[len(prefix) :]
                    if part.isdigit():
                        numbers.append(int(part))
            next_num = max(numbers, default=0) + 1
            new_item = copy.deepcopy(body)
            new_item["id"] = next_num

        self.resources[collection].append(new_item)
        return copy.deepcopy(new_item)

    def tool_api_patch(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        self._record_request("PATCH", path)
        parts = _split_path(path)
        if len(parts) != 2:
            raise ToolFailure(f"HTTP 400: PATCH path must be '/<collection>/<id>', got '{path}'")

        collection, target_id = parts[0], parts[1]
        if collection in self.readonly:
            raise ToolFailure(f"HTTP 405: collection '{collection}' is read-only")
        if collection not in self.resources:
            raise ToolFailure(f"HTTP 404: collection '{collection}' not found")
        if not isinstance(body, dict):
            raise ToolFailure("HTTP 400: bad body, expected JSON object")

        target_item: dict[str, Any] | None = None
        for item in self.resources[collection]:
            if str(item.get("id")) == target_id:
                target_item = item
                break

        if target_item is None:
            raise ToolFailure(f"HTTP 404: item '{target_id}' not found in '{collection}'")

        if "id" in body and str(body["id"]) != target_id:
            raise ToolFailure("HTTP 400: cannot change item id")

        for k, v in body.items():
            if k != "id":
                target_item[k] = v

        return copy.deepcopy(target_item)

    def tool_api_delete(self, path: str) -> str:
        self._record_request("DELETE", path)
        parts = _split_path(path)
        if len(parts) != 2:
            raise ToolFailure(f"HTTP 400: DELETE path must be '/<collection>/<id>', got '{path}'")

        collection, target_id = parts[0], parts[1]
        if collection in self.readonly:
            raise ToolFailure(f"HTTP 405: collection '{collection}' is read-only")
        if collection not in self.resources:
            raise ToolFailure(f"HTTP 404: collection '{collection}' not found")

        found_idx: int | None = None
        for idx, item in enumerate(self.resources[collection]):
            if str(item.get("id")) == target_id:
                found_idx = idx
                break

        if found_idx is None:
            raise ToolFailure(f"HTTP 404: item '{target_id}' not found in '{collection}'")

        self.resources[collection].pop(found_idx)
        self.deleted.append(f"/{collection}/{target_id}")
        return "deleted"
