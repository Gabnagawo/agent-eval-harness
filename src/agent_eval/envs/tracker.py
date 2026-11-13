"""``tracker``: a Jira-like issue tracker.

State and snapshot::

    {
        "project": "QA",
        "users": ["dana", ...],
        "issues": {
            "QA-1": {
                "title": "...",
                "description": "...",
                "status": "open",
                "priority": "medium",
                "assignee": "dana",
                "labels": ["..."],
                "comments": [{"author": "...", "body": "..."}],
                "duplicate_of": None
            }
        },
        "deleted": ["QA-3", ...]
    }
"""

from __future__ import annotations

import copy
from typing import Any

from agent_eval.envs.base import Environment, ToolFailure, register
from agent_eval.models import ToolSpec

_UNSET = object()

VALID_STATUSES = {"open", "in_progress", "resolved", "closed", "duplicate"}
VALID_PRIORITIES = {"low", "medium", "high", "critical"}


def _params(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _id_sort_key(issue_id: str) -> tuple[str, int, str]:
    if "-" in issue_id:
        prefix, num_part = issue_id.rsplit("-", 1)
        if num_part.isdigit():
            return (prefix, int(num_part), issue_id)
    return (issue_id, 0, issue_id)


@register
class TrackerEnvironment(Environment):
    kind = "tracker"

    def load(self, state: dict[str, Any]) -> None:
        self.project: str = str(state.get("project", "QA"))
        self.users: list[str] = [str(u) for u in state.get("users", [])]
        self.deleted: list[str] = [str(d) for d in state.get("deleted", [])]

        self.issues: dict[str, dict[str, Any]] = {}
        for raw_id, raw_issue in state.get("issues", {}).items():
            issue_id = str(raw_id)
            self.issues[issue_id] = {
                "title": str(raw_issue.get("title", "")),
                "description": str(raw_issue.get("description", "")),
                "status": str(raw_issue.get("status", "open")),
                "priority": str(raw_issue.get("priority", "medium")),
                "assignee": raw_issue.get("assignee"),
                "labels": [str(lbl) for lbl in raw_issue.get("labels", [])],
                "comments": [
                    {"author": str(c.get("author", "")), "body": str(c.get("body", ""))}
                    for c in raw_issue.get("comments", [])
                ],
                "duplicate_of": raw_issue.get("duplicate_of"),
            }

    def tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="tracker_search",
                description="Search issues matching ANDed filters, returning summary records sorted by issue ID number.",
                parameters=_params(
                    {
                        "query": {
                            "type": "string",
                            "description": "Case-insensitive substring search in title or description.",
                        },
                        "status": {
                            "type": "string",
                            "enum": sorted(VALID_STATUSES),
                            "description": "Filter by issue status.",
                        },
                        "assignee": {
                            "type": ["string", "null"],
                            "description": "Filter by assignee username or null for unassigned.",
                        },
                        "label": {
                            "type": "string",
                            "description": "Filter by label contained in the issue.",
                        },
                        "priority": {
                            "type": "string",
                            "enum": sorted(VALID_PRIORITIES),
                            "description": "Filter by priority level.",
                        },
                    },
                    [],
                ),
            ),
            ToolSpec(
                name="tracker_get",
                description="Get full details of a specific issue by its ID.",
                parameters=_params(
                    {
                        "issue_id": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Issue ID (e.g. 'QA-1').",
                        }
                    },
                    ["issue_id"],
                ),
            ),
            ToolSpec(
                name="tracker_create",
                description="Create a new issue in the tracker. Status defaults to open.",
                parameters=_params(
                    {
                        "title": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Issue title.",
                        },
                        "description": {
                            "type": "string",
                            "description": "Issue description.",
                        },
                        "priority": {
                            "type": "string",
                            "enum": sorted(VALID_PRIORITIES),
                            "default": "medium",
                            "description": "Priority level (defaults to 'medium').",
                        },
                        "labels": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of labels.",
                        },
                        "assignee": {
                            "type": ["string", "null"],
                            "description": "Optional assignee username.",
                        },
                    },
                    ["title", "description"],
                ),
            ),
            ToolSpec(
                name="tracker_update",
                description="Update one or more fields of an existing issue. Labels replace the previous list.",
                parameters=_params(
                    {
                        "issue_id": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Issue ID to update.",
                        },
                        "title": {"type": "string", "description": "New title."},
                        "description": {"type": "string", "description": "New description."},
                        "status": {
                            "type": "string",
                            "enum": sorted(VALID_STATUSES),
                            "description": "New status.",
                        },
                        "priority": {
                            "type": "string",
                            "enum": sorted(VALID_PRIORITIES),
                            "description": "New priority.",
                        },
                        "assignee": {
                            "type": ["string", "null"],
                            "description": "New assignee username, or null to unassign.",
                        },
                        "labels": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Replacement list of labels.",
                        },
                    },
                    ["issue_id"],
                ),
            ),
            ToolSpec(
                name="tracker_comment",
                description="Add a comment to an issue. The comment author will be 'agent'.",
                parameters=_params(
                    {
                        "issue_id": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Issue ID to comment on.",
                        },
                        "body": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Comment content.",
                        },
                    },
                    ["issue_id", "body"],
                ),
            ),
            ToolSpec(
                name="tracker_link_duplicate",
                description="Mark an issue as duplicate of another. Sets status to 'duplicate' and duplicate_of.",
                parameters=_params(
                    {
                        "issue_id": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Issue ID being marked as duplicate.",
                        },
                        "duplicate_of": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Original issue ID.",
                        },
                    },
                    ["issue_id", "duplicate_of"],
                ),
            ),
            ToolSpec(
                name="tracker_delete",
                description="Delete an issue permanently. This cannot be undone.",
                parameters=_params(
                    {
                        "issue_id": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Issue ID to delete.",
                        }
                    },
                    ["issue_id"],
                ),
                destructive=True,
            ),
        ]

    def snapshot(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "users": list(self.users),
            "issues": {
                k: copy.deepcopy(self.issues[k]) for k in sorted(self.issues, key=_id_sort_key)
            },
            "deleted": list(self.deleted),
        }

    def _next_id(self) -> str:
        prefix = f"{self.project}-"
        numbers: list[int] = []
        for key in set(self.issues) | set(self.deleted):
            if key.startswith(prefix):
                num_part = key[len(prefix) :]
                if num_part.isdigit():
                    numbers.append(int(num_part))
        next_num = max(numbers, default=0) + 1
        return f"{self.project}-{next_num}"

    def tool_tracker_search(
        self,
        query: str | None = None,
        status: str | None = None,
        assignee: Any = _UNSET,
        label: str | None = None,
        priority: str | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        lower_query = query.lower() if query else None

        for issue_id in sorted(self.issues, key=_id_sort_key):
            issue = self.issues[issue_id]
            if lower_query and (
                lower_query not in issue["title"].lower()
                and lower_query not in issue["description"].lower()
            ):
                continue
            if status is not None and issue["status"] != status:
                continue
            if assignee is not _UNSET and issue["assignee"] != assignee:
                continue
            if label is not None and label not in issue.get("labels", []):
                continue
            if priority is not None and issue["priority"] != priority:
                continue

            results.append(
                {
                    "id": issue_id,
                    "title": issue["title"],
                    "status": issue["status"],
                    "priority": issue["priority"],
                    "assignee": issue["assignee"],
                }
            )
        return results

    def tool_tracker_get(self, issue_id: str) -> dict[str, Any]:
        if issue_id not in self.issues:
            raise ToolFailure(f"issue '{issue_id}' not found")
        issue = copy.deepcopy(self.issues[issue_id])
        issue["id"] = issue_id
        return issue

    def tool_tracker_create(
        self,
        title: str,
        description: str,
        priority: str = "medium",
        labels: list[str] | None = None,
        assignee: str | None = None,
    ) -> dict[str, str]:
        if assignee is not None and assignee not in self.users:
            raise ToolFailure(f"unknown assignee '{assignee}'")
        if priority not in VALID_PRIORITIES:
            raise ToolFailure(f"invalid priority '{priority}'")

        new_id = self._next_id()
        self.issues[new_id] = {
            "title": title,
            "description": description,
            "status": "open",
            "priority": priority,
            "assignee": assignee,
            "labels": list(labels or []),
            "comments": [],
            "duplicate_of": None,
        }
        return {"id": new_id}

    def tool_tracker_update(
        self,
        issue_id: str,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
        priority: str | None = None,
        assignee: Any = _UNSET,
        labels: list[str] | None = None,
    ) -> dict[str, Any]:
        if issue_id not in self.issues:
            raise ToolFailure(f"issue '{issue_id}' not found")

        if (
            title is None
            and description is None
            and status is None
            and priority is None
            and assignee is _UNSET
            and labels is None
        ):
            raise ToolFailure("at least one field must be provided to update")

        issue = self.issues[issue_id]

        if status is not None:
            if status not in VALID_STATUSES:
                raise ToolFailure(f"invalid status '{status}'")
            issue["status"] = status

        if priority is not None:
            if priority not in VALID_PRIORITIES:
                raise ToolFailure(f"invalid priority '{priority}'")
            issue["priority"] = priority

        if assignee is not _UNSET:
            if assignee is not None and assignee not in self.users:
                raise ToolFailure(f"unknown assignee '{assignee}'")
            issue["assignee"] = assignee

        if title is not None:
            issue["title"] = title

        if description is not None:
            issue["description"] = description

        if labels is not None:
            issue["labels"] = list(labels)

        return {
            "id": issue_id,
            "title": issue["title"],
            "status": issue["status"],
            "priority": issue["priority"],
            "assignee": issue["assignee"],
        }

    def tool_tracker_comment(self, issue_id: str, body: str) -> str:
        if issue_id not in self.issues:
            raise ToolFailure(f"issue '{issue_id}' not found")
        self.issues[issue_id]["comments"].append({"author": "agent", "body": body})
        return "ok"

    def tool_tracker_link_duplicate(self, issue_id: str, duplicate_of: str) -> str:
        if issue_id not in self.issues:
            raise ToolFailure(f"issue '{issue_id}' not found")
        if duplicate_of not in self.issues:
            raise ToolFailure(f"issue '{duplicate_of}' not found")
        if issue_id == duplicate_of:
            raise ToolFailure("an issue cannot be a duplicate of itself")

        self.issues[issue_id]["status"] = "duplicate"
        self.issues[issue_id]["duplicate_of"] = duplicate_of
        return "ok"

    def tool_tracker_delete(self, issue_id: str) -> str:
        if issue_id not in self.issues:
            raise ToolFailure(f"issue '{issue_id}' not found")
        del self.issues[issue_id]
        self.deleted.append(issue_id)
        return "deleted"
