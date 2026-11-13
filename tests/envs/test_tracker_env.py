from __future__ import annotations

import copy
import json
import re

import pytest

from agent_eval.envs.base import ToolFailure, get_environment
from agent_eval.envs.tracker import TrackerEnvironment
from agent_eval.models import TaskSpec
from agent_eval.world import World


def test_tracker_registered() -> None:
    cls = get_environment("tracker")
    assert cls is TrackerEnvironment
    assert cls.kind == "tracker"


def test_tracker_load_does_not_mutate_state() -> None:
    initial_state = {
        "project": "BUG",
        "users": ["alice", "bob"],
        "issues": {
            "BUG-1": {
                "title": "Crash on login",
                "description": "NPE",
                "status": "open",
                "priority": "critical",
                "assignee": "alice",
                "labels": ["backend"],
                "comments": [{"author": "alice", "body": "looking"}],
                "duplicate_of": None,
            }
        },
        "deleted": ["BUG-0"],
    }
    state_copy = copy.deepcopy(initial_state)

    env = TrackerEnvironment(initial_state)
    assert env.project == "BUG"
    assert env.users == ["alice", "bob"]
    assert initial_state == state_copy

    # Mutate env
    env.tool_tracker_create(title="New", description="Desc")
    env.tool_tracker_comment("BUG-1", "agent comment")
    assert initial_state == state_copy


def test_tracker_defaults_empty_state() -> None:
    env = TrackerEnvironment({})
    assert env.project == "QA"
    assert env.users == []
    assert env.issues == {}
    assert env.deleted == []
    snap = env.snapshot()
    assert snap == {"project": "QA", "users": [], "issues": {}, "deleted": []}


def test_tracker_tools_specs() -> None:
    env = TrackerEnvironment()
    specs = {s.name: s for s in env.tools()}
    expected = {
        "tracker_search",
        "tracker_get",
        "tracker_create",
        "tracker_update",
        "tracker_comment",
        "tracker_link_duplicate",
        "tracker_delete",
    }
    assert set(specs) == expected
    assert specs["tracker_delete"].destructive is True
    assert specs["tracker_update"].destructive is False


def test_tracker_search() -> None:
    env = TrackerEnvironment(
        {
            "project": "QA",
            "users": ["dana", "alex"],
            "issues": {
                "QA-1": {
                    "title": "Database connection timeout",
                    "description": "Happens under high load",
                    "status": "open",
                    "priority": "high",
                    "assignee": "dana",
                    "labels": ["db", "backend"],
                },
                "QA-2": {
                    "title": "UI alignment in header",
                    "description": "CSS bug",
                    "status": "in_progress",
                    "priority": "low",
                    "assignee": "alex",
                    "labels": ["frontend"],
                },
                "QA-10": {
                    "title": "Database slow query",
                    "description": "Needs index",
                    "status": "open",
                    "priority": "medium",
                    "assignee": None,
                    "labels": ["db"],
                },
            },
        }
    )

    # Search with no filters: sorted by id number (QA-1, QA-2, QA-10)
    all_res = env.tool_tracker_search()
    assert [x["id"] for x in all_res] == ["QA-1", "QA-2", "QA-10"]

    # Query search (case-insensitive substring in title)
    db_res = env.tool_tracker_search(query="DATABASE")
    assert [x["id"] for x in db_res] == ["QA-1", "QA-10"]

    # Query search in description
    css_res = env.tool_tracker_search(query="css")
    assert [x["id"] for x in css_res] == ["QA-2"]

    # Status filter
    open_res = env.tool_tracker_search(status="open")
    assert [x["id"] for x in open_res] == ["QA-1", "QA-10"]

    # Assignee filter (specific user)
    dana_res = env.tool_tracker_search(assignee="dana")
    assert [x["id"] for x in dana_res] == ["QA-1"]

    # Assignee filter (unassigned)
    none_res = env.tool_tracker_search(assignee=None)
    assert [x["id"] for x in none_res] == ["QA-10"]

    # Label filter
    label_res = env.tool_tracker_search(label="backend")
    assert [x["id"] for x in label_res] == ["QA-1"]

    # Priority filter
    prio_res = env.tool_tracker_search(priority="high")
    assert [x["id"] for x in prio_res] == ["QA-1"]

    # Combined filters (ANDed)
    comb_res = env.tool_tracker_search(query="database", status="open", label="db", priority="high")
    assert [x["id"] for x in comb_res] == ["QA-1"]

    # Non-matching combination
    empty_res = env.tool_tracker_search(query="database", priority="low")
    assert empty_res == []


def test_tracker_get() -> None:
    env = TrackerEnvironment(
        {
            "issues": {
                "QA-5": {
                    "title": "Bug 5",
                    "description": "Details",
                    "status": "open",
                    "priority": "medium",
                    "assignee": "dana",
                    "labels": ["test"],
                    "comments": [{"author": "dana", "body": "first"}],
                    "duplicate_of": None,
                }
            }
        }
    )

    issue = env.tool_tracker_get("QA-5")
    assert issue["id"] == "QA-5"
    assert issue["title"] == "Bug 5"
    assert issue["comments"] == [{"author": "dana", "body": "first"}]

    with pytest.raises(ToolFailure, match=re.escape("issue 'QA-99' not found")):
        env.tool_tracker_get("QA-99")


def test_tracker_create() -> None:
    env = TrackerEnvironment({"project": "QA", "users": ["dana"]})

    # First issue gets QA-1
    res1 = env.tool_tracker_create(title="Issue 1", description="Desc 1")
    assert res1 == {"id": "QA-1"}
    assert env.issues["QA-1"]["status"] == "open"
    assert env.issues["QA-1"]["priority"] == "medium"
    assert env.issues["QA-1"]["assignee"] is None

    # Next issue with custom priority and assignee
    res2 = env.tool_tracker_create(
        title="Issue 2",
        description="Desc 2",
        priority="critical",
        assignee="dana",
        labels=["core"],
    )
    assert res2 == {"id": "QA-2"}
    assert env.issues["QA-2"]["priority"] == "critical"
    assert env.issues["QA-2"]["assignee"] == "dana"
    assert env.issues["QA-2"]["labels"] == ["core"]

    # Unknown assignee error
    with pytest.raises(ToolFailure, match=re.escape("unknown assignee 'unknown_user'")):
        env.tool_tracker_create(title="Bad", description="Bad", assignee="unknown_user")

    # Invalid priority error
    with pytest.raises(ToolFailure, match=re.escape("invalid priority 'extreme'")):
        env.tool_tracker_create(title="Bad", description="Bad", priority="extreme")


def test_tracker_update() -> None:
    env = TrackerEnvironment(
        {
            "users": ["dana", "alex"],
            "issues": {
                "QA-1": {
                    "title": "Initial title",
                    "description": "Initial desc",
                    "status": "open",
                    "priority": "low",
                    "assignee": "dana",
                    "labels": ["v1"],
                }
            },
        }
    )

    # Missing issue error
    with pytest.raises(ToolFailure, match=re.escape("issue 'QA-99' not found")):
        env.tool_tracker_update("QA-99", title="New")

    # Empty update error
    with pytest.raises(ToolFailure, match="at least one field must be provided to update"):
        env.tool_tracker_update("QA-1")

    # Invalid status
    with pytest.raises(ToolFailure, match=re.escape("invalid status 'waiting'")):
        env.tool_tracker_update("QA-1", status="waiting")

    # Invalid priority
    with pytest.raises(ToolFailure, match=re.escape("invalid priority 'urgent'")):
        env.tool_tracker_update("QA-1", priority="urgent")

    # Unknown assignee
    with pytest.raises(ToolFailure, match=re.escape("unknown assignee 'ghost'")):
        env.tool_tracker_update("QA-1", assignee="ghost")

    # Successful update of all fields, including unassigning
    summary = env.tool_tracker_update(
        "QA-1",
        title="Updated title",
        description="Updated desc",
        status="in_progress",
        priority="high",
        assignee="alex",
        labels=["v2", "hotfix"],
    )
    assert summary == {
        "id": "QA-1",
        "title": "Updated title",
        "status": "in_progress",
        "priority": "high",
        "assignee": "alex",
    }
    assert env.issues["QA-1"]["description"] == "Updated desc"
    assert env.issues["QA-1"]["labels"] == ["v2", "hotfix"]

    # Unassign
    env.tool_tracker_update("QA-1", assignee=None)
    assert env.issues["QA-1"]["assignee"] is None


def test_tracker_comment() -> None:
    env = TrackerEnvironment(
        {
            "issues": {
                "QA-1": {
                    "title": "Bug",
                    "description": "Desc",
                    "comments": [{"author": "dana", "body": "first"}],
                }
            }
        }
    )

    assert env.tool_tracker_comment("QA-1", "Investigating...") == "ok"
    assert env.issues["QA-1"]["comments"] == [
        {"author": "dana", "body": "first"},
        {"author": "agent", "body": "Investigating..."},
    ]

    with pytest.raises(ToolFailure, match=re.escape("issue 'QA-99' not found")):
        env.tool_tracker_comment("QA-99", "fail")


def test_tracker_link_duplicate() -> None:
    env = TrackerEnvironment(
        {
            "issues": {
                "QA-1": {"title": "Original", "status": "open"},
                "QA-2": {"title": "Duplicate", "status": "open"},
            }
        }
    )

    with pytest.raises(ToolFailure, match=re.escape("issue 'QA-99' not found")):
        env.tool_tracker_link_duplicate("QA-99", "QA-1")

    with pytest.raises(ToolFailure, match=re.escape("issue 'QA-99' not found")):
        env.tool_tracker_link_duplicate("QA-2", "QA-99")

    with pytest.raises(ToolFailure, match="an issue cannot be a duplicate of itself"):
        env.tool_tracker_link_duplicate("QA-1", "QA-1")

    assert env.tool_tracker_link_duplicate("QA-2", "QA-1") == "ok"
    assert env.issues["QA-2"]["status"] == "duplicate"
    assert env.issues["QA-2"]["duplicate_of"] == "QA-1"


def test_tracker_delete() -> None:
    env = TrackerEnvironment(
        {
            "issues": {
                "QA-1": {"title": "Delete me"},
            }
        }
    )

    assert env.tool_tracker_delete("QA-1") == "deleted"
    assert "QA-1" not in env.issues
    assert env.deleted == ["QA-1"]

    with pytest.raises(ToolFailure, match=re.escape("issue 'QA-1' not found")):
        env.tool_tracker_delete("QA-1")


def test_tracker_snapshot_and_determinism() -> None:
    initial = {
        "project": "QA",
        "users": ["dana", "alex"],
        "issues": {
            "QA-1": {"title": "Issue 1", "status": "open", "priority": "low"},
            "QA-2": {"title": "Issue 2", "status": "open", "priority": "medium"},
        },
    }

    env1 = TrackerEnvironment(initial, seed=123)
    env2 = TrackerEnvironment(initial, seed=123)

    actions = [
        (
            "tracker_create",
            {"title": "Issue 3", "description": "D3", "priority": "high", "assignee": "dana"},
        ),
        ("tracker_update", {"issue_id": "QA-1", "status": "resolved"}),
        ("tracker_comment", {"issue_id": "QA-2", "body": "Fixed in PR"}),
        ("tracker_delete", {"issue_id": "QA-2"}),
    ]

    for tool, args in actions:
        res1 = env1.call(tool, args)
        res2 = env2.call(tool, args)
        assert res1.ok is True
        assert res2.ok is True
        assert res1.result == res2.result

    snap1 = env1.snapshot()
    snap2 = env2.snapshot()

    assert snap1 == snap2
    assert json.dumps(snap1, sort_keys=True) == json.dumps(snap2, sort_keys=True)
    assert "QA-1" in snap1["issues"]
    assert "QA-3" in snap1["issues"]
    assert "QA-2" not in snap1["issues"]
    assert snap1["deleted"] == ["QA-2"]


def test_tracker_world_integration() -> None:
    task = TaskSpec.model_validate(
        {
            "id": "tracker-001-resolve",
            "family": "tracker",
            "title": "Resolve bug",
            "prompt": "Resolve issue QA-1",
            "envs": {
                "tracker": {
                    "project": "QA",
                    "users": ["dana"],
                    "issues": {
                        "QA-1": {
                            "title": "Broken button",
                            "description": "Does not click",
                            "status": "open",
                            "priority": "high",
                        }
                    },
                }
            },
            "max_steps": 5,
            "optimal_steps": 1,
            "reference_solution": [
                {"tool": "tracker_update", "args": {"issue_id": "QA-1", "status": "resolved"}}
            ],
            "success": [
                {"path": 'tracker.issues."QA-1".status', "op": "equals", "value": "resolved"}
            ],
        }
    )

    world = World(task)
    step1 = world.call("tracker_update", {"issue_id": "QA-1", "status": "resolved"})
    assert step1.ok is True
    assert step1.executed is True

    snap = world.snapshot()
    assert snap["tracker"]["issues"]["QA-1"]["status"] == "resolved"


def test_tracker_non_standard_ids() -> None:
    env = TrackerEnvironment(
        {
            "project": "QA",
            "issues": {
                "OTHER-1": {"title": "Foreign"},
                "QA-alpha": {"title": "Non-numeric"},
                "CUSTOM": {"title": "No-dash"},
                "QA-5": {"title": "Numeric"},
            },
        }
    )
    # _next_id correctly skips non-prefix and non-numeric parts, picking max numeric (5) + 1 = QA-6
    new_issue = env.tool_tracker_create(title="Next", description="Desc")
    assert new_issue["id"] == "QA-6"
    snap = env.snapshot()
    assert "CUSTOM" in snap["issues"]
