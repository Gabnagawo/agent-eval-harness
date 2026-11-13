from __future__ import annotations

import copy
import json
import re

import pytest

from agent_eval.envs.api import ApiEnvironment
from agent_eval.envs.base import ToolFailure, get_environment
from agent_eval.models import TaskSpec
from agent_eval.world import World


def test_api_registered() -> None:
    cls = get_environment("api")
    assert cls is ApiEnvironment
    assert cls.kind == "api"


def test_api_load_does_not_mutate_state() -> None:
    initial_state = {
        "page_size": 2,
        "resources": {
            "users": [
                {"id": 1, "name": "Dana", "role": "admin"},
                {"id": 2, "name": "Alex", "role": "dev"},
            ]
        },
        "readonly": ["audit"],
        "requests": [{"method": "GET", "path": "/users"}],
        "deleted": ["/users/0"],
    }
    state_copy = copy.deepcopy(initial_state)

    env = ApiEnvironment(initial_state)
    assert env.page_size == 2
    assert env.readonly == ["audit"]
    assert initial_state == state_copy

    # Mutate env
    env.tool_api_post("/users", {"name": "Charlie"})
    env.tool_api_delete("/users/1")
    assert initial_state == state_copy


def test_api_defaults_empty_state() -> None:
    env = ApiEnvironment({})
    assert env.page_size == 5
    assert env.readonly == []
    assert env.resources == {}
    assert env.requests == []
    assert env.deleted == []
    snap = env.snapshot()
    assert snap == {
        "resources": {},
        "requests": [],
        "deleted": [],
    }


def test_api_tools_specs() -> None:
    env = ApiEnvironment()
    specs = {s.name: s for s in env.tools()}
    expected = {
        "api_get",
        "api_post",
        "api_patch",
        "api_delete",
    }
    assert set(specs) == expected
    assert specs["api_delete"].destructive is True
    assert specs["api_post"].destructive is False


def test_api_get_collection_and_pagination() -> None:
    env = ApiEnvironment(
        {
            "page_size": 3,
            "resources": {
                "users": [
                    {"id": 1, "name": "User 1", "role": "engineer"},
                    {"id": 2, "name": "User 2", "role": "manager"},
                    {"id": 3, "name": "User 3", "role": "engineer"},
                    {"id": 4, "name": "User 4", "role": "engineer"},
                    {"id": 5, "name": "User 5", "role": "designer"},
                ],
                "empty": [],
            },
        }
    )

    # Empty collection
    empty_res = env.tool_api_get("/empty")
    assert empty_res == {
        "items": [],
        "page": 1,
        "total_pages": 0,
        "next_page": None,
    }

    # Page 1 (items 1, 2, 3)
    p1 = env.tool_api_get("/users", {"page": 1})
    assert len(p1["items"]) == 3
    assert [i["id"] for i in p1["items"]] == [1, 2, 3]
    assert p1["page"] == 1
    assert p1["total_pages"] == 2
    assert p1["next_page"] == 2

    # Page 2 (items 4, 5) - last page
    p2 = env.tool_api_get("/users", {"page": 2})
    assert len(p2["items"]) == 2
    assert [i["id"] for i in p2["items"]] == [4, 5]
    assert p2["page"] == 2
    assert p2["total_pages"] == 2
    assert p2["next_page"] is None

    # Filtering by field before paging
    filtered = env.tool_api_get("/users", {"role": "engineer", "page": 1})
    assert len(filtered["items"]) == 3
    assert [i["id"] for i in filtered["items"]] == [1, 3, 4]
    assert filtered["total_pages"] == 1
    assert filtered["next_page"] is None

    # Filter with no matches
    no_matches = env.tool_api_get("/users", {"role": "nonexistent"})
    assert no_matches["items"] == []
    assert no_matches["total_pages"] == 0
    assert no_matches["next_page"] is None

    # Filter with field absent in item
    absent_filter = env.tool_api_get("/users", {"missing_field": "val"})
    assert absent_filter["items"] == []

    # Invalid page number
    with pytest.raises(ToolFailure, match=re.escape("HTTP 400: invalid page number '0'")):
        env.tool_api_get("/users", {"page": 0})
    with pytest.raises(ToolFailure, match=re.escape("HTTP 400: invalid page number 'abc'")):
        env.tool_api_get("/users", {"page": "abc"})


def test_api_get_item_and_errors() -> None:
    env = ApiEnvironment(
        {
            "resources": {
                "items": [{"id": "item-1", "value": 100}],
            }
        }
    )

    # Success item get
    res = env.tool_api_get("/items/item-1")
    assert res == {"id": "item-1", "value": 100}

    # Unknown item in collection
    with pytest.raises(
        ToolFailure, match=re.escape("HTTP 404: item 'item-99' not found in 'items'")
    ):
        env.tool_api_get("/items/item-99")

    # Unknown collection
    with pytest.raises(ToolFailure, match=re.escape("HTTP 404: collection 'ghost' not found")):
        env.tool_api_get("/ghost")
    with pytest.raises(ToolFailure, match=re.escape("HTTP 404: collection 'ghost' not found")):
        env.tool_api_get("/ghost/1")

    # Invalid path
    with pytest.raises(ToolFailure, match=re.escape("HTTP 400: invalid path '/'")):
        env.tool_api_get("/")
    with pytest.raises(ToolFailure, match=re.escape("HTTP 400: invalid path '/a/b/c'")):
        env.tool_api_get("/a/b/c")


def test_api_post() -> None:
    env = ApiEnvironment(
        {
            "resources": {
                "users": [
                    {"id": 1, "name": "Dana"},
                    {"id": 5, "name": "Alex"},
                    {"id": "non-num", "name": "Zoe"},
                ],
            },
            "readonly": ["readonly_col"],
            "deleted": ["/users/6"],
        }
    )

    # Readonly collection
    with pytest.raises(
        ToolFailure, match=re.escape("HTTP 405: collection 'readonly_col' is read-only")
    ):
        env.tool_api_post("/readonly_col", {"title": "forbidden"})

    # Unknown collection
    with pytest.raises(ToolFailure, match=re.escape("HTTP 404: collection 'ghost' not found")):
        env.tool_api_post("/ghost", {"name": "test"})

    # Invalid path
    with pytest.raises(
        ToolFailure, match=re.escape("HTTP 400: POST path must be '/<collection>', got '/users/1'")
    ):
        env.tool_api_post("/users/1", {"name": "test"})

    # Duplicate ID
    with pytest.raises(ToolFailure, match=re.escape("HTTP 409: duplicate id '1'")):
        env.tool_api_post("/users", {"id": 1, "name": "Duplicate"})

    # Invalid body type
    with pytest.raises(ToolFailure, match="HTTP 400: bad body"):
        env.tool_api_post("/users", "not a dict")  # type: ignore[arg-type]

    # Deleted items with non-matching collection and non-digit id
    env.deleted.extend(["/other/1", "/users/non-digit"])

    # Auto-generate ID: max existing numeric is 5, deleted has /users/6, next is 7
    created = env.tool_api_post("/users", {"name": "Charlie"})
    assert created["id"] == 7
    assert created["name"] == "Charlie"
    assert len(env.resources["users"]) == 4

    # Post with custom string ID
    created_custom = env.tool_api_post("/users", {"id": "custom-id", "name": "Dave"})
    assert created_custom["id"] == "custom-id"
    assert len(env.resources["users"]) == 5


def test_api_patch() -> None:
    env = ApiEnvironment(
        {
            "resources": {
                "users": [{"id": 1, "name": "Dana", "role": "dev"}],
            },
            "readonly": ["readonly_col"],
        }
    )

    # Readonly collection
    with pytest.raises(
        ToolFailure, match=re.escape("HTTP 405: collection 'readonly_col' is read-only")
    ):
        env.tool_api_patch("/readonly_col/1", {"role": "lead"})

    # Unknown collection
    with pytest.raises(ToolFailure, match=re.escape("HTTP 404: collection 'ghost' not found")):
        env.tool_api_patch("/ghost/1", {"role": "lead"})

    # Invalid path
    with pytest.raises(
        ToolFailure,
        match=re.escape("HTTP 400: PATCH path must be '/<collection>/<id>', got '/users'"),
    ):
        env.tool_api_patch("/users", {"role": "lead"})

    # Item not found
    with pytest.raises(ToolFailure, match=re.escape("HTTP 404: item '99' not found in 'users'")):
        env.tool_api_patch("/users/99", {"role": "lead"})

    # Cannot change id
    with pytest.raises(ToolFailure, match="HTTP 400: cannot change item id"):
        env.tool_api_patch("/users/1", {"id": 2, "role": "lead"})

    # Invalid body type
    with pytest.raises(ToolFailure, match="HTTP 400: bad body"):
        env.tool_api_patch("/users/1", 123)  # type: ignore[arg-type]

    # Successful patch with matching id in body
    patched_same_id = env.tool_api_patch("/users/1", {"id": 1, "status": "active"})
    assert patched_same_id["status"] == "active"

    # Successful patch (shallow merge)
    patched = env.tool_api_patch("/users/1", {"role": "lead", "department": "QA"})
    assert patched == {
        "id": 1,
        "name": "Dana",
        "role": "lead",
        "department": "QA",
        "status": "active",
    }
    assert env.resources["users"][0] == {
        "id": 1,
        "name": "Dana",
        "role": "lead",
        "department": "QA",
        "status": "active",
    }


def test_api_delete() -> None:
    env = ApiEnvironment(
        {
            "resources": {
                "users": [{"id": 1, "name": "Dana"}],
            },
            "readonly": ["readonly_col"],
        }
    )

    # Readonly collection
    with pytest.raises(
        ToolFailure, match=re.escape("HTTP 405: collection 'readonly_col' is read-only")
    ):
        env.tool_api_delete("/readonly_col/1")

    # Unknown collection
    with pytest.raises(ToolFailure, match=re.escape("HTTP 404: collection 'ghost' not found")):
        env.tool_api_delete("/ghost/1")

    # Invalid path
    with pytest.raises(
        ToolFailure,
        match=re.escape("HTTP 400: DELETE path must be '/<collection>/<id>', got '/users'"),
    ):
        env.tool_api_delete("/users")

    # Item not found
    with pytest.raises(ToolFailure, match=re.escape("HTTP 404: item '99' not found in 'users'")):
        env.tool_api_delete("/users/99")

    # Successful delete
    res = env.tool_api_delete("/users/1")
    assert res == "deleted"
    assert env.resources["users"] == []
    assert env.deleted == ["/users/1"]


def test_api_snapshot_and_determinism() -> None:
    initial = {
        "page_size": 2,
        "resources": {
            "users": [
                {"id": 2, "name": "Alex"},
                {"id": 1, "name": "Dana"},
            ]
        },
    }

    env1 = ApiEnvironment(initial, seed=42)
    env2 = ApiEnvironment(initial, seed=42)

    actions = [
        ("api_get", {"path": "/users"}),
        ("api_post", {"path": "/users", "body": {"name": "Charlie"}}),
        ("api_patch", {"path": "/users/1", "body": {"name": "Dana Updated"}}),
        ("api_delete", {"path": "/users/2"}),
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

    # Check resources in snapshot is mapping keyed by id string
    assert "users" in snap1["resources"]
    assert "1" in snap1["resources"]["users"]
    assert "3" in snap1["resources"]["users"]
    assert "2" not in snap1["resources"]["users"]
    assert snap1["deleted"] == ["/users/2"]
    assert len(snap1["requests"]) == 4


def test_api_world_integration() -> None:
    task = TaskSpec.model_validate(
        {
            "id": "api-001-post-user",
            "family": "api",
            "title": "Create user",
            "prompt": "Create user Alice",
            "envs": {
                "api": {
                    "resources": {
                        "users": [{"id": 1, "name": "Dana"}],
                    }
                }
            },
            "max_steps": 5,
            "optimal_steps": 1,
            "reference_solution": [
                {"tool": "api_post", "args": {"path": "/users", "body": {"name": "Alice"}}}
            ],
            "success": [
                {
                    "path": 'api.resources.users."2".name',
                    "op": "equals",
                    "value": "Alice",
                }
            ],
        }
    )

    world = World(task)
    step1 = world.call("api_post", {"path": "/users", "body": {"name": "Alice"}})
    assert step1.ok is True
    assert step1.executed is True

    snap = world.snapshot()
    assert snap["api"]["resources"]["users"]["2"]["name"] == "Alice"
