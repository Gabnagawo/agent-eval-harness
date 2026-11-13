from __future__ import annotations

import copy
import json
import re

import pytest

from agent_eval.envs.base import ToolFailure, get_environment
from agent_eval.envs.files import FilesEnvironment
from agent_eval.models import TaskSpec
from agent_eval.world import World


def test_files_registered() -> None:
    cls = get_environment("files")
    assert cls is FilesEnvironment
    assert cls.kind == "files"


def test_files_load_does_not_mutate_state() -> None:
    initial_state = {
        "files": {"/a.txt": "hello", "/b.txt": "world"},
        "readonly": ["/ro/*"],
        "deleted": ["/old.txt"],
    }
    state_copy = copy.deepcopy(initial_state)

    env = FilesEnvironment(initial_state)
    assert env.files == {"/a.txt": "hello", "/b.txt": "world"}
    assert initial_state == state_copy

    # Mutate env, ensure initial_state is completely unchanged
    env.tool_files_write("/c.txt", "new")
    env.tool_files_delete("/a.txt")
    assert initial_state == state_copy


def test_files_defaults_empty_state() -> None:
    env = FilesEnvironment({})
    assert env.files == {}
    assert env.readonly == []
    assert env.deleted == []
    snap = env.snapshot()
    assert snap == {"files": {}, "readonly": [], "deleted": []}


def test_files_tools_specs() -> None:
    env = FilesEnvironment()
    specs = {s.name: s for s in env.tools()}
    expected = {
        "files_list",
        "files_read",
        "files_search",
        "files_write",
        "files_append",
        "files_move",
        "files_delete",
    }
    assert set(specs) == expected
    assert specs["files_delete"].destructive is True
    assert specs["files_write"].destructive is False

    for s in specs.values():
        assert s.parameters["type"] == "object"
        assert s.parameters["additionalProperties"] is False
        assert isinstance(s.parameters["properties"], dict)


def test_files_list() -> None:
    env = FilesEnvironment(
        {
            "files": {
                "/src/main.py": "print(1)",
                "/src/util.py": "def f(): pass",
                "/docs/readme.md": "hi",
                "/etc/config.json": "{}",
            }
        }
    )
    # Default prefix "/" lists all absolute paths, sorted
    assert env.tool_files_list() == [
        "/docs/readme.md",
        "/etc/config.json",
        "/src/main.py",
        "/src/util.py",
    ]
    # Specific prefix
    assert env.tool_files_list(prefix="/src") == ["/src/main.py", "/src/util.py"]
    # Non-matching prefix
    assert env.tool_files_list(prefix="/var") == []


def test_files_read() -> None:
    env = FilesEnvironment({"files": {"/docs/readme.md": "content here"}})
    assert env.tool_files_read("/docs/readme.md") == "content here"

    with pytest.raises(ToolFailure, match=re.escape("file '/missing.txt' not found")):
        env.tool_files_read("/missing.txt")


def test_files_search() -> None:
    env = FilesEnvironment(
        {
            "files": {
                "/src/a.py": "def foo():\n    return 'SEARCH_ME'\n",
                "/src/b.py": "search_me_again = 1\n",
                "/docs/note.txt": "nothing relevant\nSEARCH_ME in docs\n",
                "/other/skip.txt": "search_me but prefix filtered",
            }
        }
    )

    hits = env.tool_files_search(query="search_me", prefix="/src")
    assert len(hits) == 2
    assert hits[0] == {"path": "/src/a.py", "line": 2, "text": "    return 'SEARCH_ME'"}
    assert hits[1] == {"path": "/src/b.py", "line": 1, "text": "search_me_again = 1"}

    # Search with default prefix "/"
    all_hits = env.tool_files_search(query="SEARCH_ME")
    assert len(all_hits) == 4

    # Search with empty query
    with pytest.raises(ToolFailure, match="search query cannot be empty"):
        env.tool_files_search(query="")

    # Search exceeding 20 hits is capped at 20
    long_content = "\n".join(f"hit line {i}" for i in range(30))
    big_env = FilesEnvironment({"files": {"/big.txt": long_content}})
    hits_capped = big_env.tool_files_search(query="hit line")
    assert len(hits_capped) == 20
    assert hits_capped[0]["line"] == 1
    assert hits_capped[19]["line"] == 20


def test_files_write() -> None:
    env = FilesEnvironment({"files": {"/a.txt": "initial"}, "readonly": ["/ro/*"]})

    # Create new
    assert env.tool_files_write("/b.txt", "new file") == "created"
    assert env.files["/b.txt"] == "new file"

    # Overwrite existing
    assert env.tool_files_write("/a.txt", "updated") == "overwritten"
    assert env.files["/a.txt"] == "updated"

    # Readonly failure
    with pytest.raises(ToolFailure, match="permission denied"):
        env.tool_files_write("/ro/secret.txt", "blocked")


def test_files_append() -> None:
    env = FilesEnvironment(
        {"files": {"/a.txt": "first", "/ro/cant_append.txt": "locked"}, "readonly": ["/ro/*"]}
    )

    assert env.tool_files_append("/a.txt", " second") == "ok"
    assert env.files["/a.txt"] == "first second"

    with pytest.raises(ToolFailure, match=re.escape("file '/missing.txt' not found")):
        env.tool_files_append("/missing.txt", "more")

    with pytest.raises(ToolFailure, match="permission denied"):
        env.tool_files_append("/ro/cant_append.txt", "more")


def test_files_move() -> None:
    env = FilesEnvironment(
        {
            "files": {
                "/src/old.txt": "content",
                "/src/existing.txt": "already there",
                "/ro/locked.txt": "locked",
            },
            "readonly": ["/ro/*"],
        }
    )

    # Success
    assert env.tool_files_move("/src/old.txt", "/src/new.txt") == "ok"
    assert "/src/old.txt" not in env.files
    assert env.files["/src/new.txt"] == "content"

    # Source missing
    with pytest.raises(ToolFailure, match=re.escape("file '/src/old.txt' not found")):
        env.tool_files_move("/src/old.txt", "/src/another.txt")

    # Destination exists
    with pytest.raises(
        ToolFailure, match=re.escape("destination '/src/existing.txt' already exists")
    ):
        env.tool_files_move("/src/new.txt", "/src/existing.txt")

    # Source is readonly
    with pytest.raises(ToolFailure, match="permission denied"):
        env.tool_files_move("/ro/locked.txt", "/src/free.txt")

    # Destination is readonly
    with pytest.raises(ToolFailure, match="permission denied"):
        env.tool_files_move("/src/new.txt", "/ro/destination.txt")


def test_files_delete() -> None:
    env = FilesEnvironment(
        {
            "files": {"/a.txt": "1", "/b.txt": "2", "/ro/lock.txt": "ro"},
            "readonly": ["/ro/*"],
        }
    )

    assert env.tool_files_delete("/a.txt") == "deleted"
    assert "/a.txt" not in env.files
    assert env.deleted == ["/a.txt"]

    assert env.tool_files_delete("/b.txt") == "deleted"
    assert env.deleted == ["/a.txt", "/b.txt"]

    # Missing file
    with pytest.raises(ToolFailure, match=re.escape("file '/missing.txt' not found")):
        env.tool_files_delete("/missing.txt")

    # Readonly
    with pytest.raises(ToolFailure, match="permission denied"):
        env.tool_files_delete("/ro/lock.txt")


def test_files_snapshot_and_determinism() -> None:
    initial = {
        "files": {
            "/z.txt": "z",
            "/a.txt": "a",
            "/m.txt": "m",
        },
        "readonly": ["/ro/*"],
    }

    env1 = FilesEnvironment(initial, seed=42)
    env2 = FilesEnvironment(initial, seed=42)

    actions = [
        ("files_write", {"path": "/b.txt", "content": "b"}),
        ("files_append", {"path": "/a.txt", "content": " extra"}),
        ("files_move", {"source": "/z.txt", "destination": "/c.txt"}),
        ("files_delete", {"path": "/m.txt"}),
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
    # Check sorted keys in files snapshot
    assert list(snap1["files"].keys()) == ["/a.txt", "/b.txt", "/c.txt"]
    assert snap1["deleted"] == ["/m.txt"]


def test_files_world_integration() -> None:
    task = TaskSpec.model_validate(
        {
            "id": "files-001-basic",
            "family": "files",
            "title": "Basic file test",
            "prompt": "Create a file",
            "envs": {
                "files": {
                    "files": {"/start.txt": "start"},
                    "readonly": ["/ro/*"],
                }
            },
            "max_steps": 5,
            "optimal_steps": 1,
            "reference_solution": [
                {"tool": "files_write", "args": {"path": "/done.txt", "content": "done"}}
            ],
            "success": [{"path": 'files.files."/done.txt"', "op": "equals", "value": "done"}],
        }
    )

    world = World(task)
    step1 = world.call("files_write", {"path": "/done.txt", "content": "done"})
    assert step1.ok is True
    assert step1.executed is True

    snap = world.snapshot()
    assert snap["files"]["files"]["/done.txt"] == "done"
    assert snap["files"]["files"]["/start.txt"] == "start"
