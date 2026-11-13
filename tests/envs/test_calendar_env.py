from __future__ import annotations

import copy
import json
import re

import pytest

from agent_eval.envs.base import ToolFailure, get_environment
from agent_eval.envs.calendar import CalendarEnvironment
from agent_eval.models import TaskSpec
from agent_eval.world import World


def test_calendar_registered() -> None:
    cls = get_environment("calendar")
    assert cls is CalendarEnvironment
    assert cls.kind == "calendar"


def test_calendar_load_does_not_mutate_state() -> None:
    initial_state = {
        "now": "2026-03-02T09:00",
        "people": {"dana": {"working_hours": ["09:00", "17:00"]}},
        "events": {
            "evt-1": {
                "title": "Sync",
                "start": "2026-03-02T10:00",
                "end": "2026-03-02T10:30",
                "attendees": ["dana"],
                "location": "Room A",
            }
        },
        "cancelled": ["evt-0"],
    }
    state_copy = copy.deepcopy(initial_state)

    env = CalendarEnvironment(initial_state)
    assert env.now == "2026-03-02T09:00"
    assert env.people == {"dana": {"working_hours": ["09:00", "17:00"]}}
    assert initial_state == state_copy

    # Mutate env
    env.tool_calendar_create(
        title="New",
        start="2026-03-02T11:00",
        end="2026-03-02T12:00",
        attendees=["dana"],
    )
    env.tool_calendar_cancel("evt-1")
    assert initial_state == state_copy


def test_calendar_defaults_empty_state() -> None:
    env = CalendarEnvironment({})
    assert env.now == "2026-01-01T09:00"
    assert env.people == {}
    assert env.events == {}
    assert env.cancelled == []
    snap = env.snapshot()
    assert snap == {
        "now": "2026-01-01T09:00",
        "people": {},
        "events": {},
        "cancelled": [],
    }


def test_calendar_tools_specs() -> None:
    env = CalendarEnvironment()
    specs = {s.name: s for s in env.tools()}
    expected = {
        "calendar_now",
        "calendar_list",
        "calendar_get",
        "calendar_find_free",
        "calendar_create",
        "calendar_update",
        "calendar_cancel",
    }
    assert set(specs) == expected
    assert specs["calendar_cancel"].destructive is True
    assert specs["calendar_create"].destructive is False


def test_calendar_now() -> None:
    env = CalendarEnvironment({"now": "2026-05-10T14:30"})
    assert env.tool_calendar_now() == "2026-05-10T14:30"


def test_calendar_list() -> None:
    env = CalendarEnvironment(
        {
            "events": {
                "evt-1": {
                    "title": "Morning standup",
                    "start": "2026-03-02T09:00",
                    "end": "2026-03-02T09:30",
                    "attendees": ["dana", "alex"],
                },
                "evt-2": {
                    "title": "Design review",
                    "start": "2026-03-02T14:00",
                    "end": "2026-03-02T15:00",
                    "attendees": ["alex"],
                },
                "evt-3": {
                    "title": "Next day retro",
                    "start": "2026-03-03T11:00",
                    "end": "2026-03-03T12:00",
                    "attendees": ["dana"],
                },
                "evt-4": {
                    "title": "Out of range",
                    "start": "2026-03-10T10:00",
                    "end": "2026-03-10T11:00",
                    "attendees": ["dana"],
                },
            }
        }
    )

    # Date range error
    with pytest.raises(ToolFailure, match="date_to cannot be earlier than date_from"):
        env.tool_calendar_list(date_from="2026-03-05", date_to="2026-03-01")

    # List all in 2-day range
    events = env.tool_calendar_list(date_from="2026-03-02", date_to="2026-03-03")
    assert [e["id"] for e in events] == ["evt-1", "evt-2", "evt-3"]

    # Filter by attendee
    dana_events = env.tool_calendar_list(
        date_from="2026-03-02", date_to="2026-03-03", attendee="dana"
    )
    assert [e["id"] for e in dana_events] == ["evt-1", "evt-3"]


def test_calendar_get() -> None:
    env = CalendarEnvironment(
        {
            "events": {
                "evt-1": {
                    "title": "Sprint Planning",
                    "start": "2026-03-02T10:00",
                    "end": "2026-03-02T11:30",
                    "attendees": ["dana"],
                    "location": "Boardroom",
                }
            }
        }
    )

    res = env.tool_calendar_get("evt-1")
    assert res == {
        "id": "evt-1",
        "title": "Sprint Planning",
        "start": "2026-03-02T10:00",
        "end": "2026-03-02T11:30",
        "attendees": ["dana"],
        "location": "Boardroom",
    }

    with pytest.raises(ToolFailure, match=re.escape("event 'evt-99' not found")):
        env.tool_calendar_get("evt-99")


def test_calendar_find_free() -> None:
    env = CalendarEnvironment(
        {
            "people": {
                "dana": {"working_hours": ["09:00", "17:00"]},
                "alex": {"working_hours": ["10:00", "18:00"]},
            },
            "events": {
                "evt-1": {
                    "title": "Meeting 1",
                    "start": "2026-03-02T11:00",
                    "end": "2026-03-02T12:00",
                    "attendees": ["dana"],
                },
                "evt-2": {
                    "title": "Meeting 2 (touching)",
                    "start": "2026-03-02T12:00",
                    "end": "2026-03-02T13:00",
                    "attendees": ["alex"],
                },
                "evt-3": {
                    "title": "Late Meeting",
                    "start": "2026-03-02T16:30",
                    "end": "2026-03-02T17:30",
                    "attendees": ["dana", "alex"],
                },
            },
        }
    )

    # Empty attendees
    with pytest.raises(ToolFailure, match="attendees list cannot be empty"):
        env.tool_calendar_find_free(attendees=[], date="2026-03-02", duration_minutes=30)

    # Unknown person
    with pytest.raises(ToolFailure, match=re.escape("unknown person 'ghost'")):
        env.tool_calendar_find_free(
            attendees=["dana", "ghost"], date="2026-03-02", duration_minutes=30
        )

    # Find free windows for Dana and Alex:
    # Add an event with unrelated attendee and an event outside working hours
    env.events["evt-other"] = {
        "title": "Other",
        "start": "2026-03-02T10:00",
        "end": "2026-03-02T11:00",
        "attendees": ["other_person"],
    }
    env.events["evt-early"] = {
        "title": "Early",
        "start": "2026-03-02T06:00",
        "end": "2026-03-02T07:00",
        "attendees": ["dana"],
    }

    # Free slots with evt-3 present (ends at 17:00, no trailing gap)
    free = env.tool_calendar_find_free(
        attendees=["dana", "alex"], date="2026-03-02", duration_minutes=45
    )
    assert free == [
        {"start": "2026-03-02T10:00", "end": "2026-03-02T11:00"},
        {"start": "2026-03-02T13:00", "end": "2026-03-02T16:30"},
    ]

    # Remove evt-3 so there is a trailing gap until 17:00
    del env.events["evt-3"]
    # Add an event starting exactly at window start (10:00) and an event ending at 16:50 (trailing gap 10 min < 30 min)
    env.events["evt-start"] = {
        "title": "Start",
        "start": "2026-03-02T10:00",
        "end": "2026-03-02T10:30",
        "attendees": ["dana"],
    }
    env.events["evt-end"] = {
        "title": "End",
        "start": "2026-03-02T16:00",
        "end": "2026-03-02T16:50",
        "attendees": ["alex"],
    }
    free_edges = env.tool_calendar_find_free(
        attendees=["dana", "alex"], date="2026-03-02", duration_minutes=30
    )
    # Gaps: 10:30 to 11:00 (30 min), 13:00 to 16:00 (180 min); trailing 16:50 to 17:00 is 10 min (<30 min, filtered out)
    assert free_edges == [
        {"start": "2026-03-02T10:30", "end": "2026-03-02T11:00"},
        {"start": "2026-03-02T13:00", "end": "2026-03-02T16:00"},
    ]
    del env.events["evt-start"]
    del env.events["evt-end"]

    free_with_trailing = env.tool_calendar_find_free(
        attendees=["dana", "alex"], date="2026-03-02", duration_minutes=45
    )
    assert free_with_trailing == [
        {"start": "2026-03-02T10:00", "end": "2026-03-02T11:00"},
        {"start": "2026-03-02T13:00", "end": "2026-03-02T17:00"},
    ]

    # Duration filter: 90 min (only second slot fits)
    long_free = env.tool_calendar_find_free(
        attendees=["dana", "alex"], date="2026-03-02", duration_minutes=90
    )
    assert long_free == [{"start": "2026-03-02T13:00", "end": "2026-03-02T17:00"}]

    # Incompatible working hours
    env_shift = CalendarEnvironment(
        {
            "people": {
                "night": {"working_hours": ["00:00", "08:00"]},
                "day": {"working_hours": ["09:00", "17:00"]},
            }
        }
    )
    assert (
        env_shift.tool_calendar_find_free(
            attendees=["night", "day"], date="2026-03-02", duration_minutes=15
        )
        == []
    )


def test_calendar_create() -> None:
    env = CalendarEnvironment(
        {
            "people": {
                "dana": {"working_hours": ["09:00", "17:00"]},
                "alex": {"working_hours": ["09:00", "17:00"]},
            },
            "events": {
                "evt-1": {
                    "title": "Existing",
                    "start": "2026-03-02T10:00",
                    "end": "2026-03-02T11:00",
                    "attendees": ["dana"],
                }
            },
        }
    )

    # Invalid time interval
    with pytest.raises(ToolFailure, match="event end must be strictly after start"):
        env.tool_calendar_create(
            title="Bad",
            start="2026-03-02T10:00",
            end="2026-03-02T09:00",
            attendees=["dana"],
        )
    with pytest.raises(ToolFailure, match="event end must be strictly after start"):
        env.tool_calendar_create(
            title="Zero",
            start="2026-03-02T10:00",
            end="2026-03-02T10:00",
            attendees=["dana"],
        )

    # Unknown attendee
    with pytest.raises(ToolFailure, match=re.escape("unknown attendee 'nobody'")):
        env.tool_calendar_create(
            title="Bad Att",
            start="2026-03-02T11:00",
            end="2026-03-02T12:00",
            attendees=["nobody"],
        )

    # Conflict with existing event
    with pytest.raises(ToolFailure, match="conflict with event 'evt-1'"):
        env.tool_calendar_create(
            title="Overlap",
            start="2026-03-02T10:30",
            end="2026-03-02T11:30",
            attendees=["dana"],
        )

    # Touching ends do not conflict (11:00 is end of evt-1 and start of new)
    res = env.tool_calendar_create(
        title="Follow-up",
        start="2026-03-02T11:00",
        end="2026-03-02T12:00",
        attendees=["dana", "alex"],
        location="Room 2",
    )
    assert res == {"id": "evt-2"}
    assert env.events["evt-2"]["title"] == "Follow-up"
    assert env.events["evt-2"]["location"] == "Room 2"


def test_calendar_update() -> None:
    env = CalendarEnvironment(
        {
            "people": {
                "dana": {"working_hours": ["09:00", "17:00"]},
                "alex": {"working_hours": ["09:00", "17:00"]},
            },
            "events": {
                "evt-1": {
                    "title": "Meeting 1",
                    "start": "2026-03-02T10:00",
                    "end": "2026-03-02T11:00",
                    "attendees": ["dana"],
                },
                "evt-2": {
                    "title": "Meeting 2",
                    "start": "2026-03-02T14:00",
                    "end": "2026-03-02T15:00",
                    "attendees": ["alex"],
                },
            },
        }
    )

    # Missing event
    with pytest.raises(ToolFailure, match=re.escape("event 'evt-99' not found")):
        env.tool_calendar_update("evt-99", title="Test")

    # Empty update
    with pytest.raises(ToolFailure, match="at least one field must be provided to update"):
        env.tool_calendar_update("evt-1")

    # Invalid end <= start
    with pytest.raises(ToolFailure, match="event end must be strictly after start"):
        env.tool_calendar_update("evt-1", start="2026-03-02T12:00", end="2026-03-02T11:00")

    # Unknown attendee
    with pytest.raises(ToolFailure, match=re.escape("unknown attendee 'nobody'")):
        env.tool_calendar_update("evt-1", attendees=["nobody"])

    # Conflict with evt-1 when moving evt-2 to overlap with Dana
    with pytest.raises(ToolFailure, match="conflict with event 'evt-1'"):
        env.tool_calendar_update(
            "evt-2",
            start="2026-03-02T10:30",
            end="2026-03-02T11:30",
            attendees=["dana"],
        )

    # Update evt-1 itself without conflicting with itself
    updated = env.tool_calendar_update(
        "evt-1",
        title="Updated Standup",
        start="2026-03-02T10:15",
        end="2026-03-02T11:15",
        location="Zoom",
    )
    assert updated["id"] == "evt-1"
    assert updated["title"] == "Updated Standup"
    assert updated["start"] == "2026-03-02T10:15"
    assert updated["end"] == "2026-03-02T11:15"
    assert updated["location"] == "Zoom"

    # Update evt-1 without changing title
    updated_notitle = env.tool_calendar_update(
        "evt-1",
        location="Conference Hall",
    )
    assert updated_notitle["title"] == "Updated Standup"
    assert updated_notitle["location"] == "Conference Hall"


def test_calendar_cancel() -> None:
    env = CalendarEnvironment(
        {
            "events": {
                "evt-1": {"title": "Delete me"},
            }
        }
    )

    assert env.tool_calendar_cancel("evt-1") == "cancelled"
    assert "evt-1" not in env.events
    assert env.cancelled == ["evt-1"]

    with pytest.raises(ToolFailure, match=re.escape("event 'evt-1' not found")):
        env.tool_calendar_cancel("evt-1")


def test_calendar_snapshot_and_determinism() -> None:
    initial = {
        "now": "2026-03-02T09:00",
        "people": {
            "dana": {"working_hours": ["09:00", "17:00"]},
        },
        "events": {
            "evt-1": {
                "title": "E1",
                "start": "2026-03-02T09:00",
                "end": "2026-03-02T10:00",
                "attendees": ["dana"],
            },
            "evt-2": {
                "title": "E2",
                "start": "2026-03-02T14:00",
                "end": "2026-03-02T15:00",
                "attendees": ["dana"],
            },
        },
    }

    env1 = CalendarEnvironment(initial, seed=123)
    env2 = CalendarEnvironment(initial, seed=123)

    actions = [
        (
            "calendar_create",
            {
                "title": "E3",
                "start": "2026-03-02T11:00",
                "end": "2026-03-02T12:00",
                "attendees": ["dana"],
            },
        ),
        ("calendar_update", {"event_id": "evt-1", "title": "E1 Renamed"}),
        ("calendar_cancel", {"event_id": "evt-2"}),
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
    assert "evt-1" in snap1["events"]
    assert "evt-3" in snap1["events"]
    assert "evt-2" not in snap1["events"]
    assert snap1["cancelled"] == ["evt-2"]


def test_calendar_non_standard_ids() -> None:
    env = CalendarEnvironment(
        {
            "people": {"dana": {"working_hours": ["09:00", "17:00"]}},
            "events": {
                "OTHER-1": {
                    "title": "Foreign",
                    "start": "2026-03-02T09:00",
                    "end": "2026-03-02T10:00",
                    "attendees": ["dana"],
                },
                "evt-alpha": {
                    "title": "Alpha",
                    "start": "2026-03-02T10:00",
                    "end": "2026-03-02T11:00",
                    "attendees": ["dana"],
                },
                "NO_DASH": {
                    "title": "NoDash",
                    "start": "2026-03-02T11:00",
                    "end": "2026-03-02T12:00",
                    "attendees": ["dana"],
                },
                "evt-5": {
                    "title": "Numeric",
                    "start": "2026-03-02T13:00",
                    "end": "2026-03-02T14:00",
                    "attendees": ["dana"],
                },
            },
        }
    )

    created = env.tool_calendar_create(
        title="Next",
        start="2026-03-02T15:00",
        end="2026-03-02T16:00",
        attendees=["dana"],
    )
    assert created["id"] == "evt-6"
    snap = env.snapshot()
    assert "NO_DASH" in snap["events"]


def test_calendar_world_integration() -> None:
    task = TaskSpec.model_validate(
        {
            "id": "calendar-001-create",
            "family": "calendar",
            "title": "Schedule a meeting",
            "prompt": "Schedule a meeting with Dana",
            "envs": {
                "calendar": {
                    "now": "2026-03-02T09:00",
                    "people": {"dana": {"working_hours": ["09:00", "17:00"]}},
                }
            },
            "max_steps": 5,
            "optimal_steps": 1,
            "reference_solution": [
                {
                    "tool": "calendar_create",
                    "args": {
                        "title": "Sync",
                        "start": "2026-03-02T10:00",
                        "end": "2026-03-02T10:30",
                        "attendees": ["dana"],
                    },
                }
            ],
            "success": [
                {
                    "path": 'calendar.events."evt-1".title',
                    "op": "equals",
                    "value": "Sync",
                }
            ],
        }
    )

    world = World(task)
    step1 = world.call(
        "calendar_create",
        {
            "title": "Sync",
            "start": "2026-03-02T10:00",
            "end": "2026-03-02T10:30",
            "attendees": ["dana"],
        },
    )
    assert step1.ok is True
    assert step1.executed is True

    snap = world.snapshot()
    assert snap["calendar"]["events"]["evt-1"]["title"] == "Sync"
