"""``calendar``: scheduling and calendar management.

State and snapshot::

    {
        "now": "2026-03-02T09:00",
        "people": {
            "dana": {"working_hours": ["09:00", "17:00"]}
        },
        "events": {
            "evt-1": {
                "title": "...",
                "start": "2026-03-02T10:00",
                "end": "2026-03-02T11:00",
                "attendees": ["dana"],
                "location": "Room 101"
            }
        },
        "cancelled": ["evt-2", ...]
    }
"""

from __future__ import annotations

import copy
from datetime import datetime
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


def _evt_sort_key(event_id: str) -> tuple[str, int, str]:
    if "-" in event_id:
        prefix, num = event_id.rsplit("-", 1)
        if num.isdigit():
            return (prefix, int(num), event_id)
    return (event_id, 0, event_id)


@register
class CalendarEnvironment(Environment):
    kind = "calendar"

    def load(self, state: dict[str, Any]) -> None:
        self.now: str = str(state.get("now", "2026-01-01T09:00"))
        self.people: dict[str, dict[str, Any]] = {}
        for person_name, person_data in state.get("people", {}).items():
            wh = person_data.get("working_hours", ["09:00", "17:00"])
            self.people[str(person_name)] = {"working_hours": [str(wh[0]), str(wh[1])]}

        self.cancelled: list[str] = [str(c) for c in state.get("cancelled", [])]
        self.events: dict[str, dict[str, Any]] = {}
        for raw_id, raw_evt in state.get("events", {}).items():
            event_id = str(raw_id)
            self.events[event_id] = {
                "title": str(raw_evt.get("title", "")),
                "start": str(raw_evt.get("start", "")),
                "end": str(raw_evt.get("end", "")),
                "attendees": [str(a) for a in raw_evt.get("attendees", [])],
                "location": raw_evt.get("location"),
            }

    def tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="calendar_now",
                description="Get the current date-time in local ISO format (YYYY-MM-DDTHH:MM).",
                parameters=_params({}, []),
            ),
            ToolSpec(
                name="calendar_list",
                description="List calendar events between date_from and date_to (inclusive), optionally filtered by attendee.",
                parameters=_params(
                    {
                        "date_from": {
                            "type": "string",
                            "description": "Start date in YYYY-MM-DD format (inclusive).",
                        },
                        "date_to": {
                            "type": "string",
                            "description": "End date in YYYY-MM-DD format (inclusive).",
                        },
                        "attendee": {
                            "type": "string",
                            "description": "Optional attendee username to filter by.",
                        },
                    },
                    ["date_from", "date_to"],
                ),
            ),
            ToolSpec(
                name="calendar_get",
                description="Get full event details by event ID.",
                parameters=_params(
                    {
                        "event_id": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Event ID (e.g. 'evt-1').",
                        }
                    },
                    ["event_id"],
                ),
            ),
            ToolSpec(
                name="calendar_find_free",
                description="Find available free time windows within everyone's working hours for the specified attendees.",
                parameters=_params(
                    {
                        "attendees": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "description": "List of attendees who must attend.",
                        },
                        "date": {
                            "type": "string",
                            "description": "Date in YYYY-MM-DD format.",
                        },
                        "duration_minutes": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Required minimum duration in minutes.",
                        },
                    },
                    ["attendees", "date", "duration_minutes"],
                ),
            ),
            ToolSpec(
                name="calendar_create",
                description="Create a new calendar event. Checks for schedule conflicts and attendee validity.",
                parameters=_params(
                    {
                        "title": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Event title.",
                        },
                        "start": {
                            "type": "string",
                            "description": "Start datetime in YYYY-MM-DDTHH:MM format.",
                        },
                        "end": {
                            "type": "string",
                            "description": "End datetime in YYYY-MM-DDTHH:MM format.",
                        },
                        "attendees": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of attendee usernames.",
                        },
                        "location": {
                            "type": ["string", "null"],
                            "description": "Optional location or room.",
                        },
                    },
                    ["title", "start", "end"],
                ),
            ),
            ToolSpec(
                name="calendar_update",
                description="Update fields of an existing calendar event. Checks for schedule conflicts.",
                parameters=_params(
                    {
                        "event_id": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Event ID to update.",
                        },
                        "title": {"type": "string", "description": "New title."},
                        "start": {"type": "string", "description": "New start datetime."},
                        "end": {"type": "string", "description": "New end datetime."},
                        "attendees": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "New list of attendees.",
                        },
                        "location": {
                            "type": ["string", "null"],
                            "description": "New location or room.",
                        },
                    },
                    ["event_id"],
                ),
            ),
            ToolSpec(
                name="calendar_cancel",
                description="Cancel and remove an event permanently. This cannot be undone.",
                parameters=_params(
                    {
                        "event_id": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Event ID to cancel.",
                        }
                    },
                    ["event_id"],
                ),
                destructive=True,
            ),
        ]

    def snapshot(self) -> dict[str, Any]:
        return {
            "now": self.now,
            "people": {k: copy.deepcopy(v) for k, v in sorted(self.people.items())},
            "events": {
                k: copy.deepcopy(self.events[k]) for k in sorted(self.events, key=_evt_sort_key)
            },
            "cancelled": list(self.cancelled),
        }

    def _next_id(self) -> str:
        prefix = "evt-"
        numbers: list[int] = []
        for key in set(self.events) | set(self.cancelled):
            if key.startswith(prefix):
                part = key[len(prefix) :]
                if part.isdigit():
                    numbers.append(int(part))
        next_num = max(numbers, default=0) + 1
        return f"evt-{next_num}"

    def _check_conflict(
        self, start: str, end: str, attendees: list[str], ignore_id: str | None = None
    ) -> None:
        attendees_set = set(attendees)
        for other_id, other_evt in self.events.items():
            if other_id == ignore_id:
                continue
            if not attendees_set.intersection(other_evt.get("attendees", [])):
                continue
            # Two events conflict when intervals overlap (touching ends do not overlap)
            if start < other_evt["end"] and end > other_evt["start"]:
                raise ToolFailure(
                    f"conflict with event '{other_id}' ('{other_evt['title']}') from "
                    f"{other_evt['start']} to {other_evt['end']}"
                )

    def tool_calendar_now(self) -> str:
        return self.now

    def tool_calendar_list(
        self, date_from: str, date_to: str, attendee: str | None = None
    ) -> list[dict[str, Any]]:
        if date_to < date_from:
            raise ToolFailure("date_to cannot be earlier than date_from")

        results: list[dict[str, Any]] = []
        range_start = f"{date_from}T00:00"
        range_end = f"{date_to}T23:59:59"

        sorted_events = sorted(
            self.events.items(),
            key=lambda item: (item[1]["start"], _evt_sort_key(item[0])),
        )

        for evt_id, evt in sorted_events:
            if evt["start"] <= range_end and evt["end"] >= range_start:
                if attendee is not None and attendee not in evt.get("attendees", []):
                    continue
                results.append(
                    {
                        "id": evt_id,
                        "title": evt["title"],
                        "start": evt["start"],
                        "end": evt["end"],
                        "attendees": list(evt.get("attendees", [])),
                    }
                )
        return results

    def tool_calendar_get(self, event_id: str) -> dict[str, Any]:
        if event_id not in self.events:
            raise ToolFailure(f"event '{event_id}' not found")
        data = copy.deepcopy(self.events[event_id])
        data["id"] = event_id
        return data

    def tool_calendar_find_free(
        self, attendees: list[str], date: str, duration_minutes: int
    ) -> list[dict[str, str]]:
        if not attendees:
            raise ToolFailure("attendees list cannot be empty")
        for person in attendees:
            if person not in self.people:
                raise ToolFailure(f"unknown person '{person}'")

        starts = [self.people[p]["working_hours"][0] for p in attendees]
        ends = [self.people[p]["working_hours"][1] for p in attendees]
        common_start = max(starts)
        common_end = min(ends)

        win_start = f"{date}T{common_start}"
        win_end = f"{date}T{common_end}"
        if win_start >= win_end:
            return []

        # Busy intervals intersecting with window
        busy_intervals: list[tuple[str, str]] = []
        attendees_set = set(attendees)
        for evt in self.events.values():
            if attendees_set.intersection(evt.get("attendees", [])):
                busy_start = max(evt["start"], win_start)
                busy_end = min(evt["end"], win_end)
                if busy_start < busy_end:
                    busy_intervals.append((busy_start, busy_end))

        # Merge busy intervals
        busy_intervals.sort()
        merged: list[tuple[str, str]] = []
        for b_start, b_end in busy_intervals:
            if not merged:
                merged.append((b_start, b_end))
            else:
                prev_start, prev_end = merged[-1]
                if b_start <= prev_end:
                    merged[-1] = (prev_start, max(prev_end, b_end))
                else:
                    merged.append((b_start, b_end))

        # Extract free slots
        free_windows: list[dict[str, str]] = []
        curr = win_start
        for b_start, b_end in merged:
            if curr < b_start:
                dur = (
                    datetime.fromisoformat(b_start) - datetime.fromisoformat(curr)
                ).total_seconds() / 60
                if dur >= duration_minutes:
                    free_windows.append({"start": curr, "end": b_start})
            curr = max(curr, b_end)

        if curr < win_end:
            dur = (
                datetime.fromisoformat(win_end) - datetime.fromisoformat(curr)
            ).total_seconds() / 60
            if dur >= duration_minutes:
                free_windows.append({"start": curr, "end": win_end})

        return free_windows

    def tool_calendar_create(
        self,
        title: str,
        start: str,
        end: str,
        attendees: list[str] | None = None,
        location: str | None = None,
    ) -> dict[str, str]:
        if end <= start:
            raise ToolFailure("event end must be strictly after start")

        att_list = list(attendees or [])
        for person in att_list:
            if person not in self.people:
                raise ToolFailure(f"unknown attendee '{person}'")

        self._check_conflict(start, end, att_list)

        new_id = self._next_id()
        self.events[new_id] = {
            "title": title,
            "start": start,
            "end": end,
            "attendees": att_list,
            "location": location,
        }
        return {"id": new_id}

    def tool_calendar_update(
        self,
        event_id: str,
        title: str | None = None,
        start: str | None = None,
        end: str | None = None,
        attendees: list[str] | None = None,
        location: str | None = None,
    ) -> dict[str, Any]:
        if event_id not in self.events:
            raise ToolFailure(f"event '{event_id}' not found")

        if (
            title is None
            and start is None
            and end is None
            and attendees is None
            and location is None
        ):
            raise ToolFailure("at least one field must be provided to update")

        evt = self.events[event_id]
        new_start = start if start is not None else evt["start"]
        new_end = end if end is not None else evt["end"]

        if new_end <= new_start:
            raise ToolFailure("event end must be strictly after start")

        new_attendees = list(attendees) if attendees is not None else list(evt["attendees"])
        for person in new_attendees:
            if person not in self.people:
                raise ToolFailure(f"unknown attendee '{person}'")

        self._check_conflict(new_start, new_end, new_attendees, ignore_id=event_id)

        if title is not None:
            evt["title"] = title
        evt["start"] = new_start
        evt["end"] = new_end
        evt["attendees"] = new_attendees
        if location is not None:
            evt["location"] = location

        res = copy.deepcopy(evt)
        res["id"] = event_id
        return res

    def tool_calendar_cancel(self, event_id: str) -> str:
        if event_id not in self.events:
            raise ToolFailure(f"event '{event_id}' not found")
        del self.events[event_id]
        self.cancelled.append(event_id)
        return "cancelled"
