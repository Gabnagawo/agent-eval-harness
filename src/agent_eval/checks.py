"""Deterministic success checker: JMESPath lookups plus a small set of comparison ops."""

from __future__ import annotations

import re
from typing import Any

import jmespath
from jmespath.exceptions import JMESPathError

from agent_eval.models import Check, CheckResult


class CheckError(ValueError):
    """A check that cannot be evaluated (bad JMESPath, bad regex, wrong value type)."""


def _fold(value: Any, ignore_case: bool) -> Any:
    return value.casefold() if ignore_case and isinstance(value, str) else value


def _contains(actual: Any, value: Any, ignore_case: bool) -> bool:
    if actual is None:
        return False
    if isinstance(actual, str):
        return str(_fold(value, ignore_case)) in _fold(actual, ignore_case)
    if isinstance(actual, dict):
        return value in actual
    if isinstance(actual, list):
        return any(_fold(item, ignore_case) == _fold(value, ignore_case) for item in actual)
    return False


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CheckError(f"{what} must be a number, got {value!r}")
    return float(value)


def _passes(check: Check, actual: Any) -> bool:
    op, value, ci = check.op, check.value, check.ignore_case
    if op == "exists":
        return actual is not None
    if op == "absent":
        return actual is None
    if op == "equals":
        return bool(_fold(actual, ci) == _fold(value, ci))
    if op == "not_equals":
        return bool(_fold(actual, ci) != _fold(value, ci))
    if op == "contains":
        return _contains(actual, value, ci)
    if op == "not_contains":
        return not _contains(actual, value, ci)
    if op == "matches":
        if actual is None:
            return False
        try:
            return re.search(str(value), str(actual), re.IGNORECASE if ci else 0) is not None
        except re.error as exc:
            raise CheckError(f"bad regex {value!r}: {exc}") from exc
    if op == "length":
        return isinstance(actual, str | list | dict) and len(actual) == value
    if op == "in":
        if not isinstance(value, list):
            raise CheckError("'in' needs a list value")
        return any(_fold(actual, ci) == _fold(item, ci) for item in value)
    # gte / lte
    if actual is None:
        return False
    left, right = _number(actual, "the looked-up value"), _number(value, "'value'")
    return left >= right if op == "gte" else left <= right


def evaluate(check: Check, data: Any) -> CheckResult:
    """Evaluate one check against ``data`` (a world snapshot, or the final answer text)."""
    actual = data
    if check.path is not None:
        try:
            actual = jmespath.search(check.path, data)
        except JMESPathError as exc:
            raise CheckError(f"bad JMESPath {check.path!r}: {exc}") from exc
    return CheckResult(check=check, passed=_passes(check, actual), actual=actual)


def evaluate_all(checks: list[Check], data: Any) -> list[CheckResult]:
    return [evaluate(check, data) for check in checks]
