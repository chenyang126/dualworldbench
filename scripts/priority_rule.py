"""Shared TextWorld Type-C priority rule.

The prompt and deterministic solvers define priority as a state constraint:
while at least one Type-C task is carried, an A/B task may not be delivered.
Picking up another task is allowed, and an unfinished C task is accounted for
as undelivered rather than being relabelled as a priority violation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def has_priority_violation(
    logs: Iterable[Mapping[str, Any]],
    type_of: Mapping[str, str | None],
) -> bool:
    """Return whether an execution delivers A/B while carrying any Type C."""
    carried_c: set[str] = set()
    for event in logs:
        action_type = event.get("type")
        task_id = event.get("id")
        task_type = type_of.get(task_id) if task_id is not None else None

        if action_type == "pickup" and task_type == "C":
            carried_c.add(task_id)
        elif action_type == "deliver":
            if task_type in {"A", "B"} and carried_c:
                return True
            if task_type == "C":
                carried_c.discard(task_id)
    return False
