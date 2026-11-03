from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BasePlanner(ABC):
    @abstractmethod
    def plan(self, context: Dict[str, Any]) -> List[str]:
        """Return a node sequence (e.g. ["START", "P1", "D1"]) for the given context.

        Expanding it into an executable point sequence is the evaluator's job, so
        that every modality is scored on the same output space.
        """
        raise NotImplementedError
