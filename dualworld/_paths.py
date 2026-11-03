"""Locate the checkout independently of the caller's working directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def bootstrap():
    # The inherited engines use sibling imports. Keep their implementations
    # intact and expose them through this small compatibility layer.
    for path in (ROOT, ROOT / "scripts", ROOT / "simworld/nav_planner",
                 ROOT / "textworld/agent_planner", ROOT / "qa_gt"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
