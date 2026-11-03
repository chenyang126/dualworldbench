from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "outputs" / "simworld"
NAV_PLANNER_DIR = REPO_ROOT / "simworld" / "nav_planner"
