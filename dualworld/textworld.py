"""TextWorld task loading and custom-planner execution interfaces."""

import json
from pathlib import Path

from ._paths import ROOT, bootstrap

bootstrap()
from common import BasePlanner
from simulator import EventDrivenSimulator
from priority_rule import has_priority_violation

REGIMES = ("A_L1", "A_L2", "M_L1", "M_L2")
TASK_COUNTS = (5, 8, 10, 12, 15)


def load_task(regime, task_count, seed=1):
    """Return one released raw episode, including simulator configuration."""
    if regime not in REGIMES or task_count not in TASK_COUNTS or seed not in range(1, 11):
        raise ValueError("Expected a released regime, task count and seed (1–10).")
    path = ROOT / f"data/textworld/task/{regime}/task_{task_count}_seed{seed}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(planner, task, output_dir):
    """Execute BasePlanner.plan(context), save logs, and apply official R/T.

    Use a separate output directory for each episode to preserve its logs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = task.get("config", {})
    simulator = EventDrivenSimulator(
        planner, speed=config.get("speed", task.get("SPEED", 500.0)) / 60.0,
        capacity=config.get("capacity", task.get("CAPACITY", 5)),
        start_pos={"x": config.get("start_x", task.get("CUR_X", 0.0)),
                   "y": config.get("start_y", task.get("CUR_Y", 0.0))},
        result_file=str(output_dir / "result.json"),
        planlog_file=str(output_dir / "plans.json"))
    simulator.run(task["tasks"])
    result = simulator.save_logs()
    invalid = has_priority_violation(result["logs"], {t["id"]: t["type"] for t in task["tasks"]})
    scores = {"R": 0.0 if invalid else result["summary"]["total_revenue"],
              "T": 9999.0 if invalid else result["summary"]["final_time"],
              "priority_violation": invalid, "raw_summary": result["summary"]}
    (output_dir / "scores.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    return scores
