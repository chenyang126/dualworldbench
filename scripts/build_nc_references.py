"""Compute exact distance-only pickup/delivery references for scored subsets.

This optional exponential DP leaves released data unchanged. Its route begins
at the task's current position and ends at the last delivery, with no return leg.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simworld/nav_planner"))
from scene_config import get_scene_config, load_task_json
from scene_eval import SceneMatrixEvaluator


def exact_subset_cost(context, task_ids, evaluator):
    """Minimize distance over all precedence- and capacity-feasible orders.

    Each layer has one more completed operation. A state retains the picked
    mask, delivered mask and last stop; future costs depend only on that state.
    Deadlines/rewards are not constraints in this distance-only NC reference.
    """
    if context.get("carried"):
        raise ValueError("This builder requires an initially empty load.")
    lookup = {t["id"]: t for t in context["tasks"]}
    tasks = [lookup[task_id] for task_id in sorted(set(task_ids))]
    n = len(tasks)
    if not n:
        return 0.0
    capacity = int(context["max_capacity"])
    if capacity < 1:
        raise ValueError("Capacity must be positive.")
    stops = [t["pickup"] for t in tasks] + [t["deliver"] for t in tasks]
    def distance(a, b):
        # Match node_sequence_to_actions: coincident coordinates emit no move.
        if float(a["x"]) == float(b["x"]) and float(a["y"]) == float(b["y"]):
            return 0.0
        return evaluator.distance_between(a, b)["total_distance_m"]
    initial = [distance(context["pos"], p) for p in stops]
    distances = [[distance(a, b) for b in stops] for a in stops]
    full = (1 << n) - 1
    layer = {(0, 0, -1): 0.0}
    for _ in range(2 * n):
        next_layer = {}
        for (picked, delivered, last), cost in layer.items():
            load = picked.bit_count() - delivered.bit_count()
            for i in range(n):
                bit = 1 << i
                if not picked & bit and load < capacity:
                    target = i
                    state = (picked | bit, delivered, target)
                elif picked & bit and not delivered & bit:
                    target = n + i
                    state = (picked, delivered | bit, target)
                else:
                    continue
                candidate = cost + (initial[target] if last == -1 else distances[last][target])
                if candidate < next_layer.get(state, float("inf")):
                    next_layer[state] = candidate
        layer = next_layer
    return min(cost for (picked, delivered, last), cost in layer.items()
               if picked == full and delivered == full)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True, help="Output of score_simworld.py")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/simworld/exact_references.json")
    parser.add_argument("--recompute", action="store_true", help="Also replace known released costs")
    parser.add_argument("--max-tasks", type=int, default=10, help="Skip larger subsets; DP grows exponentially")
    args = parser.parse_args()
    if args.max_tasks < 1:
        parser.error("max-tasks must be positive")
    released = json.loads((ROOT / "data/simworld/reference/subset_reference.json").read_text())
    result = {"method": "exact_subset_dp", "costs": {}}
    if args.output.exists():
        result = json.loads(args.output.read_text())
        if result.get("method") != "exact_subset_dp":
            raise ValueError("Output already exists with a different reference method.")
    subsets = {}
    for row in json.loads(args.scores.read_text())["records"]:
        ids = sorted(set(row.get("delivered_ids", [])))
        if ids:
            key = f"{row['scene']}|{row['sample_id']}|{','.join(ids)}"
            subsets[key] = (row["scene"], row["sample_id"], ids)
    evaluators = {}
    for key, (scene, sample_id, ids) in sorted(subsets.items()):
        if key in result["costs"] or (key in released and not args.recompute):
            continue
        if len(ids) > args.max_tasks:
            print(f"Skipped {key}: {len(ids)} tasks exceeds --max-tasks", flush=True)
            continue
        if scene not in evaluators:
            evaluators[scene] = SceneMatrixEvaluator(scene)
        context = load_task_json(str(get_scene_config(scene).task_dir / f"{sample_id}.json"), scene)
        cost = exact_subset_cost(context, ids, evaluators[scene])
        result["costs"][key] = cost
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output)
        print(f"Computed {key}: {cost:.6f} m", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {len(result['costs'])} exact subset references: {args.output}")


if __name__ == "__main__":
    main()
