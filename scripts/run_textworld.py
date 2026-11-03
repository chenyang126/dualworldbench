"""Run the released TextWorld episodes and apply the paper's priority score."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "textworld/agent_planner"))
from simulator import EventDrivenSimulator
from planner_algo import AlgoPlanner
from planner_dp import DPPlanner
from priority_rule import has_priority_violation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("llm", "algo", "dp"), required=True)
    parser.add_argument("--model", help="Model ID required for --mode llm")
    parser.add_argument("--regimes", nargs="+", choices=("A_L1", "A_L2", "M_L1", "M_L2"),
                        default=["A_L1", "A_L2", "M_L1", "M_L2"])
    parser.add_argument("--task-counts", nargs="+", type=int, choices=(5, 8, 10, 12, 15),
                        default=[5, 8, 10, 12, 15])
    parser.add_argument("--seeds", nargs="+", type=int, choices=range(1, 11), default=[1])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/textworld")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.mode == "llm" and not args.model:
        parser.error("--mode llm requires --model")
    jobs = [(regime, n, seed) for regime in args.regimes
            for n in args.task_counts for seed in args.seeds]
    if args.dry_run:
        print(json.dumps({"episodes": len(jobs), "mode": args.mode, "model": args.model}))
        return
    if args.mode == "llm":
        from planner_gpt import GPTPlanner
        # Fail on absent credentials before creating results.
        from model_endpoints import resolve
        resolve(args.model)
        factory = lambda: GPTPlanner(model=args.model)
    else:
        factory = DPPlanner if args.mode == "dp" else AlgoPlanner
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scored = []
    for regime, n, seed in jobs:
        task_path = ROOT / f"data/textworld/task/{regime}/task_{n}_seed{seed}.json"
        data = json.loads(task_path.read_text())
        name = f"{regime}_n{n}_seed{seed}"
        result_path = args.output_dir / f"{name}_result.json"
        plan_path = args.output_dir / f"{name}_plans.json"
        cfg = data.get("config", {})
        sim = EventDrivenSimulator(
            factory(), speed=cfg.get("speed", data.get("SPEED", 500.0)) / 60.0,
            capacity=cfg.get("capacity", data.get("CAPACITY", 5)),
            start_pos={"x": cfg.get("start_x", data.get("CUR_X", 0.0)),
                       "y": cfg.get("start_y", data.get("CUR_Y", 0.0))},
            result_file=str(result_path), planlog_file=str(plan_path))
        sim.run(data["tasks"])
        result = sim.save_logs()
        invalid = has_priority_violation(result["logs"], {t["id"]: t["type"] for t in data["tasks"]})
        row = {"regime": regime, "task_count": n, "seed": seed,
               "model": args.model or args.mode, "priority_violation": invalid,
               "R": 0.0 if invalid else result["summary"]["total_revenue"],
               "T": 9999.0 if invalid else result["summary"]["final_time"],
               "raw_summary": result["summary"], "result_file": result_path.name,
               "plan_file": plan_path.name}
        scored.append(row)
        (args.output_dir / "scores.json").write_text(json.dumps(scored, indent=2) + "\n")
        print(f"{name}: R={row['R']}, T={row['T']}, priority_violation={invalid}")


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
