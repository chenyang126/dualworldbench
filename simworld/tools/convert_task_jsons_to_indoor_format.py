#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def is_new_format(payload: dict) -> bool:
    return "tasks" in payload and "params" in payload and "summary" in payload


def convert_payload(payload: dict) -> dict:
    pickup_points = payload.get("pickup_points", [])
    delivery_points = payload.get("delivery_points", [])
    if len(pickup_points) != len(delivery_points):
        raise ValueError("pickup_points and delivery_points length mismatch")

    start = payload.get("start_point", {})
    num_tasks = int(payload.get("n", len(pickup_points)))
    seed = int(payload.get("seed", 0))

    tasks = []
    for idx, (pickup, delivery) in enumerate(zip(pickup_points, delivery_points), start=1):
        tasks.append(
            {
                "id": f"A-{idx:04d}",
                "type": "A",
                "type_name": "Standard Delivery",
                "title": "",
                "trigger_time": 0.0,
                "pickup": {
                    "x": float(pickup["x"]),
                    "y": float(pickup["y"]),
                },
                "deliver": {
                    "x": float(delivery["x"]),
                    "y": float(delivery["y"]),
                },
                "distance_m": None,
                "end_time": 99999999999999.0,
                "reward": None,
                "priority": False,
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "params": {
            "num_tasks": num_tasks,
            "area_size": None,
            "min_gap": None,
            "max_gap": None,
            "ratio": [1, 0, 0],
            "ratio_effective": [1, 0, 0],
            "seed": seed,
        },
        "summary": {
            "count_total": len(tasks),
            "count_A": len(tasks),
            "count_B": 0,
            "count_C": 0,
        },
        "LEVEL": 1,
        "SPEED": 500.0,
        "NOW_TIME": 0,
        "CUR_X": float(start.get("x", 0.0)),
        "CUR_Y": float(start.get("y", 0.0)),
        "CAPACITY": 5,
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert legacy sim task JSON files to the indoor task schema.")
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Task root directory to convert in place; use a working copy",
    )
    args = parser.parse_args()

    task_files = sorted(args.root.glob("*/*_task/n*-*.json"))
    converted = 0
    skipped = 0

    for path in task_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if is_new_format(payload):
            skipped += 1
            continue
        new_payload = convert_payload(payload)
        path.write_text(json.dumps(new_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        converted += 1

    print(f"converted={converted} skipped={skipped}")


if __name__ == "__main__":
    main()
