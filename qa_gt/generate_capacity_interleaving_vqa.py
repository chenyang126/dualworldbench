#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from scene_registry import load_qa_scene_configs, select_qa_scene_configs


THIS_DIR = Path(__file__).resolve().parent
SIM_TASK_ROOT = THIS_DIR.parent
NAV_PLANNER_DIR = SIM_TASK_ROOT / "simworld" / "nav_planner"

if str(NAV_PLANNER_DIR) not in sys.path:
    sys.path.insert(0, str(NAV_PLANNER_DIR))

from scene_eval import SceneMatrixEvaluator  # type: ignore  # noqa: E402


ORDER_NAME_POOL = [
    "Task A",
    "Task B",
    "Task C",
    "Task D",
    "Task E",
]


@dataclass(frozen=True)
class Node:
    label: str
    x: float
    y: float

    def as_coord(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y}


@dataclass(frozen=True)
class TaskRef:
    task_index: int
    pickup: Node
    deliver: Node

    @property
    def pickup_label(self) -> str:
        return self.pickup.label

    @property
    def deliver_label(self) -> str:
        return self.deliver.label


@dataclass(frozen=True)
class RoutePlan:
    sequence: tuple[str, ...]
    total_distance_m: float


@dataclass(frozen=True)
class CandidateQuestion:
    selected_tasks: tuple[TaskRef, ...]
    capacity: int
    optimal_plan: RoutePlan
    alternative_optimal_plans: tuple[RoutePlan, ...]
    acceptable_plans: tuple[RoutePlan, ...]
    distance_tolerance_ratio: float
    max_acceptable_distance_m: float
    second_best_plan: RoutePlan | None
    best_simple_plan: RoutePlan
    uses_interleaving: bool
    uses_pickup_batching: bool
    interleaving_gain_m: float
    uniqueness_gap_m: float
    score: tuple[float, float, float, float, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate capacity/interleaving planning VQA questions and GT.")
    parser.add_argument(
        "--output",
        type=Path,
        default=THIS_DIR.parent / "outputs" / "generated" / "vqa" / "capacity_interleaving_vqa.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Optional subset of scenes, e.g. rooms campus",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed used to break ties between equally good candidates",
    )
    parser.add_argument(
        "--num-orders",
        type=int,
        default=3,
        help="How many pickup-delivery orders to include in each question",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=None,
        help="Capacity used in the question; default is min(2, num_orders)",
    )
    parser.add_argument(
        "--min-uniqueness-gap",
        type=float,
        default=0.5,
        help="Prefer questions whose best route beats the second-best route by at least this margin",
    )
    parser.add_argument(
        "--distance-tolerance-ratio",
        type=float,
        default=0.10,
        help="Relative distance tolerance for accepted answers; 0.10 means within 10%% of the optimum",
    )
    parser.add_argument(
        "--questions-per-case",
        type=int,
        default=3,
        help="How many distinct task-combination candidates to keep per case",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_case_files(task_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in task_dir.glob("n*-*.json")
        if path.is_file() and path.stem.startswith("n")
    )


def ensure_image_path(image_dir: Path, stem: str) -> Path:
    image_path = image_dir / f"{stem}.png"
    if not image_path.exists():
        raise FileNotFoundError(f"找不到图片: {image_path}")
    return image_path


def build_start_and_tasks(task_payload: Dict[str, Any]) -> tuple[Node, List[TaskRef]]:
    pos = task_payload.get("pos", {})
    start = Node(label="START", x=float(pos["x"]), y=float(pos["y"]))
    tasks: List[TaskRef] = []
    for idx, task in enumerate(task_payload.get("tasks", []), start=1):
        pickup = task["pickup"]
        deliver = task["deliver"]
        tasks.append(
            TaskRef(
                task_index=idx,
                pickup=Node(label=f"P{idx}", x=float(pickup["x"]), y=float(pickup["y"])),
                deliver=Node(label=f"D{idx}", x=float(deliver["x"]), y=float(deliver["y"])),
            )
        )
    return start, tasks


def route_info(evaluator: SceneMatrixEvaluator, src: Node, dst: Node) -> Dict[str, Any]:
    return evaluator.distance_between(src.as_coord(), dst.as_coord())


def precompute_distance_cache(
    evaluator: SceneMatrixEvaluator,
    start: Node,
    tasks: Sequence[TaskRef],
) -> tuple[Dict[str, Dict[str, float]], Dict[str, Node]]:
    nodes: Dict[str, Node] = {start.label: start}
    for task in tasks:
        nodes[task.pickup.label] = task.pickup
        nodes[task.deliver.label] = task.deliver

    cache: Dict[str, Dict[str, float]] = {label: {} for label in nodes}
    labels = list(nodes.keys())
    for src_label in labels:
        for dst_label in labels:
            if src_label == dst_label:
                cache[src_label][dst_label] = 0.0
            else:
                cache[src_label][dst_label] = float(
                    route_info(evaluator, nodes[src_label], nodes[dst_label])["total_distance_m"]
                )
    return cache, nodes


def route_distance(sequence: Sequence[str], distance_cache: Dict[str, Dict[str, float]]) -> float:
    total = 0.0
    for src, dst in zip(sequence, sequence[1:]):
        total += distance_cache[src][dst]
    return total


def enumerate_feasible_sequences(selected_tasks: Sequence[TaskRef], capacity: int) -> Iterable[tuple[str, ...]]:
    task_ids = tuple(task.task_index for task in selected_tasks)
    pickup_map = {task.task_index: task.pickup.label for task in selected_tasks}
    deliver_map = {task.task_index: task.deliver.label for task in selected_tasks}

    def dfs(prefix: List[str], picked: set[int], delivered: set[int], carrying: set[int]) -> Iterable[tuple[str, ...]]:
        if len(delivered) == len(task_ids):
            yield tuple(["START"] + prefix)
            return

        if len(carrying) < capacity:
            for task_id in task_ids:
                if task_id in picked:
                    continue
                picked.add(task_id)
                carrying.add(task_id)
                prefix.append(pickup_map[task_id])
                yield from dfs(prefix, picked, delivered, carrying)
                prefix.pop()
                carrying.remove(task_id)
                picked.remove(task_id)

        for task_id in task_ids:
            if task_id not in carrying or task_id in delivered:
                continue
            delivered.add(task_id)
            carrying.remove(task_id)
            prefix.append(deliver_map[task_id])
            yield from dfs(prefix, picked, delivered, carrying)
            prefix.pop()
            carrying.add(task_id)
            delivered.remove(task_id)

    yield from dfs(prefix=[], picked=set(), delivered=set(), carrying=set())


def simple_block_sequences(selected_tasks: Sequence[TaskRef]) -> List[tuple[str, ...]]:
    sequences: List[tuple[str, ...]] = []
    for perm in itertools.permutations(selected_tasks):
        labels = ["START"]
        for task in perm:
            labels.append(task.pickup.label)
            labels.append(task.deliver.label)
        sequences.append(tuple(labels))
    return sequences


def consecutive_pickups_before_first_delivery(sequence: Sequence[str]) -> int:
    count = 0
    for label in sequence[1:]:
        if label.startswith("D"):
            break
        if label.startswith("P"):
            count += 1
    return count


def choose_candidates(
    start: Node,
    tasks: Sequence[TaskRef],
    evaluator: SceneMatrixEvaluator,
    random_seed: int,
    scene_name: str,
    case_id: str,
    num_orders: int,
    capacity: int,
    min_uniqueness_gap: float,
    distance_tolerance_ratio: float,
    questions_per_case: int,
) -> List[CandidateQuestion]:
    if len(tasks) < num_orders:
        raise ValueError(f"scene={scene_name} case={case_id} 任务数不足，无法选出 {num_orders} 笔订单")

    distance_cache, _ = precompute_distance_cache(evaluator, start, tasks)
    candidates: List[CandidateQuestion] = []

    for selected_tasks in itertools.combinations(tasks, num_orders):
        all_plans = [
            RoutePlan(sequence=sequence, total_distance_m=route_distance(sequence, distance_cache))
            for sequence in enumerate_feasible_sequences(selected_tasks, capacity)
        ]
        all_plans.sort(key=lambda plan: (plan.total_distance_m, plan.sequence))
        optimal_distance = all_plans[0].total_distance_m
        alternative_optimal_plans = tuple(
            plan for plan in all_plans if math.isclose(plan.total_distance_m, optimal_distance, abs_tol=1e-9)
        )
        optimal_plan = alternative_optimal_plans[0]
        max_acceptable_distance_m = optimal_distance * (1.0 + distance_tolerance_ratio)
        acceptable_plans = tuple(
            plan
            for plan in all_plans
            if plan.total_distance_m <= max_acceptable_distance_m + 1e-9
        )
        second_best_plan = next(
            (
                plan
                for plan in all_plans
                if not math.isclose(plan.total_distance_m, optimal_distance, abs_tol=1e-9)
            ),
            None,
        )

        simple_sequences = simple_block_sequences(selected_tasks)
        simple_plans = [
            RoutePlan(sequence=sequence, total_distance_m=route_distance(sequence, distance_cache))
            for sequence in simple_sequences
        ]
        simple_plans.sort(key=lambda plan: (plan.total_distance_m, plan.sequence))
        best_simple_plan = simple_plans[0]

        uses_interleaving = optimal_plan.sequence not in {plan.sequence for plan in simple_plans}
        pickup_batching_count = consecutive_pickups_before_first_delivery(optimal_plan.sequence)
        uses_pickup_batching = pickup_batching_count >= 2
        interleaving_gain_m = best_simple_plan.total_distance_m - optimal_plan.total_distance_m
        uniqueness_gap_m = (
            second_best_plan.total_distance_m - optimal_plan.total_distance_m if second_best_plan is not None else float("inf")
        )

        selected_task_signature = ",".join(f"T{task.task_index}" for task in selected_tasks)
        score = (
            1.0 if uses_interleaving else 0.0,
            1.0 if uses_pickup_batching else 0.0,
            interleaving_gain_m,
            uniqueness_gap_m if uniqueness_gap_m >= min_uniqueness_gap else -1.0,
            selected_task_signature,
        )
        candidates.append(
            CandidateQuestion(
                selected_tasks=tuple(selected_tasks),
                capacity=capacity,
                optimal_plan=optimal_plan,
                alternative_optimal_plans=alternative_optimal_plans,
                acceptable_plans=acceptable_plans,
                distance_tolerance_ratio=distance_tolerance_ratio,
                max_acceptable_distance_m=max_acceptable_distance_m,
                second_best_plan=second_best_plan,
                best_simple_plan=best_simple_plan,
                uses_interleaving=uses_interleaving,
                uses_pickup_batching=uses_pickup_batching,
                interleaving_gain_m=interleaving_gain_m,
                uniqueness_gap_m=uniqueness_gap_m,
                score=score,
            )
        )

    rng = random.Random(f"{random_seed}:{scene_name}:{case_id}:{num_orders}:{capacity}")
    rng.shuffle(candidates)
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates[:questions_per_case]


def sequence_to_text(sequence: Sequence[str]) -> str:
    return " -> ".join(sequence)


def question_text(capacity: int, current_node: str, selected_tasks: Sequence[TaskRef]) -> str:
    task_desc = []
    for idx, task in enumerate(selected_tasks):
        name = ORDER_NAME_POOL[idx] if idx < len(ORDER_NAME_POOL) else f"Task {idx + 1}"
        task_desc.append(f"{name}: pick up at {task.pickup.label}, deliver to {task.deliver.label}")
    joined = ", ".join(task_desc)
    return (
        f"You start at {current_node} with an empty load. Your carrying capacity is {capacity}. "
        f"Complete the following {len(selected_tasks)} orders while never exceeding capacity and never delivering before pickup: {joined}. "
        "Find the visiting sequence that minimizes the total walking distance. "
        "Output exactly one line containing only the node labels in visit order, joined by ' -> '."
    )


def generate_records_for_scene(
    scene_name: str,
    config: Any,
    random_seed: int,
    num_orders: int,
    capacity: int,
    min_uniqueness_gap: float,
    distance_tolerance_ratio: float,
    questions_per_case: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    evaluator = SceneMatrixEvaluator(scene=scene_name)
    records: List[Dict[str, Any]] = []
    shortfall_cases: List[Dict[str, Any]] = []

    for task_path in discover_case_files(config.task_dir):
        payload = load_json(task_path)
        normalized_payload = config.task_loader(payload)
        start, tasks = build_start_and_tasks(normalized_payload)
        if len(tasks) < num_orders:
            print(
                f"skip {scene_name}/{task_path.stem}: only {len(tasks)} task(s), "
                f"cannot build a {num_orders}-order question"
            )
            shortfall_cases.append(
                {
                    "scene": scene_name,
                    "case_id": task_path.stem,
                    "requested": questions_per_case,
                    "generated": 0,
                    "reason": f"fewer_than_{num_orders}_orders",
                }
            )
            continue
        image_path = ensure_image_path(config.image_dir, task_path.stem)
        candidates = choose_candidates(
            start=start,
            tasks=tasks,
            evaluator=evaluator,
            random_seed=random_seed,
            scene_name=scene_name,
            case_id=task_path.stem,
            num_orders=num_orders,
            capacity=capacity,
            min_uniqueness_gap=min_uniqueness_gap,
            distance_tolerance_ratio=distance_tolerance_ratio,
            questions_per_case=questions_per_case,
        )
        if len(candidates) < questions_per_case:
            shortfall_cases.append(
                {
                    "scene": scene_name,
                    "case_id": task_path.stem,
                    "requested": questions_per_case,
                    "generated": len(candidates),
                }
            )

        for variant_index, candidate in enumerate(candidates, start=1):
            selected_nodes = [start.label]
            selected_task_payload = []
            for task in candidate.selected_tasks:
                selected_nodes.extend([task.pickup.label, task.deliver.label])
                selected_task_payload.append(
                    {
                        "task_index": task.task_index,
                        "pickup": task.pickup.label,
                        "deliver": task.deliver.label,
                    }
                )

            node_coordinates = {
                start.label: {"x": round(start.x, 6), "y": round(start.y, 6)},
            }
            for task in candidate.selected_tasks:
                node_coordinates[task.pickup.label] = {"x": round(task.pickup.x, 6), "y": round(task.pickup.y, 6)}
                node_coordinates[task.deliver.label] = {"x": round(task.deliver.x, 6), "y": round(task.deliver.y, 6)}

            records.append(
                {
                    "question_id": f"{scene_name}/{task_path.stem}/capacity_interleaving_{variant_index}",
                    "scene": scene_name,
                    "case_id": task_path.stem,
                    "image_path": str(image_path),
                    "task_json_path": str(task_path),
                    "question_type": "capacity_interleaving_optimization",
                    "template_name": "Capacity & Interleaving",
                    "level": 3,
                    "output_format": "Return exactly one line containing only node labels joined by ' -> '",
                    "evaluation_metrics": ["distance_within_relative_tolerance"],
                    "question_variant_index": variant_index,
                    "capacity": capacity,
                    "current_node": start.label,
                    "num_orders": num_orders,
                    "distance_tolerance_ratio": round(candidate.distance_tolerance_ratio, 6),
                    "selected_tasks": selected_task_payload,
                    "question": question_text(capacity, start.label, candidate.selected_tasks),
                    "gt_sequence": list(candidate.optimal_plan.sequence),
                    "gt_answer": sequence_to_text(candidate.optimal_plan.sequence),
                    "gt_total_distance_m": round(candidate.optimal_plan.total_distance_m, 6),
                    "max_acceptable_distance_m": round(candidate.max_acceptable_distance_m, 6),
                    "acceptable_sequences": [list(plan.sequence) for plan in candidate.acceptable_plans],
                    "acceptable_distances_m": [
                        round(plan.total_distance_m, 6) for plan in candidate.acceptable_plans
                    ],
                    "acceptable_sequence_count": len(candidate.acceptable_plans),
                    "gt_alternative_sequences": [list(plan.sequence) for plan in candidate.alternative_optimal_plans],
                    "best_simple_sequence": list(candidate.best_simple_plan.sequence),
                    "best_simple_distance_m": round(candidate.best_simple_plan.total_distance_m, 6),
                    "second_best_sequence": list(candidate.second_best_plan.sequence) if candidate.second_best_plan else None,
                    "second_best_distance_m": (
                        round(candidate.second_best_plan.total_distance_m, 6) if candidate.second_best_plan else None
                    ),
                    "metadata": {
                        "uses_interleaving": candidate.uses_interleaving,
                        "uses_pickup_batching": candidate.uses_pickup_batching,
                        "interleaving_gain_m": round(candidate.interleaving_gain_m, 6),
                        "uniqueness_gap_m": (
                            round(candidate.uniqueness_gap_m, 6)
                            if math.isfinite(candidate.uniqueness_gap_m)
                            else None
                        ),
                        "distance_tolerance_ratio": round(candidate.distance_tolerance_ratio, 6),
                        "max_acceptable_distance_m": round(candidate.max_acceptable_distance_m, 6),
                        "acceptable_sequence_count": len(candidate.acceptable_plans),
                    },
                    "node_coordinates": node_coordinates,
                }
            )

    return records, shortfall_cases


def main() -> None:
    args = parse_args()
    if args.num_orders < 2:
        raise ValueError("--num-orders 必须至少为 2")

    capacity = args.capacity if args.capacity is not None else min(2, args.num_orders)
    if capacity < 1:
        raise ValueError("--capacity 必须至少为 1")
    if capacity > args.num_orders:
        raise ValueError("--capacity 不能大于 --num-orders")
    if args.questions_per_case < 1:
        raise ValueError("--questions-per-case 必须至少为 1")

    scene_configs = select_qa_scene_configs(load_qa_scene_configs(), args.scenes)
    scene_names = list(scene_configs.keys())

    all_records: List[Dict[str, Any]] = []
    question_counts_by_scene: Dict[str, int] = {}
    shortfall_cases: List[Dict[str, Any]] = []

    for scene_name in scene_names:
        records, scene_shortfalls = generate_records_for_scene(
            scene_name=scene_name,
            config=scene_configs[scene_name],
            random_seed=args.random_seed,
            num_orders=args.num_orders,
            capacity=capacity,
            min_uniqueness_gap=args.min_uniqueness_gap,
            distance_tolerance_ratio=args.distance_tolerance_ratio,
            questions_per_case=args.questions_per_case,
        )
        all_records.extend(records)
        question_counts_by_scene[scene_name] = len(records)
        shortfall_cases.extend(scene_shortfalls)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question_family": "Logical Planning Layer / Capacity & Interleaving",
        "template_count_per_image": args.questions_per_case,
        "template_selection_policy": "top_interleaving_candidates_per_image",
        "scene_source": "active_simworld_scene_configs",
        "random_seed": args.random_seed,
        "num_orders": args.num_orders,
        "capacity": capacity,
        "questions_per_case": args.questions_per_case,
        "distance_tolerance_ratio": args.distance_tolerance_ratio,
        "default_evaluation_metrics": ["distance_within_relative_tolerance"],
        "question_counts_by_scene": question_counts_by_scene,
        "case_counts_by_scene": {
            scene_name: len(discover_case_files(scene_configs[scene_name].task_dir))
            for scene_name in scene_names
        },
        "shortfall_case_count": len(shortfall_cases),
        "shortfall_cases": shortfall_cases,
        "total_questions": len(all_records),
        "records": all_records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(all_records)} questions to {args.output}")


if __name__ == "__main__":
    main()
