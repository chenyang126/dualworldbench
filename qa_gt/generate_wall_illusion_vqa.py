#!/usr/bin/env python3

from __future__ import annotations

import argparse
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


@dataclass(frozen=True)
class Node:
    label: str
    x: float
    y: float

    def as_coord(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y}


@dataclass(frozen=True)
class CandidateQuestion:
    anchor: Node
    target_b: Node
    target_c: Node
    euclidean_ab: float
    euclidean_ac: float
    topo_ab: float
    topo_ac: float
    longer_target: str
    inversion: bool
    score: tuple[float, float, float, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate topology-aware wall-illusion VQA questions and GT.")
    parser.add_argument(
        "--output",
        type=Path,
        default=THIS_DIR.parent / "outputs" / "generated" / "vqa" / "wall_illusion_vqa.json",
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
        help="Seed used to break ties and randomize target order",
    )
    parser.add_argument(
        "--min-topo-gap",
        type=float,
        default=1.0,
        help="Minimum topological distance gap (meters) between the two compared routes",
    )
    parser.add_argument(
        "--min-euclid-gap",
        type=float,
        default=0.5,
        help="Minimum Euclidean distance gap (meters) for a strong visual illusion candidate",
    )
    parser.add_argument(
        "--questions-per-case",
        type=int,
        default=3,
        help="How many distinct wall-illusion candidates to keep per case",
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


def build_nodes(task_payload: Dict[str, Any]) -> List[Node]:
    start = task_payload.get("pos", {})
    nodes = [Node(label="START", x=float(start["x"]), y=float(start["y"]))]
    for idx, task in enumerate(task_payload.get("tasks", []), start=1):
        pickup = task["pickup"]
        deliver = task["deliver"]
        nodes.append(Node(label=f"P{idx}", x=float(pickup["x"]), y=float(pickup["y"])))
        nodes.append(Node(label=f"D{idx}", x=float(deliver["x"]), y=float(deliver["y"])))
    return nodes


def euclidean_distance(a: Node, b: Node) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def route_info(evaluator: SceneMatrixEvaluator, src: Node, dst: Node) -> Dict[str, Any]:
    return evaluator.distance_between(src.as_coord(), dst.as_coord())


def compare_pair(
    evaluator: SceneMatrixEvaluator,
    anchor: Node,
    b: Node,
    c: Node,
    min_topo_gap: float,
    min_euclid_gap: float,
) -> CandidateQuestion | None:
    if len({anchor.label, b.label, c.label}) < 3:
        return None

    euclidean_ab = euclidean_distance(anchor, b)
    euclidean_ac = euclidean_distance(anchor, c)
    topo_ab = float(route_info(evaluator, anchor, b)["total_distance_m"])
    topo_ac = float(route_info(evaluator, anchor, c)["total_distance_m"])

    topo_diff = topo_ab - topo_ac
    euclid_diff = euclidean_ab - euclidean_ac
    abs_topo_gap = abs(topo_diff)
    abs_euclid_gap = abs(euclid_diff)

    if abs_topo_gap < min_topo_gap:
        return None

    inversion = topo_diff * euclid_diff < 0 and abs_euclid_gap >= min_euclid_gap
    if topo_ab == topo_ac:
        return None

    longer_target = b.label if topo_ab > topo_ac else c.label
    shorter_visual_target = b.label if euclidean_ab < euclidean_ac else c.label
    score = (
        1.0 if inversion else 0.0,
        abs_topo_gap,
        abs_euclid_gap,
        f"{anchor.label}|{b.label}|{c.label}|{longer_target}|{shorter_visual_target}",
    )

    return CandidateQuestion(
        anchor=anchor,
        target_b=b,
        target_c=c,
        euclidean_ab=euclidean_ab,
        euclidean_ac=euclidean_ac,
        topo_ab=topo_ab,
        topo_ac=topo_ac,
        longer_target=longer_target,
        inversion=inversion,
        score=score,
    )


def iter_candidates(
    evaluator: SceneMatrixEvaluator,
    nodes: Sequence[Node],
    min_topo_gap: float,
    min_euclid_gap: float,
) -> Iterable[CandidateQuestion]:
    for anchor in nodes:
        others = [node for node in nodes if node.label != anchor.label]
        for idx, b in enumerate(others):
            for c in others[idx + 1 :]:
                candidate = compare_pair(
                    evaluator=evaluator,
                    anchor=anchor,
                    b=b,
                    c=c,
                    min_topo_gap=min_topo_gap,
                    min_euclid_gap=min_euclid_gap,
                )
                if candidate is not None:
                    yield candidate


def choose_candidates(
    evaluator: SceneMatrixEvaluator,
    nodes: Sequence[Node],
    random_seed: int,
    scene_name: str,
    case_id: str,
    min_topo_gap: float,
    min_euclid_gap: float,
    questions_per_case: int,
) -> List[CandidateQuestion]:
    candidates = list(iter_candidates(evaluator, nodes, min_topo_gap, min_euclid_gap))
    if not candidates:
        raise ValueError(
            f"scene={scene_name} case={case_id} 未找到满足最小拓扑差阈值的可用比较题"
        )

    rng = random.Random(f"{random_seed}:{scene_name}:{case_id}")
    rng.shuffle(candidates)
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates[:questions_per_case]


def maybe_swap_targets(
    candidate: CandidateQuestion,
    random_seed: int,
    scene_name: str,
    case_id: str,
    variant_index: int,
) -> tuple[Node, Node, float, float]:
    rng = random.Random(f"swap:{random_seed}:{scene_name}:{case_id}:{variant_index}")
    if rng.random() < 0.5:
        return candidate.target_b, candidate.target_c, candidate.topo_ab, candidate.topo_ac
    return candidate.target_c, candidate.target_b, candidate.topo_ac, candidate.topo_ab


def question_text(anchor: str, b: str, c: str) -> str:
    return (
        f"Considering only valid walking routes that do not pass through black walls, which path is longer: "
        f"{anchor} -> {b} or {anchor} -> {c}? "
        f"Output exactly one line in the form '{anchor} -> X is longer', where X is either {b} or {c}."
    )


def gt_answer(anchor: str, longer_target: str) -> str:
    return f"{anchor} -> {longer_target} is longer"


def generate_records_for_scene(
    scene_name: str,
    config: Any,
    random_seed: int,
    min_topo_gap: float,
    min_euclid_gap: float,
    questions_per_case: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    evaluator = SceneMatrixEvaluator(scene=scene_name)
    records: List[Dict[str, Any]] = []
    shortfall_cases: List[Dict[str, Any]] = []

    for task_path in discover_case_files(config.task_dir):
        payload = load_json(task_path)
        normalized_payload = config.task_loader(payload)
        nodes = build_nodes(normalized_payload)
        if len(nodes) < 3:
            print(f"skip {scene_name}/{task_path.stem}: only {len(nodes)} node available")
            shortfall_cases.append(
                {
                    "scene": scene_name,
                    "case_id": task_path.stem,
                    "requested": questions_per_case,
                    "generated": 0,
                    "reason": "fewer_than_three_nodes",
                }
            )
            continue
        image_path = ensure_image_path(config.image_dir, task_path.stem)
        try:
            candidates = choose_candidates(
                evaluator=evaluator,
                nodes=nodes,
                random_seed=random_seed,
                scene_name=scene_name,
                case_id=task_path.stem,
                min_topo_gap=min_topo_gap,
                min_euclid_gap=min_euclid_gap,
                questions_per_case=questions_per_case,
            )
        except ValueError as exc:
            print(f"skip {scene_name}/{task_path.stem}: {exc}")
            shortfall_cases.append(
                {
                    "scene": scene_name,
                    "case_id": task_path.stem,
                    "requested": questions_per_case,
                    "generated": 0,
                    "reason": str(exc),
                }
            )
            continue
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
            first_target, second_target, first_topo, second_topo = maybe_swap_targets(
                candidate=candidate,
                random_seed=random_seed,
                scene_name=scene_name,
                case_id=task_path.stem,
                variant_index=variant_index,
            )
            longer_target = first_target.label if first_topo > second_topo else second_target.label

            records.append(
                {
                    "question_id": f"{scene_name}/{task_path.stem}/wall_illusion_{variant_index}",
                    "scene": scene_name,
                    "case_id": task_path.stem,
                    "image_path": str(image_path),
                    "task_json_path": str(task_path),
                    "question_type": "visual_distance_vs_topological_distance",
                    "template_name": "Wall Illusion Test",
                    "level": 2,
                    "output_format": f"Return exactly one line in the form '{candidate.anchor.label} -> X is longer'",
                    "evaluation_metrics": ["exact_match"],
                    "question_variant_index": variant_index,
                    "question": question_text(candidate.anchor.label, first_target.label, second_target.label),
                    "anchor_node": candidate.anchor.label,
                    "candidate_nodes": [first_target.label, second_target.label],
                    "gt_answer": gt_answer(candidate.anchor.label, longer_target),
                    "gt_longer_route": f"{candidate.anchor.label} -> {longer_target}",
                    "gt_shorter_route": (
                        f"{candidate.anchor.label} -> {second_target.label}"
                        if longer_target == first_target.label
                        else f"{candidate.anchor.label} -> {first_target.label}"
                    ),
                    "distances_m": {
                        f"{candidate.anchor.label}->{first_target.label}": round(first_topo, 6),
                        f"{candidate.anchor.label}->{second_target.label}": round(second_topo, 6),
                    },
                    "visual_distances_m": {
                        f"{candidate.anchor.label}->{first_target.label}": round(
                            euclidean_distance(candidate.anchor, first_target), 6
                        ),
                        f"{candidate.anchor.label}->{second_target.label}": round(
                            euclidean_distance(candidate.anchor, second_target), 6
                        ),
                    },
                    "illusion_metadata": {
                        "is_visual_topology_inversion": candidate.inversion,
                        "selected_by": "top_questions_per_case_by_inversion_then_gap",
                        "requested_min_topo_gap": min_topo_gap,
                        "requested_min_euclid_gap": min_euclid_gap,
                        "actual_topo_gap": round(abs(first_topo - second_topo), 6),
                        "actual_visual_gap": round(
                            abs(
                                euclidean_distance(candidate.anchor, first_target)
                                - euclidean_distance(candidate.anchor, second_target)
                            ),
                            6,
                        ),
                    },
                    "node_coordinates": {
                        candidate.anchor.label: {"x": round(candidate.anchor.x, 6), "y": round(candidate.anchor.y, 6)},
                        first_target.label: {"x": round(first_target.x, 6), "y": round(first_target.y, 6)},
                        second_target.label: {"x": round(second_target.x, 6), "y": round(second_target.y, 6)},
                    },
                    "matrix_points": {
                        route_key: {
                            "src_label": route_value["src_label"],
                            "dst_label": route_value["dst_label"],
                            "src_offset_m": round(float(route_value["src_offset_m"]), 6),
                            "dst_offset_m": round(float(route_value["dst_offset_m"]), 6),
                            "graph_distance_m": round(float(route_value["graph_distance_m"]), 6),
                            "total_distance_m": round(float(route_value["total_distance_m"]), 6),
                        }
                        for route_key, route_value in {
                            f"{candidate.anchor.label}->{first_target.label}": route_info(evaluator, candidate.anchor, first_target),
                            f"{candidate.anchor.label}->{second_target.label}": route_info(evaluator, candidate.anchor, second_target),
                        }.items()
                    },
                }
            )

    return records, shortfall_cases


def main() -> None:
    args = parse_args()
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
            min_topo_gap=args.min_topo_gap,
            min_euclid_gap=args.min_euclid_gap,
            questions_per_case=args.questions_per_case,
        )
        all_records.extend(records)
        question_counts_by_scene[scene_name] = len(records)
        shortfall_cases.extend(scene_shortfalls)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question_family": "Spatial Topology Layer / Visual Distance vs. Topological Distance",
        "template_count_per_image": args.questions_per_case,
        "template_selection_policy": "top_wall_illusion_candidates_per_image",
        "scene_source": "active_simworld_scene_configs",
        "random_seed": args.random_seed,
        "min_topo_gap": args.min_topo_gap,
        "min_euclid_gap": args.min_euclid_gap,
        "questions_per_case": args.questions_per_case,
        "default_evaluation_metrics": ["exact_match"],
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
