#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from scene_registry import load_qa_scene_configs, select_qa_scene_configs


THIS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Node:
    label: str
    x: float
    y: float

    @property
    def category(self) -> str:
        if self.label == "START":
            return "START"
        return self.label[0]


@dataclass(frozen=True)
class QuestionTemplate:
    template_id: str
    axis_key: str
    axis_en: str
    order_en: str
    require_start: bool


TEMPLATES: Sequence[QuestionTemplate] = (
    QuestionTemplate(
        template_id="y_desc",
        axis_key="y",
        axis_en="absolute vertical position (Y-axis)",
        order_en="from top to bottom",
        require_start=True,
    ),
    QuestionTemplate(
        template_id="y_asc",
        axis_key="y",
        axis_en="absolute vertical position (Y-axis)",
        order_en="from bottom to top",
        require_start=False,
    ),
    QuestionTemplate(
        template_id="x_asc",
        axis_key="x",
        axis_en="absolute horizontal position (X-axis)",
        order_en="from left to right",
        require_start=True,
    ),
    QuestionTemplate(
        template_id="x_desc",
        axis_key="x",
        axis_en="absolute horizontal position (X-axis)",
        order_en="from right to left",
        require_start=False,
    ),
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate absolute-axis-sorting VQA questions and GT.")
    parser.add_argument(
        "--output",
        type=Path,
        default=THIS_DIR.parent / "outputs" / "generated" / "vqa" / "absolute_axis_sorting_vqa.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=5,
        help="How many nodes to include in each question; default randomly samples 5 nodes",
    )
    parser.add_argument(
        "--min-axis-gap",
        type=float,
        default=1.0,
        help="Minimum gap on the queried axis to avoid ambiguous GT",
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
        help="Seed used to pick one template per image",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_points_file(path: Path | None) -> List[tuple[float, float]]:
    if path is None or not path.exists():
        return []

    points: List[tuple[float, float]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not (line.startswith("(") and line.endswith(")")):
            continue
        x_str, y_str = line[1:-1].split(",")
        points.append((float(x_str), float(y_str)))
    return points


def matches_point_pool(node: Node, point_pool: Sequence[tuple[float, float]], eps: float = 1e-6) -> bool:
    for px, py in point_pool:
        if abs(node.x - px) <= eps and abs(node.y - py) <= eps:
            return True
    return False


def build_nodes(task_payload: Dict[str, Any]) -> List[Node]:
    start = task_payload.get("pos", {})
    nodes = [Node(label="START", x=float(start["x"]), y=float(start["y"]))]
    for idx, task in enumerate(task_payload.get("tasks", []), start=1):
        pickup = task["pickup"]
        deliver = task["deliver"]
        nodes.append(Node(label=f"P{idx}", x=float(pickup["x"]), y=float(pickup["y"])))
        nodes.append(Node(label=f"D{idx}", x=float(deliver["x"]), y=float(deliver["y"])))
    return nodes


def axis_value(node: Node, axis_key: str) -> float:
    return node.y if axis_key == "y" else node.x


def other_axis_value(node: Node, axis_key: str) -> float:
    return node.x if axis_key == "y" else node.y


def diversity_score(nodes: Iterable[Node]) -> int:
    return len({node.category for node in nodes})


def question_text(template: QuestionTemplate, nodes: Sequence[Node]) -> str:
    labels = ", ".join(node.label for node in nodes)
    return (
        f"Use only the map image and compare the nodes by their {template.axis_en}. "
        f"Order {labels} {template.order_en}. "
        "Ignore walls, corridors, connectivity, and any numbering pattern in the labels. "
        "Output exactly one line containing only the ordered node labels joined by ' -> '."
    )


def gt_order(nodes: Sequence[Node], axis_key: str, descending: bool) -> List[Node]:
    return sorted(
        nodes,
        key=lambda node: (
            axis_value(node, axis_key),
            other_axis_value(node, axis_key),
            node.label,
        ),
        reverse=descending,
    )


def order_descending_for_scene(template: QuestionTemplate, config: Any) -> bool:
    if template.axis_key == "x":
        x_increases_to_right = bool(getattr(config, "image_x_increases_to_right", True))
        return not x_increases_to_right if template.template_id == "x_asc" else x_increases_to_right

    y_increases_upward = bool(getattr(config, "image_y_increases_upward", True))
    return y_increases_upward if template.template_id == "y_desc" else not y_increases_upward


def subset_quality(nodes: Sequence[Node], axis_key: str) -> tuple[float, float, int, str]:
    ordered_values = sorted(axis_value(node, axis_key) for node in nodes)
    min_gap = min(
        ordered_values[idx + 1] - ordered_values[idx]
        for idx in range(len(ordered_values) - 1)
    )
    span = ordered_values[-1] - ordered_values[0]
    diversity = diversity_score(nodes)
    signature = ",".join(sorted(node.label for node in nodes))
    return (min_gap, span, diversity, signature)


def select_subset(
    nodes: Sequence[Node],
    axis_key: str,
    subset_size: int,
    min_axis_gap: float,
    require_start: bool,
    rng: random.Random,
    max_attempts: int = 128,
) -> tuple[List[Node], bool, float]:
    if subset_size >= len(nodes):
        subset = list(nodes)
        actual_min_gap = subset_quality(subset, axis_key)[0]
        return subset, actual_min_gap < min_axis_gap, actual_min_gap

    required_nodes = [node for node in nodes if require_start and node.label == "START"]
    optional_nodes = [node for node in nodes if node.label != "START" or not require_start]
    sample_size = subset_size - len(required_nodes)
    if sample_size < 0 or sample_size > len(optional_nodes):
        raise ValueError(
            f"无法采样节点集合: axis={axis_key}, subset_size={subset_size}, require_start={require_start}"
        )

    best_subset: List[Node] | None = None
    best_quality: tuple[float, float, int, str] | None = None
    strict_subset: List[Node] | None = None
    strict_quality: tuple[float, float, int, str] | None = None

    for _ in range(max_attempts):
        sampled_optional = rng.sample(optional_nodes, sample_size)
        subset = required_nodes + sampled_optional
        quality = subset_quality(subset, axis_key)
        min_gap = quality[0]

        if best_quality is None or quality > best_quality:
            best_subset = list(subset)
            best_quality = quality

        if min_gap >= min_axis_gap and (strict_quality is None or quality > strict_quality):
            strict_subset = list(subset)
            strict_quality = quality

    if strict_subset is not None:
        return strict_subset, False, strict_quality[0]
    if best_subset is not None and best_quality is not None:
        return best_subset, True, best_quality[0]
    raise ValueError(f"找不到可用节点集合: axis={axis_key}, require_start={require_start}")


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


def render_answer(labels: Sequence[str]) -> str:
    return " -> ".join(labels)


def choose_template_for_case(
    scene_name: str,
    case_id: str,
    random_seed: int,
) -> QuestionTemplate:
    rng = random.Random(f"{random_seed}:{scene_name}:{case_id}")
    return rng.choice(list(TEMPLATES))


def make_case_rng(random_seed: int, scene_name: str, case_id: str, template_id: str) -> random.Random:
    return random.Random(f"{random_seed}:{scene_name}:{case_id}:{template_id}")


def generate_records_for_scene(
    scene_name: str,
    config: Any,
    subset_size: int,
    min_axis_gap: float,
    random_seed: int,
) -> List[Dict[str, Any]]:
    task_files = discover_case_files(config.task_dir)
    point_pool = parse_points_file(getattr(config, "points_file", None))
    records: List[Dict[str, Any]] = []

    for task_path in task_files:
        payload = load_json(task_path)
        normalized_payload = config.task_loader(payload)
        nodes = build_nodes(normalized_payload)
        if len(nodes) < 2:
            print(f"skip {scene_name}/{task_path.stem}: only {len(nodes)} node available")
            continue
        image_path = ensure_image_path(config.image_dir, task_path.stem)
        effective_subset_size = min(subset_size, len(nodes))
        template = choose_template_for_case(scene_name, task_path.stem, random_seed)
        case_rng = make_case_rng(random_seed, scene_name, task_path.stem, template.template_id)

        if effective_subset_size >= len(nodes):
            subset = list(nodes)
            gap_relaxed = False
            actual_min_gap = subset_quality(subset, template.axis_key)[0]
        else:
            subset, gap_relaxed, actual_min_gap = select_subset(
                nodes=nodes,
                axis_key=template.axis_key,
                subset_size=effective_subset_size,
                min_axis_gap=min_axis_gap,
                require_start=template.require_start,
                rng=case_rng,
            )

        ordered_nodes = gt_order(
            subset,
            template.axis_key,
            order_descending_for_scene(template, config),
        )
        subset_labels = [node.label for node in subset]
        answer_labels = [node.label for node in ordered_nodes]
        axis_values = {
            node.label: round(axis_value(node, template.axis_key), 6) for node in subset
        }
        coordinates = {
            node.label: {"x": round(node.x, 6), "y": round(node.y, 6)} for node in subset
        }

        records.append(
            {
                "question_id": f"{scene_name}/{task_path.stem}/{template.template_id}",
                "scene": scene_name,
                "case_id": task_path.stem,
                "image_path": str(image_path),
                "task_json_path": str(task_path),
                "points_file_path": str(getattr(config, "points_file", "")) if getattr(config, "points_file", None) else None,
                "question_type": "absolute_axis_sorting",
                "template_name": "Absolute Axis Sorting",
                "template_id": template.template_id,
                "level": 1,
                "axis": template.axis_key.upper(),
                "axis_description": template.axis_en,
                "order_description": template.order_en,
                "output_format": "Return exactly one line containing only node labels joined by ' -> '",
                "evaluation_metrics": [
                    "exact_match",
                    "lcs_length",
                    "lcs_precision",
                    "lcs_recall",
                    "lcs_f1",
                ],
                "scene_image_axis_convention": {
                    "x_increases_to_right": bool(getattr(config, "image_x_increases_to_right", True)),
                    "y_increases_upward": bool(getattr(config, "image_y_increases_upward", True)),
                },
                "node_list": subset_labels,
                "uses_all_nodes": effective_subset_size >= len(nodes),
                "subset_sampling_policy": "random_sample",
                "requested_min_axis_gap": min_axis_gap,
                "actual_min_axis_gap": round(actual_min_gap, 6),
                "min_axis_gap_relaxed": gap_relaxed,
                "question": question_text(template, subset),
                "gt_order": answer_labels,
                "gt_answer": render_answer(answer_labels),
                "node_coordinates": coordinates,
                "queried_axis_values": axis_values,
                "points_file_validation": {
                    node.label: matches_point_pool(node, point_pool) if point_pool else None
                    for node in subset
                },
            }
        )

    return records


def main() -> None:
    args = parse_args()
    scene_configs = select_qa_scene_configs(load_qa_scene_configs(), args.scenes)

    if args.subset_size is not None and args.subset_size < 2:
        raise ValueError("--subset-size 必须至少为 2")
    scene_names = list(scene_configs.keys())

    all_records: List[Dict[str, Any]] = []
    scene_counts: Dict[str, int] = {}

    for scene_name in scene_names:
        records = generate_records_for_scene(
            scene_name=scene_name,
            config=scene_configs[scene_name],
            subset_size=args.subset_size,
            min_axis_gap=args.min_axis_gap,
            random_seed=args.random_seed,
        )
        all_records.extend(records)
        scene_counts[scene_name] = len(records)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question_family": "Perception Layer / Absolute Axis Sorting",
        "template_count_per_image": 1,
        "template_selection_policy": "random_one_of_four",
        "scene_source": "active_simworld_scene_configs",
        "random_seed": args.random_seed,
        "subset_size": args.subset_size,
        "default_subset_policy": "random_subset",
        "min_axis_gap": args.min_axis_gap,
        "default_evaluation_metrics": [
            "exact_match",
            "lcs_length",
            "lcs_precision",
            "lcs_recall",
            "lcs_f1",
        ],
        "scene_image_axis_convention_by_scene": {
            scene_name: {
                "x_increases_to_right": bool(getattr(scene_configs[scene_name], "image_x_increases_to_right", True)),
                "y_increases_upward": bool(getattr(scene_configs[scene_name], "image_y_increases_upward", True)),
            }
            for scene_name in scene_names
        },
        "question_counts_by_scene": scene_counts,
        "case_counts_by_scene": {
            scene_name: len(discover_case_files(scene_configs[scene_name].task_dir))
            for scene_name in scene_names
        },
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
