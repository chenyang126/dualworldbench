#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import json
import math
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
class AnchorQuestion:
    anchor: Node
    candidates: tuple[Node, ...]
    extreme: str
    answer: Node
    distance_gap: float
    min_distance_gap_relaxed: bool


@dataclass(frozen=True)
class PairQuestion:
    nodes: tuple[Node, ...]
    extreme: str
    answer_pair: tuple[Node, Node]
    distance_gap: float
    min_distance_gap_relaxed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Euclidean-distance perception VQA questions and GT.")
    parser.add_argument(
        "--output",
        type=Path,
        default=THIS_DIR.parent / "outputs" / "generated" / "vqa" / "euclidean_distance_vqa.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=5,
        help="How many candidate nodes to include in each question",
    )
    parser.add_argument(
        "--min-distance-gap",
        type=float,
        default=1.0,
        help="Preferred minimum distance gap between the winner and runner-up",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Optional subset of scenes, e.g. indoor l1_corner_hall",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed used for deterministic template choices and sampling",
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
        raise FileNotFoundError(f"Cannot find image: {image_path}")
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


def diversity_score(nodes: Iterable[Node]) -> int:
    return len({node.category for node in nodes})


def choose_extreme(random_seed: int, scene_name: str, case_id: str, template_id: str, options: Sequence[str]) -> str:
    rng = random.Random(f"{random_seed}:{scene_name}:{case_id}:{template_id}")
    return rng.choice(list(options))


def extreme_node(
    anchor: Node,
    candidates: Sequence[Node],
    extreme: str,
) -> tuple[Node, float]:
    ranked = sorted(
        candidates,
        key=lambda node: (euclidean_distance(anchor, node), node.label),
        reverse=extreme == "farthest",
    )
    if len(ranked) < 2:
        return ranked[0], float("inf")
    best_distance = euclidean_distance(anchor, ranked[0])
    runner_up_distance = euclidean_distance(anchor, ranked[1])
    return ranked[0], abs(best_distance - runner_up_distance)


def pair_distances(nodes: Sequence[Node]) -> List[tuple[tuple[Node, Node], float]]:
    pairs = []
    for a, b in itertools.combinations(nodes, 2):
        pairs.append(((a, b), euclidean_distance(a, b)))
    return pairs


def extreme_pair(nodes: Sequence[Node], extreme: str) -> tuple[tuple[Node, Node], float]:
    ranked = sorted(
        pair_distances(nodes),
        key=lambda item: (item[1], item[0][0].label, item[0][1].label),
        reverse=extreme == "longest",
    )
    if len(ranked) < 2:
        return ranked[0][0], float("inf")
    return ranked[0][0], abs(ranked[0][1] - ranked[1][1])


def select_anchor_question(
    nodes: Sequence[Node],
    subset_size: int,
    min_distance_gap: float,
    extreme: str,
    rng: random.Random,
    max_attempts: int = 256,
) -> AnchorQuestion:
    if len(nodes) < 3:
        raise ValueError("Need at least 3 nodes to build an anchor distance question")

    effective_subset_size = min(subset_size, len(nodes) - 1)
    best_question: AnchorQuestion | None = None
    best_quality: tuple[float, float, int, str] | None = None
    strict_question: AnchorQuestion | None = None
    strict_quality: tuple[float, float, int, str] | None = None

    for _ in range(max_attempts):
        anchor = rng.choice(list(nodes))
        optional_nodes = [node for node in nodes if node.label != anchor.label]
        candidates = tuple(rng.sample(optional_nodes, effective_subset_size))
        answer, gap = extreme_node(anchor, candidates, extreme)
        distances = [euclidean_distance(anchor, node) for node in candidates]
        quality = (
            gap,
            max(distances) - min(distances),
            diversity_score((anchor, *candidates)),
            f"{anchor.label}|{','.join(sorted(node.label for node in candidates))}",
        )
        question = AnchorQuestion(
            anchor=anchor,
            candidates=candidates,
            extreme=extreme,
            answer=answer,
            distance_gap=gap,
            min_distance_gap_relaxed=gap < min_distance_gap,
        )
        if best_quality is None or quality > best_quality:
            best_question = question
            best_quality = quality
        if gap >= min_distance_gap and (strict_quality is None or quality > strict_quality):
            strict_question = question
            strict_quality = quality

    if strict_question is not None:
        return strict_question
    if best_question is not None:
        return best_question
    raise ValueError("Cannot sample a usable anchor distance question")


def select_pair_question(
    nodes: Sequence[Node],
    subset_size: int,
    min_distance_gap: float,
    extreme: str,
    rng: random.Random,
    max_attempts: int = 256,
) -> PairQuestion:
    if len(nodes) < 3:
        raise ValueError("Need at least 3 nodes to build a pair distance question")

    effective_subset_size = min(subset_size, len(nodes))
    best_question: PairQuestion | None = None
    best_quality: tuple[float, float, int, str] | None = None
    strict_question: PairQuestion | None = None
    strict_quality: tuple[float, float, int, str] | None = None

    for _ in range(max_attempts):
        subset = tuple(rng.sample(list(nodes), effective_subset_size))
        answer_pair, gap = extreme_pair(subset, extreme)
        distances = [distance for _, distance in pair_distances(subset)]
        quality = (
            gap,
            max(distances) - min(distances),
            diversity_score(subset),
            ",".join(sorted(node.label for node in subset)),
        )
        question = PairQuestion(
            nodes=subset,
            extreme=extreme,
            answer_pair=answer_pair,
            distance_gap=gap,
            min_distance_gap_relaxed=gap < min_distance_gap,
        )
        if best_quality is None or quality > best_quality:
            best_question = question
            best_quality = quality
        if gap >= min_distance_gap and (strict_quality is None or quality > strict_quality):
            strict_question = question
            strict_quality = quality

    if strict_question is not None:
        return strict_question
    if best_question is not None:
        return best_question
    raise ValueError("Cannot sample a usable pair distance question")


def anchor_question_text(question: AnchorQuestion) -> str:
    labels = ", ".join(node.label for node in question.candidates)
    relation = "closest to" if question.extreme == "closest" else "farthest from"
    return (
        "Use only the map image and compare straight-line Euclidean distances. "
        f"Among {labels}, which node is {relation} {question.anchor.label}? "
        "Ignore walls, corridors, connectivity, and any numbering pattern in the labels. "
        "Output exactly one line containing only the node label."
    )


def pair_question_text(question: PairQuestion) -> str:
    labels = ", ".join(node.label for node in question.nodes)
    relation = "shortest" if question.extreme == "shortest" else "longest"
    return (
        "Use only the map image and compare straight-line Euclidean distances. "
        f"Among {labels}, which pair of nodes has the {relation} Euclidean distance? "
        "Ignore walls, corridors, connectivity, and any numbering pattern in the labels. "
        "Output exactly one line containing only the two node labels joined by ' -> '."
    )


def node_coordinates(nodes: Iterable[Node]) -> Dict[str, Dict[str, float]]:
    return {
        node.label: {"x": round(node.x, 6), "y": round(node.y, 6)}
        for node in nodes
    }


def generate_records_for_scene(
    scene_name: str,
    config: Any,
    subset_size: int,
    min_distance_gap: float,
    random_seed: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for task_path in discover_case_files(config.task_dir):
        payload = load_json(task_path)
        normalized_payload = config.task_loader(payload)
        nodes = build_nodes(normalized_payload)
        if len(nodes) < 3:
            print(f"skip {scene_name}/{task_path.stem}: only {len(nodes)} node(s) available")
            continue
        image_path = ensure_image_path(config.image_dir, task_path.stem)

        anchor_extreme = choose_extreme(
            random_seed=random_seed,
            scene_name=scene_name,
            case_id=task_path.stem,
            template_id="anchor_euclidean_choice",
            options=("closest", "farthest"),
        )
        anchor_rng = random.Random(f"{random_seed}:{scene_name}:{task_path.stem}:anchor_euclidean_choice")
        anchor_question = select_anchor_question(
            nodes=nodes,
            subset_size=subset_size,
            min_distance_gap=min_distance_gap,
            extreme=anchor_extreme,
            rng=anchor_rng,
        )
        anchor_nodes = (anchor_question.anchor, *anchor_question.candidates)
        anchor_distances = {
            f"{anchor_question.anchor.label}->{node.label}": round(euclidean_distance(anchor_question.anchor, node), 6)
            for node in anchor_question.candidates
        }
        records.append(
            {
                "question_id": f"{scene_name}/{task_path.stem}/anchor_euclidean_choice",
                "scene": scene_name,
                "case_id": task_path.stem,
                "image_path": str(image_path),
                "task_json_path": str(task_path),
                "question_type": "euclidean_anchor_extreme_choice",
                "template_name": "Anchor Euclidean Choice",
                "template_id": "anchor_euclidean_choice",
                "level": 1,
                "output_format": "Return exactly one line containing only one node label",
                "evaluation_metrics": ["exact_match"],
                "distance_extreme": anchor_question.extreme,
                "anchor_node": anchor_question.anchor.label,
                "candidate_nodes": [node.label for node in anchor_question.candidates],
                "question": anchor_question_text(anchor_question),
                "gt_answer_node": anchor_question.answer.label,
                "gt_answer": anchor_question.answer.label,
                "candidate_distances_m": anchor_distances,
                "actual_distance_gap": round(anchor_question.distance_gap, 6),
                "requested_min_distance_gap": min_distance_gap,
                "min_distance_gap_relaxed": anchor_question.min_distance_gap_relaxed,
                "node_coordinates": node_coordinates(anchor_nodes),
            }
        )

        pair_extreme = choose_extreme(
            random_seed=random_seed,
            scene_name=scene_name,
            case_id=task_path.stem,
            template_id="euclidean_pair_extreme",
            options=("shortest", "longest"),
        )
        pair_rng = random.Random(f"{random_seed}:{scene_name}:{task_path.stem}:euclidean_pair_extreme")
        pair_question = select_pair_question(
            nodes=nodes,
            subset_size=subset_size,
            min_distance_gap=min_distance_gap,
            extreme=pair_extreme,
            rng=pair_rng,
        )
        pair_a, pair_b = pair_question.answer_pair
        pair_distance_payload = {
            f"{a.label}->{b.label}": round(distance, 6)
            for (a, b), distance in pair_distances(pair_question.nodes)
        }
        records.append(
            {
                "question_id": f"{scene_name}/{task_path.stem}/euclidean_pair_extreme",
                "scene": scene_name,
                "case_id": task_path.stem,
                "image_path": str(image_path),
                "task_json_path": str(task_path),
                "question_type": "euclidean_pair_extreme_choice",
                "template_name": "Euclidean Pair Extreme",
                "template_id": "euclidean_pair_extreme",
                "level": 1,
                "output_format": "Return exactly one line containing only two node labels joined by ' -> '",
                "evaluation_metrics": ["exact_match"],
                "distance_extreme": pair_question.extreme,
                "node_list": [node.label for node in pair_question.nodes],
                "question": pair_question_text(pair_question),
                "gt_pair": [pair_a.label, pair_b.label],
                "gt_answer": f"{pair_a.label} -> {pair_b.label}",
                "pair_distances_m": pair_distance_payload,
                "actual_distance_gap": round(pair_question.distance_gap, 6),
                "requested_min_distance_gap": min_distance_gap,
                "min_distance_gap_relaxed": pair_question.min_distance_gap_relaxed,
                "node_coordinates": node_coordinates(pair_question.nodes),
            }
        )

    return records


def main() -> None:
    args = parse_args()
    if args.subset_size < 2:
        raise ValueError("--subset-size must be at least 2")

    scene_configs = select_qa_scene_configs(load_qa_scene_configs(), args.scenes)
    scene_names = list(scene_configs.keys())

    all_records: List[Dict[str, Any]] = []
    question_counts_by_scene: Dict[str, int] = {}

    for scene_name in scene_names:
        records = generate_records_for_scene(
            scene_name=scene_name,
            config=scene_configs[scene_name],
            subset_size=args.subset_size,
            min_distance_gap=args.min_distance_gap,
            random_seed=args.random_seed,
        )
        all_records.extend(records)
        question_counts_by_scene[scene_name] = len(records)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question_family": "Perception Layer / Euclidean Distance",
        "template_count_per_image": 2,
        "template_selection_policy": "one_anchor_extreme_and_one_pair_extreme_per_image",
        "scene_source": "active_simworld_scene_configs",
        "random_seed": args.random_seed,
        "subset_size": args.subset_size,
        "min_distance_gap": args.min_distance_gap,
        "default_evaluation_metrics": ["exact_match"],
        "question_counts_by_scene": question_counts_by_scene,
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
