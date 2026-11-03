from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List


NAV_PLANNER_DIR = Path(__file__).resolve().parent
SIM_PLANNER_ROOT = NAV_PLANNER_DIR.parent
REPO_ROOT = SIM_PLANNER_ROOT.parent
SIM_TASK_ROOT = REPO_ROOT / "data" / "simworld" / "task"


def _format_task_id(index: int) -> str:
    return f"A-{index:04d}"


def _normalize_xyz(point: Dict[str, Any]) -> Dict[str, float]:
    return {
        "x": float(point.get("x", 0.0)),
        "y": float(point.get("y", 0.0)),
        "z": float(point.get("z", 0.0)),
    }


def normalize_indoor_task_data(raw: Dict[str, Any]) -> Dict[str, Any]:
    tasks = [
        {
            "id": task["id"],
            "type": task.get("type", 0),
            "type_name": task.get("type_name"),
            "title": task.get("title"),
            "pickup": {"x": float(task["pickup"]["x"]), "y": float(task["pickup"]["y"])},
            "deliver": {"x": float(task["deliver"]["x"]), "y": float(task["deliver"]["y"])},
            "end_time": task.get("end_time"),
            "reward": task.get("reward"),
            "distance_m": task.get("distance_m"),
            "trigger_time": float(task.get("trigger_time", 0.0)),
            "priority": bool(task.get("priority", False)),
        }
        for task in raw.get("tasks", [])
    ]

    return {
        "now": float(raw.get("NOW_TIME", 0.0)),
        "pos": {
            "x": float(raw.get("CUR_X", 0.0)),
            "y": float(raw.get("CUR_Y", 0.0)),
        },
        "speed": float(raw.get("SPEED", 1.0)),
        "max_capacity": int(raw.get("CAPACITY", 5)),
        "carried": [],
        "tasks": tasks,
        "raw_task_data": raw,
    }


def normalize_point_pair_task_data(raw: Dict[str, Any]) -> Dict[str, Any]:
    pickup_points = raw.get("pickup_points", [])
    delivery_points = raw.get("delivery_points", [])

    if len(pickup_points) != len(delivery_points):
        raise ValueError("pickup_points 和 delivery_points 数量不一致，无法构建任务对")

    tasks: List[Dict[str, Any]] = []
    for idx, (pickup, deliver) in enumerate(zip(pickup_points, delivery_points), start=1):
        tasks.append(
            {
                "id": _format_task_id(idx),
                "type": 0,
                "type_name": "pickup_delivery",
                "title": f"Task {_format_task_id(idx)}",
                "pickup": {"x": float(pickup["x"]), "y": float(pickup["y"])},
                "deliver": {"x": float(deliver["x"]), "y": float(deliver["y"])},
                "end_time": raw.get("deadline"),
                "reward": raw.get("default_reward", 1.0),
                "distance_m": None,
                "trigger_time": 0.0,
                "priority": False,
            }
        )

    start = _normalize_xyz(raw.get("start_point", {}))
    return {
        "now": float(raw.get("NOW_TIME", raw.get("now_time", 0.0))),
        "pos": {"x": start["x"], "y": start["y"]},
        # Honor the uppercase keys too: some point-pair files carry the same
        # SPEED / CAPACITY fields as the task-list schema, and silently
        # substituting a default would make total_time_s incomparable across
        # samples of one scene.
        "speed": float(raw.get("SPEED", raw.get("speed", 1.0))),
        "max_capacity": int(raw.get("CAPACITY", raw.get("capacity", max(1, len(tasks))))),
        "carried": [],
        "tasks": tasks,
        "raw_task_data": raw,
    }


def detect_task_loader(raw: Dict[str, Any]) -> Callable[[Dict[str, Any]], Dict[str, Any]] | None:
    """Pick the normalizer that matches this file's actual shape.

    A scene directory is not guaranteed to be schema-homogeneous (indoor mixes
    both forms), and the two schemas share no keys, so the file itself is a more
    reliable signal than the scene config. Returns None when neither shape is
    recognizable, letting the caller fall back and then fail loudly.
    """
    if isinstance(raw.get("tasks"), list) and raw["tasks"]:
        return normalize_indoor_task_data
    if isinstance(raw.get("pickup_points"), list) and raw["pickup_points"]:
        return normalize_point_pair_task_data
    return None


@dataclass(frozen=True)
class SceneConfig:
    name: str
    task_dir: Path
    image_dirname: str
    task_loader: Callable[[Dict[str, Any]], Dict[str, Any]]
    evaluator_module: str | None = None
    matrix_csv: Path | None = None
    point_mapping_csv: Path | None = None
    points_file: Path | None = None
    image_x_increases_to_right: bool = True
    image_y_increases_upward: bool = True

    @property
    def image_dir(self) -> Path:
        return self.task_dir / self.image_dirname

    def default_task_dir(self) -> str:
        return str(self.task_dir)

    def default_image_dir(self) -> str:
        return str(self.image_dir)


SCENE_CONFIGS: Dict[str, SceneConfig] = {
    "indoor": SceneConfig(
        name="indoor",
        task_dir=SIM_TASK_ROOT / "indoor",
        image_dirname="indoor_image",
        task_loader=normalize_indoor_task_data,
        evaluator_module="scene_eval",
        matrix_csv=SIM_TASK_ROOT / "indoor" / "pairwise_distance_matrix_with_origin_m.csv",
        point_mapping_csv=SIM_TASK_ROOT / "indoor" / "point_nav_mapping_with_origin.csv",
    ),
    "forest": SceneConfig(
        name="forest",
        task_dir=SIM_TASK_ROOT / "forest" / "forest_task",
        image_dirname="forest_image",
        task_loader=normalize_indoor_task_data,
        evaluator_module="scene_eval",
        matrix_csv=SIM_TASK_ROOT / "forest" / "pairwise_distance_matrix_with_origin_m.csv",
        points_file=SIM_TASK_ROOT / "forest" / "forest_points.txt",
    ),
    "campus": SceneConfig(
        name="campus",
        task_dir=SIM_TASK_ROOT / "campus" / "campus_task",
        image_dirname="campus_image",
        task_loader=normalize_indoor_task_data,
        evaluator_module="scene_eval",
        matrix_csv=SIM_TASK_ROOT / "campus" / "pairwise_distance_matrix_with_origin_m.csv",
        points_file=SIM_TASK_ROOT / "campus" / "campus_points.txt",
    ),
    "typical": SceneConfig(
        name="typical",
        task_dir=SIM_TASK_ROOT / "typical",
        image_dirname="typical_image",
        task_loader=normalize_indoor_task_data,
        evaluator_module="scene_eval",
        matrix_csv=SIM_TASK_ROOT / "typical" / "pairwise_distance_matrix_with_origin_m.csv",
        points_file=SIM_TASK_ROOT / "typical" / "typical_points.txt",
    ),
    "typical_islands": SceneConfig(
        name="typical_islands",
        task_dir=SIM_TASK_ROOT / "typical_islands" / "typical_islands_task",
        image_dirname="typical_islands_image",
        task_loader=normalize_indoor_task_data,
        evaluator_module="scene_eval",
        matrix_csv=SIM_TASK_ROOT / "typical_islands" / "pairwise_distance_matrix_with_origin_m.csv",
        points_file=SIM_TASK_ROOT / "typical_islands" / "typical_islands_points.txt",
    ),
    "winding_road": SceneConfig(
        name="winding_road",
        task_dir=SIM_TASK_ROOT / "winding_road" / "winding_road_task",
        image_dirname="winding_road_image",
        task_loader=normalize_indoor_task_data,
        evaluator_module="scene_eval",
        matrix_csv=SIM_TASK_ROOT / "winding_road" / "pairwise_distance_matrix_with_origin_m.csv",
        points_file=SIM_TASK_ROOT / "winding_road" / "winding_road_points.txt",
    ),
    "rooms": SceneConfig(
        name="rooms",
        task_dir=SIM_TASK_ROOT / "rooms" / "rooms_task",
        image_dirname="rooms_image",
        task_loader=normalize_indoor_task_data,
        evaluator_module="scene_eval",
        matrix_csv=SIM_TASK_ROOT / "rooms" / "pairwise_distance_matrix_with_origin_m.csv",
        points_file=SIM_TASK_ROOT / "rooms" / "rooms_points.txt",
    ),
    "rooms_mixed": SceneConfig(
        name="rooms_mixed",
        task_dir=SIM_TASK_ROOT / "rooms_mixed" / "rooms_mixed_task",
        image_dirname="rooms_mixed_image",
        task_loader=normalize_indoor_task_data,
        evaluator_module="scene_eval",
        matrix_csv=SIM_TASK_ROOT / "rooms_mixed" / "pairwise_distance_matrix_with_origin_m.csv",
        points_file=SIM_TASK_ROOT / "rooms_mixed" / "rooms_mixed_points.txt",
    ),
    "branch_corridor": SceneConfig(
        name="branch_corridor",
        task_dir=SIM_TASK_ROOT / "branch_corridor" / "branch_corridor_task",
        image_dirname="branch_corridor_image",
        task_loader=normalize_indoor_task_data,
        evaluator_module="scene_eval",
        matrix_csv=SIM_TASK_ROOT / "branch_corridor" / "pairwise_distance_matrix_with_origin_m.csv",
        points_file=SIM_TASK_ROOT / "branch_corridor" / "branch_corridor_points.txt",
    ),
    "l1_spine_rooms": SceneConfig(
        name="l1_spine_rooms",
        task_dir=SIM_TASK_ROOT / "l1_spine_rooms",
        image_dirname="l1_spine_rooms_image",
        task_loader=normalize_point_pair_task_data,
        evaluator_module="scene_eval",
        matrix_csv=SIM_TASK_ROOT / "l1_spine_rooms" / "pairwise_distance_matrix_with_origin_m.csv",
        points_file=SIM_TASK_ROOT / "l1_spine_rooms" / "l1_spine_rooms_points.txt",
    ),
    "l1_cross_suite": SceneConfig(
        name="l1_cross_suite",
        task_dir=SIM_TASK_ROOT / "l1_cross_suite",
        image_dirname="l1_cross_suite_image",
        task_loader=normalize_point_pair_task_data,
        evaluator_module="scene_eval",
        matrix_csv=SIM_TASK_ROOT / "l1_cross_suite" / "pairwise_distance_matrix_with_origin_m.csv",
        points_file=SIM_TASK_ROOT / "l1_cross_suite" / "l1_cross_suite_points.txt",
    ),
    "l1_offset_wing": SceneConfig(
        name="l1_offset_wing",
        task_dir=SIM_TASK_ROOT / "l1_offset_wing",
        image_dirname="l1_offset_wing_image",
        task_loader=normalize_point_pair_task_data,
        evaluator_module="scene_eval",
        matrix_csv=SIM_TASK_ROOT / "l1_offset_wing" / "pairwise_distance_matrix_with_origin_m.csv",
        points_file=SIM_TASK_ROOT / "l1_offset_wing" / "l1_offset_wing_points.txt",
        image_x_increases_to_right=False,
        image_y_increases_upward=False,
    ),
    "l1_dual_room": SceneConfig(
        name="l1_dual_room",
        task_dir=SIM_TASK_ROOT / "l1_dual_room",
        image_dirname="l1_dual_room_image",
        task_loader=normalize_point_pair_task_data,
        evaluator_module="scene_eval",
        matrix_csv=SIM_TASK_ROOT / "l1_dual_room" / "pairwise_distance_matrix_with_origin_m.csv",
        points_file=SIM_TASK_ROOT / "l1_dual_room" / "l1_dual_room_points.txt",
    ),
    "l1_corner_hall": SceneConfig(
        name="l1_corner_hall",
        task_dir=SIM_TASK_ROOT / "l1_corner_hall",
        image_dirname="l1_corner_hall_image",
        task_loader=normalize_point_pair_task_data,
        evaluator_module="scene_eval",
        matrix_csv=SIM_TASK_ROOT / "l1_corner_hall" / "pairwise_distance_matrix_with_origin_m.csv",
        points_file=SIM_TASK_ROOT / "l1_corner_hall" / "l1_corner_hall_points.txt",
    ),
    "l2_loop_rooms": SceneConfig(
        name="l2_loop_rooms",
        task_dir=SIM_TASK_ROOT / "l2_loop_rooms",
        image_dirname="l2_loop_rooms_image",
        task_loader=normalize_point_pair_task_data,
        evaluator_module="scene_eval",
        matrix_csv=SIM_TASK_ROOT / "l2_loop_rooms" / "pairwise_distance_matrix_with_origin_m.csv",
        points_file=SIM_TASK_ROOT / "l2_loop_rooms" / "l2_loop_rooms_points.txt",
        image_x_increases_to_right=False,
        image_y_increases_upward=False,
    ),
    "l2_dual_spine": SceneConfig(
        name="l2_dual_spine",
        task_dir=SIM_TASK_ROOT / "l2_dual_spine",
        image_dirname="l2_dual_spine_image",
        task_loader=normalize_point_pair_task_data,
        evaluator_module="scene_eval",
        matrix_csv=SIM_TASK_ROOT / "l2_dual_spine" / "pairwise_distance_matrix_with_origin_m.csv",
        points_file=SIM_TASK_ROOT / "l2_dual_spine" / "l2_dual_spine_points.txt",
        image_x_increases_to_right=False,
        image_y_increases_upward=False,
    ),
    "l2_ring_hub": SceneConfig(
        name="l2_ring_hub",
        task_dir=SIM_TASK_ROOT / "l2_ring_hub",
        image_dirname="l2_ring_hub_image",
        task_loader=normalize_point_pair_task_data,
        evaluator_module="scene_eval",
        matrix_csv=SIM_TASK_ROOT / "l2_ring_hub" / "pairwise_distance_matrix_with_origin_m.csv",
        points_file=SIM_TASK_ROOT / "l2_ring_hub" / "l2_ring_hub_points.txt",
        image_x_increases_to_right=False,
        image_y_increases_upward=False,
    ),
    "l2_offset_blocks": SceneConfig(
        name="l2_offset_blocks",
        task_dir=SIM_TASK_ROOT / "l2_offset_blocks",
        image_dirname="l2_offset_blocks_image",
        task_loader=normalize_point_pair_task_data,
        evaluator_module="scene_eval",
        matrix_csv=SIM_TASK_ROOT / "l2_offset_blocks" / "pairwise_distance_matrix_with_origin_m.csv",
        points_file=SIM_TASK_ROOT / "l2_offset_blocks" / "l2_offset_blocks_points.txt",
        image_x_increases_to_right=False,
        image_y_increases_upward=False,
    ),
    "l2_cluster_corridor": SceneConfig(
        name="l2_cluster_corridor",
        task_dir=SIM_TASK_ROOT / "l2_cluster_corridor",
        image_dirname="l2_cluster_corridor_image",
        task_loader=normalize_point_pair_task_data,
        evaluator_module="scene_eval",
        matrix_csv=SIM_TASK_ROOT / "l2_cluster_corridor" / "pairwise_distance_matrix_with_origin_m.csv",
        points_file=SIM_TASK_ROOT / "l2_cluster_corridor" / "l2_cluster_corridor_points.txt",
        image_x_increases_to_right=False,
        image_y_increases_upward=False,
    ),
}


# Only expose scenes whose data ships in this release.
SCENE_CONFIGS = {k: v for k, v in SCENE_CONFIGS.items() if v.task_dir.is_dir()}

def get_scene_config(scene: str) -> SceneConfig:
    try:
        return SCENE_CONFIGS[scene]
    except KeyError as exc:
        raise ValueError(f"不支持的 scene: {scene}") from exc


def infer_scene_from_task_dir(task_dir: str | None) -> str | None:
    if not task_dir:
        return None

    resolved = os.path.abspath(task_dir)
    for scene_name, config in SCENE_CONFIGS.items():
        if resolved.startswith(os.path.abspath(config.default_task_dir())):
            return scene_name
    return None


def load_task_json(json_path: str, scene: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    config = get_scene_config(scene)
    loader = detect_task_loader(raw) or config.task_loader
    normalized = loader(raw)
    # Before shape detection, a file written in the scene's *other* schema
    # normalized to zero tasks with no error at all: the planner was prompted
    # with an empty task list and the sample scored 0 completions as if the
    # model had failed. Refuse to hand back an empty task set instead.
    if not normalized.get("tasks") and not normalized.get("carried"):
        raise ValueError(
            f"任务文件解析后没有任何任务，可能是无法识别的格式: {json_path} "
            f"(顶层字段: {', '.join(sorted(raw)[:12])})"
        )
    normalized["task_file"] = json_path
    return normalized
