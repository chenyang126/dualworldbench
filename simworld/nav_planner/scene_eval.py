#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from node_sequence import build_node_mapping, canonicalize_node_sequence, node_sequence_to_actions
from scene_config import SCENE_CONFIGS, get_scene_config, infer_scene_from_task_dir, load_task_json


@dataclass
class PointEntry:
    point_label: str
    point_id: int
    point_x: float
    point_y: float
    access_distance_m: float


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


class SceneMatrixEvaluator:
    def __init__(
        self,
        scene: str,
        default_reward: float = 1.0,
        coord_tolerance: float = 1e-2,
        max_anchor_snap_m: float = 0.75,
    ) -> None:
        self.scene = scene
        self.config = get_scene_config(scene)
        self.default_reward = default_reward
        self.coord_tolerance = coord_tolerance
        self.max_anchor_snap_m = max_anchor_snap_m
        self.matrix = self._load_distance_matrix()
        self.points = self._load_point_mapping()

    def _require_path(self, path: Optional[os.PathLike | str], name: str) -> str:
        if not path:
            raise FileNotFoundError(f"scene={self.scene} 未配置 {name}")
        path_str = os.fspath(path)
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"scene={self.scene} 缺少 {name}: {path_str}")
        return path_str

    def _load_distance_matrix(self) -> Dict[str, Dict[str, float]]:
        csv_path = self._require_path(self.config.matrix_csv, "distance matrix")
        matrix: Dict[str, Dict[str, float]] = {}
        with open(csv_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            headers = [h for h in reader.fieldnames if h != "point"] if reader.fieldnames else []
            for row in reader:
                src = row["point"]
                matrix[src] = {dst: float(row[dst]) for dst in headers}
        return matrix

    def _load_indoor_point_mapping(self) -> List[PointEntry]:
        csv_path = self._require_path(self.config.point_mapping_csv, "point mapping csv")
        points: List[PointEntry] = []
        with open(csv_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                point_id = int(row["point_id"])
                points.append(
                    PointEntry(
                        point_label=f"P{point_id:02d}",
                        point_id=point_id,
                        point_x=float(row["point_x"]),
                        point_y=float(row["point_y"]),
                        access_distance_m=float(row.get("access_distance_m", 0.0)),
                    )
                )
        return points

    def _load_points_file_mapping(self) -> List[PointEntry]:
        points_file = self._require_path(self.config.points_file, "points file")
        points: List[PointEntry] = []
        with open(points_file, "r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle, start=1):
                nums = re.findall(r"-?\d+(?:\.\d+)?", line)
                if len(nums) < 2:
                    continue
                points.append(
                    PointEntry(
                        point_label=f"P{idx:02d}",
                        point_id=idx,
                        point_x=float(nums[0]),
                        point_y=float(nums[1]),
                        access_distance_m=0.0,
                    )
                )
        return points

    def _load_point_mapping(self) -> List[PointEntry]:
        if self.config.point_mapping_csv:
            return self._load_indoor_point_mapping()
        if self.config.points_file:
            return self._load_points_file_mapping()
        raise FileNotFoundError(f"scene={self.scene} 未配置点位映射资源")

    def resolve_point_label(self, coord: Dict[str, float]) -> Tuple[str, float]:
        best_entry: Optional[PointEntry] = None
        best_distance = float("inf")

        x = float(coord["x"])
        y = float(coord["y"])
        for entry in self.points:
            distance = math.hypot(x - entry.point_x, y - entry.point_y)
            if distance < best_distance:
                best_distance = distance
                best_entry = entry

        if best_entry is None:
            raise ValueError(f"无法为坐标 {coord} 找到矩阵点位")
        if best_distance > self.max_anchor_snap_m:
            raise ValueError(
                f"坐标 {coord} 距离最近导航锚点 {best_entry.point_label} 超过允许阈值 "
                f"({best_distance:.3f} m > {self.max_anchor_snap_m:.3f} m)"
            )
        return best_entry.point_label, best_distance + best_entry.access_distance_m

    def distance_between(self, src_coord: Dict[str, float], dst_coord: Dict[str, float]) -> Dict[str, Any]:
        src_label, src_offset = self.resolve_point_label(src_coord)
        dst_label, dst_offset = self.resolve_point_label(dst_coord)

        if src_label not in self.matrix or dst_label not in self.matrix[src_label]:
            raise KeyError(f"距离矩阵缺失点对 {src_label} -> {dst_label}")

        graph_distance = self.matrix[src_label][dst_label]
        total_distance = graph_distance + src_offset + dst_offset
        return {
            "src_label": src_label,
            "dst_label": dst_label,
            "src_offset_m": src_offset,
            "dst_offset_m": dst_offset,
            "graph_distance_m": graph_distance,
            "total_distance_m": total_distance,
        }

    def _task_lookup(self, normalized_task_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return {task["id"]: task for task in normalized_task_data.get("tasks", [])}

    def evaluate_node_sequence(
        self,
        normalized_task_data: Dict[str, Any],
        node_sequence: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Score a node sequence, the shared output space of all modalities.

        The sequence is expanded here into an executable point sequence, so the
        move-before-op formatting is supplied identically for every modality and
        only the ordering decisions are attributed to the model. Labels are
        canonicalized first (idempotent for freshly parsed plans) so that a
        hand-written or replayed file is not penalized for label spelling.
        """
        canonical_sequence = canonicalize_node_sequence(node_sequence)
        node_mapping = build_node_mapping(normalized_task_data)
        actions, unresolved = node_sequence_to_actions(
            canonical_sequence,
            node_mapping,
            normalized_task_data.get("pos", {}),
        )

        result = self.evaluate_actions(
            normalized_task_data=normalized_task_data,
            actions=actions,
            metadata=metadata,
        )
        # An unresolvable label is a real planning error (a node that does not
        # exist in this scene), so it must be charged rather than dropped.
        result["violations"] = [
            f"unknown_node_{label}_at_{index}" for label, index in unresolved
        ] + result["violations"]
        result["feasible"] = len(result["violations"]) == 0
        result["node_sequence"] = list(canonical_sequence)
        if list(canonical_sequence) != [str(node) for node in node_sequence]:
            result["raw_node_sequence"] = [str(node) for node in node_sequence]
        result["expanded_actions"] = actions
        return result

    def evaluate_actions(
        self,
        normalized_task_data: Dict[str, Any],
        actions: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        task_lookup = self._task_lookup(normalized_task_data)
        current_pos = {
            "x": float(normalized_task_data.get("pos", {}).get("x", 0.0)),
            "y": float(normalized_task_data.get("pos", {}).get("y", 0.0)),
        }
        speed = float(normalized_task_data.get("speed", 1.0))
        max_capacity = int(normalized_task_data.get("max_capacity", 1))

        carried_tasks = normalized_task_data.get("carried", [])
        carried: set[str] = {
            str(task["id"])
            for task in carried_tasks
            if isinstance(task, dict) and task.get("id")
        }
        picked: set[str] = set(carried)
        delivered: set[str] = set()
        violations: List[str] = []
        move_segments: List[Dict[str, Any]] = []

        total_distance_m = 0.0
        total_reward = 0.0

        for index, action in enumerate(actions):
            action_type = action.get("action")
            if action_type == "move":
                target = action.get("to")
                if not isinstance(target, dict) or "x" not in target or "y" not in target:
                    violations.append(f"invalid_move_action_at_{index}")
                    continue
                try:
                    segment = self.distance_between(current_pos, target)
                except Exception:
                    violations.append(f"invalid_move_target_off_graph_at_{index}")
                    continue
                segment["index"] = index
                segment["from"] = current_pos.copy()
                segment["to"] = {"x": float(target["x"]), "y": float(target["y"])}
                move_segments.append(segment)
                total_distance_m += segment["total_distance_m"]
                current_pos = {"x": float(target["x"]), "y": float(target["y"])}
                continue

            if action_type not in {"pickup", "deliver"}:
                violations.append(f"unknown_action_{action_type}_at_{index}")
                continue

            task_id = action.get("task_id")
            if task_id not in task_lookup:
                violations.append(f"unknown_task_{task_id}_at_{index}")
                continue

            task = task_lookup[task_id]
            expected_coord = task["pickup"] if action_type == "pickup" else task["deliver"]
            at_expected_coord = (
                abs(current_pos["x"] - float(expected_coord["x"])) <= self.coord_tolerance
                and abs(current_pos["y"] - float(expected_coord["y"])) <= self.coord_tolerance
            )
            if not at_expected_coord:
                violations.append(f"{action_type}_without_arrival_{task_id}_at_{index}")

            if action_type == "pickup":
                pickup_valid = True
                if task_id in picked:
                    violations.append(f"duplicate_pickup_{task_id}_at_{index}")
                    pickup_valid = False
                if len(carried) >= max_capacity:
                    violations.append(f"capacity_exceeded_at_{index}")
                    pickup_valid = False
                if not at_expected_coord:
                    pickup_valid = False

                if pickup_valid:
                    picked.add(task_id)
                    carried.add(task_id)
            else:
                delivery_valid = True
                if task_id not in picked:
                    violations.append(f"deliver_before_pickup_{task_id}_at_{index}")
                    delivery_valid = False
                if task_id in delivered:
                    violations.append(f"duplicate_delivery_{task_id}_at_{index}")
                    delivery_valid = False
                if task_id not in carried:
                    violations.append(f"deliver_without_carry_{task_id}_at_{index}")
                    delivery_valid = False
                if not at_expected_coord:
                    delivery_valid = False

                if delivery_valid:
                    delivered.add(task_id)
                    carried.discard(task_id)
                    reward = task.get("reward")
                    total_reward += float(reward) if reward is not None else self.default_reward

        completed_tasks = len(delivered)
        total_time_s = total_distance_m / speed if speed > 0 else float("inf")
        feasible = len(violations) == 0
        reward_rate = total_reward / total_time_s if total_time_s > 0 and math.isfinite(total_time_s) else 0.0

        return {
            "feasible": feasible,
            "completed_tasks": completed_tasks,
            # Which tasks were validly delivered, not just how many. Needed to
            # normalise travel against the optimal route for the subset a plan
            # actually served, rather than for the full task set.
            "delivered_ids": sorted(delivered),
            "total_tasks": len(task_lookup),
            "completion_rate": completed_tasks / len(task_lookup) if task_lookup else 0.0,
            "total_distance_m": round(total_distance_m, 3),
            "total_time_s": round(total_time_s, 3) if math.isfinite(total_time_s) else None,
            "total_reward": round(total_reward, 3),
            "reward_rate": round(reward_rate, 6),
            "violations": violations,
            "move_segments": move_segments,
            "metadata": metadata or {},
        }


def evaluate_files(
    task_json: str,
    planning_json: str,
    output_path: Optional[str] = None,
    default_reward: float = 1.0,
    scene: Optional[str] = None,
) -> Dict[str, Any]:
    scene = scene or infer_scene_from_task_dir(task_json) or infer_scene_from_task_dir(planning_json) or "indoor"
    normalized_task_data = load_task_json(task_json, scene)
    planning_data = load_json(planning_json)
    evaluator = SceneMatrixEvaluator(scene=scene, default_reward=default_reward)

    if "node_sequence" in planning_data:
        result = evaluator.evaluate_node_sequence(
            normalized_task_data=normalized_task_data,
            node_sequence=planning_data.get("node_sequence", []),
            metadata=planning_data.get("metadata", {}),
        )
    else:
        # Legacy plan file written before the output space was unified.
        result = evaluator.evaluate_actions(
            normalized_task_data=normalized_task_data,
            actions=planning_data.get("actions", []),
            metadata=planning_data.get("metadata", {}),
        )
    result["task_file"] = task_json
    result["planning_file"] = planning_json
    result["scene"] = scene

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Scene planning evaluator based on pairwise distance matrices.")
    parser.add_argument("--scene", choices=sorted(SCENE_CONFIGS), default=None, help="场景名称")
    parser.add_argument("--task-json", required=True, help="任务 json 文件路径")
    parser.add_argument("--planning-json", required=True, help="规划结果 json 文件路径")
    parser.add_argument("--output", default=None, help="评测结果输出路径")
    parser.add_argument("--default-reward", type=float, default=1.0, help="当任务 reward 缺失时的默认奖励")
    args = parser.parse_args()

    result = evaluate_files(
        task_json=args.task_json,
        planning_json=args.planning_json,
        output_path=args.output,
        default_reward=args.default_reward,
        scene=args.scene,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
