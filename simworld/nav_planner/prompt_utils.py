from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, List, Optional


logger = logging.getLogger("PromptUtils")


def get_task_sequence_number(task_id: str) -> Optional[int]:
    try:
        return int(str(task_id).split("-")[-1])
    except (ValueError, IndexError, AttributeError):
        return None


def summarize_tasks_for_prompt(tasks: List[Dict[str, Any]], include_coords: bool = True) -> List[Dict[str, Any]]:
    summary = []
    for task in tasks:
        item = {
            "id": task.get("id"),
            "type": task.get("type"),
            "reward": task.get("reward"),
            "deadline": task.get("end_time"),
        }
        if include_coords:
            if task.get("pickup") is not None:
                item["pickup"] = task.get("pickup")
            if task.get("deliver") is not None:
                item["deliver"] = task.get("deliver")
        summary.append(item)
    return summary


def build_mapping_lines(context: Dict[str, Any]) -> List[str]:
    lines: List[str] = []

    for task in context.get("carried", []):
        task_num = get_task_sequence_number(task.get("id"))
        if task_num is None:
            continue
        deliver = task.get("deliver", {})
        lines.append(
            f"Task {task['id']} (carried): deliver at D{task_num} -> "
            f"(x={deliver.get('x', 0.0):.2f}, y={deliver.get('y', 0.0):.2f})"
        )

    for task in context.get("tasks", []):
        task_num = get_task_sequence_number(task.get("id"))
        if task_num is None:
            continue
        pickup = task.get("pickup", {})
        deliver = task.get("deliver", {})
        lines.append(
            f"Task {task['id']}: pickup at P{task_num} -> "
            f"(x={pickup.get('x', 0.0):.2f}, y={pickup.get('y', 0.0):.2f}); "
            f"deliver at D{task_num} -> "
            f"(x={deliver.get('x', 0.0):.2f}, y={deliver.get('y', 0.0):.2f})"
        )

    return lines


def build_previous_execution_prompt(prev_execution: Optional[Dict[str, Any]], heading: str) -> str:
    if not prev_execution:
        return ""

    try:
        waypoints = prev_execution.get("waypoints", [])
        wp_lines = []
        for wp in waypoints:
            coord = wp.get("coord", {})
            wp_lines.append(
                f"- #{wp.get('waypoint_index')}: {wp.get('task_type')} {wp.get('task_id')} "
                f"at (x={float(coord.get('x', 0.0)):.2f}, y={float(coord.get('y', 0.0)):.2f}), "
                f"elapsed={float(wp.get('elapsed_time', 0.0)):.2f}s"
            )

        if not wp_lines:
            return ""

        return (
            f"\n\n### {heading}\n"
            f"- Model used: {prev_execution.get('model')}\n"
            f"- Planner type: {prev_execution.get('planner_type')}\n"
            f"- Executed waypoints: {prev_execution.get('total_waypoints')}\n"
            f"- Total execution time (seconds): {prev_execution.get('total_time_seconds')}\n"
            "Executed waypoint sequence:\n"
            + "\n".join(wp_lines)
            + "\nUse this execution trace only to refine the route, reduce detours, and keep all original constraints valid."
        )
    except Exception as exc:
        logger.warning(f"构建上一轮执行结果提示失败，将忽略该信息: {exc}")
        return ""


def build_multimodal_user_content(prompt: str, context: Dict[str, Any], require_image: bool = False) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    image_path = context.get("image_path") or context.get("map_image")
    image_data = context.get("image_data")
    image_loaded = False

    if image_path and os.path.exists(image_path):
        try:
            print(f"正在加载地图: {image_path}")
            with open(image_path, "rb") as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode("utf-8")
            ext = os.path.splitext(image_path)[1].lower()
            mime_type = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
            }.get(ext, "image/png")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}})
            image_loaded = True
        except Exception as exc:
            logger.warning(f"加载图片失败: {exc}")
    elif isinstance(image_data, str) and image_data:
        prefix = "" if image_data.startswith("data:") else "data:image/png;base64,"
        content.append({"type": "image_url", "image_url": {"url": f"{prefix}{image_data}"}})
        image_loaded = True

    if require_image and not image_loaded:
        raise FileNotFoundError("未加载任何图片，当前模式需要地图图片")

    return content
