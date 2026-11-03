from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Dict, List


logger = logging.getLogger("ActionUtils")


def calculate_distance(p1: Dict[str, float], p2: Dict[str, float]) -> float:
    return math.hypot(p1["x"] - p2["x"], p1["y"] - p2["y"])


def interpolate_pos(start: Dict[str, float], end: Dict[str, float], ratio: float) -> Dict[str, float]:
    return {
        "x": start["x"] + (end["x"] - start["x"]) * ratio,
        "y": start["y"] + (end["y"] - start["y"]) * ratio,
    }


def extract_strict_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}

    clean_text = re.sub(r"```json\s*|\s*```", "", text).strip()
    start_idx = clean_text.find("{")
    end_idx = clean_text.rfind("}")

    if start_idx == -1 or end_idx == -1:
        start_idx = clean_text.find("[")
        end_idx = clean_text.rfind("]")

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        potential_json = clean_text[start_idx : end_idx + 1]
        try:
            return json.loads(potential_json)
        except json.JSONDecodeError:
            try:
                fixed_json = re.sub(r",\s*([\]}])", r"\1", potential_json)
                return json.loads(fixed_json)
            except Exception as exc:
                logger.error(f"Deep parsing failed: {exc}")

    try:
        return json.loads(clean_text)
    except Exception as exc:
        logger.error(f"Final fallback failed: {exc}")
        logger.debug(f"Raw text slice: {text[:200]}")
        return {}


def normalize_action_output(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []

    for action in actions:
        action_type = action.get("type") or action.get("action")
        if not action_type:
            continue

        clean_action = {"action": action_type}
        if action_type == "move":
            target = action.get("to")
            if not isinstance(target, dict) or "x" not in target or "y" not in target:
                continue
            clean_action["to"] = {"x": float(target["x"]), "y": float(target["y"])}
        elif action_type in {"pickup", "deliver"}:
            task_id = action.get("id") or action.get("task_id")
            if not task_id:
                continue
            clean_action["task_id"] = task_id
        else:
            continue

        normalized.append(clean_action)

    return normalized


def parse_json_actions(text: str) -> List[Dict[str, Any]]:
    data = extract_strict_json(text)
    if not data:
        return []
    return normalize_action_output(data.get("actions", []))
