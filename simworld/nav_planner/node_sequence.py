"""Shared symbolic output space for all planning modalities.

A fully controlled comparison requires the same simplified output format during
generation for every modality. So meta / text / image all emit one ordered node
sequence, e.g. ["START", "P1", "D1"], and nothing else. The deterministic
expansion into an executable point sequence (an exact-coordinate move before
every pickup / delivery) happens once, at evaluation time, so no modality is
scored on its ability to format executable actions.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from action_utils import extract_strict_json, normalize_action_output


logger = logging.getLogger("NodeSequence")

# Models drift on label spelling even when the ordering they intend is
# unambiguous: "P 1", "P_1", "P01", "P#1", "Pickup P1" all mean P1. Tolerate a
# short separator and leading zeros so that drift is normalized rather than
# charged downstream as a node that does not exist. The \b guard keeps this from
# firing inside longer words ("MAP1" has no boundary before the P).
_NODE_TOKEN = re.compile(r"\b([PD])[\s_#\-.:]{0,2}0*(\d+)\b", re.IGNORECASE)
_PROSE_TOKEN = re.compile(
    r"\b(START)\b|\b([PD])[\s_#\-.:]{0,2}0*(\d+)\b", re.IGNORECASE
)
_START_ALIASES = {"START", "BEGIN", "ORIGIN", "CURRENT", "ROBOT"}
_FENCE_PATTERN = re.compile(r"```[a-z]*\s*|\s*```")
_DICT_LABEL_KEYS = ("node", "label", "id", "name", "value", "step")


def canonicalize_node_label(raw: Any) -> Optional[str]:
    """Map one raw sequence entry onto its canonical node label.

    Returns "START", a canonical "P{n}" / "D{n}", or None when the entry carries
    no node claim at all (filler such as "PLAN" or "DONE"). Because every
    modality is scored on ordering alone, label spelling must be normalized here;
    a None result is dropped rather than penalized, whereas a well-formed label
    that simply does not exist in this scene (e.g. P99) is preserved so the
    evaluator can still charge it as a planning error.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return None
    if isinstance(raw, dict):
        for key in _DICT_LABEL_KEYS:
            if key in raw:
                return canonicalize_node_label(raw[key])
        return None

    text = str(raw).strip().upper()
    if not text:
        return None
    if text in _START_ALIASES:
        return "START"

    match = _NODE_TOKEN.search(text)
    if match:
        return f"{match.group(1).upper()}{int(match.group(2))}"
    return None


def canonicalize_node_sequence(nodes: List[Any]) -> List[str]:
    """Canonicalize a whole sequence, dropping entries that name no node."""
    canonical: List[str] = []
    dropped: List[str] = []
    for node in nodes:
        label = canonicalize_node_label(node)
        if label is None:
            dropped.append(str(node))
            continue
        canonical.append(label)
    if dropped:
        logger.warning(f"忽略 {len(dropped)} 个非节点条目: {', '.join(dropped[:8])}")
    return canonical


def get_task_sequence_number(task_id: Any) -> Optional[int]:
    """Extract the trailing task number, e.g. "A-0001" -> 1."""
    try:
        return int(str(task_id).split("-")[-1])
    except (ValueError, IndexError, AttributeError):
        return None


def build_node_mapping(task_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Resolve the P{n}/D{n} labels onto task ids and their target coordinates.

    Accepts a normalized task context (see scene_config), so both the planner and
    the evaluator derive the same mapping from the same task file.
    """
    mapping: Dict[str, Dict[str, Any]] = {}

    for task in task_data.get("tasks", []):
        task_num = get_task_sequence_number(task.get("id"))
        if task_num is None:
            logger.warning(f"无法解析任务编号: {task.get('id')}")
            continue
        mapping[f"P{task_num}"] = {"id": task["id"], "type": "pickup", "coord": task["pickup"]}
        mapping[f"D{task_num}"] = {"id": task["id"], "type": "deliver", "coord": task["deliver"]}

    for task in task_data.get("carried", []):
        task_num = get_task_sequence_number(task.get("id"))
        if task_num is None:
            logger.warning(f"无法解析任务编号: {task.get('id')}")
            continue
        mapping[f"D{task_num}"] = {"id": task["id"], "type": "deliver", "coord": task["deliver"]}

    return mapping


def project_actions_to_nodes(actions: List[Dict[str, Any]]) -> List[str]:
    """Collapse an executable action list onto the shared node space.

    Only used to rescue a model that answered in the legacy action format, and to
    replay legacy planning results through the unified pipeline.
    """
    nodes: List[str] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        op = action.get("type") or action.get("action")
        if op not in {"pickup", "deliver"}:
            continue
        task_num = get_task_sequence_number(action.get("id") or action.get("task_id"))
        if task_num is None:
            continue
        nodes.append(f"{'P' if op == 'pickup' else 'D'}{task_num}")
    return nodes


def _first_sequence_value(data: Dict[str, Any]) -> Optional[List[Any]]:
    """Find the list a model actually put its sequence in.

    Models wrap the answer under whatever key the prompt example suggested to
    them ("sequence", but also "nodes" / "plan" / "order"), sometimes nested one
    level deep. Accepting the synonyms costs nothing and keeps a well-ordered
    plan from being thrown away over its container name.
    """
    for key in ("sequence", "nodes", "node_sequence", "plan", "order", "route", "path"):
        value = data.get(key)
        if isinstance(value, list):
            return value
    for value in data.values():
        if isinstance(value, dict):
            nested = _first_sequence_value(value)
            if nested is not None:
                return nested
    return None


def _looks_like_actions(nodes: List[Any]) -> bool:
    """True when the list holds executable actions rather than node labels.

    A model that ignores the node format usually falls back to the legacy action
    shape, sometimes under a sequence-like key. Those entries carry a task id
    ("A-0001") that no node label pattern matches, so without this check the
    whole plan would canonicalize to nothing.
    """
    dicts = [node for node in nodes if isinstance(node, dict)]
    if not dicts or len(dicts) < len(nodes) / 2:
        return False
    return all(("action" in node or "type" in node) for node in dicts)


def parse_node_sequence(text: str) -> List[str]:
    """Extract the node sequence from a raw model response.

    Every modality funnels through here, so this is the single place where label
    spelling is normalized. Only ordering is attributed to the model: a mangled
    but recognizable label is repaired, a non-node token is dropped, and a
    well-formed label naming a task that does not exist survives to be charged
    by the evaluator.
    """
    data = extract_strict_json(text)

    nodes: List[Any]
    if isinstance(data, list):
        nodes = data
    elif isinstance(data, dict):
        sequence = _first_sequence_value(data)
        if sequence is not None:
            nodes = sequence
        elif isinstance(data.get("actions"), list):
            nodes = data["actions"]
        else:
            nodes = []
    else:
        nodes = []

    if nodes and _looks_like_actions(nodes):
        logger.warning("模型未按统一节点格式输出，已从 actions 投影回节点序列")
        return project_actions_to_nodes(nodes)

    if not nodes:
        # No usable JSON: fall back to scanning the prose for node tokens. The
        # same canonicalization applies, so "P 1" reads identically either way.
        clean_text = _FENCE_PATTERN.sub("", text or "").strip()
        prose: List[str] = []
        for start, kind, number in _PROSE_TOKEN.findall(clean_text):
            prose.append("START" if start else f"{kind.upper()}{int(number)}")
        return prose

    return canonicalize_node_sequence(nodes)


def node_sequence_to_actions(
    node_sequence: List[str],
    node_mapping: Dict[str, Dict[str, Any]],
    start_pos: Dict[str, float],
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, int]]]:
    """Expand a node sequence into an executable point sequence.

    Returns the normalized actions plus the (label, index) pairs that could not be
    resolved, so the evaluator can charge them as violations instead of silently
    dropping them.
    """
    actions: List[Dict[str, Any]] = []
    unresolved: List[Tuple[str, int]] = []
    current_pos = {
        "x": float(start_pos.get("x", 0.0)),
        "y": float(start_pos.get("y", 0.0)),
    }

    for index, node in enumerate(node_sequence):
        if node == "START":
            continue

        node_info = node_mapping.get(node)
        if not node_info:
            logger.warning(f"未找到节点映射: {node}")
            unresolved.append((node, index))
            continue

        target = {"x": float(node_info["coord"]["x"]), "y": float(node_info["coord"]["y"])}
        if current_pos != target:
            actions.append({"type": "move", "to": dict(target)})
            current_pos = dict(target)
        actions.append({"type": node_info["type"], "id": node_info["id"]})

    return normalize_action_output(actions), unresolved
