#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model_endpoints import resolve as resolve_endpoint, temperature_kwargs


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
ICLR_VQA_DIR = REPO_ROOT / "data" / "vqa"
# nav_planner has lived at two different depths across repo layouts; take the
# first one that actually exists rather than assuming a fixed relative path.
_NAV_CANDIDATES = [
    REPO_ROOT / "simworld" / "nav_planner",
    REPO_ROOT.parent / "nav_planner",
    REPO_ROOT / "nav_planner",
]
NAV_PLANNER_DIR = next((p for p in _NAV_CANDIDATES if p.is_dir()), _NAV_CANDIDATES[0])

if str(NAV_PLANNER_DIR) not in sys.path:
    sys.path.insert(0, str(NAV_PLANNER_DIR))

from prompt_utils import build_multimodal_user_content  # type: ignore  # noqa: E402
from scene_eval import SceneMatrixEvaluator  # type: ignore  # noqa: E402


DATASET_PRESETS = {
    "absolute": ICLR_VQA_DIR / "task" / "absolute_axis_sorting_vqa.json",
    "euclidean": ICLR_VQA_DIR / "task" / "euclidean_distance_vqa.json",
    "wall": ICLR_VQA_DIR / "task" / "wall_illusion_vqa.json",
    "capacity": ICLR_VQA_DIR / "task" / "capacity_interleaving_vqa.json",
}
DATASET_PRESET_GROUPS = {
    "l1": ("absolute", "euclidean"),
    "l2": ("wall",),
    "l3": ("capacity",),
}

NODE_PATTERN = re.compile(r"\b(?:START|P\d+|D\d+)\b", re.IGNORECASE)
EVALUATION_POLICY_BY_TYPE = {
    "absolute_axis_sorting": {
        "primary": "exact_match",
        "auxiliary": ["lcs_length", "lcs_precision", "lcs_recall", "lcs_f1"],
    },
    "visual_distance_vs_topological_distance": {
        "primary": "exact_match",
        "auxiliary": [],
    },
    "euclidean_anchor_extreme_choice": {
        "primary": "exact_match",
        "auxiliary": [],
    },
    "euclidean_pair_extreme_choice": {
        "primary": "exact_match",
        "auxiliary": [],
    },
    "capacity_interleaving_optimization": {
        "primary": "distance_within_relative_tolerance",
        "auxiliary": [],
    },
}


SCENE_EVALUATOR_CACHE: Dict[str, SceneMatrixEvaluator] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VQA datasets against an OpenAI-compatible API.")
    parser.add_argument("--api-base", default=None, help="API base URL; defaults to OPENAI_BASE_URL or the official OpenAI API")
    parser.add_argument("--api-key", default=None, help="API key; defaults to OPENAI_API_KEY")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument(
        "--vqa-set",
        nargs="+",
        required=True,
        help="Dataset preset(s): absolute / euclidean / wall / capacity / l1 / l2 / l3, or direct JSON path(s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path; default writes under outputs/vqa/",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max-questions", type=int, default=None, help="Optional cap on total number of questions to run")
    parser.add_argument("--limit-per-set", type=int, default=None, help="Optional cap per dataset")
    parser.add_argument(
        "--scene",
        nargs="+",
        default=None,
        help="Optional scene filter, e.g. campus indoor l1_corner_hall",
    )
    parser.add_argument("--timeout", type=float, default=500.0, help="Per-request timeout in seconds")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests")
    parser.add_argument("--parallelism", type=int, default=10, help="How many requests to run concurrently")
    parser.add_argument("--max-attempts", type=int, default=3, help="Application-level retry attempts per question")
    parser.add_argument("--retry-delay", type=float, default=2.0, help="Initial retry delay in seconds")
    parser.add_argument(
        "--question-id",
        nargs="*",
        default=None,
        help="Optional question_id filter; when provided only these questions are run",
    )
    return parser.parse_args()


def resolve_dataset_paths(vqa_set_args: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for item in vqa_set_args:
        if item in DATASET_PRESET_GROUPS:
            for preset_name in DATASET_PRESET_GROUPS[item]:
                path = DATASET_PRESETS[preset_name]
                if not path.exists():
                    raise FileNotFoundError(f"找不到 VQA 集合文件: {preset_name}")
                paths.append(path.resolve())
            continue

        preset = DATASET_PRESETS.get(item)
        path = preset if preset is not None else Path(item)
        if not path.exists():
            raise FileNotFoundError(f"找不到 VQA 集合文件: {item}")
        paths.append(path.resolve())
    return paths


SIMWORLD_IMAGE_ROOT = REPO_ROOT / "data" / "simworld" / "task"


def localize_image_path(stored: str | None) -> str | None:
    """Resolve repository-relative dataset paths independently of the CWD."""
    if not stored:
        return stored
    path = Path(stored)
    candidate = path if path.is_absolute() else REPO_ROOT / path
    if not candidate.is_file():
        raise FileNotFoundError(f"Missing VQA image: {stored}")
    return str(candidate.resolve())


def load_dataset(
    path: Path,
    question_id_filter: set[str] | None,
    scene_filter: set[str] | None,
    limit_per_set: int | None,
) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if scene_filter is not None:
        records = [record for record in records if record.get("scene") in scene_filter]
    if question_id_filter is not None:
        records = [record for record in records if record["question_id"] in question_id_filter]
    if limit_per_set is not None:
        records = records[:limit_per_set]
    for record in records:
        record["image_path"] = localize_image_path(record.get("image_path"))
        record["_dataset_path"] = str(path)
        record["_dataset_family"] = payload.get("question_family")
    return records


def build_system_prompt(question_type: str) -> str:
    if question_type == "absolute_axis_sorting":
        return (
            "You are a strict VQA answerer. Return exactly one line containing only the final node ordering. "
            "Use the format NODE -> NODE -> NODE. Do not add explanations, numbering, or markdown."
        )
    if question_type == "visual_distance_vs_topological_distance":
        return (
            "You are a strict VQA answerer. Return exactly one line identifying the longer route. "
            "Use the format START -> P1 is longer. Do not add explanations or extra text."
        )
    if question_type == "euclidean_anchor_extreme_choice":
        return (
            "You are a strict VQA answerer. Return exactly one line containing only one node label. "
            "Do not add explanations, numbering, or markdown."
        )
    if question_type == "euclidean_pair_extreme_choice":
        return (
            "You are a strict VQA answerer. Return exactly one line containing only two node labels. "
            "Use the format NODE -> NODE. Do not add explanations, numbering, or markdown."
        )
    if question_type == "capacity_interleaving_optimization":
        return (
            "You are a strict VQA answerer. Return exactly one line containing only the optimal visit sequence. "
            "Use the format START -> P5 -> P1 -> D1 -> D2. Do not add explanations, numbering, or markdown."
        )
    return "You are a strict VQA answerer. Output only the final answer and no explanation."


def build_user_prompt(record: Dict[str, Any]) -> str:
    output_format = record.get("output_format")
    prompt = (
        "Please answer the following question based on the provided map image.\n"
        f"Question type: {record['question_type']}\n"
        f"Question: {record['question']}\n"
    )
    if output_format:
        prompt += f"Output format: {output_format}\n"
    prompt += "Output only the final answer and do not provide any reasoning."
    return prompt


def build_messages(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    prompt = build_user_prompt(record)
    content = build_multimodal_user_content(
        prompt=prompt,
        context={"image_path": record["image_path"]},
        require_image=True,
    )
    return [
        {"role": "system", "content": build_system_prompt(record["question_type"])},
        {"role": "user", "content": content},
    ]


def extract_nodes(text: str) -> List[str]:
    return [match.upper() for match in NODE_PATTERN.findall(text or "")]


def lcs_length(seq_a: Sequence[str], seq_b: Sequence[str]) -> int:
    rows = len(seq_a) + 1
    cols = len(seq_b) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        for j in range(1, cols):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def build_sequence_metrics(parsed_nodes: Sequence[str], gt_nodes: Sequence[str]) -> Dict[str, Any]:
    lcs = lcs_length(parsed_nodes, gt_nodes)
    pred_len = len(parsed_nodes)
    gt_len = len(gt_nodes)
    precision = lcs / pred_len if pred_len else 0.0
    recall = lcs / gt_len if gt_len else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    return {
        "complete_sequence_match": list(parsed_nodes) == list(gt_nodes),
        "lcs_length": lcs,
        "pred_length": pred_len,
        "gt_length": gt_len,
        "lcs_precision": precision,
        "lcs_recall": recall,
        "lcs_f1": f1,
    }


def evaluate_absolute(record: Dict[str, Any], raw_response: str) -> Dict[str, Any]:
    parsed_nodes = extract_nodes(raw_response)
    gt_nodes = list(record["gt_order"])
    sequence_metrics = build_sequence_metrics(parsed_nodes, gt_nodes)
    return {
        "parsed_answer": " -> ".join(parsed_nodes) if parsed_nodes else raw_response.strip(),
        "is_correct": sequence_metrics["complete_sequence_match"],
        "judge_detail": {
            "expected_nodes": gt_nodes,
            "parsed_nodes": parsed_nodes,
        },
        "sequence_metrics": sequence_metrics,
    }


def evaluate_wall(record: Dict[str, Any], raw_response: str) -> Dict[str, Any]:
    parsed_nodes = extract_nodes(raw_response)
    gt_route = str(record["gt_longer_route"])
    gt_nodes = extract_nodes(gt_route)
    parsed_target = None
    if len(parsed_nodes) >= 2:
        if parsed_nodes[0] == record["anchor_node"]:
            parsed_target = parsed_nodes[1]
        else:
            parsed_target = parsed_nodes[-1]
    parsed_answer = (
        f"{record['anchor_node']} -> {parsed_target} is longer" if parsed_target is not None else raw_response.strip()
    )
    gt_target = gt_nodes[-1] if gt_nodes else None
    return {
        "parsed_answer": parsed_answer,
        "is_correct": parsed_target == gt_target,
        "judge_detail": {
            "expected_longer_target": gt_target,
            "parsed_target": parsed_target,
        },
        "sequence_metrics": None,
    }


def evaluate_euclidean_anchor(record: Dict[str, Any], raw_response: str) -> Dict[str, Any]:
    parsed_nodes = extract_nodes(raw_response)
    expected_node = str(record["gt_answer_node"]).upper()
    return {
        "parsed_answer": parsed_nodes[0] if parsed_nodes else raw_response.strip(),
        "is_correct": parsed_nodes == [expected_node],
        "judge_detail": {
            "expected_node": expected_node,
            "parsed_nodes": parsed_nodes,
        },
        "sequence_metrics": None,
    }


def evaluate_euclidean_pair(record: Dict[str, Any], raw_response: str) -> Dict[str, Any]:
    parsed_nodes = extract_nodes(raw_response)
    expected_pair = [str(node).upper() for node in record["gt_pair"]]
    is_correct = len(parsed_nodes) == 2 and set(parsed_nodes) == set(expected_pair)
    return {
        "parsed_answer": " -> ".join(parsed_nodes) if parsed_nodes else raw_response.strip(),
        "is_correct": is_correct,
        "judge_detail": {
            "expected_pair": expected_pair,
            "parsed_nodes": parsed_nodes,
            "match_mode": "unordered_pair_exact_match",
        },
        "sequence_metrics": None,
    }


def evaluate_capacity(record: Dict[str, Any], raw_response: str) -> Dict[str, Any]:
    parsed_nodes = extract_nodes(raw_response)
    required_nodes = {record["current_node"]}
    pickup_by_label = {}
    deliver_by_label = {}
    for task in record.get("selected_tasks", []):
        pickup_label = task["pickup"]
        deliver_label = task["deliver"]
        task_index = task["task_index"]
        required_nodes.add(pickup_label)
        required_nodes.add(deliver_label)
        pickup_by_label[pickup_label] = task_index
        deliver_by_label[deliver_label] = task_index

    violations: List[str] = []
    if not parsed_nodes:
        violations.append("empty_sequence")
    else:
        if parsed_nodes[0] != record["current_node"]:
            violations.append("must_start_from_current_node")
        if len(parsed_nodes) != len(required_nodes):
            violations.append("wrong_sequence_length")
        extra_nodes = [node for node in parsed_nodes if node not in required_nodes]
        if extra_nodes:
            violations.append(f"unexpected_nodes:{','.join(extra_nodes)}")
        duplicate_nodes = [node for node in parsed_nodes if parsed_nodes.count(node) > 1]
        if duplicate_nodes:
            violations.append(f"duplicate_nodes:{','.join(sorted(set(duplicate_nodes)))}")
        if set(parsed_nodes) != required_nodes:
            missing_nodes = sorted(required_nodes - set(parsed_nodes))
            if missing_nodes:
                violations.append(f"missing_nodes:{','.join(missing_nodes)}")

    carrying = set()
    picked = set()
    delivered = set()
    capacity = int(record["capacity"])
    if not violations:
        for label in parsed_nodes[1:]:
            if label in pickup_by_label:
                task_index = pickup_by_label[label]
                if task_index in picked:
                    violations.append(f"duplicate_pickup:T{task_index}")
                    break
                if len(carrying) >= capacity:
                    violations.append("capacity_overflow")
                    break
                picked.add(task_index)
                carrying.add(task_index)
                continue
            if label in deliver_by_label:
                task_index = deliver_by_label[label]
                if task_index not in picked:
                    violations.append(f"deliver_before_pickup:T{task_index}")
                    break
                if task_index in delivered:
                    violations.append(f"duplicate_delivery:T{task_index}")
                    break
                if task_index not in carrying:
                    violations.append(f"deliver_without_carry:T{task_index}")
                    break
                carrying.remove(task_index)
                delivered.add(task_index)
                continue
            violations.append(f"unexpected_label:{label}")
            break

    total_distance_m = None
    relative_error = None
    max_acceptable_distance_m = float(
        record.get(
            "max_acceptable_distance_m",
            float(record["gt_total_distance_m"]) * (1.0 + float(record.get("distance_tolerance_ratio", 0.10))),
        )
    )
    tolerance_ratio = float(record.get("distance_tolerance_ratio", 0.10))
    if not violations and parsed_nodes:
        evaluator = SCENE_EVALUATOR_CACHE.get(record["scene"])
        if evaluator is None:
            evaluator = SceneMatrixEvaluator(scene=record["scene"])
            SCENE_EVALUATOR_CACHE[record["scene"]] = evaluator
        coords = record["node_coordinates"]
        try:
            total_distance_m = 0.0
            for src_label, dst_label in zip(parsed_nodes, parsed_nodes[1:]):
                total_distance_m += float(
                    evaluator.distance_between(coords[src_label], coords[dst_label])["total_distance_m"]
                )
        except Exception as exc:
            violations.append(f"distance_eval_failed:{exc}")
            total_distance_m = None

    optimal_distance_m = float(record["gt_total_distance_m"])
    if total_distance_m is not None and optimal_distance_m > 0:
        relative_error = (total_distance_m - optimal_distance_m) / optimal_distance_m

    is_correct = (
        not violations
        and total_distance_m is not None
        and total_distance_m <= max_acceptable_distance_m + 1e-6
    )
    return {
        "parsed_answer": " -> ".join(parsed_nodes) if parsed_nodes else raw_response.strip(),
        "is_correct": bool(is_correct),
        "judge_detail": {
            "parsed_nodes": parsed_nodes,
            "expected_optimal_sequence": list(record["gt_sequence"]),
            "expected_optimal_distance_m": optimal_distance_m,
            "distance_tolerance_ratio": tolerance_ratio,
            "max_acceptable_distance_m": max_acceptable_distance_m,
            "parsed_total_distance_m": round(total_distance_m, 6) if total_distance_m is not None else None,
            "relative_error": round(relative_error, 6) if relative_error is not None else None,
            "acceptable_sequence_count": record.get("acceptable_sequence_count"),
            "match_mode": "distance_within_relative_tolerance",
            "violations": violations,
        },
        "sequence_metrics": None,
    }


def evaluate_record(record: Dict[str, Any], raw_response: str) -> Dict[str, Any]:
    question_type = record["question_type"]
    if question_type == "absolute_axis_sorting":
        return evaluate_absolute(record, raw_response)
    if question_type == "visual_distance_vs_topological_distance":
        return evaluate_wall(record, raw_response)
    if question_type == "euclidean_anchor_extreme_choice":
        return evaluate_euclidean_anchor(record, raw_response)
    if question_type == "euclidean_pair_extreme_choice":
        return evaluate_euclidean_pair(record, raw_response)
    if question_type == "capacity_interleaving_optimization":
        return evaluate_capacity(record, raw_response)
    return {
        "parsed_answer": raw_response.strip(),
        "is_correct": False,
        "judge_detail": {"reason": f"unsupported question_type={question_type}"},
        "sequence_metrics": None,
    }


class EmptyModelResponse(RuntimeError):
    """Gateway returned a zero-length body; retryable, not a model answer."""


def call_model(
    client: OpenAI,
    model: str,
    record: Dict[str, Any],
    temperature: float,
    timeout: float,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=build_messages(record),
        timeout=timeout,
        **temperature_kwargs(model, temperature),
    )
    content = response.choices[0].message.content or ""
    # The gateway intermittently returns finish_reason=stop with a zero-length
    # body. Returning it would be judged as a wrong answer, charging a transport
    # glitch to the model's accuracy; raise so the caller's retry loop runs.
    if not content.strip():
        raise EmptyModelResponse(f"empty content for {record.get('question_id')}")
    return content


def run_single_record(
    index: int,
    total: int,
    record: Dict[str, Any],
    api_base: str,
    api_key: str,
    model: str,
    temperature: float,
    timeout: float,
    max_attempts: int,
    retry_delay: float,
    sleep: float,
) -> Dict[str, Any]:
    print(f"[{index}/{total}] {record['question_id']}")
    started_at = time.time()
    raw_response = ""
    error = None
    attempts_used = 0
    judged: Dict[str, Any] = {
        "parsed_answer": None,
        "is_correct": False,
        "judge_detail": {},
        "sequence_metrics": None,
    }

    client = OpenAI(base_url=api_base, api_key=api_key, max_retries=3)

    for attempt in range(1, max_attempts + 1):
        attempts_used = attempt
        try:
            raw_response = call_model(
                client=client,
                model=model,
                record=record,
                temperature=temperature,
                timeout=timeout,
            )
            judged = evaluate_record(record, raw_response)
            error = None
            break
        except Exception as exc:
            error = str(exc)
            judged = {
                "parsed_answer": None,
                "is_correct": False,
                "judge_detail": {"error": error, "attempt": attempt},
                "sequence_metrics": None,
            }
            if attempt < max_attempts:
                backoff = retry_delay * (2 ** (attempt - 1))
                print(f"  retry {attempt}/{max_attempts - 1} for {record['question_id']} after {backoff:.1f}s: {error}")
                time.sleep(backoff)

    if sleep > 0:
        time.sleep(sleep)

    return {
        "index": index,
        "question_id": record["question_id"],
        "scene": record["scene"],
        "case_id": record["case_id"],
        "question_type": record["question_type"],
        "dataset_path": record["_dataset_path"],
        "dataset_family": record["_dataset_family"],
        "image_path": record["image_path"],
        "question": record["question"],
        "gt_answer": record.get("gt_answer"),
        "raw_response": raw_response,
        "parsed_answer": judged["parsed_answer"],
        "is_correct": bool(judged["is_correct"]),
        "judge_detail": judged["judge_detail"],
        "sequence_metrics": judged.get("sequence_metrics"),
        "latency_s": round(time.time() - started_at, 3),
        "error": error,
        "attempts_used": attempts_used,
    }


def _compact_name(parts: Sequence[str], fallback: str) -> str:
    cleaned = [
        re.sub(r"[^A-Za-z0-9._-]+", "_", part).strip("._-")
        for part in parts
        if part
    ]
    cleaned = [part for part in cleaned if part]
    return "-".join(cleaned) if cleaned else fallback


def make_output_path(model: str, vqa_set_args: Sequence[str], scene_filter: Sequence[str] | None) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
    dataset_name = _compact_name(
        [Path(item).stem if item not in DATASET_PRESETS else item for item in vqa_set_args],
        fallback="all-vqa",
    )
    scene_name = _compact_name(scene_filter or [], fallback="all-scenes")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / "outputs" / "vqa"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"vqa_run_{safe_model}_{dataset_name}_{scene_name}_{timestamp}.json"


def write_progress(
    output_path: Path,
    args: argparse.Namespace,
    dataset_paths: Sequence[Path],
    results: Sequence[Dict[str, Any]],
    planned_total: int,
) -> Dict[str, Any]:
    ordered_results = sorted(results, key=lambda item: item["index"])
    correct_count = sum(1 for item in ordered_results if item["is_correct"])
    accuracy_by_type = {
        qtype: (
            sum(1 for item in ordered_results if item["question_type"] == qtype and item["is_correct"])
            / sum(1 for item in ordered_results if item["question_type"] == qtype)
        )
        for qtype in sorted({item["question_type"] for item in ordered_results})
    }

    sequence_items = [item for item in ordered_results if item.get("sequence_metrics") is not None]
    sequence_metrics_by_type: Dict[str, Dict[str, Any]] = {}
    for qtype in sorted({item["question_type"] for item in sequence_items}):
        metrics_list = [item["sequence_metrics"] for item in sequence_items if item["question_type"] == qtype]
        complete_match_count = sum(1 for metric in metrics_list if metric["complete_sequence_match"])
        sequence_metrics_by_type[qtype] = {
            "count": len(metrics_list),
            "complete_match_count": complete_match_count,
            "complete_match_rate": complete_match_count / len(metrics_list) if metrics_list else 0.0,
            "avg_lcs_length": sum(metric["lcs_length"] for metric in metrics_list) / len(metrics_list) if metrics_list else 0.0,
            "avg_lcs_recall": sum(metric["lcs_recall"] for metric in metrics_list) / len(metrics_list) if metrics_list else 0.0,
            "avg_lcs_f1": sum(metric["lcs_f1"] for metric in metrics_list) / len(metrics_list) if metrics_list else 0.0,
        }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_base": args.api_base,
        "model": args.model,
        "scene_filter": args.scene,
        "parallelism": args.parallelism,
        "max_attempts": args.max_attempts,
        "retry_delay": args.retry_delay,
        "dataset_paths": [str(path) for path in dataset_paths],
        "planned_total_questions": planned_total,
        "completed_questions": len(ordered_results),
        "total_questions": len(ordered_results),
        "correct_count": correct_count,
        "accuracy": (correct_count / len(ordered_results)) if ordered_results else 0.0,
        "error_count": sum(1 for item in ordered_results if item["error"] is not None),
        "question_counts_by_type": {
            qtype: sum(1 for item in ordered_results if item["question_type"] == qtype)
            for qtype in sorted({item["question_type"] for item in ordered_results})
        },
        "accuracy_by_type": accuracy_by_type,
        "evaluation_policy_by_type": {
            qtype: EVALUATION_POLICY_BY_TYPE.get(qtype, {"primary": "exact_match", "auxiliary": []})
            for qtype in sorted({item["question_type"] for item in ordered_results})
        },
        "sequence_metrics_by_type": sequence_metrics_by_type,
        "in_progress": len(ordered_results) < planned_total,
    }

    payload = {"summary": summary, "results": ordered_results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(output_path)
    return summary


def main() -> None:
    args = parse_args()
    # Use the user-configured OpenAI-compatible endpoint.
    if args.api_key:
        routed_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        routed_key = args.api_key
    else:
        routed_base, routed_key = resolve_endpoint(args.model)
    args.api_base = args.api_base or routed_base
    api_key = args.api_key or routed_key
    if not api_key:
        raise ValueError("请通过 --api-key 或 OPENAI_API_KEY 提供 API key")

    question_id_filter = set(args.question_id) if args.question_id else None
    scene_filter = set(args.scene) if args.scene else None
    dataset_paths = resolve_dataset_paths(args.vqa_set)

    records: List[Dict[str, Any]] = []
    for dataset_path in dataset_paths:
        records.extend(load_dataset(dataset_path, question_id_filter, scene_filter, args.limit_per_set))

    if args.max_questions is not None:
        records = records[: args.max_questions]

    output_path = (
        args.output.resolve()
        if args.output is not None
        else make_output_path(args.model, args.vqa_set, args.scene)
    )
    planned_total = len(records)

    if args.parallelism < 1:
        raise ValueError("--parallelism 必须至少为 1")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts 必须至少为 1")

    results: List[Dict[str, Any]] = []
    write_progress(output_path, args, dataset_paths, results, planned_total)
    total = len(records)

    if args.parallelism == 1:
        for idx, record in enumerate(records, start=1):
            result = run_single_record(
                index=idx,
                total=total,
                record=record,
                api_base=args.api_base,
                api_key=api_key,
                model=args.model,
                temperature=args.temperature,
                timeout=args.timeout,
                max_attempts=args.max_attempts,
                retry_delay=args.retry_delay,
                sleep=args.sleep,
            )
            results.append(result)
            write_progress(output_path, args, dataset_paths, results, planned_total)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallelism) as executor:
            futures = [
                executor.submit(
                    run_single_record,
                    idx,
                    total,
                    record,
                    args.api_base,
                    api_key,
                    args.model,
                    args.temperature,
                    args.timeout,
                    args.max_attempts,
                    args.retry_delay,
                    args.sleep,
                )
                for idx, record in enumerate(records, start=1)
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                write_progress(output_path, args, dataset_paths, results, planned_total)

    summary = write_progress(output_path, args, dataset_paths, results, planned_total)
    print(f"wrote {len(results)} results to {output_path}")
    print(f"accuracy={summary['accuracy']:.4f} ({summary['correct_count']}/{len(results)})")


if __name__ == "__main__":
    main()
