#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib
import json
import os
import sys
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from node_sequence import build_node_mapping, node_sequence_to_actions
from planner import get_planner_class
from scene_config import SCENE_CONFIGS, get_scene_config, infer_scene_from_task_dir, load_task_json


DEFAULT_SCENE = "indoor"
TASK_DIR = get_scene_config(DEFAULT_SCENE).default_task_dir()

PLANNER_REGISTRY = {
    "meta": {"class": get_planner_class("meta"), "use_image": True, "label": "ActionPlanner (图片+坐标)"},
    "text": {"class": get_planner_class("text"), "use_image": False, "label": "TextPlanner (仅文本)"},
    "image": {"class": get_planner_class("image"), "use_image": True, "label": "ImagePlanner (纯图像)"},
}


def list_sample_ids(task_dir: str) -> list[str]:
    sample_ids = []
    for name in sorted(os.listdir(task_dir)):
        if not name.endswith(".json"):
            continue
        if name.endswith("_task_points.json") or "_task_points_" in name:
            continue
        sample_ids.append(os.path.splitext(name)[0])
    return sample_ids


def planning_output_path(task_dir: str, sample_id: str, model: str, planner_type: str, refined: bool = False) -> str:
    suffix = f"{model}_{planner_type}"
    if refined:
        suffix = f"{suffix}_refined"
    return os.path.join(task_dir, "planning_result", f"planning_result_{sample_id}_{suffix}.json")


def find_image_for_sample(sample_id: str, image_dir: str) -> str:
    image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".webp"]
    for ext in image_extensions:
        image_path = os.path.join(image_dir, f"{sample_id}{ext}")
        if os.path.exists(image_path):
            return image_path
    raise FileNotFoundError(f"未找到样本 {sample_id} 对应的图片文件")


def resolve_scene(scene: Optional[str], task_dir: Optional[str]) -> str:
    if scene:
        return scene
    inferred = infer_scene_from_task_dir(task_dir)
    return inferred or DEFAULT_SCENE


def resolve_task_bundle(task_dir: Optional[str] = None, sample_id: Optional[str] = None, scene: str = DEFAULT_SCENE) -> Dict[str, Optional[str]]:
    config = get_scene_config(scene)
    task_dir = task_dir or config.default_task_dir()

    if sample_id:
        json_path = os.path.join(task_dir, f"{sample_id}.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"未找到样本 {sample_id} 对应的任务文件: {json_path}")
        image_path = None
        try:
            image_path = find_image_for_sample(sample_id, config.default_image_dir())
        except FileNotFoundError:
            image_path = None
        return {
            "scene": scene,
            "sample_id": sample_id,
            "task_dir": task_dir,
            "json_path": json_path,
            "image_path": image_path,
        }

    sample_ids = list_sample_ids(task_dir)
    if not sample_ids:
        raise FileNotFoundError(f"在 {task_dir} 中未找到 JSON 文件")
    sample_id = sample_ids[0]
    json_path = os.path.join(task_dir, f"{sample_id}.json")

    image_path = None
    try:
        image_path = find_image_for_sample(sample_id, config.default_image_dir())
    except FileNotFoundError:
        image_path = None

    return {
        "scene": scene,
        "sample_id": sample_id,
        "task_dir": task_dir,
        "json_path": json_path,
        "image_path": image_path,
    }


def load_task_context(task_dir: str, use_image: bool = True, sample_id: str | None = None, scene: str = DEFAULT_SCENE) -> dict:
    bundle = resolve_task_bundle(task_dir=task_dir, sample_id=sample_id, scene=scene)
    json_path = bundle["json_path"]
    print(f"读取任务文件: {json_path}")

    context = load_task_json(json_path, scene)
    context.update(
        {
            "last_interrupt": False,
            "scene": scene,
            "sample_id": bundle["sample_id"],
            "task_count": len(context.get("tasks", [])),
        }
    )

    if use_image:
        image_path = bundle.get("image_path")
        if image_path:
            print(f"读取地图图片: {image_path}")
            context["image_path"] = image_path
        else:
            print("警告: 未找到图片文件，将仅使用坐标信息")
            context["image_path"] = None
    else:
        print("模式: 仅使用坐标信息（不加载图片）")
        context["image_path"] = None

    return context


def print_context_summary(context: dict) -> None:
    print("\n" + "=" * 60)
    print("任务上下文摘要")
    print("=" * 60)
    print(f"场景: {context.get('scene', 'unknown')}")
    print(f"当前时间: {context['now']:.2f} s")
    print(f"当前位置: ({context['pos']['x']:.2f}, {context['pos']['y']:.2f})")
    print(f"速度: {context['speed']:.2f} m/s")
    print(f"最大载货量: {context['max_capacity']}")
    print(f"已携带任务数: {len(context['carried'])}")
    print(f"可用任务数: {len(context['tasks'])}")
    print(f"地图图片: {context.get('image_path', '未提供')}")

    if context["tasks"]:
        print("\n任务列表:")
        for idx, task in enumerate(context["tasks"], start=1):
            print(f"  {idx}. {task['id']} (Type {task.get('type')})")
            print(f"     取货: ({task['pickup']['x']:.2f}, {task['pickup']['y']:.2f})")
            print(f"     送货: ({task['deliver']['x']:.2f}, {task['deliver']['y']:.2f})")
            print(f"     reward={task.get('reward')} deadline={task.get('end_time')}")

    print("=" * 60 + "\n")


def print_node_sequence(node_sequence: list, actions: list) -> None:
    print("=" * 60)
    print("生成的节点序列")
    print("=" * 60)
    if not node_sequence:
        print("未生成任何节点（可能规划失败）")
        return

    print(" -> ".join(node_sequence))
    print("-" * 60)
    print("展开后的执行动作")
    for idx, action in enumerate(actions, start=1):
        action_type = action.get("action", "unknown")
        if action_type == "move":
            target = action.get("to", {})
            print(f"{idx}. MOVE -> ({target.get('x', 0):.2f}, {target.get('y', 0):.2f})")
        elif action_type in {"pickup", "deliver"}:
            print(f"{idx}. {action_type.upper()} -> {action.get('task_id', 'unknown')}")
        else:
            print(f"{idx}. {action}")
    print("=" * 60 + "\n")


def save_planning_result(
    node_sequence: list,
    actions: list,
    output_dir: str | None = None,
    model_name: str | None = None,
    metadata: dict | None = None,
    sample_id: str | None = None,
) -> str:
    if output_dir is None:
        output_dir = TASK_DIR

    output_dir = os.path.join(output_dir, "planning_result")
    os.makedirs(output_dir, exist_ok=True)

    file_stem = "planning_result"
    if sample_id:
        file_stem = f"{file_stem}_{sample_id}"

    filename = f"{file_stem}_{model_name}.json" if model_name else f"{file_stem}.json"
    output_file = os.path.join(output_dir, filename)
    # node_sequence is what the model actually produced and what gets graded;
    # actions is the deterministic expansion, kept so downstream execution
    # (publish_waypoints.py) can consume the file unchanged.
    payload = {"node_sequence": node_sequence, "actions": actions}
    if metadata:
        payload["metadata"] = metadata

    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"\n规划结果已保存到: {output_file}")
    return output_file


def load_execution_result(nav_result_path: str) -> dict:
    if not nav_result_path:
        return {}
    if not os.path.exists(nav_result_path):
        print(f"警告: 未找到导航结果文件: {nav_result_path}")
        return {}

    try:
        with open(nav_result_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        print(f"警告: 读取导航结果文件失败: {exc}")
        return {}

    summary = {
        "model": data.get("model"),
        "planner_type": data.get("planner_type"),
        "planning_result_file": data.get("planning_result_file"),
        "total_waypoints": data.get("total_waypoints"),
        "total_time_seconds": data.get("total_time_seconds"),
        "waypoints": data.get("waypoints", []),
    }
    print(f"已加载导航执行结果摘要，共 {len(summary.get('waypoints', []))} 个路点")
    return summary


def maybe_evaluate(
    *,
    scene: str,
    task_file: str,
    planning_json: str,
    output_path: str,
    default_reward: float,
) -> Optional[dict]:
    evaluator_module_name = get_scene_config(scene).evaluator_module
    if not evaluator_module_name:
        print(f"警告: scene={scene} 暂未配置统一评测器，跳过 evaluate")
        return None

    evaluator_module = importlib.import_module(evaluator_module_name)
    evaluate_files = getattr(evaluator_module, "evaluate_files")
    return evaluate_files(
        task_json=task_file,
        planning_json=planning_json,
        output_path=output_path,
        default_reward=default_reward,
        scene=scene,
    )


def main(
    planner_type: str = "meta",
    model: str = "gpt-4o",
    task_dir: str | None = None,
    nav_result_path: str | None = None,
    sample_id: str | None = None,
    evaluate: bool = False,
    default_reward: float = 1.0,
    scene: str | None = None,
):
    scene = resolve_scene(scene, task_dir)
    task_dir = task_dir or get_scene_config(scene).default_task_dir()

    if planner_type not in PLANNER_REGISTRY:
        raise ValueError(f"不支持的 planner_type: {planner_type}")

    planner_config = PLANNER_REGISTRY[planner_type]
    use_image = planner_config["use_image"]
    planner_name = planner_config["label"]

    print(f"{planner_name} - scene={scene}")
    print("=" * 60)

    try:
        context = load_task_context(task_dir, use_image=use_image, sample_id=sample_id, scene=scene)
        if nav_result_path:
            prev_execution = load_execution_result(nav_result_path)
            if prev_execution:
                context["prev_execution"] = prev_execution
                print("已将第一轮执行结果加入到规划上下文中，将提示模型进行路线优化。")

        print_context_summary(context)

        print(f"创建 {planner_name} 实例...")
        planner = planner_config["class"](model=model, scene=scene)

        print("正在调用 GPT 生成规划（这可能需要一些时间）...")
        node_sequence = planner.plan(context)
        # The same deterministic expansion the evaluator applies, materialized
        # here only so the saved file stays executable by the ROS publisher.
        actions, unresolved = node_sequence_to_actions(
            node_sequence,
            build_node_mapping(context),
            context["pos"],
        )
        if unresolved:
            print(f"警告: {len(unresolved)} 个节点无法映射到任务，将在评测中记为违规: "
                  f"{', '.join(label for label, _ in unresolved)}")
        print_node_sequence(node_sequence, actions)

        if node_sequence:
            model_suffix = f"{planner.model}_{planner_type}"
            refined = bool(nav_result_path)
            if refined:
                model_suffix = f"{model_suffix}_refined"
            planning_path = save_planning_result(
                node_sequence,
                actions,
                task_dir,
                model_suffix,
                metadata={
                    "model": planner.model,
                    "planner_type": planner_type,
                    "scene": context.get("scene"),
                    "sample_id": context.get("sample_id"),
                    "task_file": context.get("task_file"),
                    "image_path": context.get("image_path"),
                },
                sample_id=context.get("sample_id"),
            )

            if evaluate:
                eval_output_path = os.path.join(
                    task_dir,
                    "planning_result",
                    f"evaluation_{context.get('sample_id')}_{model_suffix}.json",
                )
                eval_result = maybe_evaluate(
                    scene=scene,
                    task_file=context["task_file"],
                    planning_json=planning_path,
                    output_path=eval_output_path,
                    default_reward=default_reward,
                )
                if eval_result is not None:
                    print("=" * 60)
                    print("矩阵评测结果")
                    print("=" * 60)
                    print(json.dumps(eval_result, ensure_ascii=False, indent=2))
                    print("=" * 60)
        else:
            print("\n警告: 未生成任何节点，规划可能失败")

        return node_sequence
    except FileNotFoundError as exc:
        print(f"错误: {exc}")
        return None
    except Exception as exc:
        print(f"发生错误: {exc}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="通用规划器入口，支持 meta/text/image 三种模式和多类场景。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--type", "-t", choices=["meta", "text", "image"], default="meta", help="规划器类型")
    parser.add_argument("--model", "-m", type=str, default="gpt-5", help="模型名称")
    parser.add_argument("--scene", choices=sorted(SCENE_CONFIGS), default=None, help="场景名称")
    parser.add_argument("--task-dir", "-d", type=str, default=None, help=f"任务目录路径 (默认: {TASK_DIR})")
    parser.add_argument("--sample-id", "-s", type=str, default=None, help="指定样本 ID，例如 n5-1")
    parser.add_argument("--nav-result", "-r", type=str, default=None, help="第一轮导航执行结果 JSON")
    parser.add_argument("--evaluate", action="store_true", help="规划完成后执行统一评测")
    parser.add_argument("--default-reward", type=float, default=1.0, help="reward 缺失时的默认奖励")
    parser.add_argument("task_dir_positional", nargs="?", type=str, default=None, help="任务目录路径（向后兼容）")
    args = parser.parse_args()

    task_dir = args.task_dir_positional or args.task_dir
    if task_dir:
        print(f"使用指定的任务目录: {task_dir}")

    node_sequence = main(
        planner_type=args.type,
        model=args.model,
        task_dir=task_dir,
        nav_result_path=args.nav_result,
        sample_id=args.sample_id,
        evaluate=args.evaluate,
        default_reward=args.default_reward,
        scene=args.scene,
    )

    if node_sequence:
        print(f"\n规划完成！共生成 {len(node_sequence)} 个节点。")
    else:
        print("\n规划失败，请检查错误信息。")
