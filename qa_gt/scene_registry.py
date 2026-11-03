from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Sequence


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
SIM_TASK_ROOT = REPO_ROOT / "data" / "simworld" / "task"
SIMWORLD_ROOT = SIM_TASK_ROOT
SCENE_CONFIG_PATH = REPO_ROOT / "simworld" / "nav_planner" / "scene_config.py"


def load_qa_scene_configs() -> Dict[str, Any]:
    spec = importlib.util.spec_from_file_location("qa_scene_config_local", SCENE_CONFIG_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 scene config: {SCENE_CONFIG_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return dict(module.SCENE_CONFIGS)


def is_active_simworld_scene(config: Any) -> bool:
    task_dir = Path(config.task_dir)
    try:
        task_dir.resolve().relative_to(SIMWORLD_ROOT.resolve())
    except ValueError:
        return False
    return task_dir.exists()


def select_qa_scene_configs(
    scene_configs: Dict[str, Any],
    requested_scenes: Sequence[str] | None = None,
) -> Dict[str, Any]:
    active_scene_configs = {
        name: config
        for name, config in sorted(scene_configs.items())
        if is_active_simworld_scene(config)
    }

    if requested_scenes is None:
        return active_scene_configs

    unknown_scenes = [scene for scene in requested_scenes if scene not in scene_configs]
    if unknown_scenes:
        raise ValueError(f"未知场景: {', '.join(unknown_scenes)}")

    inactive_scenes = [scene for scene in requested_scenes if scene not in active_scene_configs]
    if inactive_scenes:
        raise ValueError(
            "以下场景不属于当前可用的 simworld 数据集: "
            + ", ".join(inactive_scenes)
        )

    return {scene: active_scene_configs[scene] for scene in requested_scenes}
