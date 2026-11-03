"""Verify the shipped datasets, image paths, navigation resources and hashes."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simworld/nav_planner"))
from scene_config import SCENE_CONFIGS, load_task_json
from scene_eval import SceneMatrixEvaluator
from grid import keep_sample


def main():
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for relative, expected in manifest["sha256"].items():
        path = ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Missing or changed data file: {relative}")
    count = 0
    for scene, config in SCENE_CONFIGS.items():
        evaluator = SceneMatrixEvaluator(scene)
        for path in config.task_dir.glob("n*-*.json"):
            context = load_task_json(str(path), scene)
            for task in context["tasks"]:
                evaluator.distance_between(context["pos"], task["pickup"])
                evaluator.distance_between(task["pickup"], task["deliver"])
            if keep_sample(scene, path.stem):
                count += 1
    assert count == 189, count
    question_count = 0
    for path in (ROOT / "data/vqa/task").glob("*.json"):
        for r in json.loads(path.read_text())["records"]:
            for key in ("image_path", "task_json_path", "points_file_path"):
                if r.get(key) and not (ROOT / r[key]).is_file():
                    raise FileNotFoundError(f"{r['question_id']}: {key}={r[key]}")
            question_count += 1
    assert question_count == 1665, question_count
    text_count = len(list((ROOT / "data/textworld/task").glob("*/*.json")))
    assert text_count == 200, text_count
    print(json.dumps({"scenes": len(SCENE_CONFIGS), "simworld_instances": count,
                      "simworld_cells_per_model": count * 3,
                      "vqa_questions": question_count, "textworld_instances": text_count,
                      "hashed_files": len(manifest["sha256"]), "status": "passed"}, indent=2))


if __name__ == "__main__":
    main()
