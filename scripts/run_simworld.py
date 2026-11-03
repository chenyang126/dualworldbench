from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List

from repo_paths import DATA_DIR, NAV_PLANNER_DIR, SCRIPT_DIR

sys.path.insert(0, str(NAV_PLANNER_DIR))

from node_sequence import build_node_mapping  # noqa: E402
from planner import get_planner_class  # noqa: E402
from run_planner import list_sample_ids, resolve_task_bundle  # noqa: E402
from scene_config import get_scene_config, load_task_json  # noqa: E402
from scene_eval import SceneMatrixEvaluator  # noqa: E402

sys.path.insert(0, str(SCRIPT_DIR))
from grid import MODELS, MODALITIES, available_scenes, select_samples  # noqa: E402

logging.basicConfig(level=logging.CRITICAL, format="%(levelname)s %(name)s: %(message)s")

FORMAT_VIOLATION_MARKERS = ("invalid_move", "off_graph", "without_arrival")

# Retry only transport failures. A model that answers unparseably is not retried:
# giving it extra attempts would inflate it relative to the reference protocol.
MAX_ATTEMPTS = 4
BACKOFF_S = (5.0, 15.0, 40.0)

_write_lock = threading.Lock()
_print_lock = threading.Lock()


def job_key(job: Dict[str, str]) -> str:
    return f"{job['scene']}|{job['sample_id']}|{job['modality']}|{job['model']}"


def build_jobs(scenes: List[str], models: List[str], modalities=MODALITIES) -> List[Dict[str, str]]:
    jobs: List[Dict[str, str]] = []
    for scene in scenes:
        ids = select_samples(scene, list_sample_ids(get_scene_config(scene).default_task_dir()))
        for sample_id in ids:
            for modality in modalities:
                for model in models:
                    jobs.append(
                        {
                            "scene": scene,
                            "sample_id": sample_id,
                            "modality": modality,
                            "model": model,
                        }
                    )
    # Interleave so that a partial run is still balanced across the grid rather
    # than a complete first scene and nothing else, and so concurrent workers do
    # not all hammer one scene's images at once.
    random.Random(20260804).shuffle(jobs)
    return jobs


def load_done(path: str) -> Dict[str, Dict[str, Any]]:
    """Read the checkpoint, keeping the last record for each cell."""
    done: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn final line from a hard kill
            key = f"{rec['scene']}|{rec['sample_id']}|{rec['planner_type']}|{rec['model']}"
            done[key] = rec
    return done


def load_context(scene: str, sample_id: str, modality: str) -> Dict[str, Any]:
    bundle = resolve_task_bundle(sample_id=sample_id, scene=scene)
    context = load_task_json(bundle["json_path"], scene)
    context.update(
        {
            "last_interrupt": False,
            "scene": scene,
            "sample_id": sample_id,
            "task_count": len(context.get("tasks", [])),
            "image_path": bundle.get("image_path") if modality != "text" else None,
        }
    )
    return context


def run_job(job: Dict[str, str], evaluators: Dict[str, SceneMatrixEvaluator]) -> Dict[str, Any]:
    scene, model, sample_id, modality = (
        job["scene"], job["model"], job["sample_id"], job["modality"]
    )
    record: Dict[str, Any] = {
        "scene": scene,
        "sample_id": sample_id,
        "planner_type": modality,
        "model": model,
        "status": "error",
    }
    started = time.time()
    try:
        context = load_context(scene, sample_id, modality)
        record["total_tasks_loaded"] = len(context.get("tasks", []))
        record["task_schema"] = (
            "point_pair" if "pickup_points" in context.get("raw_task_data", {}) else "task_list"
        )
        record["node_mapping_size"] = len(build_node_mapping(context))

        planner = get_planner_class(modality)(model=model, scene=scene)
        node_sequence: List[str] = []
        attempts = 0
        api_errors: List[str] = []
        for attempt in range(MAX_ATTEMPTS):
            attempts = attempt + 1
            node_sequence = planner.plan(context)
            if node_sequence:
                break
            err = getattr(planner, "last_error", None)
            if err is None:
                break  # call succeeded, model output unparseable -> do not retry
            api_errors.append(err)
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)])

        record["attempts"] = attempts
        record["node_sequence"] = node_sequence
        record["elapsed_s"] = round(time.time() - started, 2)
        if api_errors:
            record["api_errors"] = api_errors[-MAX_ATTEMPTS:]

        if not node_sequence:
            record["status"] = "api_error" if api_errors else "empty_plan"
            # Keep the body that failed to parse. Without it, telling a genuine
            # model failure apart from a gateway hiccup means grepping stdout of
            # a 48-worker run, where the lines are interleaved and unattributable.
            raw = getattr(planner, "last_raw", None)
            if raw is not None:
                record["raw_response"] = raw[:2000]
            return record

        result = evaluators[scene].evaluate_node_sequence(
            context, node_sequence, metadata={"model": model, "planner_type": modality}
        )
        record.update(
            {
                "status": "ok",
                "feasible": result["feasible"],
                "completed_tasks": result["completed_tasks"],
                "total_tasks": result["total_tasks"],
                "completion_rate": result["completion_rate"],
                "total_distance_m": result["total_distance_m"],
                "total_time_s": result.get("total_time_s"),
                "total_reward": result.get("total_reward"),
                "reward_rate": result.get("reward_rate"),
                "violations": result["violations"],
                "format_violations": [
                    v for v in result["violations"]
                    if any(m in v for m in FORMAT_VIOLATION_MARKERS)
                ],
                "num_actions": len(result.get("expanded_actions", [])),
            }
        )
        # expanded_actions is ~30x the size of everything else; the smoke run
        # already established the expansion is well formed, so keep only the
        # count here and the node sequence it came from.
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()[-1200:]
        record["elapsed_s"] = round(time.time() - started, 2)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--checkpoint", default=str(DATA_DIR / "full_records.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="仅跑前 N 个未完成作业（用于探测并发上限）")
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument("--dry-run", action="store_true", help="List job count without API calls")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")

    scenes = args.scenes or available_scenes()
    jobs = build_jobs(scenes, args.models, args.modalities)
    if args.dry_run:
        print(json.dumps({"jobs": len(jobs), "scenes": scenes, "models": args.models,
                          "modalities": args.modalities}, indent=2))
        return 0
    from model_endpoints import resolve
    for model in args.models:
        resolve(model)
    os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint)), exist_ok=True)
    done = load_done(args.checkpoint)
    # A previous api_error / error cell is worth retrying on resume; ok and
    # empty_plan are settled results.
    settled = {"ok", "empty_plan"}
    todo = [j for j in jobs if done.get(job_key(j), {}).get("status") not in settled]
    if args.limit:
        todo = todo[: args.limit]

    evaluators = {s: SceneMatrixEvaluator(scene=s) for s in scenes}
    print(
        f"场景 {len(scenes)}  模型 {len(args.models)}  模态 {len(args.modalities)}\n"
        f"总格子 {len(jobs)}，已完成 {len(jobs)-len([j for j in jobs if done.get(job_key(j),{}).get('status') not in settled])}，"
        f"本次要跑 {len(todo)}，并发 {args.workers}",
        flush=True,
    )
    if not todo:
        print("没有待跑作业。")
        return 0

    counts: Counter = Counter()
    start = time.time()
    handle = open(args.checkpoint, "a", encoding="utf-8")
    completed = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_job, j, evaluators): j for j in todo}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    record = future.result()
                except Exception as exc:  # never lose a cell to a worker crash
                    record = dict(
                        scene=job["scene"], sample_id=job["sample_id"],
                        planner_type=job["modality"], model=job["model"],
                        status="error", error=f"worker: {type(exc).__name__}: {exc}",
                    )
                with _write_lock:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                completed += 1
                counts[record["status"]] += 1
                if completed % 25 == 0 or record["status"] not in {"ok", "empty_plan"}:
                    rate = completed / max(1e-9, time.time() - start)
                    eta = (len(todo) - completed) / rate / 60 if rate else 0
                    with _print_lock:
                        print(
                            f"[{completed}/{len(todo)}] {dict(counts)} "
                            f"{rate*60:.1f}/min ETA {eta:.0f}min  "
                            f"last={record['scene']}/{record['sample_id']}/"
                            f"{record['planner_type']}/{record['model']}={record['status']}"
                            + (f" {str(record.get('api_errors') or record.get('error'))[:120]}"
                               if record["status"] not in {"ok", "empty_plan"} else ""),
                            flush=True,
                        )
    finally:
        handle.close()

    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    print(f"\n完成 {completed} 个，用时 {(time.time()-start)/60:.1f} min，{dict(counts)}")
    print(f"检查点: {args.checkpoint}  ({stamp})")

    # Consolidate into the reference run's flat all_records.json shape.
    final = load_done(args.checkpoint)
    all_records = sorted(
        final.values(),
        key=lambda r: (r["scene"], r["model"], r["sample_id"], r["planner_type"]),
    )
    # Derive the merged path from the checkpoint: a probe run pointed at a
    # scratch checkpoint must not overwrite the real run's merged records.
    stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    stem = stem[: -len("_records")] if stem.endswith("_records") else stem
    path = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), f"{stem}_all_records.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(all_records, fh, ensure_ascii=False, indent=2)
    print(f"合并记录: {path}  共 {len(all_records)} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
