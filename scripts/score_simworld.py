"""Score external node-sequence predictions without making API calls."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simworld" / "nav_planner"))
from node_sequence import canonicalize_node_sequence
from scene_config import get_scene_config, load_task_json
from scene_eval import SceneMatrixEvaluator
from grid import MODALITIES, available_scenes, group_of, keep_sample


def expected_cells(scenes, modalities):
    cells = set()
    for scene in scenes:
        for path in get_scene_config(scene).task_dir.glob("n*-*.json"):
            if keep_sample(scene, path.stem):
                for modality in modalities:
                    cells.add((scene, path.stem, modality))
    return cells


def summarize(rows):
    if not rows:
        return {"samples": 0}
    nc = [r["normalized_cost"] for r in rows if r.get("normalized_cost") is not None]
    return {
        "samples": len(rows),
        "F": sum(bool(r.get("feasible")) for r in rows) / len(rows),
        "C": statistics.mean(r.get("completion_rate", 0.0) for r in rows),
        "SR": sum(bool(r.get("feasible")) and r.get("completion_rate", 0.0) == 1.0
                  for r in rows) / len(rows),
        "NC": statistics.mean(nc) if nc else None,
        "nc_samples": len(nc),
        "mean_violations": statistics.mean(len(r.get("violations", [])) for r in rows),
    }


def read_predictions(path):
    """Keep the last checkpoint for a cell, matching the benchmark protocol."""
    cells = {}
    with path.open(encoding="utf-8") as stream:
        for i, line in enumerate(stream, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["scene"], r["sample_id"], r.get("planner_type", "meta"),
                   r.get("model", "custom"))
            cells[key] = r
    return cells


def score_predictions(predictions, scenes, modalities, allow_partial=False, reference_file=None):
    expected = expected_cells(scenes, modalities)
    selected = {k: r for k, r in predictions.items()
                if k[:3] in expected}
    if not selected:
        raise ValueError("No predictions match the selected benchmark grid.")
    models = sorted({k[3] for k in selected})
    coverage = {}
    for model in models:
        supplied = {k[:3] for k in selected if k[3] == model}
        coverage[model] = {"expected": len(expected), "provided": len(supplied),
                           "complete": supplied == expected}
        if supplied != expected and not allow_partial:
            raise ValueError(f"{model}: {len(supplied)}/{len(expected)} cells; "
                             "supply the complete grid or pass --allow-partial.")
    references = json.loads((ROOT / "data/simworld/reference/subset_reference.json").read_text())
    exact_keys = set()
    if reference_file:
        extra = json.loads(Path(reference_file).read_text())
        if extra.get("method") != "exact_subset_dp":
            raise ValueError("Expected an exact_subset_dp reference file.")
        for key, cost in extra["costs"].items():
            if not isinstance(cost, (int, float)) or not 0 <= cost < float("inf"):
                raise ValueError(f"Invalid reference cost for {key}")
        references.update(extra["costs"])
        exact_keys = set(extra["costs"])
    evaluators = {}
    rows = []
    for (scene, sample_id, modality, model), prediction in sorted(selected.items()):
        base = {"scene": scene, "sample_id": sample_id,
                "planner_type": modality, "model": model}
        nodes = prediction.get("node_sequence", prediction.get("sequence", []))
        status = prediction.get("status", "ok")
        if status == "ok" and nodes:
            if not isinstance(nodes, list):
                raise ValueError(f"Node sequence must be a list: {scene}/{sample_id}/{model}")
            nodes = canonicalize_node_sequence(nodes)
        if status != "ok" or not nodes:
            base.update(status=status if status != "ok" else "empty_plan",
                        feasible=False, completion_rate=0.0, normalized_cost=None,
                        violations=["no_evaluable_plan"])
        else:
            context = load_task_json(str(get_scene_config(scene).task_dir / f"{sample_id}.json"), scene)
            if scene not in evaluators:
                evaluators[scene] = SceneMatrixEvaluator(scene)
            result = evaluators[scene].evaluate_node_sequence(context, canonicalize_node_sequence(nodes))
            # Denominator depends on exactly the validly delivered task subset.
            ids = result["delivered_ids"]
            ref_key = f"{scene}|{sample_id}|{','.join(ids)}"
            reference_cost = references.get(ref_key) if ids else None
            base.update(result)
            base.update(status="ok", reference_distance_m=reference_cost,
                        reference_method="exact_subset_dp" if ref_key in exact_keys else
                        ("released_reference" if reference_cost is not None else None),
                        normalized_cost=result["total_distance_m"] / reference_cost
                        if reference_cost else None)
        rows.append(base)
    buckets = defaultdict(list)
    task_buckets = defaultdict(list)
    for row in rows:
        for group in ("All", group_of(row["scene"])):
            buckets[(row["model"], row["planner_type"], group)].append(row)
        n = int(row["sample_id"].split("-")[0][1:])
        task_buckets[(row["model"], row["planner_type"], n)].append(row)
    scene_summary = [{"model": m, "modality": mo, "scene_group": g, **summarize(rs)}
                     for (m, mo, g), rs in sorted(buckets.items())]
    scaling = [{"model": m, "modality": mo, "task_count": n, **summarize(rs)}
               for (m, mo, n), rs in sorted(task_buckets.items())]
    averages = []
    for model in models:
        for modality in modalities:
            per = [s for s in scaling if s["model"] == model and s["modality"] == modality]
            if not per:
                continue
            entry = {"model": model, "modality": modality, "task_counts": len(per)}
            for metric in ("F", "C", "SR", "NC"):
                values = [s[metric] for s in per if s[metric] is not None]
                entry[metric] = statistics.mean(values) if values else None
            averages.append(entry)
    return {"coverage": coverage, "excluded_records": len(predictions) - len(selected),
            "nc_reference": "released_routes_with_explicit_exact_overrides" if reference_file else
            "released_reference_routes_not_all_certified_optimal",
            "scene_summary": scene_summary, "task_count_summary": scaling,
            "scaling_average": averages, "records": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/simworld/score.json")
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--reference-file", type=Path, help="Exact reference overrides from build_nc_references.py")
    args = parser.parse_args()
    result = score_predictions(read_predictions(args.predictions),
                               args.scenes or available_scenes(), args.modalities, args.allow_partial,
                               args.reference_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"coverage": result["coverage"], "scaling_average": result["scaling_average"]}, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
