"""Regression checks for the public scoring contract; no API calls."""

import json
import itertools
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "simworld/nav_planner"))
from scene_config import get_scene_config, load_task_json
from scene_eval import SceneMatrixEvaluator
from node_sequence import parse_node_sequence
from score_simworld import expected_cells, score_predictions
from priority_rule import has_priority_violation
from model_endpoints import resolve
from build_nc_references import exact_subset_cost
from run_simworld import load_context
from planner import get_planner_class
from prompt_registry import get_planner_prompt
from scene_config import SCENE_CONFIGS


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.context = load_task_json(str(get_scene_config("typical").task_dir / "n5-1.json"), "typical")
        self.evaluator = SceneMatrixEvaluator("typical")

    def test_complete_plan_with_extra_invalid_delivery_preserves_completion(self):
        nodes = ["START"] + [label for i in range(1, 6) for label in (f"P{i}", f"D{i}")]
        valid = self.evaluator.evaluate_node_sequence(self.context, nodes)
        invalid = self.evaluator.evaluate_node_sequence(self.context, ["D1"] + nodes)
        self.assertTrue(valid["feasible"])
        self.assertFalse(invalid["feasible"])
        self.assertEqual(invalid["completion_rate"], 1.0)
        self.assertEqual(invalid["delivered_ids"], valid["delivered_ids"])

    def test_unknown_label_is_a_violation(self):
        result = self.evaluator.evaluate_node_sequence(self.context, ["START", "P99"])
        self.assertFalse(result["feasible"])
        self.assertEqual(result["completion_rate"], 0)

    def test_partial_legal_plan_is_feasible(self):
        result = self.evaluator.evaluate_node_sequence(self.context, ["START", "P1", "D1"])
        self.assertTrue(result["feasible"])
        self.assertEqual(result["completion_rate"], 0.2)

    def test_grid_and_missing_prediction_guard(self):
        self.assertEqual(len(expected_cells(list(__import__('scene_config').SCENE_CONFIGS), ["meta", "text", "image"])), 567)
        key = ("typical", "n5-1", "meta", "custom")
        prediction = {key: {"sequence": ["START", "P1", "D1"]}}
        with self.assertRaises(ValueError):
            score_predictions(prediction, ["typical"], ["meta"])

    def test_no_plan_scores_zero(self):
        key = ("typical", "n5-1", "meta", "custom")
        scored = score_predictions({key: {"sequence": []}}, ["typical"], ["meta"], True)
        row = scored["records"][0]
        self.assertFalse(row["feasible"])
        self.assertEqual(row["completion_rate"], 0)

    def test_non_node_entries_do_not_form_an_evaluable_plan(self):
        key = ("typical", "n5-1", "meta", "custom")
        scored = score_predictions({key: {"sequence": ["DONE"]}}, ["typical"], ["meta"], True)
        self.assertFalse(scored["records"][0]["feasible"])
        self.assertEqual(scored["records"][0]["status"], "empty_plan")

    def test_label_normalization_keeps_order(self):
        self.assertEqual(parse_node_sequence('{"sequence":["START","P 01","D_1"]}'), ["START", "P1", "D1"])

    def test_priority_rule_allows_pickups_but_not_non_c_delivery(self):
        logs = [{"type": "pickup", "id": "C"}, {"type": "pickup", "id": "A"}]
        types = {"A": "A", "C": "C"}
        self.assertFalse(has_priority_violation(logs, types))
        self.assertTrue(has_priority_violation(logs + [{"type": "deliver", "id": "A"}], types))

    def test_credentials_are_required_without_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                resolve("custom")

    def test_every_scene_modality_builds_messages_without_legacy_templates(self):
        for scene in SCENE_CONFIGS:
            for modality in ("meta", "text", "image"):
                with self.subTest(scene=scene, modality=modality):
                    planner_class = get_planner_class(modality)
                    planner = planner_class.__new__(planner_class)
                    planner.scene = scene
                    planner.prompt_template = get_planner_prompt(scene, modality)
                    context = load_context(scene, "n5-1", modality)
                    prompt = planner.build_prompt(context)
                    self.assertIn('"sequence"', prompt)
                    messages = planner.build_messages(prompt, context)
                    self.assertEqual(messages[0]["role"], "user")

    def test_exact_reference_matches_exhaustive_execution(self):
        context = dict(self.context, max_capacity=2)
        ids = [t["id"] for t in context["tasks"][:3]]
        labels = [label for i in range(1, 4) for label in (f"P{i}", f"D{i}")]
        minimum = float("inf")
        for order in itertools.permutations(labels):
            result = self.evaluator.evaluate_node_sequence(context, list(order))
            if result["feasible"] and set(result["delivered_ids"]) == set(ids):
                minimum = min(minimum, result["total_distance_m"])
        self.assertAlmostEqual(exact_subset_cost(context, ids, self.evaluator), minimum)

    def test_exact_reference_skips_move_at_coincident_coordinates(self):
        class OffsetEvaluator:
            def distance_between(self, a, b):
                return {"total_distance_m": abs(a["x"] - b["x"]) + 2.0}
        origin = {"x": 0, "y": 0}
        context = {"pos": origin, "max_capacity": 1, "tasks": [
            {"id": "A-0001", "pickup": origin, "deliver": {"x": 3, "y": 0}}]}
        self.assertEqual(exact_subset_cost(context, ["A-0001"], OffsetEvaluator()), 5.0)


if __name__ == "__main__":
    unittest.main()
