"""Verify cross-suite adapters and CLI preserve the evaluation contract."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dualworld import simworld, textworld, vqa
from planner_algo import AlgoPlanner


class PublicInterfacesTests(unittest.TestCase):
    def test_standard_grid_and_prompt_for_each_modality(self):
        scenes = simworld.list_scenes()
        self.assertEqual(len(scenes), 13)
        self.assertEqual(sum(len(simworld.list_samples(s)) for s in scenes), 189)
        for modality in ("text", "image", "meta"):
            messages = simworld.build_messages("typical", "n5-1", modality)
            self.assertEqual(messages[0]["role"], "user")
            encoded = json.dumps(messages)
            self.assertEqual('"image_url"' in encoded, modality != "text")
        with self.assertRaises(ValueError):
            simworld.load_task("typical", "n5-4")

    def test_external_plan_score_requires_explicit_partial_coverage(self):
        path = ROOT / "examples/simworld_predictions.jsonl"
        with self.assertRaises(ValueError):
            simworld.score_file(path)
        result = simworld.score_file(path, allow_partial=True)
        self.assertEqual(len(result["records"]), 3)
        self.assertFalse(result["coverage"]["example-model"]["complete"])

    def test_textworld_execution_matches_known_episode_and_saves_scores(self):
        task = textworld.load_task("A_L1", 5, 1)
        with tempfile.TemporaryDirectory() as directory:
            scores = textworld.evaluate(AlgoPlanner(), task, directory)
            self.assertAlmostEqual(scores["R"], 92.7)
            self.assertFalse(scores["priority_violation"])
            self.assertEqual(json.loads((Path(directory) / "scores.json").read_text()), scores)
            self.assertTrue((Path(directory) / "result.json").is_file())
            self.assertTrue((Path(directory) / "plans.json").is_file())

    def test_vqa_deduplicates_sets_and_judges_released_example_answers(self):
        questions = vqa.load_questions(("l1", "l2", "l3", "absolute"))
        self.assertEqual(len(questions), 1665)
        by_id = {q["question_id"]: q for q in questions}
        for line in (ROOT / "examples/vqa_predictions.jsonl").read_text().splitlines():
            answer = json.loads(line)
            record = by_id[answer["question_id"]]
            self.assertTrue(vqa.judge(record, answer["answer"])["is_correct"])
            self.assertTrue(Path(record["image_path"]).is_file())

    def test_cli_works_outside_checkout_and_propagates_argument_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            entry = "import sys; sys.path.insert(0, sys.argv.pop(1)); from dualworld.__main__ import main; raise SystemExit(main())"
            command = [sys.executable, "-c", entry, str(ROOT)]
            result = subprocess.run(command + ["simworld", "--models", "mock", "--dry-run"],
                                    cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("567", result.stdout)
            bad = subprocess.run(command + ["textworld", "--mode", "invalid"],
                                 cwd=directory, capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)


if __name__ == "__main__":
    unittest.main()
