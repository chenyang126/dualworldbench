"""Score externally generated VQA answers using the released question IDs."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qa_gt"))
from run_vqa_api import evaluate_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/vqa/score.json")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    records = {}
    for path in (ROOT / "data/vqa/task").glob("*.json"):
        for r in json.loads(path.read_text())["records"]:
            records[r["question_id"]] = r
    answers = {}
    for line in args.predictions.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            qid = r["question_id"]
            if qid not in records:
                raise ValueError(f"Unknown question ID: {qid}")
            answers[qid] = r["answer"]
    if not answers:
        raise ValueError("No answers supplied.")
    if len(answers) != len(records) and not args.allow_partial:
        raise ValueError(f"Provided {len(answers)}/{len(records)} answers; pass --allow-partial for a subset.")
    output = []
    buckets = defaultdict(list)
    for qid, answer in sorted(answers.items()):
        r = records[qid]
        result = evaluate_record(r, str(answer))
        output.append({"question_id": qid, "answer": answer, **result})
        buckets[r["question_type"]].append(result["is_correct"])
    summary = {"provided": len(answers), "expected": len(records),
               "complete": len(answers) == len(records),
               "accuracy": sum(r["is_correct"] for r in output) / len(output),
               "by_question_type": {k: {"count": len(v), "accuracy": sum(v) / len(v)}
                                    for k, v in sorted(buckets.items())}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "results": output}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
