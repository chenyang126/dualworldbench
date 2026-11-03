# Three public module interfaces

Run Python from this checkout with Python 3.10+. Install `requirements.txt`
first. These are source-checkout interfaces; the data ships alongside the
code, and no wheel or server deployment is required.

| Module | Python interface | CLI | Inputs | Outputs |
|---|---|---|---|---|
| SimWorld | `dualworld.simworld` | `python -m dualworld simworld` | Scene, sample, text/image/meta prompt | Node sequence; F/C/SR/NC and coverage |
| TextWorld | `dualworld.textworld` | `python -m dualworld textworld` | Episode and `BasePlanner.plan(context)` | Executed actions, logs, R/T |
| DualWorld-VQA | `dualworld.vqa` | `python -m dualworld dualworld-vqa` | Question ID, question text and image | Answer, correctness, auxiliary metrics |

`meta` means multimodal. These are benchmark library/command interfaces;
the project website can link to the repository and download archive. API
evaluation connects to the user's OpenAI-compatible model endpoint.

## SimWorld

```python
from dualworld import simworld

scenes = simworld.list_scenes()
samples = simworld.list_samples("typical")
context = simworld.load_task("typical", "n5-1", "text")
messages = simworld.build_messages("typical", "n5-1", "text")
scores = simworld.score_file(
    "examples/simworld_predictions.jsonl", allow_partial=True)
```

`load_task` returns normalized `pos`, `tasks`, `max_capacity` and scene
metadata. The evaluator context contains raw task resources; use only the
canonical `build_messages` output for model inference. `list_samples`
enforces the standard 189-instance grid. `score_file` accepts optional
`scenes`, `modalities` and `reference_file`, returns a dictionary, and does
not write output or call the model. Use the CLI to save a score file.

JSONL prediction contract:

```json
{"scene":"typical","sample_id":"n5-1","planner_type":"text","model":"my-model","sequence":["START","P1","D1"]}
```

`planner_type` is `text`, `image` or `meta`. Task node labels are `P1`, `D1`,
etc.; the engine maps these to coordinates. See [protocol.md](protocol.md)
for partial delivery credit and reference-cost qualifications.

## TextWorld

```python
from dualworld import textworld
from planner_algo import AlgoPlanner  # Compatibility path initialized above.

task = textworld.load_task("A_L1", 5, seed=1)
scores = textworld.evaluate(AlgoPlanner(), task, "outputs/textworld/python_smoke")
print(scores["R"], scores["T"])
```

For your model, subclass `textworld.BasePlanner` and implement
`plan(context)`. Context includes `speed`, `now`, `pos`, `max_capacity`,
`carried`, `tasks` and `last_interrupt`. Return a list of actions:

```json
[
  {"action":"move","to":{"x":1.0,"y":2.0}},
  {"action":"pickup","task_id":"A-0001"},
  {"action":"move","to":{"x":3.0,"y":4.0}},
  {"action":"deliver","task_id":"A-0001"}
]
```

The simulator may call the planner repeatedly when new tasks arrive or
plans end. Use actual task coordinates. Episode files contain `SPEED` in
meters/minute; the wrapper converts to the simulator's meters/second.
`evaluate` writes `result.json`, `plans.json`, and `scores.json`, and returns
final R/T together with `priority_violation` and `raw_summary`. Select a
different output directory per episode. The Type-C scoring rule is the
same as `scripts/run_textworld.py`.

## DualWorld-VQA

```python
from dualworld import vqa

questions = vqa.load_questions("l1", scenes=["typical"])
record = questions[0]
messages = vqa.build_messages(record)
result = vqa.judge(record, "YOUR_MODEL_ANSWER")
```

Set names are `l1`, `l2`, `l3`, or `absolute`, `euclidean`, `wall`,
`capacity`. `load_questions` resolves image paths independently of the
working directory and deduplicates question IDs if sets overlap.
Records contain annotations for evaluation; send only the canonical
messages to a model. `judge` returns `is_correct` and question-specific
metrics using the existing evaluator. External prediction files contain
`question_id` and `answer`, one JSON object per line.

## Unified commands

```bash
python -m dualworld --help
python -m dualworld validate
python -m dualworld simworld --models YOUR_MODEL_ID --dry-run
python -m dualworld textworld --mode algo --regimes A_L1 --task-counts 5 --seeds 1
python -m dualworld dualworld-vqa --help
python -m dualworld score-simworld --predictions examples/simworld_predictions.jsonl --allow-partial
python -m dualworld score-vqa --predictions examples/vqa_predictions.jsonl --allow-partial
```

The dispatcher passes suite arguments unchanged and propagates its exit
code. The original scripts remain supported. API runners use the common
environment configuration in `.env.example`. No live API request is
necessary for validation or offline scoring.
