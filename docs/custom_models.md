# Integrating a model

## OpenAI-compatible server

Start your server separately, then set `OPENAI_BASE_URL` to its Chat Completions API root and `OPENAI_API_KEY` to its token (or a non-empty placeholder). Run the public API runners with the server's model identifier. `NO_TEMPERATURE_MODELS` is a comma-separated list of identifiers that reject an explicit temperature parameter.

## External predictions

For SimWorld, provide node sequences keyed by scene, sample, modality and model. See `examples/simworld_predictions.jsonl`. For VQA, provide final answer strings keyed by released `question_id`; see `examples/vqa_predictions.jsonl`. The scorers do not call a model API. They reject partial grids unless explicitly allowed.

Use the original question text and map for VQA; do not include `gt_*`, accepted answers, or reference routes in model inputs. For comparable SimWorld results, follow `simworld/nav_planner/prompt_registry.py` for the selected scene and modality instead of giving extra navigation-distance information to one condition.

## Python TextWorld adapter

Create a planner subclass and pass it to the original simulator:

```python
import sys
from pathlib import Path

root = Path.cwd()  # Run from the repository root.
sys.path.insert(0, str(root / "textworld/agent_planner"))
from common import BasePlanner
from simulator import EventDrivenSimulator

class MyPlanner(BasePlanner):
    def plan(self, context):
        # context: speed, now, pos, max_capacity, carried, tasks, last_interrupt
        # Implement model inference and return normalized actions, for example:
        task = context["tasks"][0]
        return [
            {"action": "move", "to": task["pickup"]},
            {"action": "pickup", "task_id": task["id"]},
            {"action": "move", "to": task["deliver"]},
            {"action": "deliver", "task_id": task["id"]},
        ]
```

The snippet illustrates action shape only: a complete implementation must handle empty task pools, carried tasks, capacity, priorities, and repeated replanning. Use the task configuration as in `scripts/run_textworld.py`, save execution logs, and apply `has_priority_violation` to obtain the official R/T score. Simulator-measured travel time and rewards are authoritative; a model's declared time or reward is not the score.
