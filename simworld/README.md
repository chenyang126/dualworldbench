# SimWorld

Public adapter: [`dualworld/simworld.py`](../dualworld/simworld.py).
Data: `data/simworld/task/` and `data/simworld/reference/`.

`nav_planner/` contains the scene registry, canonical prompts, model client,
node parser, action conversion and navigation-distance evaluator.
Run `python -m dualworld simworld --help`; use
`python -m dualworld score-simworld --help` for offline replay.

See [module interfaces](../docs/interfaces.md) and [protocol](../docs/protocol.md).
