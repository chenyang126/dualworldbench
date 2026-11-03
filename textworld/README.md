# TextWorld

Public adapter: [`dualworld/textworld.py`](../dualworld/textworld.py).
Data: `data/textworld/task/{A_L1,A_L2,M_L1,M_L2}/`.

`agent_planner/` contains the event-driven simulator, planner contract, API
planner, DFS and inherited DP/offline reference baselines.
`task_generation/` contains the episode generator.
Run `python -m dualworld textworld --help`.

See [module interfaces](../docs/interfaces.md) and [protocol](../docs/protocol.md).
