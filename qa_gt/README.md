# DualWorld-VQA

Public adapter: [`dualworld/vqa.py`](../dualworld/vqa.py).
Data: `data/vqa/task/`; map and navigation resources are shared with SimWorld.

`run_vqa_api.py` provides canonical question prompts, API inference and judges.
The four `generate_*_vqa.py` scripts generate the four question families;
`scene_registry.py` provides scene discovery. The internal `qa_gt` directory
name is preserved to keep existing imports and generators compatible.
Run `python -m dualworld dualworld-vqa --help` or
`python -m dualworld score-vqa --help`.

See [module interfaces](../docs/interfaces.md) and [protocol](../docs/protocol.md).
