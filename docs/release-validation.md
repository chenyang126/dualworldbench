# Release validation

## Standalone GitHub extraction — 2026-11-03

Validated with Python 3.12.14. The current extraction adds public adapters,
a unified command dispatcher, five optional scene tools and source mapping.

| Check | Result |
|---|---|
| Dataset hashes and resource paths | All 625 entries passed |
| Released evaluation sets | 13 scenes / 189 SimWorld instances / 567 cells per model; 200 TextWorld episodes; 1,665 VQA questions |
| Offline regression and public adapter tests | 17 passed |
| Public Python interfaces | Grid selection, multimodal prompt construction, external scoring, VQA set deduplication and known-answer judging passed |
| Python TextWorld execution | Known A_L1 episode matched R=92.7; execution/plan/score logs saved |
| Local DFS via unified CLI | All four regimes completed at n=5, seed=1 without priority violations |
| Local mock API via unified CLI | 27 SimWorld cells across all modalities; resume made zero extra requests; one TextWorld episode; five VQA question types passed |
| CLI portability | Renamed/other working directory dispatch and nonzero argument-error exit verified |
| Optional scene tools | All five compile; standard-library tool help commands verified |

The mock server runs locally and does not call a paid provider. Optional
pointcloud/BEV tools require NumPy/Pillow and user-supplied original assets;
full raw-scene regeneration was not performed. The original source folder
and existing public baseline were left intact.

## Inherited public baseline validation — 2026-10-26

The following records describe checks documented by the existing public
baseline. Historical full-result replays were not repeated for this extraction.

Validated on 2026-10-26 with Python 3.12 and OpenAI SDK 2.54.0.

| Check | Result |
|---|---|
| Data hashes and resource paths | All 625 manifest entries passed |
| Evaluation grid | 13 scenes, 189 SimWorld instances, 567 modality cells/model |
| Other datasets | 200 TextWorld episodes and 1,665 VQA questions |
| Offline regression tests | 12 passed, including all 39 scene/modality prompt builds |
| Historical SimWorld replay | 3,402 plans; no F/C mismatches; no distance difference |
| VQA ground-truth replay | 1,665/1,665 accepted by the judges |
| Local DFS smoke | Completed n=5, seed=1 in all four TextWorld regimes |
| Exact NC reference | Matched exhaustive three-task execution; five-task CLI build/rescore passed |
| Local mock API integration | 27 SimWorld cells across all modalities; resume made zero additional requests; one TextWorld episode; five VQA question types |
| VQA credential override | Explicit CLI key works without OPENAI_API_KEY |
| Portability | Ten checks passed from a renamed checkout and a working directory outside the repository |

The API integration used a local mock Chat Completions server. It verifies
request construction, image loading, response handling and scoring; it does
not measure model capability or certify compatibility with a paid provider.
Historical model responses and local verification outputs are excluded from
the public repository.

To repeat the standard offline checks:

```bash
python scripts/validate_data.py
python -m unittest discover -s tests -v
python scripts/score_simworld.py --predictions examples/simworld_predictions.jsonl \
  --allow-partial --output outputs/simworld/example_scores.json
python scripts/score_vqa.py --predictions examples/vqa_predictions.jsonl \
  --allow-partial --output outputs/vqa/example_scores.json
```

See the README and protocol for reference-cost and inherited-baseline
limitations. Ground-truth replay checks annotation/judge consistency, not
independent correctness of every annotation.
