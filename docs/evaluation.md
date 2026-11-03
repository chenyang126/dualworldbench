# Evaluation guide

**Can Agents Plan Deliveries across Symbolic and Grounded Worlds?**

DualWorldBench evaluates multi-goal pickup-and-delivery planning in three complementary settings:

- **TextWorld:** symbolic scheduling with task rewards, deadlines, capacity constraints, and online task release.
- **SimWorld:** grounded route planning from text, annotated bird's-eye-view maps, or both, using the same node-sequence output interface.
- **DualWorld-VQA:** diagnostic questions about visual geometry, navigable topology, and constrained route planning.

[Project website](https://dualworld.netlify.app/) · [中文说明](../README.zh-CN.md) · [Evaluation protocol](protocol.md) · [Module interfaces](interfaces.md) · [GitHub publishing](publishing.zh-CN.md)

This release contains evaluation code, portable task data, navigation-distance resources, and offline scorers. It runs on the supplied maps and distance matrices; **Gazebo, ROS, a GPU, and live robot control are not required**. A GPU is only needed if you choose to host a model locally. Historical experiments, private endpoint configuration, model response archives, and manuscript files are excluded.

## Installation

Python **3.10 or newer** is required. Python 3.12 was used for release validation.

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/validate_data.py
python -m unittest discover -s tests -v
```

On Windows, activate the environment with `.venv\Scripts\activate`. Run the commands from the repository root unless indicated otherwise.

For API evaluation, export your own credentials and endpoint:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

Use the exact model identifier supported by your endpoint in `--model` / `--models`. Any OpenAI-compatible Chat Completions server can be used, including a local server. For an unauthenticated local server, set a non-empty placeholder API key. Image and multimodal evaluation require a server that accepts image inputs. `.env.example` documents the variables; scripts do not automatically load `.env` files.

## Unified entry point and Python interfaces

The repository provides a single command entry point for all three suites:

```bash
python -m dualworld --help
python -m dualworld validate
python -m dualworld simworld --models YOUR_MODEL_ID --dry-run
python -m dualworld textworld --mode algo --regimes A_L1 --task-counts 5 --seeds 1
python -m dualworld dualworld-vqa --help
```

Import `dualworld.simworld`, `dualworld.textworld`, or `dualworld.vqa` for
local data loading, canonical model inputs and evaluator adapters. See
[interfaces and prediction schemas](interfaces.md). Run from the
checkout; the original script commands below remain supported.

## Included data

| Suite | Released evaluation set |
|---|---|
| TextWorld | 200 task files: 4 regimes × 5 task counts × 10 seeds |
| SimWorld | 189 standard instances in 13 scenes; 567 cells per model across 3 modalities |
| DualWorld-VQA | 1,665 questions: 555 perception, 555 topology, 555 planning |

The SimWorld data folder has 191 task files. The standard grid uses only instances 1–3 for each task count in the open scene (`typical`), giving 9 open instances and 15 instances in each remaining scene. VQA retains its original 185 source cases, including two extra open cases. Do not apply the SimWorld allowlist to the VQA question files.

## Evaluate SimWorld

The internal modality names are `text`, `image`, and `meta` (**multimodal**). All three prompts request:

```json
{"sequence": ["START", "P1", "P3", "D3", "D1"]}
```

The evaluator supplies coordinates and movement actions. The model decides node ordering; it does not generate executable coordinates. Pickup must precede its matching delivery, load must respect capacity, and duplicate task actions are invalid.

Preview the full evaluation grid without calling an API:

```bash
python scripts/run_simworld.py --models YOUR_MODEL_ID --dry-run
```

Run a small API check:

```bash
python scripts/run_simworld.py \
  --models YOUR_MODEL_ID --scenes typical --modalities text \
  --limit 1 --workers 1 --checkpoint outputs/simworld/smoke_records.jsonl
```

Run all 567 cells for a model:

```bash
python scripts/run_simworld.py \
  --models YOUR_MODEL_ID --workers 1 \
  --checkpoint outputs/simworld/full_records.jsonl

python scripts/score_simworld.py \
  --predictions outputs/simworld/full_records.jsonl \
  --output outputs/simworld/scores.json
```

Rerun the same command to resume: completed and unparseable model responses are retained; infrastructure-error cells are retried. Use a fresh checkpoint for a different endpoint, prompt, or generation configuration, even if the model ID is unchanged. The API runner retries transport failures, rather than giving a model extra attempts to repair its plan.

### Use your own model without the API runner

Write one JSON object per line with `scene`, `sample_id`, `planner_type`, `model`, and `sequence` (or `node_sequence`):

```json
{"scene":"typical","sample_id":"n5-1","planner_type":"text","model":"my-model","sequence":["START","P1","D1"]}
```

Then run:

```bash
python scripts/score_simworld.py \
  --predictions examples/simworld_predictions.jsonl \
  --allow-partial --output outputs/simworld/example_scores.json
```

The scorer replays the sequence instead of trusting metrics supplied by a model. It keeps the last record per model–scene–sample–modality cell. A complete selected grid is required unless `--allow-partial` is passed; partial summaries explicitly report coverage. `--scenes` and `--modalities` select a smaller evaluation grid. Partial evaluation is useful for development and is not a full benchmark result.

### Metrics

- **F:** fraction of violation-free, successfully evaluated runs.
- **C:** mean fraction of validly delivered tasks. Valid deliveries retain credit even when other actions are invalid; therefore **C can exceed F**.
- **SR:** fraction of runs that are both violation-free and complete.
- **NC:** traveled distance divided by a released reference-route cost for the exact subset of validly delivered tasks. The scorer reports `nc_samples`; zero-delivery runs and unseen subsets without a released reference are excluded from NC, while remaining in F/C/SR.

**Reference-cost qualification:** the inherited reference builder uses multi-start local search, with exact checks reported for small instances. The shipped scalar references do not certify optimality for every larger instance. NC should therefore be described as normalization against the released reference routes, not an unconditional theoretical optimality gap. Ratios below 1 are possible if a model finds a better route. New delivered subsets need new independently validated references before comparing NC. See [protocol details](protocol.md).

For new delivered subsets, compute missing distance-only references and rescore:

```bash
python scripts/build_nc_references.py --scores outputs/simworld/scores.json \
  --output outputs/simworld/exact_references.json
python scripts/score_simworld.py --predictions outputs/simworld/full_records.jsonl \
  --reference-file outputs/simworld/exact_references.json \
  --output outputs/simworld/scores_with_references.json
```

The builder enumerates precedence/capacity states exactly under the supplied navigation distances. It has exponential time/memory cost, saves completed subsets for resuming, and defaults to a 10-task limit. It never edits shipped data. By default it fills only missing costs; `--recompute` also computes existing subsets. Replacing historical reference costs changes the NC denominator: report the reference file and each record's `reference_method`, and use the same reference policy for compared models. Add `--allow-partial` when rescoring a development subset.

`scene_summary` pools samples within each scene group. `task_count_summary` pools by task-count label. `scaling_average` is the unweighted average of the available task-count columns, matching the paper's table convention. A filename's `n` is a dataset label; the actual task count is read from its JSON.

## Evaluate TextWorld

The regimes are `A_L1`, `A_L2`, `M_L1`, and `M_L2`. A uses Type-A tasks; M mixes A/B/C. L1 is static with non-binding deadlines; L2 combines online release with deadlines. Published main comparisons use seed 1; all 10 generated seeds are included for further experiments.

Run a local DFS baseline with no API key:

```bash
python scripts/run_textworld.py \
  --mode algo --regimes A_L1 --task-counts 5 --seeds 1 \
  --output-dir outputs/textworld/dfs_smoke
```

Evaluate your model on the 20 seed-1 episodes:

```bash
python scripts/run_textworld.py \
  --mode llm --model YOUR_MODEL_ID \
  --output-dir outputs/textworld/my_model_seed1
```

The output includes execution logs, replanning logs, and `scores.json` containing cumulative reward **R** and simulated completion time **T**. The official Type-C rule is applied to executed logs: delivering A/B while carrying any C scores **R=0, T=9999**. Intervening pickups are allowed by this scoring rule. Raw simulator metrics are retained alongside these final scores.

`--mode dp` exposes the inherited online DP implementation, and `textworld/agent_planner/optimal_sky.py` contains the inherited offline solver. They are provided as **reference baselines**, not newly verified exact solvers. Their Type-C action restriction is stronger than the scoring rule, and their single-label reward/time DP does not establish optimality for general deadline settings. Do not treat them as certified theoretical upper bounds without additional verification.

To add a Python planner, implement `BasePlanner.plan(context)` from `textworld/agent_planner/common.py`, returning normalized `move`, `pickup`, and `deliver` actions, and pass the planner to `EventDrivenSimulator`. See [custom model integration](custom_models.md).

## Evaluate DualWorld-VQA

Run a small image-input API check:

```bash
python qa_gt/run_vqa_api.py \
  --model YOUR_MODEL_ID --vqa-set l1 l2 l3 \
  --max-questions 5 --parallelism 1 --output outputs/vqa/smoke.json
```

Evaluate all 1,665 questions:

```bash
python qa_gt/run_vqa_api.py \
  --model YOUR_MODEL_ID --vqa-set l1 l2 l3 \
  --parallelism 1 --output outputs/vqa/full.json
```

The runner saves progress, but a rerun starts a new evaluation; it does not resume like the SimWorld JSONL runner. For external models, submit one JSONL answer per `question_id`:

```json
{"question_id":"QUESTION_ID_FROM_DATASET","answer":"YOUR_ANSWER"}
```

```bash
python scripts/score_vqa.py \
  --predictions examples/vqa_predictions.jsonl \
  --allow-partial --output outputs/vqa/example_scores.json
```

Perception/topology questions use exact categorical or sequence scoring, with LCS auxiliary metrics for axis sorting. Planning questions require the complete valid task-node set, precedence and capacity compliance, and distance within the stored **10%** relative tolerance of the ground-truth route. Overall accuracy is question-weighted. The API runner uses the same judges as the offline scorer.

## Repository structure

```text
data/                      # Licensed task data, maps, matrices and references
  simworld/task/           # Scene assets and pickup–delivery task files
  simworld/reference/      # Released delivered-subset route reference costs
  textworld/task/          # Four regimes, five sizes, ten seeds
  vqa/task/                # Four question-family JSON files
  manifest.json            # Data integrity hashes
dualworld/                 # Unified CLI and public adapters for all three suites
simworld/tools/            # Optional point-cloud/map/distance asset tools
simworld/nav_planner/      # Prompts, parser, graph-distance evaluator
textworld/                 # Event-driven simulator, planners, task generator
qa_gt/                     # VQA generators, scene registry, API runner and judges
scripts/                   # Public runners, offline scorers, data validation
examples/                  # Small external-prediction examples
tests/                     # Offline scoring-contract regression checks
docs/                      # Protocol and custom-model integration
outputs/                   # User-generated results (ignored by Git)
```

The public release preserves the active prompt registry and evaluator semantics. Old JSON prompt templates and ROS publishing tools are omitted so there is a single active prompt source. Task/question generators are included, but the shipped files are the benchmark evaluation set: regenerating them creates a new dataset unless its contents are verified against `data/manifest.json`.

VQA generators write to `outputs/generated/vqa/` by default, preserving the shipped question files. Optional scene-asset generators require `requirements-tools.txt` and user-supplied raw pointcloud/world files; see [scene tools](../simworld/tools/README.md). The source extraction inventory is in [source-layout.md](source-layout.md). Release checks and their scope are recorded in [release validation](release-validation.md).

## Reproducibility and practical notes

- Configure your own endpoint; private gateways and credentials are not distributed.
- Start with one worker and a small limit. API evaluation uses your endpoint's quota and billing.
- The standard SimWorld grid has 567 cells per model; TextWorld's 20 seed-1 episodes may require multiple model calls per episode due to replanning; VQA has 1,665 requests before retries.
- Store the exact model/version, endpoint, generation settings, date, and any changed prompts when reporting results. Old paper model names are reference identifiers, not a promise of public endpoint availability.
- SDK/application retry limits and timeouts are visible in the runners. TextWorld preserves the original planner's behavior of returning an empty plan after exhausted API retries; review its logs before interpreting deadlocks as model failures.
- Image inputs are supplied as base64 by the framework. Dataset paths are repository-relative and are resolved against this checkout, so data can be moved to another machine.
- Release verification uses offline scoring, data-integrity checks, and local baseline runs. It does not verify a paid/live API provider.

## Citation

```bibtex
@misc{chen2026dualworldbench,
  title={DualWorldBench: Can Agents Plan Deliveries across Symbolic and Grounded Worlds?},
  author={Chen, Yang and Wen, Bin and Peng, Danyang and You, Hong-Jie and Shang, Zi-Qiao and Guo, Lan-Zhe},
  year={2026},
  url={https://dualworld.netlify.app/}
}
```

Update the BibTeX entry to the final publication metadata when available.

## License

Source code and documentation: **MIT**, see [LICENSE](../LICENSE).

Benchmark datasets and rendered maps under `data/`: **CC BY 4.0**, see [data/LICENSE](../data/LICENSE). Credit the authors, cite the paper, and identify changes to redistributed data. Third-party dependencies retain their own licenses.
