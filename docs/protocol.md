# Evaluation protocol

## Scope and provenance

This release packages the current evaluator and task files from the authors' working codebase. Source modules were retained where possible; endpoint configuration, dataset paths, output paths, public runner arguments, and standalone scoring wrappers were adapted for distribution. Historical experiment directories and paper-writing scripts were not included. No previously collected private model responses are needed to run the benchmark.

The 13 SimWorld scenes map to five reported groups: `typical` → Open, `l1_*` → T1 (five scenes), `l2_*` → T2 (five scenes), `indoor` → Indoor, and `campus` → Campus. `scripts/grid.py` defines the standard instance selection. The VQA dataset uses its own released question IDs and is not filtered by that allowlist.

## SimWorld execution and aggregation

All modalities request a JSON node sequence. The shared parser normalizes equivalent label spellings; it can recover node lists from legacy action replies or prose. The evaluator resolves node IDs to exact task coordinates, adds movement when necessary, and uses a precomputed navigation graph distance plus access offsets. Unknown node IDs, capacity overflow, duplicate operations, and precedence violations remain semantic errors.

An invalid pickup/delivery does not update task state. Evaluation continues so subsequent valid deliveries can still receive credit. For M selected runs, let `ok_j` indicate that a plan was successfully evaluated, `v_j` its violation count, `d_j` validly delivered tasks, and `N_j` total tasks:

```text
F  = (1/M) sum_j 1[ok_j and v_j = 0]
C  = (1/M) sum_j 1[ok_j] d_j / N_j
SR = (1/M) sum_j 1[ok_j and v_j = 0 and d_j = N_j]
```

The public scorer explicitly treats missing/unparseable plans as zero F/C/SR. It does not treat every infeasible plan as zero C. Valid but incomplete plans have F=1 and SR=0. A plan may have C=1 and F=0 if all tasks eventually receive valid deliveries despite other invalid operations.

For a delivered subset S, the NC reference key is `scene|sample_id|comma-separated-sorted-task-IDs`. Reference values are unchanged from the author's current statistics. NC is averaged only where a nonempty delivered subset has a reference. Always report the number of scored NC runs. The inherited reference builder uses heuristics for larger instances, so released costs are not certified optima for all subsets. Do not clamp NC to 1 or present a heuristic route cost as a universal lower bound. The public scorer does not generate or update references automatically.

`scripts/build_nc_references.py` optionally computes missing references using an exact distance-only DP over picked/delivered masks and last stop. It starts at the task's current position with an empty load, respects capacity and pickup precedence, ends at the last delivery, and uses the same movement costs as the evaluator (no movement between identical coordinates). It does not constrain reward/deadlines or require returning to the start. Runtime and memory grow exponentially; the default limit is ten tasks. Pass its output to the scorer with `--reference-file`. Existing released costs remain unchanged unless an override is explicitly generated with `--recompute`. Each scored record identifies its reference method; exact overrides should be computed consistently for every compared model, since a changed denominator changes historical NC.

Checkpoints are deduplicated by model, scene, sample ID and modality, keeping the last entry. By default the offline scorer requires a full selected grid. `--allow-partial` permits a development subset and labels coverage; it does not fill absent predictions with fabricated results.

## TextWorld

The simulator maintains continuous time, position, active tasks, and carried tasks. New task arrivals can interrupt motion at an interpolated position and trigger replanning. The original task files store `SPEED` in m/min; the TextWorld runner converts it to m/s, matching the existing simulator entry point. SimWorld retains its own original normalization; travel time across the worlds should not be pooled.

Reward is computed by the simulator from delivery time and type-dependent lateness penalties. Type-C priority is additionally scored from executed logs: an A/B delivery while carrying C makes the official episode score R=0 and T=9999. The wrapper preserves raw metrics for inspection. It does not change the simulator's dynamics.

Reference DFS/DP source is included for reproducibility. The inherited DP implementations block all pickups while carrying C, whereas the stated scoring rule only blocks A/B delivery. Their single reward/time label per discrete state also requires additional justification for exact optimality when future reward depends on time. These are limitations of the inherited baselines, not a certification supplied by the public release.

## VQA

The released question counts are:

| Question type | Count | Primary score |
|---|---:|---|
| Absolute axis ordering | 185 | Exact sequence; LCS auxiliary |
| Euclidean anchor choice | 185 | Exact node |
| Euclidean pair choice | 185 | Exact pair |
| Wall-illusion topology | 555 | Exact route comparison |
| Capacity interleaving | 555 | Valid sequence and distance tolerance |

Planning answers must start from the required current node, contain the complete required task-node set exactly once, obey pickup precedence and capacity, and stay at or below the stored accepted-distance threshold. The released ratio is 0.10. The scorer accepts alternate feasible routes inside that threshold rather than requiring the exact ground-truth node order.

Overall accuracy is averaged across all questions. Perception accuracy can be obtained by pooling the three 185-question types. Ground-truth data is released for transparent evaluation and must not be supplied to the tested model; runners supply question text and the map, not ground-truth answers.
