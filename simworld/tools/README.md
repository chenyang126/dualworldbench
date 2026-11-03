# Optional scene asset tools

These five scripts are extracted from the original `simworld/tools/`.
Benchmark evaluation uses the shipped maps/matrices and does not require
these tools. Original raw PLY/VGH/SDF scene assets are not in this release;
provide your own source files when building new assets.

| Script | Purpose | Input |
|---|---|---|
| `compute_far_distance_matrix.py` | Graph navigation distances | Points and VGH graph |
| `compute_sdf_path_distance_matrix.py` | Occupancy-grid navigation distances | Points and SDF world |
| `filter_ground_from_ply.py` | Remove ground points | Binary PLY cloud |
| `generate_clean_bev.py` | Generate clean BEV maps and metadata | Original scene pointcloud directory |
| `convert_task_jsons_to_indoor_format.py` | Convert legacy tasks to task-list format | Working copy of task directory |

Run `python simworld/tools/SCRIPT.py --help` for explicit file arguments.
Install `requirements-tools.txt` for PLY/BEV tools; the two distance builders
and task converter use Python's standard library. BEV generation now accepts
`--mesh-root` and `--output-root` instead of a private machine path. The
converter requires `--root` and edits that directory in place: use a copy
under `outputs/` when experimenting. Generated assets constitute a new
dataset unless verified against the shipped manifest.
