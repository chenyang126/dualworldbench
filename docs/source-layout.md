# 源码整理与对应关系

本次以已有的 `output/dualworldbench-public` 为公开提取基础，核对 `uni-embodied-benchmark-main` 中的模块源码，并新增独立的统一接口。原始工程未修改。代码哈希与逐文件来源见 [source-manifest.json](source-manifest.json)。

## 目录对应

| 原始目录 | 发布目录 | 内容 |
|---|---|---|
| `simworld/nav_planner/` | `simworld/nav_planner/` | 活跃提示词、规划器、节点解析、场景与评分 |
| `simworld/tools/` | `simworld/tools/` | 五个场景生成/转换工具，移除私有路径 |
| `textworld/agent_planner/` | `textworld/agent_planner/` | 模拟器、模型接口、DFS/DP 及离线参考求解器 |
| `textworld/task_generation/` | `textworld/task_generation/` | Python 任务生成器 |
| `qa_gt/` | `qa_gt/` | VQA 生成器、场景注册、模型接口与评判器 |
| `iclr_data/*/task/` | `data/*/task/` | 当前公开任务与必要地图资源 |
| `tool/iclr_simworld/` 等 | `scripts/` | 公共运行、网格与评分辅助脚本 |
| 新增 | `dualworld/` | 三模块 Python 适配器与统一 CLI |

公开版本通过环境变量配置模型端点，将数据引用改为相对仓库路径，运行结果写入 `outputs/`。保留已有公开版的评测语义与数据哈希。额外的原始地图点云、Gazebo/ROS 控制、论文、历史模型响应和审稿统计未纳入评测发布包。

## 原始模块中未发布的文件

| 原始文件 | 原因 |
|---|---|
| `simworld/nav_planner/benchmark_scene.py` | Original batch/output-layout or robot publishing utility; public runner replaces experiment orchestration |
| `simworld/nav_planner/merge_scene_results_into_model_layout.py` | Original batch/output-layout or robot publishing utility; public runner replaces experiment orchestration |
| `simworld/nav_planner/prompt/prompt_template.json` | Legacy prompt or generation shell helper; active Python entry point retained |
| `simworld/nav_planner/publish_waypoints.py` | Original batch/output-layout or robot publishing utility; public runner replaces experiment orchestration |
| `simworld/nav_planner/run_eval_suite.py` | Original batch/output-layout or robot publishing utility; public runner replaces experiment orchestration |
| `simworld/nav_planner/run_structured_scene_suite.py` | Original batch/output-layout or robot publishing utility; public runner replaces experiment orchestration |
| `textworld/agent_planner/planner_alog_v2.py` | Alternative planner returns legacy intent action format; active planner_algo.py retained |
| `textworld/agent_planner/prompt_template.json` | Legacy prompt or generation shell helper; active Python entry point retained |
| `textworld/task_generation/v1.sh` | Legacy prompt or generation shell helper; active Python entry point retained |
| `textworld/task_generation/v2.sh` | Legacy prompt or generation shell helper; active Python entry point retained |
| `qa_gt/vqa.md` | Legacy prompt or generation shell helper; active Python entry point retained |


`simworld/tools/` 可用于生成新场景资源；标准评测直接使用随包地图和距离矩阵。`planner_dp.py` 与 `optimal_sky.py` 保留为历史参考基线，其适用限制见 [protocol.md](protocol.md)。
