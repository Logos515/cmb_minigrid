# Step 3 执行记录：轨迹采集、数据集与离线监督

Step 3 已完成五地图族扩展和 CPU 验收。seed=42，每族生成 10 个反事实 pair，共 50 个 pair、200 条轨迹。范围限于环境扩展、数据采集、分组划分、加载和校验。

## 交付内容

- `cmb_minigrid/data/trajectory.py`：专家/探索轨迹采集、JSONL 读写、逐步回放、handoff 快照独立恢复、专家标签校验。
- `cmb_minigrid/data/dataset.py`：按 `map_family_id` 和 `pair_id` 分组的 split manifest、数据集加载器、`T+1`/`T` padding mask 和部署安全 batch。
- `cmb_minigrid/scripts/collect_data.py`、`validate_data.py`、`stats_data.py`：可配置采集、完整校验和统计 CLI。
- `configs/data.yaml`：seed、地图族列表、每族 pair 数、探索比例、输出路径和 split ratios。
- `data/samples/step3_trajectories.jsonl`、`step3_manifest.json`：5 个地图族、每族 10 对 pair，共 200 条轨迹。
- `cmb_minigrid/environments/hidden_history.py`：5 个不同的真实拓扑布局；快照携带 `map_family_id`，专家规划按实际拓扑寻路。

## 实际验证

下列命令均实际执行，退出码均为 **0**。样例已用最终布局重新生成。

```text
/data2/xuyihao/envs/cmb_minigrid/bin/python -m pytest -q
44 passed in 17.05s

/data2/xuyihao/envs/cmb_minigrid/bin/ruff check cmb_minigrid tests
All checks passed!

/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.collect_data --config configs/data.yaml --pairs-per-family 10 --output data/samples/step3_trajectories.jsonl --manifest data/samples/step3_manifest.json --force
records=200 pairs_per_family=10 families=5 pairs=50 successes=150 truncated=50
split_counts=train=120,val=40,test=40

/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.validate_data
records=200 replayed=200 labels=200 pairs=50 handoff_restore=validated split_errors=0 pair_errors=0

/data2/xuyihao/envs/cmb_minigrid/bin/python -m cmb_minigrid.scripts.stats_data
{"horizon_max": 64, "horizon_min": 18, "map_families": ["hidden_history_nav_v1", "hidden_history_nav_v2", "hidden_history_nav_v3", "hidden_history_nav_v4", "hidden_history_nav_v5"], "pairs": 50, "per_family": {"hidden_history_nav_v1": {"pairs": 10, "records": 40, "split": "test"}, "hidden_history_nav_v2": {"pairs": 10, "records": 40, "split": "val"}, "hidden_history_nav_v3": {"pairs": 10, "records": 40, "split": "train"}, "hidden_history_nav_v4": {"pairs": 10, "records": 40, "split": "train"}, "hidden_history_nav_v5": {"pairs": 10, "records": 40, "split": "train"}}, "records": 200, "roles": {"expert": 100, "exploration": 100}, "split_counts": {"test": 40, "train": 120, "val": 40}, "terminated": 150, "truncated": 50}
```

采集 CLI 支持 `--map-families` 和 `--pairs-per-family`；`--pairs` 保留为每族数量的兼容别名。小规模五族采集、JSONL round-trip、回放与标签验证已在测试中覆盖。完整校验还会检查保存的两个专家端点是否仍然构成严格反事实 pair，避免只校验单条轨迹而忽略错误配对。

| 地图族 | 真实布局 | pair 数 | 轨迹数 | split |
| --- | --- | ---: | ---: | --- |
| hidden_history_nav_v1 | 原始水平走廊、竖直分支 | 10 | 40 | test |
| hidden_history_nav_v2 | 带弯折的公共走廊、竖直分支 | 10 | 40 | val |
| hidden_history_nav_v3 | 竖直公共走廊、水平分支 | 10 | 40 | train |
| hidden_history_nav_v4 | 水平公共走廊、各分支一次弯折 | 10 | 40 | train |
| hidden_history_nav_v5 | 水平公共走廊、各分支两次弯折 | 10 | 40 | train |

额外对最终 JSONL 的三个 split 实际运行了 CPU DataLoader（退出码 0）：

```text
split=train records=120 image_shape=(4, 65, 7, 7, 3) device=cpu masks=passed privileged_input=absent
split=val records=40 image_shape=(4, 65, 7, 7, 3) device=cpu masks=passed privileged_input=absent
split=test records=40 image_shape=(4, 65, 7, 7, 3) device=cpu masks=passed privileged_input=absent
```

相同断言已纳入 `test_each_split_loads_cpu_batch`；重复运行同种子采集的全部序列和划分也由测试验证完全一致。

## 验收边界

轨迹严格保存 `obs[0:T+1]` 与 `action[0:T]`。`privileged_state` 和 `env_snapshot` 只用于生成与离线校验；`collate_trajectories` 返回的 deployment batch 不包含它们。探索样本的 `expert_actions` 由每步真实状态重新规划，失败样本使用环境真实的 `truncated` 标志。

5 个地图族均通过 pair 严格校验：两端在 handoff 的当前观测、位姿、最近 8 步观测与动作一致，早期 clue 不同，正确分支不同且目标可达。每个 pair 的两端探索/专家记录保持同一 `pair_id`，split 只按完整地图族分组，因此 family 与 pair 均无跨 split 泄漏。

环境测试还确认：五族实际格网几何互不相同；每条专家轨迹等于真实地图上的 BFS 最短动作序列；封住 handoff 格子后目标不可达，无法绕过交接点。v1 保留原 19×15 地图和默认 API，新族使用同一任务、观测与三动作接口。快照带 `map_family_id`；旧 JSON 缺少该字段时按 v1 恢复。

划分比例配置仍为 `[0.8,0.1,0.1]`；只有五个不可拆的地图族时，为保证三个正比例 split 非空，分配为 3/1/1 个族，实际轨迹比例为 60%/20%/20%。不能通过拆分地图族来凑比例。

## 未运行与下一步

这五族目前是固定几何模板；每族 10 个 seed 对应重复的专家轨迹及不同的探索后缀，不等于 50 张独立随机地图。这批数据用于 Step 3 pilot 验收，尚不能支持大规模空间泛化结论。

未运行超过本次 10 pair/族的大规模采集；未运行人类窗口渲染（当前为无窗口会话）、GPU 专用测试（本步骤要求 CPU 验收），也未运行 RSSM/策略/CMB 训练或闭环模型评估（均属于后续步骤）。下一步依赖是 Step 4 的 RSSM 实现及接口测试，继续保持普通输入与特权监督字段的边界。
