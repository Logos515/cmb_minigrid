# CMB × MiniGrid × DreamerV3-style RSSM：AI 分步实现任务书

版本：v1.0（可依次交给 Codex、Cursor、Claude Code 等编程 AI）

## 一、项目目标与边界

目标：在 MiniGrid 部分可观测环境中，验证在同一当前观测、不同早期历史的条件下，将 RSSM-A 的历史 belief 通过紧凑 CMB 传给独立训练的 RSSM-B，是否可以恢复 B 的动作选择、未来预测和真实闭环成功率。

实验最初采用**相同观测/动作接口、不同隐藏维度及随机种子的两个 RSSM**；三模型共享 Hub、Isaac Lab 四足导航均为后续扩展，不纳入第一轮最小可行性验收。

主要基线：Reset-B、Replay-N、Linear Bridge、容量匹配的 MLP Bridge、CMB、Full-history Oracle-B。指标：关键分支正确率、闭环成功率、多步预测误差、History Swap 影响、通信字节数。

**禁止的实验捷径**：不把历史标签、真实地图、正确动作、全部历史直接喂给部署阶段的 B 或 CMB；不在冻结的 A/B/Policy 上训练额外主干参数；不把隐藏 MSE 单项下降当作交接成功；不假设两模型的 CMB 已证明共享模型无关空间。

## 二、每次向编程 AI 发任务时附加的统一指令

> 你是这个研究仓库的开发者。本轮**只完成我指定的步骤**，不要自行跳到后续训练任务。开始前先检查仓库和 `docs/spec.md`、`docs/implementation_log.md`、已有配置。代码用 Python 3.11+ 与 PyTorch，环境用 Gymnasium 和 MiniGrid。每次必须：①列出将改动的文件；②实现最小可运行代码和必要测试；③运行测试及本步骤 CLI，贴出真实命令、退出码与关键输出；④更新 `docs/implementation_log.md`（实现内容、接口、已验证内容、未解决问题、下一步依赖）；⑤明确哪些测试尚未运行以及原因。不得伪造实验结果、测试成功或下载状态。采用可复现随机种子，CPU 上至少可执行 smoke test。除非明确允许，不调整任务定义、数据划分、观测接口与实验基线。若现有接口不满足需求，先指出并实施最小兼容修改。先完成验收，不做大量无关重构。

### 必须固定的数据与时间契约

- 一条轨迹为 `obs[0:T+1]` 与 `action[0:T]`。`action[t]` 是在看到 `obs[t]` 后执行、产生 `obs[t+1]` 的动作。
- `state[t]` 是 RSSM 已经融合 `obs[t]` 的后验状态。发送 CMB 后，接收 B 在时刻 `t` **不得再融合一次 `obs[t]`**。
- Oracle-B 使用与 A 相同的已发生轨迹（观测和动作）完整过滤，得到 `state_B_star[t]`；不能将仿真器特权地图当作 Oracle-B 的输入。
- 反事实配对要求：交接时局部观测、方向、任务指令及最近 `k=8` 步的观测/动作完全相同，但早期隐藏线索及正确的关键分支不同；只做视觉像素相似、不精确相同，不能算通过。
- `env_state`、隐藏变量、专家动作标签可以用于数据生成、监督及评估，但不能出现在部署期 B/Adapter 输入中。
- 数据划分按 `map_family_id` / `pair_id`，反事实配对两端必须进入同一个 split，不准按时间帧随机划分。
- 第一轮 `RSSM-A: deter=128, stoch=16×16`，`RSSM-B: deter=256, stoch=16×16`，`CMB=4×16`，局部符号视觉编码 64 维；如因实现细节调整，需更新配置并解释。

## 三、分步指令（一次只发一步）

### Step 0 — 建立项目骨架和不可变接口规范

**交给 AI 的指令：**

> 建立 `cmb_minigrid/` Python 项目骨架及 `pyproject.toml`、`README.md`、`configs/`、`docs/spec.md`、`docs/implementation_log.md`，创建 `environments/ models/ bridge/ data/ trainers/ evaluation/ tests/ scripts/` 包。实现统一配置读取、种子设置、设备选择、日志目录与实验 ID。将任务目标、六个基线、最重要的时间索引和无数据泄漏要求写入 `docs/spec.md`。给出最小 `python -m cmb_minigrid.cli --help` 和 `pytest -q`。暂不实现 RSSM 或训练。

**验收：** 安装后 CLI 可调用；空数据下测试通过；配置校验失败有可读错误；README 提供复现命令。

### Step 1 — 跑通官方 MiniGrid-Memory，统一观测/动作接口

**交给 AI 的指令：**

> 实现 Gymnasium/MiniGrid 环境工厂，使用 `MiniGrid-MemoryS13-v0` 作为 smoke test。将局部 `image` 三个符号通道、`direction`、固定任务标识封装成统一 `AgentObs`；暂时只暴露左转/右转/前进三个动作（若环境需要更多动作才能完成任务，明确说明并调整封装，不悄悄改变任务）。实现随机/脚本 rollout、种子重放、每步观测哈希及可选渲染。添加测试，证明模型普通输入中没有全局地图、隐藏标签、正确分支、专家动作；生成一段 20 步 rollout 的样例数据。不要训练网络。

**验收：** `python -m cmb_minigrid.scripts.smoke_env --seed 42` 可执行；相同种子/动作重放完全一致；观测字段、shape、dtype 明确。

### Step 2 — 实现真正的 HiddenHistoryNav 和反事实配对

**交给 AI 的指令：**

> 实现一个基于 MiniGrid 的 `HiddenHistoryNav` 自定义地图：早期可观察“左/右通道阻塞”线索，之后进入公共走廊与统一交接点；在交接点两种隐藏世界的局部观测、方向、任务指令、最近 8 个观测和动作必须完全相同，但正确后续分支不同。可以通过固定公共前缀和可控早期线索生成配对轨迹。实现 `generate_counterfactual_pair(seed)`、`validate_pair(pair,k=8)`、可复现的专家路径、保存/恢复环境 snapshot。最重要的是自动排查当前观测、步数、任务文本、动作轨迹对隐藏标签的泄漏。此步暂不训练 RSSM。

**验收：** 至少生成 100 对严格通过检查的轨迹；正反两端都存在有效可达目标；随机选择隐藏标签的无历史基线关键分支准确率约 50%（报告样本量而非强行断言）。

### Step 3 — 采集轨迹、构建数据集和离线监督

**交给 AI 的指令：**

> 实现轨迹采集器：由特权专家规划器生成正确动作标签，但存储时区分 `agent_obs` 与 `privileged_state`，后者绝不作为部署输入。加入适量探索动作/失败轨迹；保存 `obs[0:T+1]`、`actions[0:T]`、`reward`、`terminated/truncated`、`expert_actions`、`map_family_id`、`pair_id`、`handoff_t`、`env_snapshot`。实现 train/val/test 按地图族及反事实 pair 分组划分，数据集加载器、padding mask、校验工具和统计脚本。先生成小批量 smoke 数据，再提供可扩展到正式规模的 CLI。

**验收：** 训练/测试无地图族和 pair 交叉泄漏；动作与下一观测时间对齐；重放轨迹可重建相同交接快照；数据加载器 CPU batch 测试通过。

### Step 4 — 编写轻量 DreamerV3-style 离散 RSSM

**交给 AI 的指令：**

> 在 PyTorch 实现一个独立可测的轻量 RSSM（不是逐行复刻官方 JAX）：分类符号图像编码器 64 维、GRU 确定性状态、`16×16` 分组离散随机状态、Prior/Posterior 网络、符号观测重建头、奖励/continue 预测头。接口必须有 `initial(batch)`, `observe_step(prev_state,prev_action,obs)`, `imagine_step(state,action)`, `features(state)`。约定 `state[t]` 已融合 `obs[t]`，从 `t` 经 `action[t]` 更新到 `t+1`。实现 KL dynamics/representation（stop-gradient 与 free-nats 处理）、分类采样/确定性评估。分别配置 A deter=128、B deter=256。先写单元测试，不训练大模型。

**验收：** 批量及序列张量形状正确；one-hot/概率有效；损失有限、可反传；`imagine_step` 不偷看未来观测；固定 seed 可重现；有简单的单 batch 过拟合测试。

### Step 5 — 独立训练并验收 RSSM-A / RSSM-B

**交给 AI 的指令：**

> 用 Step 3 数据分别训练 A/B 两个 RSSM，独立初始化与 checkpoint。实现观测分类重建、奖励/continue 和拆分 KL 损失，按序列训练并用 mask 排除 padding/终止后位置。实现验证集 early stopping、梯度监控、日志、保存/恢复优化器状态与配置。先运行 100 步 CPU/GPU smoke 训练，再用可配置命令启动正式训练。检查在反事实配对里早期隐藏线索是否能从 `state_A/state_B` 的冻结线性 probe 中被预测；不要将 probe 输出接回主模型。

**验收：** 两模型 checkpoint 可独立加载并进行 full-history filtering；损失稳定下降或明确报告训练失败；模型当前观测对历史线索的 probe 不应作为“记忆已充分”的唯一依据。

### Step 6 — 训练 Actor-B（可选 Actor-A）并验证 Oracle/Reset 间隙

**交给 AI 的指令：**

> 冻结两个 RSSM，使用专家动作标签行为克隆训练 `Policy-B(features_B, goal)`，如用于 A 自主前缀再训练 Policy-A。训练 Actor 时只输入 RSSM full-history posterior，不能输入真实隐藏变量或专家未来动作。实现三种独立闭环：A full-history、Oracle-B full-history、Reset-B（仅从当前观测构造 B 的状态）。在 HiddenHistoryNav 测试集报告关键分支正确率和闭环成功率，并保存逐轨迹预测记录。如果 Oracle 与 Reset 不存在明确间隙，停止 Bridge 阶段，优先诊断环境泄漏、RSSM 遗忘、Actor 不具备历史决策能力。

**验收：** 模型能闭环完成任务；Oracle-B 明显高于 Reset-B；完整历史和重置的状态构造方式与时间索引一致。目标成功率可建议而不能伪造保证。

### Step 7 — 生成 Handoff Dataset 和实现基础基线

**交给 AI 的指令：**

> 对每段同一前缀历史，分别由冻结 A/B 调用 `observe_step` 完整过滤，得到同一交接点 `state_A[t]` 与 `state_B_star[t]`；B 的 Oracle 不访问仿真器特权地图。保存 `obs_t`、未来真实 `obs/actions`、pair_id、snapshot。实现 Reset、Replay-4/8/16、Linear A→B、容量匹配 MLP A→B、Oracle-B 的状态初始化逻辑。对 B 的随机离散状态，输出合法 logits/one-hot，而不是直接任意浮点。写测试保证 Replay-N 只读取最近 N 帧、不泄漏交接后的未来观测；所有方法在相同 snapshot 开始。

**验收：** 从同一交接状态可重复运行所有初始化方式；Oracle 与完整 Replay 一致（处理采样随机性）；关键时刻不会重复融合 `obs_t`。

### Step 8 — 实现并训练 CMB Bridge

**交给 AI 的指令：**

> 实现 `CMBBridge`：Sender 读取 A 的 `(deter,stoch)` 并压缩为 `4×16` latent，Receiver 将 CMB 与冻结 B 对当前观测的 64 维编码融合，输出 B 的 256 维 deter 和 `16×16` 离散 state logits。冻结 A/B/Actor-B，分两段训练：先做 Oracle-B 状态辅助损失 + 动作分布 KL，再加入共同动作条件下 K=5/10 多步预测蒸馏，并与真实未来观测核验。必须保证冻结 B 的**学生**分支仍可对 CMB 反向传播；仅 Oracle 教师分支 `no_grad`。实现训练/验证脚本、日志、检查点、梯度测试及与等容量 MLP 的参数/字节开销对照。

**验收：** `state_B_hat` 可被 B 的 policy 和 RSSM 正常消费；有非零可回传梯度；验证集动作和未来预测指标不劣于 Reset（若失败如实报告）；不把对两模型的瓶颈映射称为已证明“共享模型无关空间”。

### Step 9 — 闭环接管、History Swap 与主结果表

**交给 AI 的指令：**

> 实现 `HandoffRunner`：由历史前缀获取 `state_A[t]` 和 env snapshot，针对 Reset、Replay-N、Linear、MLP、CMB、Oracle 逐个恢复完全相同的环境快照；选择时刻 t 的 B 初始 posterior（已融合 obs_t），由 B 在真实环境内自主执行闭环动作并从 obs[t+1] 更新；记录第一关键分支、任务成功、错误动作、路径效率、碰撞/失败、通信载荷、接管耗时。实现配对 `History Swap`：在当前观测相同但早期历史不同的两端交换 CMB，测试动作是否依交换线索变化。生成表格与每地图族汇总，不得填写模拟成绩。

**验收：** 同一方法/种子运行可复现；不同方法从完全相同的物理快照起步；History Swap 确实只交换消息、不改变环境或当前观测；输出真实 CSV/JSON 报表。

### Step 10 — 统计、消融和研究报告

**交给 AI 的指令：**

> 在已通过前述门槛的实验基础上，运行 CMB 容量 `{2×16,4×16,8×16,8×32}`、历史间隔 `{8,16,32,64}`、Replay 窗口、带/不带 Action/Future Loss、正确/交换/置零 CMB 等消融。使用按 map_family/pair 聚类的 Bootstrap 做 95% 置信区间，报告至少 5 组不同 RSSM 训练种子（资源不足时先明确只完成 Pilot）。输出 `reports/experiment_report.md`、配置清单、CSV 与图表。区分：成功传递单个线索、可持续闭环预测恢复、真正跨模型共享空间，这三类结论的证据边界。

**验收：** 所有图表由真实数据生成；具有随机种子、版本、训练配置和复现命令；如没有超过 Replay/MLP，必须客观报告和分析，不允许只选成功样本。

### Step 11（可选，不阻塞主结果）— 三模型共享 Hub 与空间历史扩展

**交给 AI 的指令：**

> 当 A→B 已经稳定时，增加第三个不同隐维度/随机种子的 RSSM-C。设计每个模型的 Sender 与 Receiver Adapter，共用潜在空间；测试未直接训练的 A→C、C→B 或第三模型 checkpoint 组合，与各自单独的 Pairwise Bridge 比较。另增加多障碍/访问历史环境，避免只学会一个二元线索。明确哪些 Adapter 实际已训练、哪些 Pair 真正未见过；不能仅因载体张量形状相同就声称共享语义。

**验收：** 多模型路由矩阵、每 Pair 成功率、适配参数量、训练代价以及泛化拆分明确。

## 四、建议实施顺序和停机条件

- **先做 0→1→2→3**：环境可复现、反事实配对通过才写世界模型训练。
- **再做 4→5→6**：如果 Oracle-B 不能显著优于 Reset-B，先停下来定位，而不是强行训练 CMB。
- **最后做 7→8→9→10**：先交接可运行，再看多步闭环和因果消融。
- **Step 11 可后做**，作为真正模型无关共享 CMB 研究的下一阶段。

## 五、建议第一轮的命令约定（由 Step 0 在仓库中落实）

```bash
python -m pip install -e '.[dev]'
pytest -q
python -m cmb_minigrid.scripts.smoke_env --seed 42
python -m cmb_minigrid.scripts.build_pairs --count 100 --seed 42
python -m cmb_minigrid.scripts.train_rssm --model a --config configs/rssm_a.yaml
python -m cmb_minigrid.scripts.train_rssm --model b --config configs/rssm_b.yaml
python -m cmb_minigrid.scripts.train_policy --model b
python -m cmb_minigrid.scripts.build_handoff_dataset
python -m cmb_minigrid.scripts.train_bridge --config configs/cmb.yaml
python -m cmb_minigrid.scripts.evaluate --methods reset,replay8,linear,mlp,cmb,oracle
python -m cmb_minigrid.scripts.report
```

这些命令是建议统一的 CLI 规范，**不是表示命令已经实现**。实现后由编程 AI 根据实际运行结果更新 README。

## 六、最小验收路径

如果时间有限，先只做：自定义反事实环境 → 独立 RSSM A/B → Full-history Oracle 与 Reset 差距 → 4×16 CMB → 闭环与 Swap。不要在这个证据链建立前引入复杂机器人底层动力学、动态路由器、跨模型共享 Hub 或大量模型架构变化。
