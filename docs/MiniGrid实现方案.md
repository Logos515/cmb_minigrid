# CMB × DreamerV3 RSSM × MiniGrid：详细实验实现方案

研究目标：异构世界模型间的隐式历史记忆交接与闭环控制恢复｜第一阶段实现方案 v1.0

## 一、总体方案与预期成果

建议采用 PyTorch + MiniGrid + 轻量级 DreamerV3-style RSSM，实现一个能够独立训练、复现实验和开展消融的 CMB 原型。

第一阶段暂时不训练完整 DreamerV3，也不引入四足机器人动力学。我们保留 DreamerV3 最关键的 RSSM 状态更新机制，用 MiniGrid 构造严格的历史依赖导航任务，验证：

> 模型 A 已经通过历史观测形成 Belief，模型 B 没有经历这些历史。能否通过一个共享的隐式记忆载体 CMB，让 B 在后续预测与闭环决策中恢复接近完整历史 Oracle-B 的能力？

这与您上传的研究路线中“接收方条件化残差隐式交接”的核心假设一致。

### 1.1 最终实现的系统

MiniGrid 部分可观测环境

局部观测 \\(o_t\\)、离散动作 \\(a_t\\)、隐藏环境状态

完整历史 \\(H_t\\)

观测序列 + 已执行动作

RSSM-A

\\(s_t^A=(h_t^A,z_t^A)\\)

已有历史记忆的发送模型

Oracle RSSM-B

\\(s_t^{B,\*}\\)

完整历史教师，仅用于训练和评估

CMB Bridge

Sender Adapter → Shared Latent Tokens → Receiver Adapter

\\(s_t^A\rightarrow Z_t\rightarrow\hat s_t^B\\)

RSSM-B + Policy-B

使用恢复后的 Belief，连续选择动作并更新状态

闭环测试

导航成功率｜动作一致性｜未来预测误差｜通信成本

### 1.2 第一版必须完成的六个功能

| 模块             | 实现目标                | 核心输出             |
| -------------- | ------------------- | ---------------- |
| Environment    | 生成依赖不可见历史的导航任务      | 可复现的反事实轨迹对       |
| RSSM-A         | 独立学习历史状态表示与动力学      | \\(s_t^A\\)      |
| RSSM-B         | 独立学习导航任务和闭环决策       | Oracle-B         |
| CMB            | 将 A 的状态转换为 B 可使用的状态 | \\(\hat s_t^B\\) |
| Handoff Runner | 在指定时刻切换 A→B         | 接管后的独立闭环轨迹       |
| Evaluator      | 比较不同记忆交接方法          | 实验指标、曲线、统计结果     |

第一阶段的最小成功条件是：

\\[ J\_{\mathrm{Oracle-B}} > J\_{\mathrm{CMB-B}} > J\_{\mathrm{Reset-B}} \\]

这里是希望观察的趋势，不是需要人为强制满足的约束。CMB 是否优于 Replay 和直接 Hidden Mapping，则必须通过实验检验。

我建议用以下两个项目作为设计依据：

- [DreamerV3 官方 RSSM 实现](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/rssm.py)：具有 `initial()`、`observe()`、`imagine()` 和 `loss()` 等明确接口，使用确定性状态与分组离散随机状态。

  [image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

  GitHub

  +1

- [MiniGrid](https://github.com/Farama-Foundation/Minigrid)：使用 Gymnasium 接口，提供 7×7×3 局部符号观测及 Memory 等任务，并支持自定义地图生成。

  [image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

  GitHub

  +1



需要说明：下面给出的 PyTorch 结构和训练方案是借鉴 DreamerV3 RSSM 设计的研究原型，不是对官方 JAX DreamerV3 的逐行复现。参数、辅助损失和环境改造也是针对 CMB 新提出的实验配置。

## 二、第一阶段应该使用什么 MiniGrid 任务？

我建议采用“两级验证”，不要直接在复杂导航环境上调试 CMB。

### Level 0：MiniGrid-Memory（功能验证）

使用官方：

```
gym.make("MiniGrid-MemoryS13-v0")
```

MiniGrid-Memory 的任务是：机器人起初看到某个物体，经过走廊后，需要在岔路口选择与最初物体相匹配的目标。默认局部观测为 `7×7×3`，动作空间为 `Discrete(7)`。

[image](https://www.google.com/s2/favicons?domain=https://minigrid.farama.org\&sz=32)

Minigrid Documentation

+1



任务逻辑示意：在岔路口，仅凭当前观测无法恢复早期见到的线索。

验证流程：

模型 A 从起点开始观察，经过走廊，在岔路口将状态交给 B。B 不直接获得最初的线索，而是通过 CMB 恢复信息并选择分支。

这是一个低成本的工程检查，主要验证 Bridge 接口和历史线索传递是否正常。

但它可能只需要保存 1 bit 左右的关键决策信息，因此不能作为最终的空间记忆充分性证据。

### Level 1：自定义 HiddenHistoryNav（主实验）

我更推荐把主要研究工作放在一个自定义的 `HiddenHistoryNav` 环境中。

场景设计为：

机器人起初会观察到左侧或右侧的障碍状态。经过一段路程后，返回或到达统一交接点。此时障碍物已经位于当前视野之外，但此前观察到的状态决定哪条路线可通行。

History 1

历史观察：左侧阻塞

History 2

历史观察：右侧阻塞

隐藏历史导航示意。两种历史在交接点有相同的局部观测，但需要采取不同动作。

环境必须满足：

\\[ o_t^{(1)}=o_t^{(2)} \\]

\\[ g^{(1)}=g^{(2)} \\]

\\[ a\_{t-k:t-1}^{(1)}=a\_{t-k:t-1}^{(2)} \\]

但正确的后续决策不同：

\\[ a_t^{\*,(1)}\neq a_t^{\*,(2)} \\]

其中 \\(g\\) 是任务目标或任务指令，\\(k\\) 是最近动作窗口长度。

建议保证最后 8 个时间步的观测和动作也相同，从而使 Replay-8 无法通过短期信息区分历史。

这类实验才能比较严格地验证：CMB 是否传递了当前观测和近期动作无法恢复的历史残差。

## 三、项目代码结构

建议把环境、RSSM、CMB、控制策略和评估系统分开实现，避免后续更换模型时改动大量代码。

```
cmb_minigrid/
├── configs/
│   ├── env.yaml
│   ├── rssm_a.yaml
│   ├── rssm_b.yaml
│   └── cmb.yaml
│
├── environments/
│   ├── memory_env.py
│   ├── hidden_history_env.py
│   ├── observation_wrapper.py
│   └── snapshot.py
│
├── models/
│   ├── observation_encoder.py
│   ├── rssm.py
│   ├── world_model.py
│   ├── policy.py
│   └── prediction_heads.py
│
├── bridge/
│   ├── sender_adapter.py
│   ├── receiver_adapter.py
│   ├── cmb.py
│   └── baselines.py
│
├── data/
│   ├── collect_trajectories.py
│   ├── build_counterfactual_pairs.py
│   ├── trajectory_dataset.py
│   └── handoff_dataset.py
│
├── trainers/
│   ├── train_world_model.py
│   ├── train_policy.py
│   └── train_bridge.py
│
├── evaluation/
│   ├── handoff_runner.py
│   ├── eval_prediction.py
│   ├── eval_closed_loop.py
│   ├── eval_intervention.py
│   └── statistics.py
│
├── tests/
│   ├── test_env_pairs.py
│   ├── test_rssm_state.py
│   ├── test_state_timing.py
│   └── test_handoff.py
│
├── scripts/
│   ├── run_p0.sh
│   └── run_ablation.sh
│
└── README.md
```

建议先采用 PyTorch 实现，尽量只依赖 `torch`、`minigrid`、`gymnasium`、`numpy`、`hydra-core` 和常规日志、绘图工具。

不必首先复现 DreamerV3 的完整分布式训练与 Actor-Critic 系统。

## 四、MiniGrid 环境的具体实现

### 4.1 先统一观测格式

MiniGrid 默认返回一个字典，通常包含：

```
obs = {    "image": ...,      # [7, 7, 3]    "direction": ...,  # 0, 1, 2, 3    "mission": ...      # 任务指令}
```

`image` 的三个通道不是 RGB，而是：

\\[ (\mathrm{object\\\_id},\mathrm{color\\\_id},\mathrm{state\\\_id}) \\]

因此，建议第一版直接使用符号观测，不必引入 RGB 图像编码的额外复杂性。

在模型输入中，把三个类别通道分别做 Embedding，再送入 CNN 或 MLP。

例如：

```
object_feature = object_emb(image[..., 0])color_feature  = color_emb(image[..., 1])state_feature  = state_emb(image[..., 2])visual_feature = torch.cat(    [object_feature, color_feature, state_feature],    dim=-1)
```

后面再用一个小型卷积网络提取：

\\[ e_t=E(o_t)\in\mathbb R^{64} \\]

方向信息可以做 4 维 One-Hot，然后拼接到编码器输出中。

第一阶段所有实验保持任务指令不变，因此可以暂时不使用语言编码器。

必须避免把以下信息放入普通观测：全局地图、隐藏障碍位置、历史线索标签、正确分支，以及能够直接泄漏隐藏状态的自定义字段。

### 4.2 简化动作空间

官方 MiniGrid-Memory 有 7 个动作，但第一阶段真正重要的是：

| 动作 ID | 动作 |
| ----- | -- |
| 0     | 左转 |
| 1     | 右转 |
| 2     | 前进 |

建议通过 Action Wrapper 将可选动作限制为这三个。

在 RSSM 内部，上一时刻动作使用 One-Hot 编码。初始时刻 \\(t=0\\) 另用零向量表示没有上一动作，避免把初始化错误编码成“左转”。

等 CMB 验证成功后，再恢复完整 7 动作接口。

### 4.3 自定义环境应如何生成反事实历史？

建议给 `HiddenHistoryNav` 设置一个 `history_mode`，决定机器人早期能够观察到的隐藏变量。

例如：

```
history_mode = 0  # 左侧通道隐藏障碍history_mode = 1  # 右侧通道隐藏障碍
```

实现时，不要把这个变量直接交给 Agent。

它仅用于环境生成、专家规划和评估。

具体构造方法：

1. 生成一张固定拓扑的地图。
2. 在远离交接点的位置放置隐藏障碍。
3. 设计一段观测路径，使 A 在早期能够看到障碍物。
4. 让 A 经过一段公共走廊。
5. 将交接时刻固定在同一位置、朝向和局部视野。
6. 检查最近 8 帧局部观测及相应动作是否一致。
7. 验证两种历史对应的正确后续路线不同。

MiniGrid 官方支持通过继承 `MiniGridEnv` 并重写 `_gen_grid()` 生成地图，具体可以参照 [官方自定义环境教程](https://minigrid.farama.org/main/content/create_env_tutorial/)。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



### 4.4 反事实配对生成器

这是你们需要特别认真实现的模块。

建议每个样本对包含：

```
counterfactual_pair = {    "pair_id": 1001,    "history_1": trajectory_1,    "history_2": trajectory_2,    "handoff_step": t,    "handoff_obs_1": obs_1,    "handoff_obs_2": obs_2,    "optimal_action_1": action_1,    "optimal_action_2": action_2,}
```

然后执行自动验证：

```
import numpy as npdef validate_counterfactual_pair(pair, k=8):    h1 = pair["history_1"]    h2 = pair["history_2"]    t = pair["handoff_step"]    # 1. 当前观测完全相同    assert np.array_equal(        h1["observations"][t]["image"],        h2["observations"][t]["image"]    )    assert (        h1["observations"][t]["direction"]        == h2["observations"][t]["direction"]    )    # 2. 最近 k 步观测与动作相同    for j in range(t - k, t):        assert np.array_equal(            h1["observations"][j]["image"],            h2["observations"][j]["image"]        )        assert h1["actions"][j] == h2["actions"][j]    # 3. 历史中确实存在差异    assert pair["history_1_id"] != pair["history_2_id"]    # 4. 需要不同的后续决策    assert (        pair["optimal_action_1"]        != pair["optimal_action_2"]    )    return True
```

这是核心检查逻辑的示例。正式实现还应检查完整任务指令、交接点的位姿、终止标志、可见性以及隐藏变量是否真的在早期被 A 观察过。

另外，第一阶段应要求正确决策不同的反事实样本对，但不必让 A 和 B 在同一时间经历完全相同的全局环境状态。两种历史可以对应两张不同的隐藏地图，只要交接时提供给 B 的非记忆信息确实一致即可。

### 4.5 保存完整的 Handoff Trace

建议每条轨迹统一为：

\\[ \tau= (o_0,a_0,r_1,o_1,\ldots,a\_{T-1},r_T,o_T) \\]

同时保存环境完整状态作为评估用的独立字段。

| 字段                 | 使用目的                        |
| ------------------ | --------------------------- |
| `obs[t]`           | A/B 的正常观测                   |
| `action[t]`        | 从 \\(t\\) 到 \\(t+1\\) 执行的动作 |
| `reward[t+1]`      | 动作执行后的奖励                    |
| `terminated[t+1]`  | 是否达到终止状态                    |
| `env_state[t]`     | 仿真器真实状态，仅用于监督和评估            |
| `history_label`    | 早期线索或隐藏地图类别，仅用于诊断           |
| `handoff_t`        | 交接时刻                        |
| `expert_action[t]` | 专家规划得到的参考动作                 |
| `seed`             | 环境随机种子                      |
| `map_id`           | 环境布局标识                      |
| `pair_id`          | 对应反事实历史配对                   |

这里务必统一时间索引。`action[t]` 应当是在读取 `obs[t]` 后选择、用于产生 `obs[t+1]` 的动作。

后面所有 RSSM 更新和交接实验都使用这一约定。

## 五、RSSM 模型的实现

### 5.1 第一版架构

参考 DreamerV3 的状态设计：

\\[ s_t=(h_t,z_t) \\]

其中：

\\[ h_t\in\mathbb R^{d_h} \\]

\\[ z_t\in\\{0,1\\}^{G\times C} \\]

- \\(h_t\\) 是确定性循环状态。
- \\(G\\) 是离散随机变量的组数。
- \\(C\\) 是每组的类别数。
- \\(z_t\\) 是分组 One-Hot 随机状态。

建议配置：

| 参数                            | RSSM-A  | RSSM-B  |
| ----------------------------- | ------- | ------- |
| Observation Embedding         | 64      | 64      |
| Deterministic State \\(d_h\\) | 128     | 256     |
| Stochastic Groups \\(G\\)     | 16      | 16      |
| Categories \\(C\\)            | 16      | 16      |
| 展平随机状态维度                      | 256     | 256     |
| 总特征维度                         | 384     | 512     |
| 动力学核心                         | GRUCell | GRUCell |
| Prior 网络                      | MLP     | MLP     |
| Posterior 网络                  | MLP     | MLP     |

这是为了研究而缩小的配置，不是官方 DreamerV3 的默认大小。

官方实现的 RSSM 采用 Block GRU，并使用分组离散随机状态、Unimix 和分离的动态/表征 KL 损失。第一版可先用普通 `GRUCell` 验证接口，再在结构消融中换成 Block GRU。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



### 5.2 一个时间步如何运行？

建议实现以下计算：

\\[ h_t=f\_{\mathrm{GRU}}(h\_{t-1},z\_{t-1},a\_{t-1}) \\]

\\[ p_t=p\_\theta(z_t\mid h_t) \\]

\\[ q_t=q\_\theta(z_t\mid h_t,e_t) \\]

\\[ z_t\sim q_t \\]

最后生成特征：

\\[ f_t=[h_t;\operatorname{flatten}(z_t)] \\]

再把 \\(f_t\\) 提供给世界模型预测头与控制策略。

\\(h\_{t-1},z\_{t-1}\\)

\\(a\_{t-1}\\)

GRU Dynamics

\\(h_t\\)

Prior

\\(p(z_t\mid h_t)\\)

不使用当前观测

Posterior

\\(q(z_t\mid h_t,e_t)\\)

融合当前观测

\\(s_t=(h_t,z_t)\\)

### 5.3 RSSM 接口定义

我建议统一为以下 API：

```
class RSSM:    def initial(self, batch_size):        """返回初始 h、z 状态。"""    def observe_step(self, prev_state, prev_action, obs):        """融合新观测，得到 t 时刻 posterior 状态。"""    def imagine_step(self, state, action):        """不使用新观测，预测下一时刻 prior 状态。"""    def get_feature(self, state):        """返回拼接后的 RSSM 特征。"""    def set_state(self, state):        """校验并设置外部初始化状态。"""
```

这些是建议的 PyTorch 包装接口，并不全部对应官方同名方法。

建议在 `observe_step()` 中返回：

```
{    "state": {        "h": h_t,           # [B, deter_dim]        "z": z_t,           # [B, groups, classes]    },    "prior_logits": prior_logits,    "post_logits": post_logits,}
```

这样不仅方便训练世界模型，也方便后续直接生成 CMB 的输入与 Oracle-B 的教师状态。

### 5.4 离散随机状态的采样

第一版可以使用 PyTorch 的 Straight-Through Gumbel-Softmax：

```
import torch.nn.functional as Fdef sample_categorical(logits, training=True):    if training:        return F.gumbel_softmax(            logits,            tau=1.0,            hard=True,            dim=-1,        )    index = logits.argmax(dim=-1)    return F.one_hot(        index,        num_classes=logits.shape[-1],    ).float()
```

这能让前向得到 One-Hot 状态，同时通过近似梯度进行训练。

为了更接近 DreamerV3，还可以在概率分布中加入 `unimix=0.01`，即将模型预测分布与少量均匀分布混合。官方使用的是专门的 One-Hot 分布实现；上面的代码只是轻量原型的采样接口。

### 5.5 世界模型预测头

在 MiniGrid 中，建议使用三个预测头：

\\[ \hat o_t=D\_{\mathrm{obs}}(f_t) \\]

\\[ \hat r_t=D\_{\mathrm{reward}}(f_t) \\]

\\[ \hat c_t=D\_{\mathrm{continue}}(f_t) \\]

它们分别负责：

- 重建当前局部观测。
- 预测奖励。
- 预测是否继续执行。

对于符号 MiniGrid 图像，推荐为物体、颜色和状态通道分别使用分类交叉熵，而不是把类别 ID 当作连续 RGB 强行做 MSE。

由于 Memory 类任务的成功奖励比较稀疏，还可以增加一个独立的历史线索 Probe，例如预测早期看到的颜色或历史阻塞方向。

但应把它标记为辅助记忆监督实验，不能与仅使用世界模型原始损失的结果混为一谈。否则无法区分 RSSM 自发保留历史信息与辅助监督强制写入历史信息的作用。

## 六、RSSM-A 和 RSSM-B 应该怎样训练？

我建议采用一个两阶段训练方法：

先训练世界模型，再训练 Actor。

暂时不使用完整的 DreamerV3 Imagination Actor-Critic 训练系统，以减少初期工程量。

### 6.1 第一阶段：训练 RSSM 世界模型

构建数据集：

\\[ D=\\{\tau_1,\tau_2,\ldots,\tau_N\\} \\]

每条轨迹由专家策略和探索策略共同生成。

为了使模型既能预测成功路径，也能理解错误动作的后果，建议数据中包含成功轨迹、非最优轨迹和少量失败轨迹。

训练损失为：

\\[ \mathcal L\_{\mathrm{WM}} = \mathcal L\_{\mathrm{obs}} + \lambda_r\mathcal L\_{\mathrm{reward}} + \lambda_c\mathcal L\_{\mathrm{continue}} + \beta_d\mathcal L\_{\mathrm{dyn}} + \beta_p\mathcal L\_{\mathrm{rep}} \\]

其中：

\\[ \mathcal L\_{\mathrm{dyn}} = D\_{\mathrm{KL}} \left( \operatorname{sg}(q_t)\\|p_t \right) \\]

\\[ \mathcal L\_{\mathrm{rep}} = D\_{\mathrm{KL}} \left( q_t\\|\operatorname{sg}(p_t) \right) \\]

\\(\operatorname{sg}\\) 表示停止梯度。

这与 DreamerV3 将 dynamics KL 和 representation KL 分开优化的思想一致。正式实现可参考官方的 free-nats、损失系数与分布处理，而不要直接照搬未经验证的简化 KL 代码。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



建议的初始参数：

| 配置                | 起始值                  |
| ----------------- | -------------------- |
| Batch Size        | 32                   |
| Sequence Length   | 64                   |
| RSSM 隐状态 A/B      | 128 / 256            |
| Optimizer         | AdamW                |
| Learning Rate     | \\(3\times10^{-4}\\) |
| Gradient Clipping | 100（全局范数，先检查实际梯度尺度）  |
| KL Free Nats      | 1.0（按样本聚合后的 KL）      |
| 训练随机种子            | 至少 5 组               |

这些是调试起点，不是最终超参数。尤其需要针对 MiniGrid 的实际奖励分布、序列长度和优化稳定性重新调参。

### 6.2 第二阶段：训练闭环 Policy

Policy-B 输入：

\\[ f_t^B=[h_t^B;z_t^B] \\]

输出离散动作分布：

\\[ \pi_B(a_t\mid f_t^B,g) \\]

可以先使用行为克隆：

\\[ \mathcal L\_{\mathrm{BC}} = -\log \pi_B(a_t^\*\mid f_t^B,g) \\]

其中 \\(a_t^\*\\) 来自专家规划器。

专家规划器可以访问完整仿真器状态，但训练后的 Policy-B 不能直接接收完整地图或真实隐藏线索。它只能使用 RSSM 在部分观测历史下形成的状态。

这保证了：

- 训练时有可靠动作标签。
- 推理时仍然是部分可观测控制。
- 不会因为 CMB 提供了额外历史信息以外的特权数据而产生假阳性。

完成训练后，分别冻结 RSSM-A、RSSM-B 及 Policy-B。

### 6.3 先执行三个独立验收实验

训练 CMB 之前，必须获得如下基线：

| 测试         | 目的                |
| ---------- | ----------------- |
| A 从起点完整执行  | 确认发送模型能够记住历史线索    |
| B 从起点完整执行  | 确认接收模型具备利用历史导航的能力 |
| B 在岔路口重置状态 | 确认缺少历史会造成可测量的性能下降 |

建议将 Oracle-B 在标准记忆测试上的成功率达到 85% 以上作为初始工程门槛，并要求 Reset-B 明显低于 Oracle-B。

该数值只是建议，不是必须达到的科学标准。如果 Reset-B 与 Oracle-B 基本没有差距，应优先检查任务设置是否真正需要长期记忆，而不是马上训练 Bridge。

## 七、CMB 模块具体怎么实现？

第一版我建议使用 MLP Sender + 固定大小 Latent Carrier + MLP Receiver。

先验证方案有效，再考虑把 Sender 改成 Cross-Attention 或 Perceiver Resampler。

### 7.1 CMB 的数据流

发送方读取 A 的状态：

\\[ s_t^A=(h_t^A,z_t^A) \\]

将其展平：

\\[ x_t^A= [h_t^A;\operatorname{flatten}(z_t^A)] \\]

使用 Sender Adapter：

\\[ Z_t=E_A(x_t^A) \\]

然后，Receiver Adapter 结合 B 对当前观测的编码：

\\[ e_t^B=E_B(o_t) \\]

恢复 B 的状态：

\\[ \hat s_t^B=D_B(Z_t,e_t^B) \\]

这里 B 自己负责编码当前可见信息，CMB 主要负责补充历史信息。

建议暂时不使用 A 的全部历史观测作为 Sender 的额外输入。先验证 RSSM-A 的当前状态本身能否携带必要历史，再考虑短期时序特征和多 Token Cross-Attention。

### 7.2 Latent Carrier 维度

第一版推荐：

\\[ Z_t\in\mathbb R^{4\times16} \\]

即 64 个浮点数。

对于前面建议的 A 配置：

\\[ d(s_t^A)=128+16\times16=384 \\]

因此，64 维 CMB 确实构成了明显的表示容量约束。

建议的容量扫描：

| 配置   | 浮点数数量 | FP16 原始载荷 |
| ---- | ----- | --------- |
| 2×16 | 32    | 64 B      |
| 4×16 | 64    | 128 B     |
| 8×16 | 128   | 256 B     |
| 8×32 | 256   | 512 B     |

原先讨论过的 8×64 配置也可以保留作为高容量对照，但对于这个轻量 A，它有 512 个浮点数，已经大于 A 展平状态的 384 维，不能直接宣称它实现了维度压缩。

第一版用 MLP 输出并 reshape 成 Token 是合理的，但这些 Token 还不具有自动学出的独立空间语义。只有使用 Token 级注意力和 Slot 干预后，才进一步研究内部结构。

### 7.3 核心 PyTorch 实现

下面是一份与推荐模型配置匹配的 CMB 网络骨架。

```
            self.num_tokens,            self.token_dim,        )        # 3. Receiver 使用当前观测 + CMB        receiver_input = torch.cat(            [                tokens.flatten(1),                obs_embed_b,            ],            dim=-1,        )        output = self.receiver(receiver_input)        # 4. 解码为 B 的确定性/随机状态        h_b = output[:, :self.receiver_deter]        logits_b = output[:, self.receiver_deter:]        logits_b = logits_b.reshape(            batch_size,            self.receiver_groups,            self.receiver_classes,        )        # 5. 生成离散随机状态        if self.training:            z_b = F.gumbel_softmax(                logits_b,                tau=1.0,                hard=True,                dim=-1,            )        else:            index = logits_b.argmax(dim=-1)            z_b = F.one_hot(                index,                num_classes=self.receiver_classes,            ).float()        return {            "carrier": tokens,            "state_b": {                "h": h_b,                "z": z_b,            },            "logits_b": logits_b,        }
```

这段代码实现的是 CMB 的基本数据转换接口，还不是完整训练系统。

实际训练时应注意：使用 Bridge 生成 B 的状态后，不要立即重新用同一个 \\(o_t\\) 执行一次 `observe_step()`。因为 Receiver 已经利用 \\(o_t\\) 构造了时刻 \\(t\\) 的状态。

正确的后续流程是：

\\[ \hat s_t^B\xrightarrow{\pi_B}a_t \\]

环境执行动作后得到：

\\[ o\_{t+1} \\]

再更新：

\\[ \hat s\_{t+1}^B = \operatorname{Observe}\_B( \hat s_t^B,a_t,o\_{t+1} ) \\]

这是一个特别值得写入单元测试的时间索引问题。

### 7.4 为什么第一版先不用复杂的 Shared Hub？

目前只有 A 和 B 两个模型。

即使将中间空间命名为 CMB，数学上它仍可能只是一个 A→B 的瓶颈映射。

因此：

第一阶段可以证明跨模型状态交接和压缩记忆有效，但不能仅凭两个模型证明“真正模型无关的共享记忆空间”。

这个结论要到后续加入第三个模型 C，并验证多个 Sender/Receiver Adapter 可以共用同一个载体时才能进一步建立。

我建议第一版先验证功能，不要过早增加共享空间对齐和多个模型身份条件。

## 八、CMB 的训练数据怎么生成？

这是建议按固定协议实现的部分。

### 8.1 使用同一段真实历史，分别运行 A 和 B

对于训练轨迹：

\\[ H_t=(o_0,a_0,\ldots,o_t) \\]

先使用 A 的 RSSM 完整过滤：

\\[ s_t^A=\operatorname{Filter}\_A(H_t) \\]

然后使用 B 的 RSSM 完整过滤：

\\[ s_t^{B,\*}=\operatorname{Filter}\_B(H_t) \\]

这里的 \\(s_t^{B,\*}\\) 作为 Oracle-B 的教师状态。

关键是：A 和 B 读取同一段实际发生的观测与动作历史。

不要求过去动作一定由 A 生成。初期可以先使用专家或数据采集策略生成历史，后期再评估由 A 自己控制产生的历史。

这样可以避免 A 和 B 因为过去执行了不同动作而形成不同物理状态，使监督目标失去可比较性。

### 8.2 生成 Handoff Dataset

每个训练样本包含：

```
sample = {    "sender_state": {        "h": ...,        "z": ...,    },    "receiver_oracle_state": {        "h": ...,        "z": ...,    },    "receiver_obs": ...,    "receiver_obs_embedding": ...,    "future_actions": ...,    "future_observations": ...,    "expert_action": ...,    "hidden_history_label": ...,}
```

训练时，`receiver_obs_embedding` 也可以由冻结的 B Encoder 在线计算，避免保存重复数据。

在训练集生成过程中，每条长轨迹可以抽取多个切换点，但训练、验证和测试集必须按地图与轨迹族划分，避免相同的环境布局及其反事实配对分别出现在训练集和测试集中。

### 8.3 未来监督数据

对于每个交接点，至少保存之后 \\(K\\) 步：

\\[ (o\_{t+1:t+K},a\_{t:t+K-1}) \\]

建议：

\\[ K\in\\{5,10,20\\} \\]

这些未来动作可以来自数据采集策略或专家规划器。

在离线预测对齐时，CMB-B 与 Oracle-B 使用同一段未来动作，这样预测误差才不会混入不同策略选择动作造成的轨迹分叉。

如果一条轨迹提前终止，要用有效长度 Mask 忽略终止后的填充部分。

## 九、CMB 的损失函数怎么定义？

不建议只训练：

\\[ \\|\hat h_t^B-h_t^{B,\*}\\|^2 \\]

否则 Bridge 可能仅仅学习 B 的隐藏坐标，而没有真正恢复可用于未来控制的记忆。

建议分三层损失。

### 9.1 第一层：Oracle-B 状态辅助对齐

先定义：

\\[ \mathcal L_h= \\|\hat h_t^B-h_t^{B,\*}\\|\_2^2 \\]

对于随机状态，比较 Bridge 的类别分布与 Oracle-B 的后验分布：

\\[ \mathcal L_z= D\_{\mathrm{KL}} \left( q_B^\*(z_t)\\| q\_{\mathrm{bridge}}(z_t) \right) \\]

得到：

\\[ \mathcal L\_{\mathrm{state}} = \mathcal L_h+\lambda_z\mathcal L_z \\]

这样训练初期更容易收敛，但它只是辅助目标。

### 9.2 第二层：未来预测一致性

从：

\\[ \hat s_t^B,\quad s_t^{B,\*} \\]

分别使用 B 的 `imagine_step()` 展开相同的未来动作。

得到：

\\[ \hat p\_{t+k}^{B,\mathrm{CMB}} \\]

与：

\\[ \hat p\_{t+k}^{B,\mathrm{Oracle}} \\]

比较预测结果：

\\[ \mathcal L\_{\mathrm{future}} = \sum\_{k=1}^{K} w_k D\left( \hat p\_{t+k}^{B,\mathrm{CMB}}, \hat p\_{t+k}^{B,\mathrm{Oracle}} \right) \\]

其中 \\(D\\) 可以是预测观测类别分布的 KL 或交叉熵、未来奖励分布差异以及隐藏状态相关的辅助预测差异。

同时可以利用真实未来观测进行校验，避免仅仅让 Bridge 模仿教师模型自身的预测错误。

需要注意：训练 Bridge 时 B 的网络参数被冻结，但从 Bridge 输出状态经过 B 的未来预测网络，梯度仍应能够回传到 Bridge。

不能把 CMB-B 整条未来预测分支放在 `torch.no_grad()` 中。教师 Oracle-B 分支则可以停止梯度。

### 9.3 第三层：未来动作一致性

在时刻 \\(t\\) 比较：

\\[ \pi_B(\cdot\mid \hat s_t^B) \\]

与：

\\[ \pi_B(\cdot\mid s_t^{B,\*}) \\]

使用：

\\[ \mathcal L\_{\mathrm{action}} = D\_{\mathrm{KL}} \left( \operatorname{sg}(\pi_B^{Oracle}) \\| \pi_B^{CMB} \right) \\]

还可以让两条状态分支在相同的未来观测和动作序列下持续更新，再比较多个时间步的动作分布。

这能够避免 CMB 仅在切换的第一个动作上对齐，经过几次 RSSM 更新后就发生状态漂移。

### 9.4 总损失

建议第一版使用：

\\[ \boxed{ \mathcal L\_{\mathrm{CMB}} = \lambda_a\mathcal L\_{\mathrm{action}} + \lambda_f\mathcal L\_{\mathrm{future}} + \lambda_s\mathcal L\_{\mathrm{state}} } \\]

起始权重可以设置为：

| 损失                 | 初始建议权重 |
| ------------------ | ------ |
| Action Consistency | 1.0    |
| Future Prediction  | 1.0    |
| State Auxiliary    | 0.1    |

不同损失的原始数值尺度不同，所以这些权重不能机械套用。首先需要记录每项损失的量级与梯度，再进行归一化或调权。

第一版先通过固定 Token 大小控制容量，不必立刻加入复杂的可微通信率估计。

### 9.5 训练流程伪代码

```
# 说明性训练流程，非完整可直接运行脚本rssm_a.eval()rssm_b.eval()policy_b.eval()for model in [rssm_a, rssm_b, policy_b]:    for param in model.parameters():        param.requires_grad_(False)cmb.train()for batch in bridge_loader:    state_a = batch["sender_state"]    state_b_teacher = batch["receiver_oracle_state"]    obs_t = batch["receiver_obs"]    # B Encoder 不更新参数    with torch.no_grad():        obs_embed_b = encoder_b(obs_t)    # CMB 构造 B 的接管状态    output = cmb(state_a, obs_embed_b)    state_b_hat = output["state_b"]    # 1. 状态辅助对齐    state_loss = compute_state_loss(        output, state_b_teacher, batch    )    # 2. 动作分布一致性    # 学生分支保留对 CMB 输入的梯度    student_action = policy_b(state_b_hat)    with torch.no_grad():
```

正式训练脚本还需要实现离散概率分布 KL、随机状态对齐、终止 Mask、日志保存及验证集 Early Stopping。

建议先只使用 `State + Action` 损失确认 CMB 能够收敛，再加入 Future Prediction，观察多步稳定性是否改善。

## 十、Handoff Runner：如何真正执行 A→B 闭环交接？

这是实验从“特征映射”升级为“闭环控制恢复”的关键模块。

### 10.1 严格的交接协议

假设当前交接时刻是 \\(t\\)。

此前 A 已经完成了从 \\(0\\) 到 \\(t\\) 的历史观测更新。

此时：

\\[ s_t^A=\operatorname{Filter}\_A(H_t) \\]

CMB 输出：

\\[ \hat s_t^B=D_B(E_A(s_t^A),e_B(o_t)) \\]

随后 B 使用自己的 Policy 选择动作：

\\[ a_t\sim\pi_B(\cdot\mid\hat s_t^B) \\]

环境执行：

\\[ o\_{t+1},r\_{t+1} =\operatorname{EnvStep}(a_t) \\]

B 再更新：

\\[ \hat s\_{t+1}^B = \operatorname{Observe}\_B( \hat s_t^B,a_t,o\_{t+1} ) \\]

后续持续执行，直到任务完成或者失败。

切换完成后，不再使用 A 的内部状态，也不允许 B 重新读取完整历史。

这才能证明 B 能独立利用接收的记忆继续执行。

### 10.2 闭环执行伪代码

```
# 说明性伪代码：state_a 已在时刻 t# 使用 obs_t 完成了 posterior 更新@torch.no_grad()def run_cmb_handoff(    env,    rssm_b,    encoder_b,    policy_b,    cmb,    state_a,    obs_t,    max_steps=100,):    obs_embed_b = encoder_b(obs_t)    # 仅在交接时注入一次 CMB    result = cmb(state_a, obs_embed_b)    state_b = result["state_b"]    trajectory = []    for step in range(max_steps):        # B 根据恢复后的 belief 自主选择动作        action = policy_b.act(            state_b, deterministic=True        )        # 实际环境闭环        next_obs, reward, terminated, truncated, info = (            env.step(action)        )        # 用执行动作和新观测更新 B        state_b = rssm_b.observe_step(
```

这里的代码仅描述控制流程。实际实现中还需要做单样本 Batch 转换、NumPy/Tensor 数据转换以及动作格式处理。

### 10.3 怎样保证 Baseline 对比公平？

建议在交接点保存一个环境快照，然后对每种方法分别恢复相同的物理状态。

```
完整历史 H_t
     ↓
A 运行到交接位置
     ↓
保存 env_state(t)
     ↓
分别恢复相同环境
     ├── Reset-B      → 独立闭环
     ├── Replay-B     → 独立闭环
     ├── Linear-B     → 独立闭环
     ├── CMB-B        → 独立闭环
     └── Oracle-B     → 独立闭环
```

MiniGrid 的自定义环境可以通过保存地图对象状态、机器人位置、方向、携带物体、步数及随机数生成器状态实现快照。

对初期的静态确定性环境，也可以重新使用固定随机种子生成相同地图，然后重放交接前的动作序列。注意，仅使用相同种子不一定足以保证动态环境具有完全相同的未来随机状态。

每次恢复后，都应校验交接点的完整环境状态一致。

这一步非常重要，否则不同方法的成功率差异可能来自不同的物理初始状态，而不是 CMB。

## 十一、必须实现的对照实验

我建议第一轮至少保留六种方法。

| 方法               | 核心实现                       | 研究目的       |
| ---------------- | -------------------------- | ---------- |
| M0 Reset         | B 重置状态，并只处理当前观测            | 无历史下界      |
| M1 Replay-N      | B 利用最近 N 帧/动作重建状态          | 测试短期历史是否足够 |
| M2 Linear Bridge | A 的状态经线性映射初始化 B            | 最简单跨模型对齐   |
| M3 MLP Bridge    | A 的状态经非线性映射初始化 B           | 检验非线性映射的价值 |
| M4 CMB           | 压缩 Carrier + Receiver 条件解码 | 核心方法       |
| M5 Oracle-B      | B 读取完整历史并恢复状态              | 理想参考性能     |

其中 M2 和 M3 也应该允许利用 B 的当前观测编码，这样比较更公平。

建议将 Replay-N 分成三个窗口：

\\[ N=\\{4,8,16\\} \\]

再增加一个能够覆盖早期历史线索的 Replay-Full 作为参考。它理论上应接近 Oracle-B，因为两者获得的信息相同。

另外，对比 Linear、MLP 和 CMB 时必须报告通信预算。否则即使 CMB 发送的数据更少，性能稍低，也未必说明机制较差；反过来，如果 CMB 的消息比 MLP Baseline 更大，性能提升也不能简单归因于共享结构。

第一版还可以加一个不需要训练的显式记忆参考：

Explicit-History Baseline： 将早期颜色或历史障碍位置等真值变量直接传给 B 的专用接收模块。

但这应该作为信息充分性的参考或显式方法对照，不能把它当作与 CMB 完全相同的隐式条件。

### 11.1 为什么不把 CMB+Warm-up 作为第一轮主要对照？

因为你们首先需要证明 CMB 在交接当下就提供了有效的历史信息。

如果让 B 接管后观察很多步再评价，B 可能在新观测中重新获取信息，掩盖 Bridge 的实际作用。

因此建议先比较无额外 Warm-up 的即时交接，再单独研究 CMB+2/4 步状态校正。

## 十二、具体评价指标和计算方法

我建议把闭环成功率、动作正确率、历史因果敏感性列为主指标。

### 12.1 第一组：控制功能

| 指标                       | 定义                 |
| ------------------------ | ------------------ |
| Task Success Rate        | 完成导航任务的比例          |
| Correct Branch Rate      | 第一次关键分支选择正确的比例     |
| Collision / Failure Rate | 发生碰撞、错误终止或超时的比例    |
| Path Efficiency          | 成功轨迹长度相对于参考最短路径的效率 |
| Post-handoff Return      | 接管后获得的累计奖励         |

第一轮 MiniGrid-Memory 最重要的是 Correct Branch Rate。

因为在这一任务中，真正需要历史信息的决策可能只有一个。如果只看最终成功率，有时会混入路径长短和执行偶然性。

### 12.2 第二组：世界模型预测

在切换点，从 CMB-B 与 Oracle-B 的状态出发，施加相同未来动作。

计算：

\\[ \mathrm{PredictionError}(K) = D( \hat o\_{t+K}^{CMB}, o\_{t+K}^{true} ) \\]

也可以比较两种世界模型初始状态在相同动作下形成的未来预测分布。

推荐：

\\[ K\in\\{1,5,10,20\\} \\]

重点观察误差是否在切换之后迅速增加。

对于后续自定义的空间任务，还应增加隐藏区域占据预测准确率。

### 12.3 第三组：闭环功能恢复率

建议使用：

\\[ R\_{\mathrm{restore}} = \frac{ J\_{\mathrm{CMB}}-J\_{\mathrm{Reset}} }{ J\_{\mathrm{Oracle}}-J\_{\mathrm{Reset}} } \\]

其中 \\(J\\) 是在同一测试集上的平均性能。

例如，如果：

\\[ J\_{\mathrm{Reset}}=0.50 \\]

\\[ J\_{\mathrm{CMB}}=0.80 \\]

\\[ J\_{\mathrm{Oracle}}=0.90 \\]

那么：

\\[ R\_{\mathrm{restore}}=0.75 \\]

表示 CMB 恢复了 Reset 与 Oracle 之间约 75% 的性能差距。

这只是说明性计算，不是实验结果。

需要同时报告原始成功率。若 Oracle 与 Reset 差距很小，或者 CMB 超过 Oracle，该比值就不能直接解释为普通意义上的历史信息恢复百分比。

### 12.4 关键结果表

建议第一轮实验结束后生成：

| 方法            | 正确分支率 | 任务成功率 | 恢复率 | 消息大小       |
| ------------- | ----- | ----- | --- | ---------- |
| Reset-B       | 待测    | 待测    | 基准  | 0          |
| Replay-4      | 待测    | 待测    | 待测  | 历史窗口       |
| Replay-8      | 待测    | 待测    | 待测  | 历史窗口       |
| Linear Bridge | 待测    | 待测    | 待测  | 待测         |
| MLP Bridge    | 待测    | 待测    | 待测  | 待测         |
| CMB 4×16      | 待测    | 待测    | 待测  | 128 B FP16 |
| Oracle-B      | 待测    | 待测    | 参考  | 完整历史       |

这样能够直接判断 CMB 的实际收益，以及是否值得付出额外的 Adapter 训练成本。

## 十三、最关键的三个消融实验

我认为这三项实验应该成为论文的重点，而不仅作为附录。

### 实验 1：History Swap——交换两段历史的 CMB

选取两个当前观测相同，但正确动作不同的历史：

\\[ H_t^{left},\quad H_t^{right} \\]

生成：

\\[ Z_t^{left}=E_A(s_t^{A,left}) \\]

\\[ Z_t^{right}=E_A(s_t^{A,right}) \\]

在交接时将二者交换。

正确交接

History Left

CMB Left

B 选择左侧

错误记忆注入

History Left

CMB Right

B 可能转向右侧

需要评价：

\\[ P(a_t=\mathrm{Left}\mid Z^{left}) \\]

以及：

\\[ P(a_t=\mathrm{Left}\mid Z^{right}) \\]

如果交换 CMB 后，B 的动作按照被交换历史的内容发生可预测变化，说明 CMB 对决策具有因果影响。

但这项实验本身只能证明被编码的历史线索影响了 B，不能证明 CMB 已经保留全部空间信息。

因此还需要下一项实验。

### 实验 2：Memory Capacity——历史信息容量

改变 CMB 容量：

\\[ M\times d \in \\{2\times16,4\times16,8\times16,8\times32\\} \\]

分别测试标准 Memory 任务和多个历史变量共同决定动作的 HiddenHistoryNav 任务。

预期两种任务会出现不同特征：

- 简单颜色线索任务可能使用极低维度就能完成。
- 具有多个障碍、访问记录和历史事件的任务可能需要更多信息容量。

如果一个 32 维 CMB 就能解决单线索任务，并不意味着它足以保存复杂导航 Belief。

这也是必须尽快从标准 MiniGrid-Memory 扩展到自定义导航环境的原因。

### 实验 3：History Length——历史间隔

改变关键信息出现到交接之间的间隔：

\\[ L\in\\{8,16,32,64\\} \\]

分别观察：

- A 的记忆能力是否下降。
- Oracle-B 的性能是否下降。
- CMB-B 与 Oracle-B 的差距是否增大。
- Replay-N 在什么长度之后失去有效信息。

例如历史线索出现在 64 步之前，但 Replay-8 只读取最近 8 步。

如果 CMB-B 仍然能够正确决策，就能够支持其传递了长期历史信息的判断。

但要排除环境布局、任务难度和 Episode 长度随 \\(L\\) 增加而一起变化所造成的混杂影响。最好固定核心导航任务，仅改变无关的中间走廊长度或等待间隔。

## 十四、实验数据量与统计设计

建议先通过 Pilot 确定任务难度，再固定正式测试集。

下面是一组初始规模建议：

| 数据用途          | 建议规模     |
| ------------- | -------- |
| RSSM 训练轨迹     | 20,000 条 |
| RSSM 验证轨迹     | 2,000 条  |
| Bridge 训练交接样本 | 20,000 个 |
| Bridge 验证交接样本 | 2,000 个  |
| 正式测试反事实历史对    | 1,000 对  |
| 模型随机种子        | 至少 5 组   |

这些样本量不是基于预先计算的统计功效得出的固定要求，应根据 Pilot 的成功率方差调整。

正式测试时，应保持：

地图布局与任务配置独立划分。

不能让同一个地图种子的两条反事实历史分别进入训练集和测试集。

建议每种方法在相同的测试实例上运行，报告均值与 95% 置信区间；置信区间可按地图或反事实配对单元进行分组 Bootstrap。

另外，需要分别报告“固定一个 A/B 模型，仅改变环境种子”和“重新训练多个 A/B 模型随机种子”的结果，因为后者才能反映模型训练不稳定性对 CMB 的影响。

## 十五、项目开发顺序与验收标准

建议按照以下顺序推进，不要同时训练全部模块。

R0

环境和数据验证

实现 MiniGrid-Memory 轨迹采集、自定义环境、专家策略、反事实配对生成器。

验收： 能够自动生成当前观测一致、早期历史不同、正确动作不同的样本对；通过数据泄漏检查。

R1

独立 RSSM 和 Policy

实现 Encoder、RSSM、Decoder、Actor。分别训练 A 和 B。

验收： Full-history A/B 能解决历史依赖任务，Reset-B 明显退化；`observe_step` 和 `imagine_step` 可复现。

R2

最小 CMB 训练

构建 Oracle-B Handoff Dataset，先训练 Linear、MLP 和 4×16 CMB。

验收： CMB-B 在验证集上具备可测量的动作与未来预测恢复能力。

R3

闭环与反事实因果实验

实现 Snapshot Runner、历史交换、容量扫描及不同历史长度测试。

验收： CMB 能在独立测试环境中改善闭环控制，且交换记忆能产生符合历史变量的决策变化。

R4

异构与共享空间扩展

增加第三个 RSSM-C，并改变隐状态维度或编码器结构。

验收： 不同模型能接入公共载体，并在未参与直接配对训练的模型组合上验证泛化能力。

R0–R3 完成后，就已经能够比较系统地回答 CMB 在部分可观测导航中是否有效。

R4 则关系到你们能否进一步提出“共享 CMB 而不是 Pairwise Hidden Mapping”的架构贡献。

## 十六、最可能遇到的五个问题

| 观察到的问题                     | 可能原因                        | 优先处理办法                  |
| -------------------------- | --------------------------- | ----------------------- |
| Oracle-B 与 Reset-B 表现接近    | 环境没有真正依赖历史，或者发生信息泄漏         | 先检查反事实观测、任务指令和历史标签      |
| Oracle-B 自身也表现很差           | RSSM 没有学会保留线索，或 Policy 训练不足 | 先检查历史 Probe、行为克隆和完整历史推理 |
| CMB Hidden MSE 下降，但闭环性能无改善 | 学到了特征数值匹配，缺少功能性监督           | 增强动作一致性与多步预测损失          |
| CMB 接管第一步正确，后续迅速漂移         | 恢复状态不符合 B 的动力学分布            | 对多个未来时间步训练预测一致性         |
| CMB 与 MLP Baseline 几乎相同    | 两模型情况下 CMB 只是另一种瓶颈映射        | 增加第三个模型、跨模型组合和容量对照      |

我认为最需要警惕的是第二种问题。

如果 A 和 B 的 RSSM 本来就没有记住早期线索，那么 CMB 不可能凭空恢复这些历史信息。

因此，一定要先完成 A/B 自身的长期记忆验证，再进入 Bridge 训练。

还要避免另一个研究解释问题：如果测试只有“记住蓝色还是红色”这样的二分类信息，一个简单 MLP 就可能与 CMB 一样好。这不意味着 CMB 无效，而是当前任务的信息复杂度不足以区分两种方法。

## 十七、建议第一轮实际使用的统一配置

为了尽快开始实现，我建议先固定以下配置。

P0 最小实验配置

| 环境                  | MiniGrid-MemoryS13-v0                     |
| ------------------- | ----------------------------------------- |
| 观测                  | 7×7×3 局部符号观测 + 朝向                         |
| 动作                  | 左转、右转、前进                                  |
| RSSM-A              | GRU 128 + Categorical 16×16               |
| RSSM-B              | GRU 256 + Categorical 16×16               |
| Observation Encoder | Embedding + CNN，64 维                      |
| Policy              | MLP Actor，行为克隆训练                          |
| CMB                 | MLP Encoder + 4×16 Carrier + MLP Decoder  |
| 世界模型训练              | Observation + Reward + Continue + KL      |
| CMB 训练              | Action KL + Future Loss + State Auxiliary |
| 主要对照                | Reset、Replay、Linear、MLP、Oracle            |
| 核心指标                | 正确分支率、成功率、功能恢复率                           |

这是用于启动 Pilot 的建议配置，参数应在验证集上调整后再固定正式测试协议。

完成这一配置后，马上进入 P1 的 HiddenHistoryNav，重点增加视野外障碍、多个历史线索以及相同当前观测下的不同未来导航需求。

## 十八、最终希望形成的研究结论

整个实验应区分三个递进的结论。

第一层：历史信息可以跨模型传递。

证明在当前观测相同、历史不同的情况下，A 的 RSSM 状态经过 CMB 可以让 B 恢复正确的分支决策。

第二层：交接的信息能够持续支撑闭环控制。

证明 CMB 初始化后的 B，不仅在第一个动作上接近 Oracle-B，而且经过连续 RSSM 更新、多步预测和环境反馈后仍然保持较好性能。

第三层：共享隐式载体能适应真正的异构世界模型。

在多个具有不同隐状态结构的 RSSM 间验证共享 Carrier，通过功能性训练目标实现泛化，而不是为每一对模型分别学习专用 Hidden Mapping。

这三个结论需要分别设计实验，不能只用最终成功率一项指标同时证明。

### 我的实施建议

第一轮最值得优先实现的不是 CMB 网络，而是 `HiddenHistoryNav` 的反事实配对生成器，以及两个能可靠使用完整历史的 RSSM。

原因很简单：你们最核心的科学问题并不是“能否训练一个 Encoder/Decoder”，而是：

\\[ \boxed{ \begin{gathered} \text{当前观测相同、历史不同}\\\ \Downarrow\\\ \text{A 具有不同的历史 Belief}\\\ \Downarrow\\\ \text{CMB 传递这种差异}\\\ \Downarrow\\\ \text{B 恢复正确的未来预测和控制} \end{gathered} } \\]

只有把这个因果链条的每一步都验证清楚，后续移植到四足机器人或无人机 WAM 时，才能真正证明 CMB 机制的有效性，而不是仅仅观察到模型接管后的任务成功率有所变化。

这套方案可直接作为第一阶段的代码开发规范、实验登记表和组会执行计划。
