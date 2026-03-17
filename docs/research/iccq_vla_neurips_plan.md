# ICCQ-VLA 研究计划

最后更新：2026-03-17

## 目标投稿

- 首选目标：NeurIPS 主会。
- 截至 2026-03-17，NeurIPS 官方页面已经公布 NeurIPS 2026 将于 2026-12-06 至 2026-12-12 在澳大利亚悉尼举行，但主会论文投稿截止时间尚未在官方页面正式发布。
- 实际建议：先把接下来 4 周当作内部冲刺周期，产出一版“可投稿级”的完整方法与实验结果；后续再根据正式 CFP 做扩展和打磨。

官方参考：
- https://neurips.cc/Conferences/FutureMeetings
- https://neurips.cc/

## 暂定题目

Intervention-Calibrated Conservative Chunk-Q for Real-World VLA Post-Training

方法简称：ICCQ-VLA

## 核心想法

我们希望把当前 RECAP 风格的状态价值学习流程：

- Monte Carlo 状态价值 `V(o, l)`
- n-step advantage 估计
- 硬阈值二值化
- indicator-conditioned 的策略提取

升级成一个统一的离线强化学习框架，核心由以下部分组成：

- chunk 级别、动作条件的双 Q 函数 `Q(o, c, l)`
- 保守且可校准的离线 Q 学习
- 将人类 intervention 视为 preference supervision，而不只是 imitation 数据
- 将 intervention 的偏好信息向未来若干状态传播
- 用 Q-advantage 加权的 VLA 行为克隆做 post-training

这不是在 RECAP 上的小修小补，而是同时改动了：

- critic 的对象：`V -> Q`
- 监督几何：binary indicator -> ranking / preference constraint
- 策略改进规则：hard-threshold extraction -> 连续 advantage weighting

## 研究问题

### RQ1

在真实机器人混合质量数据上，chunk-level 的 off-policy Q 是否比 Monte Carlo-V 更适合 VLA post-training？

### RQ2

在真实 Piper 数据集上，保守化与校准化的 Q 学习是否能够降低高估、提高后续策略改进的稳定性？

### RQ3

把 human intervention 视为 preference signal，而不是普通 demo，是否更有效？

### RQ4

把 intervention preference 向未来状态传播，是否能够改善长时程任务中的 credit assignment，并降低部署时的人类接管频率？

## 数学建模

我们把任务定义为一个 language-conditioned chunk MDP。

- 观测与任务输入：`x_t = (o_t, l)`
- 长度为 `H` 的 action chunk：`c_t = [a_t, ..., a_{t+H-1}]`
- 数据集：`D = D_demo ∪ D_policy ∪ D_intervene`
- chunk 结束后的 done 标志：`d_{t+H}`
- 折扣因子：`gamma`

第一版实现建议使用：

- 单臂 Piper
- 一个主 VLA backbone，优先 `pi05`
- chunk horizon `H = 10`
- 控制频率约 10 Hz

这样一个 chunk 大约对应 1 秒控制窗口，比较适合 intervention 敏感的操作任务。

## 方法设计

### 1. Chunk-Q Critic

我们训练：

- `Q_theta1(x, c)`
- `Q_theta2(x, c)`
- `V_psi(x)`

其中：

- `Q` 评估一个给定 observation + language 下的 action chunk 好坏
- `V` 评估状态价值，用于 advantage 与 policy improvement

#### 1.1 Chunk Return

对于 horizon 为 `H` 的 chunk：

`R_t^(H) = sum_{j=0}^{H-1} gamma^j r_{t+j}`

#### 1.2 TD Target

`y_t = R_t^(H) + gamma^H * (1 - d_{t+H}) * V_bar(x_{t+H})`

#### 1.3 Q 的 TD 损失

`L_TD = E_D [ huber(Q_theta1(x_t, c_t) - y_t) + huber(Q_theta2(x_t, c_t) - y_t) ]`

#### 1.4 Expectile Value 损失

定义：

`Q_min(x_t, c_t) = min(Q_theta1(x_t, c_t), Q_theta2(x_t, c_t))`

则：

`L_V = E_D [ L_tau(Q_min(x_t, c_t) - V_psi(x_t)) ]`

其中 `L_tau` 为 expectile loss，初始建议 `tau = 0.8`。

这部分沿用 IQL 风格思路，避免对连续 chunk action 做显式全局最大化。

### 2. Conservative Regularization

离线 Q 学习常见问题是对数据支持集之外的动作过度乐观。为此，我们加入 chunk 级别的保守项。

对每个状态 `x`，构造候选 chunk 集合 `C(x)`，来源包括：

- 实际执行的 chunk
- 记录下来的 policy chunk
- 离线 shadow replay 得到的 policy chunk
- 对数据 chunk 加扰动后的 chunk

定义：

`L_cons = E_x [ log( 1 / |C(x)| * sum_{c_hat in C(x)} exp(Q_min(x, c_hat) / alpha_c) ) - Q_min(x, c_D) / alpha_c ]`

其中 `c_D` 是数据中真实记录的 chunk。

初始超参数建议：

- `alpha_c = 1.0`
- `lambda_cons in {0.01, 0.05, 0.1}`

### 3. Calibration Loss

只做递推式 TD 往往数值尺度不稳定，因此需要加一个和观测回报对齐的校准项。

令 `G_hat_t` 为当前 RECAP / `pistar06` 流程中使用的 normalized Monte Carlo target。

则：

`L_cal = E_D [ huber(Q_min(x_t, c_t) - G_hat_t) ]`

这一步把：

- TD 结构
- 绝对回报标尺

结合起来，使 Q 更适合后续策略改进。

### 4. Human Intervention as Preference

我们把 intervention onset 解释为：

- 人类不是单纯在教“这一步怎么动”
- 而是在声明“当前分支不如另一条未来分支”

定义 intervention onset `tau`：

- `is_intervention_tau = 1`
- `is_intervention_(tau-1) = 0`

对于每个 onset，在未来传播窗口 `L` 内构造 preference pair。

对 `k in {0, ..., L}`：

- 正样本 chunk：`c^+_(tau+k)`，来自人类实际执行的轨迹
- 负样本 chunk：`c^-_(tau+k)`，来自同一时刻 shadow policy 的轨迹

传播权重：

`w_k = lambda_prop^k`

初始建议：

- `L = 8`
- `lambda_prop = 0.8`

#### 4.1 Preference Ranking Loss

`L_pref = E [ w_k * log(1 + exp(-(Q_min(x_(tau+k), c^+_(tau+k)) - Q_min(x_(tau+k), c^-_(tau+k)) - m))) ]`

初始 margin：

- `m = 0.02`

这是本文最有辨识度的部分：把 intervention 从 imitation 数据提升为局部排序监督。

### 5. 总体 Critic 目标

`L_Q = L_TD + lambda_cons * L_cons + lambda_cal * L_cal + lambda_pref * L_pref + lambda_V * L_V`

初始建议：

- `lambda_cons = 0.05`
- `lambda_cal = 1.0`
- `lambda_pref = 0.5`
- `lambda_V = 1.0`

### 6. 策略改进

主方法建议使用 Q-advantage-weighted behavior cloning，而不是第一版就把 DPO 当主方法。

定义：

`A_t = Q_min(x_t, c_t) - V_psi(x_t)`

权重：

`omega_t = clip(exp(A_t / beta), omega_min, omega_max)`

初始建议：

- `beta = 0.2`
- `omega_min = 0`
- `omega_max = 10`

策略训练目标：

`L_pi = E_D [ omega_t * l_BC(pi_phi(x_t), c_t) ]`

这部分应该作为主论文的默认策略提取方式。

DPO / chiPO 等 preference-based extraction 更适合作为 ablation 或扩展实验。

## 核心算法框图

```text
Piper Human-in-the-Loop Rollouts
    |
    v
Dataset D
(obs, task, executed action, policy action,
 is_intervention, collector_policy_id, success/failure)
    |
    v
Offline Shadow Policy Replay
(恢复 intervention 窗口附近的负样本 policy chunks)
    |
    v
Chunk Transition Builder + Preference Pair Miner
(x_t, c_t, x_{t+H}, R_t^(H), G_hat_t)
(x_{tau+k}, c^+_{tau+k}, c^-_{tau+k}, w_k)
    |
    v
ICCQ Critic Training
L_TD + L_cons + L_cal + L_pref + L_V
    |
    v
Q-Advantage Computation
A_t = Q(x_t, c_t) - V(x_t)
omega_t = clip(exp(A_t / beta))
    |
    v
Weighted BC Post-Training of pi05/pi0
    |
    v
Deploy on Piper + Collect Next Round
```

## 数据需求

当前 Evo-RL 的 human-in-the-loop 采集链路实际上已经记录了这项工作所需的大部分字段：

- executed action
- `complementary_info.policy_action`
- `complementary_info.is_intervention`
- `complementary_info.collector_policy_id`
- episode success / failure label

但有一个非常重要的改进建议：

- `collector_policy_id` 必须改成可唯一恢复 checkpoint 的标识
- 最好直接存完整 checkpoint path，或者 `(repo_id, checkpoint_step, git_commit)` 这种唯一键

因为 preference mining 依赖 shadow replay，shadow replay 又依赖严格的数据 provenance。

## 实验计划

### 主对比

- RECAP baseline
- MC-V + weighted BC
- chunk-Q only
- chunk-Q + calibration
- chunk-Q + conservative regularization
- chunk-Q + intervention preference
- chunk-Q + intervention preference + future propagation
- full ICCQ-VLA

### 可选次级对比

- full ICCQ-VLA + DPO extraction
- full ICCQ-VLA + chiPO extraction

### 任务设置

至少做 2 个真实 Piper 任务，最好 3 个。

- 任务 A：短时程 pick 或 lift
- 任务 B：更精确的放置或对齐任务
- 任务 C：更长时程或更接触敏感的任务

如果当前只有一个成熟任务，至少再引入两个变体：

- 物体变化
- 目标位置变化
- 杂乱或光照变化

### 指标

真实机器人主指标：

- success rate
- time-to-success
- intervention frame ratio
- episodes with intervention
- average intervention count per episode
- success under fixed intervention budget

critic 质量指标：

- Q-to-return MAE
- calibration error
- intervention pair ranking accuracy
- Q-advantage 与 rollout quality 的 Spearman correlation

工程指标：

- training stability
- critic convergence stability
- offline shadow replay runtime

### 必做 Ablation

- 去掉 conservative loss
- 去掉 calibration loss
- 去掉 preference loss
- preference 只作用于 onset，不做 future propagation
- 传播长度 `L in {0, 3, 5, 8, 12}`
- chunk horizon `H in {5, 10, 15}`
- binary RECAP-style extraction vs weighted BC
- logged `policy_action` negatives vs offline shadow replay negatives

## Evo-RL 代码适配计划

### A. 新建 ICCQ value 家族

建议新增：

- `src/lerobot/values/iccq/configuration_iccq.py`
- `src/lerobot/values/iccq/modeling_iccq.py`
- `src/lerobot/values/iccq/processor_iccq.py`

实现策略：

- 复用当前 `pistar06` 的 observation encoder 结构
- 新增 action chunk encoder
- 输出 `Q1`、`Q2` 和 `V`

最值得复用的本地参考：

- `src/lerobot/values/pistar06/configuration_pistar06.py`
- `src/lerobot/values/pistar06/modeling_pistar06.py`
- `src/lerobot/values/pistar06/processor_pistar06.py`

### B. 新建 Q 训练与推理脚本

当前 value 流水线是围绕 `pistar06` 风格的 scalar value 写的。为了清晰，建议不要硬塞进现有 `value_train/value_infer`，而是单独新建脚本。

推荐新增：

- `src/lerobot/scripts/lerobot_q_train.py`
- `src/lerobot/scripts/lerobot_q_infer.py`
- `src/lerobot/configs/q_train.py`
- `src/lerobot/configs/q_infer.py`

原因：

- Q-learning 需要 chunk action 和 preference pair
- 输入输出字段、训练目标与现有 scalar value 流程差异较大

### C. 新建 Preference Pair 构造脚本

建议新增：

- `src/lerobot/scripts/lerobot_prepare_intervention_preferences.py`

职责：

- 扫描 `complementary_info.is_intervention`
- 检测 onset 和 release
- 按 `collector_policy_id` 加载旧 policy checkpoint
- 离线重放 shadow policy chunk
- 构造 `(x, c^+, c^-, w)` 记录
- 保存成 parquet 或 safetensors 元数据

### D. 改进 Collector Policy Provenance

建议修改：

- `src/lerobot/utils/recording_annotations.py`
- `src/lerobot/scripts/lerobot_human_inloop_record.py`

目标：

- 不再存模糊的短 policy version name
- 改成可唯一恢复的 checkpoint provenance

### E. 复用 Delta Timestamp 机制

这个仓库已有非常适合 chunk-Q 的时间索引基础设施，不要重造一套 loader。

相关文件：

- `src/lerobot/configs/policies.py`
- `src/lerobot/datasets/factory.py`

对 `ICCQConfig`，建议定义：

- `action_delta_indices = [0, 1, ..., H-1]`
- `reward_delta_indices = [0, 1, ..., H-1]`
- `observation_delta_indices = [0, H]` 或一个短时间窗口

这样 chunk 构造会非常简洁。

### F. 接入 Q-Weighted BC

现有 `lerobot_train.py` 已经支持 per-sample weighted BC，并且有 RA-BC 的参考实现。

相关文件：

- `src/lerobot/scripts/lerobot_train.py`
- `src/lerobot/configs/train.py`
- `src/lerobot/utils/rabc.py`

建议平行新增：

- `src/lerobot/utils/qbc.py`

建议新增配置项：

- `use_qbc`
- `qbc_weight_path`
- `qbc_beta`
- `qbc_clip_max`

流程：

- `lerobot_q_infer.py` 输出每帧的 Q-advantage 权重
- `lerobot_train.py` 按 `index` 读取这些权重
- 用 per-sample weighted BC 进行策略后训练

### G. 可选：在线 Shadow Logging

作为第二阶段扩展，可以改：

- `src/lerobot/scripts/recording_loop.py`

增加类似开关：

- `store_shadow_policy_during_intervention=true`

但不建议作为第一阶段内容，因为它可能扰动 Piper 上的实时控制。

### H. Piper 相关说明

Piper 低层控制栈本身不建议第一阶段大改。

已有相关配置：

- `src/lerobot/robots/piper_follower/config_piper_follower.py`
- `src/lerobot/teleoperators/piper_leader/config_piper_leader.py`

推荐策略：

- 第一阶段尽量不碰 CAN 控制环
- 把创新全部放在数据标注、离线 pair 构造、critic 训练、policy post-training 上

## 里程碑

### Milestone 1

Baseline 可复现：

- 复现当前 RECAP 风格流程
- 验证数据标注与 policy provenance

### Milestone 2

Preference mining：

- intervention onset 检测
- shadow replay
- pair dataset 导出

### Milestone 3

Critic：

- chunk-Q + V
- TD + calibration
- conservative regularization
- intervention preference propagation

### Milestone 4

Policy improvement：

- 在 `pi05` 上做 Q-weighted BC
- 完成真实机器人评估

### Milestone 5

Paper readiness：

- 完整 ablation table
- critic analysis plots
- intervention 可视化案例

## 论文提纲

### 标题

Intervention-Calibrated Conservative Chunk-Q for Real-World Vision-Language-Action Post-Training

### 摘要

- 问题：当前真实机器人上的 VLA post-training 过度依赖 state-value surrogate 和较弱的 policy extraction
- 方法：chunk-level conservative calibrated Q-learning + intervention preference propagation
- 场景：真实 Piper 机械臂上的 human-in-the-loop 校正
- 结论：更好的 credit assignment、更低 intervention rate、更高 success

### 1. 引言

- 引出 VLA post-training 的现实问题
- 说明 state-value RECAP 的局限
- 引出 human intervention 是尚未被充分利用的 ranking supervision
- 总结贡献

### 2. 相关工作

- VLA post-training 与 RECAP
- offline RL：CQL、IQL、Cal-QL
- human feedback 与 intervention learning
- preference optimization

### 3. 预备知识

- language-conditioned chunk MDP
- RECAP 风格 baseline
- intervention window 与 policy provenance 记号

### 4. 方法

- chunk-Q critic
- conservative regularization
- calibration
- intervention preference propagation
- Q-weighted behavior cloning

### 5. 实验设置

- Piper 硬件与传感器
- 任务
- 数据集
- policy backbone
- 训练细节
- 评测指标

### 6. 主结果

- 总体性能对比
- intervention reduction
- 不同任务变体下的鲁棒性

### 7. 分析

- calibration plots
- ranking accuracy
- propagation horizon ablation
- conservative coefficient ablation
- negative source ablation

### 8. 讨论

- 为什么 intervention preference 有效
- 哪些场景 chunk-Q 最关键
- 局限性与失败案例

### 9. 结论

- 总结
- 对真实世界 VLA-RL 的意义

### 附录

- 超参数
- 更多 rollout 案例
- 实现细节
- 硬件与安全说明

## 关键风险

- 如果 collector provenance 不唯一，shadow replay 可能不可靠
- chunk horizon 过长，Q-learning 会不稳定
- propagation horizon `L` 过大，会出现错误传播
- critic 质量提升未必自动转化为 policy gain，取决于 extraction 强度

## 风险缓解

- 先修 provenance
- `H = 10` 起步
- `L = 5 或 8` 起步
- 第一版 extraction 先用 weighted BC，再尝试更复杂的 preference optimization

## 第一版“可投稿”成功标准

- 方法在 Piper 上端到端跑通
- 至少 2 个真实任务
- 至少 1 个强 baseline 和 4 个有说服力的 ablation
- 相比 RECAP，在 success rate 或 intervention reduction 上有清晰提升
- 离线 critic 分析能说明 Q 比 MC-V 更准、更稳、更能排序
