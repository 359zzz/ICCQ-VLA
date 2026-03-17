# ICCQ-VLA 导师汇报摘要版

最后更新：2026-03-17

## 一、研究主题

本项目拟围绕当前 Evo-RL 中复现的 RECAP / pi*0.6 离线强化学习思路，提出一个更完整的真实机器人 VLA post-training 方法：

**ICCQ-VLA：Intervention-Calibrated Conservative Chunk-Q for Real-World VLA Post-Training**

目标平台为 **Piper 真机机械臂**，目标投稿为 **NeurIPS**。

## 二、核心问题

当前 RECAP 风格流程的核心是：

- 学习一个语言条件的状态价值 `V(o, l)`
- 基于 n-step advantage 做样本筛选
- 将 advantage 硬阈值化成二值标签
- 再做 advantage-conditioned policy extraction

这套方法的问题是：

1. 它主要是 **状态价值**，而不是 **动作价值**，难以回答“在同一观测下哪个 action chunk 更好”。
2. 它依赖 Monte Carlo 风格价值估计，长时程任务里方差较大。
3. 它把连续的优势信息压成二值标签，损失了大量排序信息。
4. 它没有充分利用真实机器人最有价值的监督之一：**human intervention**。

## 三、核心方法

本工作计划将 RECAP 升级为一个统一的离线 RL 框架，包含两条主线：

### 1. Conservative / Calibrated Off-Policy Chunk-Q

将当前的 `V(o, l)` 升级为 chunk 级别动作条件 Q：

- `Q(o, c, l)`：输入观测、语言和一个 action chunk，输出该 chunk 的价值
- 使用 twin-Q + V 结构
- 使用 TD 学习、保守正则和 calibration loss

作用：

- 从“评估状态”升级为“评估动作块”
- 更适合做策略改进
- 解决离线 RL 中 OOD 动作高估问题

### 2. Human Intervention as Preference + Future Propagation

将真实机器人中的人工接管从“普通演示数据”升级为“偏好监督”：

- 当 intervention 发生时，说明“人类执行分支优于策略执行分支”
- 用人类实际执行 chunk 作为正样本
- 用 shadow policy chunk 作为负样本
- 将该偏好向未来若干步传播，提升长时程 credit assignment

作用：

- 更充分利用真实机器人中最有信息量的边界样本
- 不只学习“这一刻该怎么做”，而是学习“哪条未来轨迹更好”

### 3. Q-Weighted BC 做策略后训练

用 `A = Q - V` 形成连续权重，而不是使用二值 indicator：

- 高质量 chunk 更高权重
- 低质量 chunk 更低权重
- 对 `pi05` 或 `pi0` 做加权行为克隆

这一步是最终把 critic 的知识转化为 VLA policy 提升的关键。

## 四、预期创新点

本文计划形成 3 个主要创新：

1. **从状态价值 RECAP 升级到 chunk-Q 的真实机器人 VLA 后训练框架**
2. **首次系统地把 human intervention 建模为可传播的 preference signal**
3. **将 conservative Q、calibration 与 intervention preference 统一到一个真实机器人 offline RL 方法中**

## 五、实验设计

实验平台：

- 单臂 Piper
- 主 backbone 优先使用 `pi05`

实验内容：

- 复现当前 RECAP baseline
- 训练 ICCQ critic
- 用 Q-weighted BC 做 policy post-training
- 在 2 到 3 个真实任务上验证

主要指标：

- success rate
- time-to-success
- intervention frame ratio
- episodes with intervention
- critic calibration / ranking accuracy

核心 ablation：

- 去掉 conservative loss
- 去掉 calibration loss
- 去掉 preference loss
- 去掉 future propagation
- 不同 chunk horizon
- 不同 propagation horizon

## 六、工程落地

Evo-RL 侧主要需要新增 3 类模块：

1. **ICCQ critic 模块**
   - 新建 `src/lerobot/values/iccq/`

2. **intervention preference 数据构造脚本**
   - 生成正负 chunk pair
   - 支持 shadow policy replay

3. **Q-weighted BC 训练适配**
   - 复用当前 `lerobot_train.py` 中 per-sample weighted loss 机制

Piper 侧预计不需要改动底层 CAN 控制逻辑，主要改动发生在：

- 数据记录 provenance
- 离线 pair 构造
- critic 训练
- policy post-training

## 七、时间安排

当前计划采用 **3 周实验 + 1 周论文** 的内部冲刺节奏：

- 第 1 周：数据整理、baseline 复现、pair mining
- 第 2 周：ICCQ critic 训练与离线验证
- 第 3 周：Q-weighted BC 与真实机器人主实验
- 第 4 周：NeurIPS 风格论文写作

## 八、当前判断

如果只从“方法扎实、工程量可控、适合真实机器人、具备 NeurIPS 潜力”四个维度综合评估，当前最值得推进的主线就是：

**Conservative / Calibrated Off-Policy Chunk-Q + Human Intervention as Preference with Future Propagation**

这条线相比直接在 RECAP 上做样本打标小改动，更像一篇完整的、兼具数学与工程深度的真实机器人 RL 论文。
