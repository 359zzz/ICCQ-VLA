# ICCQ-VLA NeurIPS 冲刺清单

最后更新：2026-03-17

这份页面可以直接贴进 Notion 作为 4 周执行清单。

## 冲刺目标

- 用 3 周时间完成 ICCQ-VLA 在 Piper 上一版“可投稿级”的实验故事。
- 第 4 周把方法与结果整理成 NeurIPS 风格完整论文草稿。

## 背景

- 目标 venue：NeurIPS 主会
- 截至 2026-03-17，NeurIPS 2026 会议日期已公布，但主会论文 deadline 尚未在官方页面发布
- 因此当前这 4 周视为内部 deadline

## 总体交付物

- RECAP baseline 在 Piper 上可复现
- 干净的人类在环数据集，带唯一 collector policy provenance
- preference pair mining 流水线
- ICCQ critic 训练流水线
- Q-weighted BC post-training 流水线
- 主实验与 ablation 的真实机器人结果
- 论文完整初稿

---

## 第 1 周：数据 + Baseline + Pair Mining

### 本周目标

- 先把数据搞干净
- 复现 baseline
- 完成 intervention preference pair 构造

### 周一

- [ ] 固定当前 Piper 实验所使用的 baseline policy / checkpoint
- [ ] 导出当前任务列表、机器人配置、相机配置、控制频率
- [ ] 对所有候选训练数据集运行 dataset report
- [ ] 检查以下字段是否存在：
  - `action`
  - `complementary_info.policy_action`
  - `complementary_info.is_intervention`
  - `complementary_info.collector_policy_id`
  - `episode_success`

### 周二

- [ ] 修改 provenance 记录逻辑，使 `collector_policy_id` 可唯一恢复 checkpoint
- [ ] 如果当前 provenance 不够唯一，重新录制一小部分验证数据
- [ ] 验证 intervention onset 是否能稳定检测

### 周三

- [ ] 在一个 Piper 任务上复现当前 RECAP / Evo-RL baseline
- [ ] 保存精确训练命令、checkpoint 路径、评测命令
- [ ] 建立 baseline 结果表：
  - success rate
  - average time-to-success
  - intervention frame ratio

### 周四

- [ ] 实现 `lerobot_prepare_intervention_preferences.py`
- [ ] 检测 intervention onset 与 release
- [ ] 用 executed actions 构造 positive chunks
- [ ] 用记录下来的 `policy_action` 构造 negative chunks

### 周五

- [ ] 加入 offline shadow replay，按 `collector_policy_id` 重放旧 policy
- [ ] 比较 logged `policy_action` 与 shadow replay action chunk 的差异
- [ ] 导出一个包含 20 到 50 组样本的小型诊断 preference 数据

### 周六

- [ ] 可视化 10 段 intervention 序列，内容包括：
  - front 或 wrist 图像
  - human chunk
  - policy/shadow chunk
  - intervention onset
- [ ] 人工 sanity check：这些 preference 是否语义合理

### 周日

- [ ] 完成本周复盘
- [ ] 固定最终 pair schema
- [ ] 写 1 页简短总结：
  - 什么跑通了
  - 什么失败了
  - 下周要先解决什么

### 第 1 周退出标准

- [ ] RECAP baseline 已复现
- [ ] pair mining 脚本可运行
- [ ] provenance 已修正
- [ ] 小规模 pair 数据已人工验证

---

## 第 2 周：ICCQ Critic

### 本周目标

- 训练一个稳定的 chunk-Q critic
- 用离线指标证明它优于 MC-V

### 周一

- [ ] 创建 `ICCQConfig`、`ICCQPolicy`、`make_iccq_pre_post_processors`
- [ ] 复用 `pistar06` 的视觉/语言/状态编码路径
- [ ] 增加 action chunk encoder
- [ ] 输出 `Q1`、`Q2` 和 `V`

### 周二

- [ ] 实现 `lerobot_q_train.py`
- [ ] 实现 `H = 10` 的 chunk TD target
- [ ] 实现 expectile `V` loss
- [ ] 训练最小版模型：
  - 只有 TD
  - 不加 conservative
  - 不加 preference

### 周三

- [ ] 加入 calibration loss，与 normalized Monte Carlo target 对齐
- [ ] 比较：
  - TD only
  - TD + calibration
- [ ] 画 Q vs return 的 calibration 图

### 周四

- [ ] 加入 conservative regularization
- [ ] 构造候选 action 集合，来源包括：
  - data chunk
  - logged policy chunk
  - shadow replay chunk
  - 扰动 chunk
- [ ] 调 `lambda_cons`

### 周五

- [ ] 加入 intervention preference ranking loss
- [ ] 加入 future propagation：`L = 8`，`lambda_prop = 0.8`
- [ ] 比较：
  - 无 preference
  - onset only
  - onset + propagation

### 周六

- [ ] 运行离线评估：
  - Q-to-return MAE
  - calibration error
  - preference ranking accuracy
  - 与 rollout quality 的 Spearman correlation

### 周日

- [ ] 完成本周复盘
- [ ] 选出最优 critic checkpoint
- [ ] 固定第 3 周使用的 critic 超参数

### 第 2 周退出标准

- [ ] ICCQ 训练稳定
- [ ] critic 离线指标优于 MC-V baseline
- [ ] preference propagation 有明确增益

---

## 第 3 周：Policy Post-Training + Real Robot

### 本周目标

- 用 ICCQ 改进 VLA policy
- 产出核心真实机器人结果表

### 周一

- [ ] 在 `lerobot_q_infer.py` 中实现 Q-weight 导出
- [ ] 为每帧保存：
  - `Q`
  - `V`
  - `A = Q - V`
  - clipped policy weight

### 周二

- [ ] 在 `lerobot_train.py` 中加入 `use_qbc` 路径
- [ ] 复用 RA-BC 的 per-sample weighted loss 机制
- [ ] 在一个任务上训练 `pi05` 的 Q-weighted BC

### 周三

- [ ] 扩展 Q-weighted BC 到全部目标任务
- [ ] 在真实 Piper 上评估训练后的 policy
- [ ] 对比：
  - 原始 policy
  - RECAP baseline

### 周四

- [ ] 运行主对比实验：
  - RECAP baseline
  - Q only
  - Q + calibration
  - Q + preference
  - full ICCQ
- [ ] 保证每个设置都有足够的 trial 数量，形成稳定结果表

### 周五

- [ ] 运行关键 ablation：
  - no conservative
  - no calibration
  - no preference
  - no propagation
  - `H in {5, 10, 15}`
  - `L in {0, 5, 8}`

### 周六

- [ ] 生成最终图表：
  - 方法框图
  - intervention 可视化图
  - critic calibration 图
  - 主结果表
  - ablation 表

### 周日

- [ ] 完成本周复盘
- [ ] 冻结所有实验结果
- [ ] 写 1 页结论摘要：
  - 主 claim
  - 最强证据
  - 当前还弱的地方

### 第 3 周退出标准

- [ ] 方法在 Piper 上端到端跑通
- [ ] 最终主表完成
- [ ] 至少有一个强结果明显优于 RECAP
- [ ] 核心 ablation 齐全

---

## 第 4 周：论文写作

### 本周目标

- 完成一版 NeurIPS 风格的完整论文草稿

### 周一：锁故事

- [ ] 固定标题、方法名、贡献点
- [ ] 固定论文提纲
- [ ] 确定主图和主表顺序

### 周二：方法与实验设置

- [ ] 写 Introduction
- [ ] 写 Preliminaries
- [ ] 写 Method section 与核心公式
- [ ] 写 Experimental Setup

### 周三：结果与分析

- [ ] 写 Main Results
- [ ] 写 Ablations
- [ ] 写 Critic Analysis
- [ ] 写 Failure Cases 与 Limitations

### 周四：润色与附录

- [ ] 写 Related Work
- [ ] 写 Conclusion
- [ ] 写 Appendix：
  - 超参数
  - 实现细节
  - 硬件细节
  - 额外 rollout 案例

### 周五：内部审稿

- [ ] 通读检查逻辑断点
- [ ] 检查所有 claim 是否有表格和图支撑
- [ ] 检查可复现性说明与命令记录
- [ ] 形成 submission-ready checklist

### 第 4 周退出标准

- [ ] 全文初稿完成
- [ ] 图表编号稳定
- [ ] 附录完整
- [ ] 内部 review 问题已处理

---

## 建议在 Notion 里创建的子页面

- [ ] `00 - Submission Tracker`
- [ ] `01 - Baseline Reproduction`
- [ ] `02 - Pair Mining`
- [ ] `03 - ICCQ Critic`
- [ ] `04 - Q-Weighted BC`
- [ ] `05 - Real Robot Eval`
- [ ] `06 - Figures and Tables`
- [ ] `07 - Paper Draft`

---

## 建议建立的 Notion 数据库

### 实验数据库

字段建议：

- [ ] Name
- [ ] Date
- [ ] Task
- [ ] Policy backbone
- [ ] Critic version
- [ ] Dataset revision
- [ ] Train command
- [ ] Eval command
- [ ] Main metrics
- [ ] Notes
- [ ] Status

### Figure Tracker

字段建议：

- [ ] Figure ID
- [ ] Purpose
- [ ] Data source
- [ ] Script path
- [ ] Owner
- [ ] Status

### Writing Tracker

字段建议：

- [ ] Section
- [ ] Status
- [ ] Owner
- [ ] Blocking issue
- [ ] Last updated

---

## 每日执行纪律

- [ ] 每个训练或评测 run 都记录精确命令与 checkpoint 路径
- [ ] 每次真实机器人评测都写一段 observation note
- [ ] 上一个 ablation 没总结前，不开始新的 ablation
- [ ] 每天结束前更新：
  - 今天完成了什么
  - 今天失败了什么
  - 明天最先要解决什么

---

## 最终验收清单

- [ ] 方法主张明显强于 RECAP
- [ ] conservative Q 的贡献被单独隔离出来
- [ ] intervention preference 的贡献被单独隔离出来
- [ ] future propagation 的贡献被单独隔离出来
- [ ] Piper 真机结果是正文核心，而不是附录补充
- [ ] 所有命令、seed、checkpoint 都已归档
