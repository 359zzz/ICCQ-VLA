# ICCQ-VLA Single-Arm Workflow

最后更新：2026-03-17

这份文档对应当前仓库里已经接好的三条链路：

- 人类接管数据采集与 provenance 标注
- intervention preference 挖掘
- `ICCQ -> q_infer -> QBC post-training`

默认对象是单臂 AgileX PiPER / PiPER-X。

如果你用的是非 X 版本，把下面命令里的：

- `piperx_follower` 改成 `piper_follower`
- `piperx_leader` 改成 `piper_leader`

## 0. 前置约定

先准备这些变量：

```bash
DATASET_ROOT=/path/to/local_dataset_root
DATASET_REPO=your_name/iccq_single_arm_round1
TASK_DESC="pick up the bottle and place it on the coaster"

FOLLOWER_CAN_PORT=can0
LEADER_CAN_PORT=can1

BASE_POLICY=/path/to/base_policy_or_checkpoint
Q_TRAIN_OUT=outputs/q_train/iccq_single_arm
Q_INFER_OUT=outputs/q_infer/iccq_single_arm
Q_VALUES=${Q_INFER_OUT}/q_values.parquet
PREFS=outputs/intervention_preferences/iccq_single_arm_prefs.parquet
QBC_OUT=outputs/train/pi05_qbc_single_arm
```

建议第一轮先全部本地落盘，先不要混进多轮 merge。

## 1. 确认单臂遥操作正常

先确认机械臂、leader、CAN 都通。

```bash
lerobot-teleoperate \
  --robot.type=piperx_follower \
  --robot.port=${FOLLOWER_CAN_PORT} \
  --robot.id=my_piperx_follower \
  --robot.require_calibration=false \
  --teleop.type=piperx_leader \
  --teleop.port=${LEADER_CAN_PORT} \
  --teleop.id=my_piperx_leader \
  --teleop.require_calibration=false
```

通过标准：

- follower 跟手正常
- gripper 同步正常
- 没有持续报 CAN / calibration 错误

## 2. 采集人类接管数据

如果你要做 ICCQ，建议直接用 `lerobot-human-inloop-record`，不要再用普通 `lerobot-record`。

它现在会自动打开：

- `episode_success` 标注
- `complementary_info.collector_policy_id`
- `complementary_info.collector_source`
- `complementary_info.is_intervention`
- `complementary_info.policy_action`（有 policy 时）

### 2.1 纯人工首轮数据

如果你还没有一个 base policy，可以先录纯人工数据：

```bash
lerobot-human-inloop-record \
  --robot.type=piperx_follower \
  --robot.port=${FOLLOWER_CAN_PORT} \
  --robot.id=my_piperx_follower \
  --robot.require_calibration=false \
  --teleop.type=piperx_leader \
  --teleop.port=${LEADER_CAN_PORT} \
  --teleop.id=my_piperx_leader \
  --teleop.require_calibration=false \
  --dataset.repo_id=${DATASET_REPO} \
  --dataset.root=${DATASET_ROOT} \
  --dataset.single_task="${TASK_DESC}" \
  --dataset.num_episodes=30 \
  --dataset.episode_time_s=25 \
  --dataset.reset_time_s=8 \
  --dataset.push_to_hub=false \
  --display_data=true
```

### 2.2 带 policy 的 HIL 数据

如果你已经有一个 base policy，要录 intervention 数据，就把 `--policy.path` 带上：

```bash
lerobot-human-inloop-record \
  --robot.type=piperx_follower \
  --robot.port=${FOLLOWER_CAN_PORT} \
  --robot.id=my_piperx_follower \
  --robot.require_calibration=false \
  --teleop.type=piperx_leader \
  --teleop.port=${LEADER_CAN_PORT} \
  --teleop.id=my_piperx_leader \
  --teleop.require_calibration=false \
  --dataset.repo_id=${DATASET_REPO} \
  --dataset.root=${DATASET_ROOT} \
  --dataset.single_task="${TASK_DESC}" \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=25 \
  --dataset.reset_time_s=8 \
  --dataset.push_to_hub=false \
  --display_data=true \
  --policy.path=${BASE_POLICY}
```

常用热键：

- `i`: human takeover 开关
- `s`: 标记成功并结束当前 episode
- `f`: 标记失败并结束当前 episode

建议：

- intervention 只在 policy 真要犯错的时候接管
- 接管后尽量把它带回正确轨道，再交还 policy 或结束 episode
- 成功 / 失败一定要按键，不要依赖默认值

## 3. 检查数据是否满足 ICCQ 要求

采完马上做一次 report：

```bash
lerobot-dataset-report --dataset ${DATASET_ROOT}
```

你至少要确认这几项：

- `episode_success` 已存在
- `complementary_info.is_intervention` 已存在
- `complementary_info.collector_policy_id` 已存在
- `complementary_info.collector_source` 已存在

如果这一步不对，后面的 pair mining 和 shadow replay 都不要继续跑。

## 4. 挖 intervention preference

这一步把 human takeover 变成 `(positive chunk, negative chunk)` 偏好对。

默认负样本来源已经改成推荐路径：

- `shadow_replay`

命令：

```bash
lerobot-prepare-intervention-preferences \
  --dataset.repo_id=${DATASET_REPO} \
  --dataset.root=${DATASET_ROOT} \
  --output_path=${PREFS} \
  --chunk_horizon=10 \
  --propagation_horizon=8 \
  --propagation_decay=0.8 \
  --negative_source=shadow_replay
```

产物：

- `${PREFS}`

这份 parquet 里会按 `index` 对齐，包含：

- `positive_action_chunk`
- `negative_action_chunk`
- `negative_action_pad`
- `propagation_weight`
- `collector_policy_id`

建议先看一下行数是否合理。太少通常说明：

- intervention 没录到
- provenance 不完整
- policy checkpoint 路径不可恢复

## 5. 训练 ICCQ critic

这一步训练 chunk-level critic。

最小可用命令：

```bash
lerobot-q-train \
  --dataset.repo_id=${DATASET_REPO} \
  --dataset.root=${DATASET_ROOT} \
  --critic.type=iccq \
  --critic.camera_features='["observation.images.front"]' \
  --critic.state_feature=observation.state \
  --critic.chunk_horizon=10 \
  --preferences.pair_path=${PREFS} \
  --batch_size=32 \
  --steps=8000 \
  --log_freq=100 \
  --save_freq=2000 \
  --output_dir=${Q_TRAIN_OUT}
```

第一轮建议不要乱改的参数：

- `critic.chunk_horizon=10`
- `critic.expectile_tau=0.8`
- `critic.conservative_alpha=1.0`
- `critic.conservative_num_perturb_samples=2`
- `critic.preference_margin=0.02`

训练完成后，后面会用 `${Q_TRAIN_OUT}` 里的 checkpoint 做 `q_infer`。

## 6. 跑 critic inference，导出 QBC 权重

这一步把每个 frame 的：

- `q1`
- `q2`
- `q_value`
- `v_value`
- `advantage`
- `qbc_weight`

导出成 parquet。

```bash
lerobot-q-infer \
  --dataset.repo_id=${DATASET_REPO} \
  --dataset.root=${DATASET_ROOT} \
  --inference.checkpoint_path=${Q_TRAIN_OUT} \
  --inference.checkpoint_ref=last \
  --runtime.batch_size=32 \
  --qbc.beta=0.2 \
  --qbc.clip_min=0.0 \
  --qbc.clip_max=10.0 \
  --output_dir=${Q_INFER_OUT}
```

产物默认是：

```bash
${Q_INFER_OUT}/q_values.parquet
```

这就是后面 `lerobot-train --use_qbc=true` 直接吃的文件。

## 7. 用 QBC 做单臂策略后训练

这里建议先从已有 base policy 继续 finetune，而不是从头训。

如果你的 backbone 是 `pi05`，一个最小命令可以是：

```bash
lerobot-train \
  --policy.path=${BASE_POLICY} \
  --dataset.repo_id=${DATASET_REPO} \
  --dataset.root=${DATASET_ROOT} \
  --batch_size=8 \
  --steps=10000 \
  --save_freq=2000 \
  --log_freq=100 \
  --use_qbc=true \
  --qbc_weight_path=${Q_VALUES} \
  --qbc_beta=0.2 \
  --qbc_clip_min=0.0 \
  --qbc_clip_max=10.0 \
  --output_dir=${QBC_OUT}
```

注意：

- `use_qbc` 和 `use_rabc` 现在是互斥的
- `qbc_weight_path` 指向上一步的 `q_values.parquet`
- 如果 `q_values.parquet` 已经带了 `qbc_weight` 列，train 会直接优先用它
- 如果你只保留了 `advantage` 列，train 也可以按 `qbc_beta / clip_*` 现场重算

## 8. 部署下一轮单臂 HIL

拿着 QBC 训练后的 checkpoint，再录下一轮：

```bash
lerobot-human-inloop-record \
  --robot.type=piperx_follower \
  --robot.port=${FOLLOWER_CAN_PORT} \
  --robot.id=my_piperx_follower \
  --robot.require_calibration=false \
  --teleop.type=piperx_leader \
  --teleop.port=${LEADER_CAN_PORT} \
  --teleop.id=my_piperx_leader \
  --teleop.require_calibration=false \
  --dataset.repo_id=${DATASET_REPO} \
  --dataset.root=${DATASET_ROOT} \
  --dataset.single_task="${TASK_DESC}" \
  --dataset.num_episodes=20 \
  --dataset.episode_time_s=25 \
  --dataset.reset_time_s=8 \
  --dataset.push_to_hub=false \
  --display_data=true \
  --policy.path=${QBC_OUT} \
  --resume=true
```

建议每一轮都重复：

1. `dataset_report`
2. `prepare_intervention_preferences`
3. `q_train`
4. `q_infer`
5. `lerobot-train --use_qbc=true`

## 9. 单臂第一轮推荐顺序

如果你现在是第一次完整跑通，建议顺序就是：

1. 准备一个 base policy
2. 用 `lerobot-human-inloop-record --policy.path=...` 采一轮单臂 HIL 数据
3. `lerobot-dataset-report --dataset ${DATASET_ROOT}`
4. `lerobot-prepare-intervention-preferences`
5. `lerobot-q-train`
6. `lerobot-q-infer`
7. `lerobot-train --use_qbc=true`
8. 再拿 QBC 后的 checkpoint 回到 `lerobot-human-inloop-record`

## 10. 当前实现的几个重要约定

为了避免后面混淆，当前代码里这几个字段的语义固定如下：

- `complementary_info.collector_policy_id`
  当前激活 policy 的 provenance
- `complementary_info.collector_source`
  这一帧实际是谁执行的，`policy` 或 `human`

也就是说：

- human takeover 帧仍然会保留原 policy provenance
- 但 source 会标成 `human`

这正是后面 shadow replay 和 preference mining 需要的结构。

## 11. 现阶段最容易踩的坑

- `collector_policy_id` 不是可恢复 checkpoint 的路径或 ID
- dataset 里没有 `episode_success`
- 录的是纯人工数据，却直接去跑 preference mining
- `q_infer` 用的不是和 `q_train` 同一份数据集
- `qbc_weight_path` 指到了旧轮次的 parquet
- 想同时开 `use_rabc` 和 `use_qbc`

如果你想，我下一步可以继续给你补一份“单臂最小实验配置清单”，把每一步建议的 episode 数、训练步数和优先级再压成一页执行表。
