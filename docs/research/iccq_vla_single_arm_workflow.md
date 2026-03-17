# ICCQ-VLA Single-Arm Workflow

最后更新：2026-03-17

这份文档对应当前仓库里已经接好的三条链路：

- 人类接管数据采集与 provenance 标注
- intervention preference 挖掘
- `ICCQ -> q_infer -> QBC post-training`

这份版本默认按你当前的真实配置来写：

- 机械臂是 `piper`，不是 `piperx`
- 右臂 `can2` 是 follower / 执行臂
- 左臂 `can1` 是 leader / 示教主臂
- 工控机负责相机、CAN、采集
- 服务器负责 `policy_server` 远端异步推理

如果你以后切回本地单机推理，可以把下面的 `lerobot-async-human-inloop-record` 换回 `lerobot-human-inloop-record`。

## 0. 前置约定

先准备这些变量：

```bash
DATASET_ROOT=/path/to/lerobot_dataset_directory
DATASET_REPO=your_name/iccq_single_arm_round1
TASK_DESC="pick up and put the bottle into the box"

FOLLOWER_CAN_PORT=can2
LEADER_CAN_PORT=can1
FOLLOWER_ID=my_piper_follower
LEADER_ID=my_piper_leader

WRIST_CAM_SERIAL=419622072329
TOP_CAM_SERIAL=420222071960

ASYNC_POLICY_SERVER=59.72.98.55:8080

# 必须填服务器端能直接读取的 checkpoint 路径。
# 强烈建议用固定步数目录，不要用会漂移的 `last/`。
BASE_POLICY_SERVER_PATH=/workspace/outputs/pi05_right_arm_5k/checkpoints/005000/pretrained_model

Q_TRAIN_OUT=outputs/q_train/iccq_single_arm
Q_INFER_OUT=outputs/q_infer/iccq_single_arm
Q_VALUES=${Q_INFER_OUT}/q_values.parquet
PREFS=outputs/intervention_preferences/iccq_single_arm_prefs.parquet
QBC_OUT=outputs/train/pi05_qbc_single_arm
```

说明：

- `DATASET_ROOT` 在当前这些脚本里指的是“最终数据集目录本身”，不是父目录。
  例如可以直接写成 `/app/下载/lerobot_datasets/ICCQ/ICCQ_Value_loop_20260317`。
- 当 `--resume=false` 时，这个目录必须还不存在；如果目录已经存在，要么删掉重录，要么改新目录名，要么确认是有效旧数据集后加 `--resume=true`。
- `BASE_POLICY_SERVER_PATH` 可以直接用你已经做好的 LoRA / finetune checkpoint。
- 不需要为了人在环路采集重新训练一版 LoRA。
- 只要服务器上的 `policy_server` 能 `from_pretrained(...)` 正常加载它，就能直接拿来做 HIL 数据采集。
- `collector_policy_id` 会默认记录这个 checkpoint 路径，所以路径最好一开始就选稳定的、可复现的。

## 1. 确认单臂遥操作正常

你工控机端可以继续复用原来跑通的 Evo-RL 那套机械臂接入方式。

先把 CAN 配好，再做一次遥操作确认。

### 1.1 配置和检查 CAN

```bash
lerobot-setup-can --mode=setup --interfaces=${LEADER_CAN_PORT},${FOLLOWER_CAN_PORT}
lerobot-setup-can --mode=test --interfaces=${LEADER_CAN_PORT},${FOLLOWER_CAN_PORT}
```

### 1.2 检查 leader-follower 跟手

```bash
lerobot-teleoperate \
  --robot.type=piper_follower \
  --robot.port=${FOLLOWER_CAN_PORT} \
  --robot.id=${FOLLOWER_ID} \
  --robot.require_calibration=false \
  --teleop.type=piper_leader \
  --teleop.port=${LEADER_CAN_PORT} \
  --teleop.id=${LEADER_ID} \
  --teleop.require_calibration=false
```

通过标准：

- follower 跟手正常
- gripper 同步正常
- 没有持续报 CAN / calibration 错误

## 2. 服务器端启动异步推理

你现在的推荐模式不是本地 `--policy.path=...`，而是：

- 服务器跑 `policy_server`
- 工控机跑 `lerobot-async-human-inloop-record`

### 2.1 建议的服务器容器

你之前的 `lerobot_openpi:v4` 镜像可以继续用，推荐新起一个容器，把当前仓库挂进去：

```bash
docker run --gpus all --network host --shm-size=32g -it \
  -v /path/to/ICCQ-VLA:/workspace/ICCQ-VLA \
  -v /path/to/outputs:/workspace/outputs \
  lerobot_openpi:v4 bash
```

容器里建议：

```bash
cd /workspace/ICCQ-VLA
pip install -e .
```

### 2.2 启动异步 `policy_server`

```bash
CUDA_VISIBLE_DEVICES=7 python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=8080 \
  --fps=30 \
  --inference_latency=0.08 \
  --obs_queue_timeout=2.0
```

这个命令本身就是异步推理服务端。

异步性来自两部分：

- `policy_server` 按 observation 队列异步生成 action chunk
- 工控机端 `lerobot-async-human-inloop-record` 通过远端 adapter 做 observation 发送、action chunk 接收和本地聚合

所以你之前那条服务器命令是对的，现在差的只是工控机端需要换成新的异步 HIL 入口。

## 3. 工控机端采集 HIL 数据

### 3.1 推荐路径：直接用你现有 LoRA checkpoint 做异步人在环路采集

```bash
lerobot-async-human-inloop-record \
  --robot.type=piper_follower \
  --robot.port=can2 \
  --robot.id=my_piper_follower \
  --robot.speed_ratio=35 \
  --robot.high_follow=false \
  --robot.startup_sleep_s=0.5 \
  --robot.require_calibration=false \
  --robot.cameras="{ wrist: {type: intelrealsense, serial_number_or_name: "419622072329", width: 640, height: 480, fps: 30, warmup_s: 2}, top: {type: intelrealsense, serial_number_or_name: "420222071960", width: 640, height: 480, fps: 30, warmup_s: 2} }" \
  --teleop.type=piper_leader \
  --teleop.port=can1 \
  --teleop.id=my_piper_leader \
  --teleop.command_speed_ratio=25 \
  --teleop.gravity_comp_control_hz=80 \
  --teleop.gravity_comp_torque_limit=0.8 \
  --teleop.prefer_ctrl_messages=true \
  --teleop.fallback_to_feedback=true \
  --teleop.startup_sleep_s=0.5 \
  --teleop.require_calibration=false \
  --dataset.repo_id=zhang/ICCQ_Value_loop_20260317 \
  --dataset.root=/app/下载/lerobot_datasets/ICCQ/ICCQ_Value_loop_20260317 \
  --dataset.single_task="pick up and put the bottle into the box" \
  --dataset.fps=30 \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=240 \
  --dataset.reset_time_s=30 \
  --dataset.push_to_hub=false \
  --dataset.vcodec=h264 \
  --async_policy.server_address=59.72.98.55:8082 \
  --async_policy.policy_type=pi05 \
  --async_policy.pretrained_name_or_path=/workspace/outputs/pi05_right_arm_5k/checkpoints/last/pretrained_model \
  --async_policy.policy_device=cuda \
  --async_policy.client_device=cpu \
  --async_policy.actions_per_chunk=36 \
  --async_policy.chunk_size_threshold=0.6 \
  --async_policy.fps=8 \
  --async_policy.aggregate_fn_name=conservative \
  --acp_inference.enable=true \
  --acp_inference.use_cfg=false \
  --play_sounds=false
```

这个入口现在会自动打开：

- `episode_success` 标注
- `complementary_info.collector_policy_id`
- `complementary_info.collector_source`
- `complementary_info.is_intervention`
- `complementary_info.policy_action`

其中字段语义固定为：

- `collector_policy_id` 记录当前远端 policy checkpoint 的 provenance
- `collector_source` 记录这一帧实际是谁执行的，`policy` 或 `human`

### 3.2 如果还没有 policy，可先录纯人工首轮数据

如果某一轮你想先做纯人工冷启动，也可以继续用本地入口：

```bash
lerobot-human-inloop-record \
  --robot.type=piper_follower \
  --robot.port=${FOLLOWER_CAN_PORT} \
  --robot.id=${FOLLOWER_ID} \
  --robot.require_calibration=false \
  --teleop.type=piper_leader \
  --teleop.port=${LEADER_CAN_PORT} \
  --teleop.id=${LEADER_ID} \
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

常用热键：

- `i`: human takeover 开关
- `s`: 标记成功并结束当前 episode
- `f`: 标记失败并结束当前 episode

建议：

- intervention 只在 policy 真要犯错的时候接管
- 接管后尽量把它带回正确轨道，再交还 policy 或结束 episode
- 成功 / 失败一定要按键，不要依赖默认值

## 4. 检查数据是否满足 ICCQ 要求

采完马上做一次 report：

```bash
lerobot-dataset-report --dataset ${DATASET_REPO} --root ${DATASET_ROOT}
```

你至少要确认这几项：

- `episode_success` 已存在
- `complementary_info.is_intervention` 已存在
- `complementary_info.collector_policy_id` 已存在
- `complementary_info.collector_source` 已存在

如果这一步不对，后面的 pair mining 和 shadow replay 都不要继续跑。

## 5. 挖 intervention preference

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

## 6. 训练 ICCQ critic

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

## 7. 跑 critic inference，导出 QBC 权重

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

## 8. 用 QBC 做单臂策略后训练

这里建议在服务器端，从已有 base policy 继续 finetune，而不是从头训。

如果你的 backbone 是 `pi05`，一个最小命令可以是：

```bash
lerobot-train \
  --policy.path=${BASE_POLICY_SERVER_PATH} \
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

## 9. 部署下一轮单臂 HIL

拿着 QBC 训练后的 checkpoint，在工控机端再录下一轮异步 HIL：

```bash
lerobot-async-human-inloop-record \
  --robot.type=piper_follower \
  --robot.port=${FOLLOWER_CAN_PORT} \
  --robot.id=${FOLLOWER_ID} \
  --robot.speed_ratio=35 \
  --robot.high_follow=false \
  --robot.startup_sleep_s=0.5 \
  --robot.require_calibration=false \
  --robot.cameras="{ wrist: {type: intelrealsense, serial_number_or_name: \"${WRIST_CAM_SERIAL}\", width: 640, height: 480, fps: 30, warmup_s: 2}, top: {type: intelrealsense, serial_number_or_name: \"${TOP_CAM_SERIAL}\", width: 640, height: 480, fps: 30, warmup_s: 2} }" \
  --teleop.type=piper_leader \
  --teleop.port=${LEADER_CAN_PORT} \
  --teleop.id=${LEADER_ID} \
  --teleop.command_speed_ratio=25 \
  --teleop.gravity_comp_control_hz=80 \
  --teleop.gravity_comp_torque_limit=0.8 \
  --teleop.prefer_ctrl_messages=true \
  --teleop.fallback_to_feedback=true \
  --teleop.startup_sleep_s=0.5 \
  --teleop.require_calibration=false \
  --dataset.repo_id=${DATASET_REPO} \
  --dataset.root=${DATASET_ROOT} \
  --dataset.single_task="${TASK_DESC}" \
  --dataset.num_episodes=20 \
  --dataset.episode_time_s=25 \
  --dataset.reset_time_s=8 \
  --dataset.push_to_hub=false \
  --dataset.vcodec=h264 \
  --async_policy.server_address=${ASYNC_POLICY_SERVER} \
  --async_policy.policy_type=pi05 \
  --async_policy.pretrained_name_or_path=${QBC_OUT}/checkpoints/last/pretrained_model \
  --async_policy.policy_device=cuda \
  --async_policy.client_device=cpu \
  --async_policy.actions_per_chunk=36 \
  --async_policy.chunk_size_threshold=0.6 \
  --async_policy.fps=8 \
  --async_policy.aggregate_fn_name=conservative \
  --acp_inference.enable=true \
  --acp_inference.use_cfg=false \
  --play_sounds=false \
  --resume=true
```

正式采集前，最好把 `last/pretrained_model` 再替换成固定步数目录，比如
`${QBC_OUT}/checkpoints/010000/pretrained_model`，避免后续继续训练后 provenance 漂移。

建议每一轮都重复：

1. `dataset_report`
2. `prepare_intervention_preferences`
3. `q_train`
4. `q_infer`
5. `lerobot-train --use_qbc=true`

## 10. 单臂第一轮推荐顺序

如果你现在是第一次完整跑通，建议顺序就是：

1. 在服务器上准备一个可加载的 base policy checkpoint
2. 启动 `python -m lerobot.async_inference.policy_server`
3. 在工控机上用 `lerobot-async-human-inloop-record` 采一轮单臂 HIL 数据
4. `lerobot-dataset-report --dataset ${DATASET_REPO} --root ${DATASET_ROOT}`
5. `lerobot-prepare-intervention-preferences`
6. `lerobot-q-train`
7. `lerobot-q-infer`
8. `lerobot-train --use_qbc=true`
9. 再拿 QBC 后的 checkpoint 回到 `lerobot-async-human-inloop-record`

## 11. 当前实现的几个重要约定

为了避免后面混淆，当前代码里这几个字段的语义固定如下：

- `complementary_info.collector_policy_id`
  当前激活 policy 的 provenance
- `complementary_info.collector_source`
  这一帧实际是谁执行的，`policy` 或 `human`

也就是说：

- human takeover 帧仍然会保留原 policy provenance
- 但 source 会标成 `human`

这正是后面 shadow replay 和 preference mining 需要的结构。

## 12. 现阶段最容易踩的坑

- `collector_policy_id` 不是可恢复 checkpoint 的路径或 ID
- 服务器端还在用 `last/`，但你又继续训练过，导致 provenance 漂移
- dataset 里没有 `episode_success`
- 录的是纯人工数据，却直接去跑 preference mining
- `q_infer` 用的不是和 `q_train` 同一份数据集
- `qbc_weight_path` 指到了旧轮次的 parquet
- 想同时开 `use_rabc` 和 `use_qbc`

如果你想，我下一步可以继续给你补一份“单臂最小实验配置清单”，把每一步建议的 episode 数、训练步数和优先级再压成一页执行表。
