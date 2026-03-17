import draccus

from lerobot.scripts.lerobot_async_human_inloop_record import AsyncHumanInloopRecordConfig


def test_async_human_inloop_record_parses_piper():
    args = [
        "--robot.type=piper_follower",
        "--robot.port=can2",
        "--robot.id=my_piper_follower",
        "--teleop.type=piper_leader",
        "--teleop.port=can1",
        "--teleop.id=my_piper_leader",
        "--dataset.repo_id=dummy/piper_demo_next_round",
        "--dataset.single_task=pick up and put the bottle into the box",
        "--dataset.num_episodes=1",
        "--dataset.episode_time_s=1",
        "--dataset.reset_time_s=1",
        "--dataset.push_to_hub=false",
        "--async_policy.server_address=127.0.0.1:8080",
        "--async_policy.policy_type=pi05",
        "--async_policy.pretrained_name_or_path=/workspace/outputs/pi05_right_arm_5k/checkpoints/last/pretrained_model_async12",
        "--async_policy.policy_device=cuda",
        "--async_policy.client_device=cpu",
        "--async_policy.actions_per_chunk=36",
        "--async_policy.chunk_size_threshold=0.6",
        "--async_policy.fps=8",
        "--async_policy.aggregate_fn_name=conservative",
        "--acp_inference.enable=true",
        "--acp_inference.use_cfg=false",
    ]

    cfg = draccus.parse(config_class=AsyncHumanInloopRecordConfig, config_path=None, args=args)

    assert cfg.robot.type == "piper_follower"
    assert cfg.teleop.type == "piper_leader"
    assert cfg.async_policy.policy_type == "pi05"
    assert cfg.policy.type == "pi05"
    assert cfg.policy_sync_to_teleop is True
    assert cfg.intervention_state_machine_enabled is True
    assert (
        cfg.collector_policy_id_policy
        == "/workspace/outputs/pi05_right_arm_5k/checkpoints/last/pretrained_model_async12"
    )
