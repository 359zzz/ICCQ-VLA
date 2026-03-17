#!/usr/bin/env python

from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.factory import make_policy_config
from lerobot.values.iccq.configuration_iccq import ICCQConfig


def test_iccq_config_from_dict():
    payload = {
        "type": "iccq",
        "chunk_horizon": 8,
        "task_field": "task",
        "camera_features": ["observation.images.front"],
        "language_repo_id": "google/gemma-3-270m",
        "vision_repo_id": "google/siglip-so400m-patch14-384",
        "dropout": 0.2,
    }
    cfg = make_policy_config(payload.pop("type"), **payload)
    assert isinstance(cfg, ICCQConfig)
    assert cfg.type == "iccq"
    assert cfg.chunk_horizon == 8
    assert cfg.reward_chunk_key == "observation.iccq.reward_chunk"
    assert cfg.observation_delta_indices == [0, 8]
    assert cfg.action_delta_indices == list(range(8))


def test_iccq_preset_uses_cosine_decay_with_warmup():
    cfg = ICCQConfig()
    scheduler_cfg = cfg.get_scheduler_preset()
    assert isinstance(scheduler_cfg, CosineDecayWithWarmupSchedulerConfig)
    assert scheduler_cfg.peak_lr == cfg.optimizer_lr
    assert scheduler_cfg.decay_lr == cfg.scheduler_decay_lr
    assert scheduler_cfg.num_warmup_steps == cfg.scheduler_warmup_steps
    assert scheduler_cfg.num_decay_steps == cfg.scheduler_decay_steps
