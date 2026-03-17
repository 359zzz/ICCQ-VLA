#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_STATE


@PreTrainedConfig.register_subclass("iccq")
@dataclass
class ICCQConfig(PreTrainedConfig):
    """Conservative chunk-Q critic for offline VLA post-training."""

    vision_repo_id: str = "google/siglip-so400m-patch14-384"
    language_repo_id: str = "google/gemma-3-270m"
    vision_revision: str | None = None
    language_revision: str | None = None

    task_field: str = "task"
    camera_features: list[str] = field(default_factory=list)
    state_feature: str = OBS_STATE
    max_state_dim: int = 32
    tokenizer_max_length: int = 200

    chunk_horizon: int = 10
    gamma: float = 0.99
    expectile_tau: float = 0.8
    conservative_alpha: float = 1.0
    conservative_num_perturb_samples: int = 2
    conservative_perturb_std: float = 0.05
    preference_margin: float = 0.02

    td_loss_weight: float = 1.0
    calibration_loss_weight: float = 1.0
    conservative_loss_weight: float = 0.05
    preference_loss_weight: float = 0.5
    value_loss_weight: float = 1.0

    reward_chunk_key: str = "observation.iccq.reward_chunk"
    reward_pad_key: str = "iccq.reward_is_pad"
    calibration_target_key: str = "observation.iccq.calibration_target"
    next_observation_pad_key: str = "iccq.next_observation_is_pad"
    preference_negative_chunk_key: str = "observation.iccq.preference_negative_chunk"
    preference_negative_pad_key: str = "iccq.preference_negative_is_pad"
    preference_weight_key: str = "observation.iccq.preference_weight"

    image_hidden_dim: int = 256
    language_hidden_dim: int = 256
    state_hidden_dim: int = 128
    obs_hidden_dim: int = 512
    action_hidden_dim: int = 256
    critic_hidden_dim: int = 512

    dropout: float = 0.1
    dtype: str = "float32"
    freeze_vision_encoder: bool = False
    freeze_language_model: bool = False
    use_gradient_checkpointing: bool = False
    push_to_hub: bool = False

    optimizer_lr: float = 5e-5
    optimizer_weight_decay: float = 1e-5
    optimizer_grad_clip_norm: float = 10.0
    scheduler_warmup_steps: int = 500
    scheduler_decay_steps: int = 8_000
    scheduler_decay_lr: float = 1e-6

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.vision_repo_id:
            raise ValueError("'critic.vision_repo_id' must be non-empty.")
        if not self.language_repo_id:
            raise ValueError("'critic.language_repo_id' must be non-empty.")
        if not self.task_field:
            raise ValueError("'critic.task_field' must be non-empty.")
        if not self.state_feature.startswith("observation."):
            raise ValueError("'critic.state_feature' must start with 'observation.'.")
        if self.max_state_dim <= 0:
            raise ValueError("'critic.max_state_dim' must be > 0.")
        if self.chunk_horizon <= 0:
            raise ValueError("'critic.chunk_horizon' must be > 0.")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("'critic.gamma' must be within (0, 1].")
        if not 0.0 < self.expectile_tau < 1.0:
            raise ValueError("'critic.expectile_tau' must be within (0, 1).")
        if self.conservative_alpha <= 0:
            raise ValueError("'critic.conservative_alpha' must be > 0.")
        if self.conservative_num_perturb_samples < 0:
            raise ValueError("'critic.conservative_num_perturb_samples' must be >= 0.")
        if self.conservative_perturb_std < 0:
            raise ValueError("'critic.conservative_perturb_std' must be >= 0.")
        if self.tokenizer_max_length <= 0:
            raise ValueError("'critic.tokenizer_max_length' must be > 0.")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("'critic.dtype' must be one of {'float32', 'bfloat16'}.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("'critic.dropout' must be within [0, 1).")
        if self.optimizer_lr <= 0:
            raise ValueError("'critic.optimizer_lr' must be > 0.")
        if self.optimizer_weight_decay < 0:
            raise ValueError("'critic.optimizer_weight_decay' must be >= 0.")
        if self.optimizer_grad_clip_norm < 0:
            raise ValueError("'critic.optimizer_grad_clip_norm' must be >= 0.")
        if self.scheduler_warmup_steps < 0:
            raise ValueError("'critic.scheduler_warmup_steps' must be >= 0.")
        if self.scheduler_decay_steps <= 0:
            raise ValueError("'critic.scheduler_decay_steps' must be > 0.")
        if self.scheduler_decay_lr < 0:
            raise ValueError("'critic.scheduler_decay_lr' must be >= 0.")

    def validate_features(self) -> None:
        return

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int] | None:
        return [0, self.chunk_horizon]

    @property
    def action_delta_indices(self) -> list[int] | None:
        return list(range(self.chunk_horizon))

    @property
    def reward_delta_indices(self) -> None:
        # Real-robot datasets do not carry dense rewards. q_train injects pseudo-reward chunks via a raw-batch hook.
        return None
