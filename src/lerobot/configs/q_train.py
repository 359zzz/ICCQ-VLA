#!/usr/bin/env python

import builtins
import datetime as dt
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig, PeftConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.value_train import ValueTargetsConfig
from lerobot.optim import OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.utils.hub import HubMixin
from lerobot.values.iccq.configuration_iccq import ICCQConfig

Q_TRAIN_CONFIG_NAME = "q_train_config.json"


@dataclass
class PreferencePairsConfig:
    pair_path: str | None = None

    def validate(self) -> None:
        if self.pair_path is not None and not self.pair_path:
            raise ValueError("'preferences.pair_path' must be non-empty when provided.")


@dataclass
class QTrainPipelineConfig(HubMixin):
    dataset: DatasetConfig
    critic: PreTrainedConfig | None = field(default_factory=ICCQConfig)
    env: Any | None = None

    output_dir: Path | None = None
    job_name: str | None = None
    resume: bool = False
    seed: int | None = 1000

    num_workers: int = 4
    batch_size: int = 32
    steps: int = 8_000
    log_freq: int = 200
    tolerance_s: float = 1e-4

    save_checkpoint: bool = True
    save_freq: int = 4_000

    use_q_training_preset: bool = True
    use_policy_training_preset: bool = field(init=False, default=True)

    optimizer: OptimizerConfig | None = None
    scheduler: LRSchedulerConfig | None = None

    targets: ValueTargetsConfig = field(default_factory=ValueTargetsConfig)
    preferences: PreferencePairsConfig = field(default_factory=PreferencePairsConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    peft: PeftConfig | None = None

    rename_map: dict[str, str] = field(default_factory=dict)
    checkpoint_path: Path | None = field(init=False, default=None)

    @property
    def policy(self) -> PreTrainedConfig | None:
        return self.critic

    def validate(self) -> None:
        critic_path = parser.get_path_arg("critic")
        if critic_path:
            cli_overrides = parser.get_cli_overrides("critic")
            self.critic = PreTrainedConfig.from_pretrained(critic_path, cli_overrides=cli_overrides)
            self.critic.pretrained_path = Path(critic_path)
        elif self.resume:
            config_path = parser.parse_arg("config_path")
            if not config_path:
                raise ValueError(
                    f"A config_path is expected when resuming. Please specify path to {Q_TRAIN_CONFIG_NAME}"
                )
            if not Path(config_path).resolve().exists():
                raise NotADirectoryError(
                    f"{config_path=} is expected to be a local path. Resuming from hub is not supported."
                )
            critic_dir = Path(config_path).parent
            if self.critic is not None:
                self.critic.pretrained_path = critic_dir
            self.checkpoint_path = critic_dir.parent

        if self.critic is None:
            raise ValueError("Critic is not configured. Please specify a critic config with `--critic.type`.")
        if self.critic.type != "iccq":
            raise ValueError(
                f"Unsupported critic type '{self.critic.type}'. "
                "Current lerobot-q-train supports only '--critic.type=iccq'."
            )

        self.targets.validate()
        self.preferences.validate()

        if not self.job_name:
            self.job_name = f"{self.critic.type}_q"

        if not self.resume and isinstance(self.output_dir, Path) and self.output_dir.is_dir():
            raise FileExistsError(
                f"Output directory {self.output_dir} already exists and resume is {self.resume}. "
                f"Please change your output directory so that {self.output_dir} is not overwritten."
            )
        elif not self.output_dir:
            now = dt.datetime.now()
            train_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/q_train") / train_dir

        self.use_policy_training_preset = self.use_q_training_preset
        if not self.use_policy_training_preset and (self.optimizer is None or self.scheduler is None):
            raise ValueError("Optimizer and Scheduler must be set when q presets are not used.")
        elif self.use_policy_training_preset and not self.resume:
            self.optimizer = self.critic.get_optimizer_preset()
            self.scheduler = self.critic.get_scheduler_preset()

        if self.critic.push_to_hub and not self.critic.repo_id:
            raise ValueError("'critic.repo_id' argument missing. Please specify it to push the model to the hub.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["critic"]

    def to_dict(self) -> dict[str, Any]:
        return draccus.encode(self)  # type: ignore[no-any-return]

    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / Q_TRAIN_CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: builtins.type["QTrainPipelineConfig"],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict[Any, Any] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs: Any,
    ) -> "QTrainPipelineConfig":
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if Q_TRAIN_CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, Q_TRAIN_CONFIG_NAME)
            else:
                print(f"{Q_TRAIN_CONFIG_NAME} not found in {Path(model_id).resolve()}")
        elif Path(model_id).is_file():
            config_file = model_id
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=Q_TRAIN_CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{Q_TRAIN_CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        cli_args = kwargs.pop("cli_args", [])
        with draccus.config_type("json"):
            return draccus.parse(cls, config_file, args=cli_args)
