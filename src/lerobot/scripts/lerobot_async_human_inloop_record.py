#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Records a dataset with remote async policy inference and local teleop intervention.

This entrypoint is designed for the common setup where:
- the robot IPC has no GPU,
- cameras + robot + teleop run locally on the IPC,
- policy inference runs remotely on a GPU server via `policy_server`.
"""

import logging
from dataclasses import dataclass, field

from lerobot.async_inference.remote_policy_adapter import (
    AsyncRemotePolicyConfig,
    RemoteAsyncPolicyAdapter,
)
from lerobot.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.reachy2_camera.configuration_reachy2_camera import Reachy2CameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_openarm_follower,
    bi_piper_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    omx_follower,
    openarm_follower,
    piper_follower,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.scripts.lerobot_human_inloop_record import _HumanInloopFailureResetController
from lerobot.scripts.lerobot_record import DatasetRecordConfig, record
from lerobot.scripts.recording_hil import ACPInferenceConfig
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_piper_leader,
    bi_so_leader,
    homunculus,
    koch_leader,
    omx_leader,
    openarm_leader,
    piper_leader,
    reachy2_teleoperator,
    so_leader,
    unitree_g1,
)
from lerobot.utils.control_utils import sanity_check_bimanual_piper_pair
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.recording_annotations import infer_collector_policy_id


@dataclass
class AsyncHumanInloopRecordConfig:
    robot: RobotConfig
    teleop: TeleoperatorConfig
    dataset: DatasetRecordConfig
    async_policy: AsyncRemotePolicyConfig
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    play_sounds: bool = True
    resume: bool = False
    policy_sync_parallel: bool = True
    intervention_toggle_key: str = "i"
    episode_success_key: str = "s"
    episode_failure_key: str = "f"
    require_episode_success_label: bool = False
    collector_policy_id_policy: str | None = None
    collector_policy_id_human: str = "human"
    acp_inference: ACPInferenceConfig = field(default_factory=ACPInferenceConfig)
    communication_retry_timeout_s: float = 2.0
    communication_retry_interval_s: float = 0.1
    policy_sync_to_teleop: bool = field(default=True, init=False)
    intervention_state_machine_enabled: bool = field(default=True, init=False)
    enable_episode_outcome_labeling: bool = field(default=True, init=False)
    default_episode_success: str | None = field(default="failure", init=False)
    enable_collector_policy_id: bool = field(default=True, init=False)

    @property
    def policy(self) -> AsyncRemotePolicyConfig:
        return self.async_policy

    def __post_init__(self):
        sanity_check_bimanual_piper_pair(self.robot, self.teleop)
        if not self.intervention_toggle_key or len(self.intervention_toggle_key) != 1:
            raise ValueError("`intervention_toggle_key` must be a single character.")
        key_bindings = {
            "episode_success_key": self.episode_success_key,
            "episode_failure_key": self.episode_failure_key,
        }
        for key_name, key_value in key_bindings.items():
            if not key_value or len(key_value) != 1:
                raise ValueError(f"`{key_name}` must be a single character.")

        normalized_keys = [
            self.intervention_toggle_key.lower(),
            self.episode_success_key.lower(),
            self.episode_failure_key.lower(),
        ]
        if len(set(normalized_keys)) != len(normalized_keys):
            raise ValueError(
                "`intervention_toggle_key`, `episode_success_key`, and `episode_failure_key` must be distinct."
            )

        if self.collector_policy_id_policy is None:
            self.collector_policy_id_policy = infer_collector_policy_id(self.async_policy)


@parser.wrap()
def async_human_inloop_record(cfg: AsyncHumanInloopRecordConfig):
    failure_reset_controller = _HumanInloopFailureResetController(cfg)
    cfg._skip_dataset_name_check = True
    cfg._on_record_connected = failure_reset_controller.on_record_connected
    cfg._on_record_episode_outcome = failure_reset_controller.on_episode_outcome

    def make_runtime_policy(robot, dataset):
        del dataset
        return RemoteAsyncPolicyAdapter(cfg.async_policy, robot), None, None

    cfg._make_runtime_policy = make_runtime_policy

    logging.info(
        "Async human-in-loop recording is enabled. Server=%s policy=%s checkpoint=%s. "
        "Press '%s' to toggle takeover, '%s' to mark success, '%s' to mark failure. "
        "Recorded `action` is the executed action, `complementary_info.policy_action` stores remote policy output. "
        "Active policy provenance is stored in `complementary_info.collector_policy_id`, "
        "and the execution source is stored in `complementary_info.collector_source`. "
        "ACP inference: enable=%s use_cfg=%s cfg_beta=%.3f.",
        cfg.async_policy.server_address,
        cfg.async_policy.policy_type,
        cfg.async_policy.pretrained_name_or_path,
        cfg.intervention_toggle_key,
        cfg.episode_success_key,
        cfg.episode_failure_key,
        cfg.acp_inference.enable,
        cfg.acp_inference.use_cfg,
        cfg.acp_inference.cfg_beta,
    )
    return record.__wrapped__(cfg)


def main():
    register_third_party_plugins()
    async_human_inloop_record()


if __name__ == "__main__":
    main()
