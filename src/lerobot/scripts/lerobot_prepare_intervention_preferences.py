#!/usr/bin/env python

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import draccus
import numpy as np
import pandas as pd
import torch

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.scripts.recording_hil import _capture_policy_runtime_state, _restore_policy_runtime_state
from lerobot.utils.constants import ACTION, OBS_PREFIX
from lerobot.utils.control_utils import predict_action
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import auto_select_torch_device, init_logging

PREFERENCE_NEGATIVE_SOURCE_LOGGED = "logged_policy_action"
PREFERENCE_NEGATIVE_SOURCE_SHADOW = "shadow_replay"


@dataclass
class PreferenceDatasetConfig:
    repo_id: str
    root: str | None = None
    episodes: list[int] | None = None
    revision: str | None = None
    download_videos: bool = True

    def validate(self) -> None:
        if not self.repo_id:
            raise ValueError("'dataset.repo_id' must be non-empty.")


@dataclass
class PreferenceRuntimeConfig:
    device: str | None = None
    batch_size: int = 1

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("'runtime.batch_size' must be > 0.")
        if self.device is None:
            self.device = auto_select_torch_device().type


@dataclass
class PreferencePipelineConfig:
    dataset: PreferenceDatasetConfig
    runtime: PreferenceRuntimeConfig = field(default_factory=PreferenceRuntimeConfig)
    output_path: Path | None = None
    chunk_horizon: int = 10
    propagation_horizon: int = 8
    propagation_decay: float = 0.8
    intervention_field: str = "complementary_info.is_intervention"
    policy_action_field: str = "complementary_info.policy_action"
    collector_policy_id_field: str = "complementary_info.collector_policy_id"
    collector_source_field: str = "complementary_info.collector_source"
    negative_source: str = PREFERENCE_NEGATIVE_SOURCE_SHADOW
    overwrite: bool = False

    def validate(self) -> None:
        self.dataset.validate()
        self.runtime.validate()
        if self.chunk_horizon <= 0:
            raise ValueError("'chunk_horizon' must be > 0.")
        if self.propagation_horizon < 0:
            raise ValueError("'propagation_horizon' must be >= 0.")
        if not 0.0 <= self.propagation_decay <= 1.0:
            raise ValueError("'propagation_decay' must be within [0, 1].")
        if self.negative_source not in {
            PREFERENCE_NEGATIVE_SOURCE_LOGGED,
            PREFERENCE_NEGATIVE_SOURCE_SHADOW,
        }:
            raise ValueError(
                f"'negative_source' must be one of "
                f"{[PREFERENCE_NEGATIVE_SOURCE_LOGGED, PREFERENCE_NEGATIVE_SOURCE_SHADOW]}."
            )
        if self.output_path is None:
            now = dt.datetime.now()
            repo_tag = self.dataset.repo_id.replace("/", "_")
            self.output_path = Path("outputs/intervention_preferences") / f"{now:%Y-%m-%d_%H-%M-%S}_{repo_tag}.parquet"
        if self.output_path.exists() and not self.overwrite:
            raise FileExistsError(
                f"Output file {self.output_path} already exists. Set '--overwrite=true' to replace it."
            )

    def to_dict(self) -> dict[str, Any]:
        return draccus.encode(self)  # type: ignore[no-any-return]


def _to_numpy_observation(item: dict[str, Any]) -> dict[str, np.ndarray]:
    observation: dict[str, np.ndarray] = {}
    for key, value in item.items():
        if not key.startswith(OBS_PREFIX) or key.endswith("_is_pad"):
            continue
        if isinstance(value, torch.Tensor):
            observation[key] = value.detach().cpu().numpy()
        else:
            observation[key] = np.asarray(value)
    return observation


def _chunk_from_sequence(sequence: np.ndarray, start_pos: int, horizon: int, end_pos: int) -> tuple[np.ndarray, np.ndarray]:
    chunk = []
    is_pad = []
    for offset in range(horizon):
        pos = start_pos + offset
        if pos > end_pos:
            pos = end_pos
            is_pad.append(True)
        else:
            is_pad.append(False)
        chunk.append(sequence[pos])
    return np.asarray(chunk, dtype=np.float32), np.asarray(is_pad, dtype=np.bool_)


def _extract_onset_positions(interventions: np.ndarray) -> list[int]:
    onsets: list[int] = []
    prev = 0.0
    for pos, value in enumerate(interventions):
        current = float(value)
        if current > 0.5 and prev <= 0.5:
            onsets.append(pos)
        prev = current
    return onsets


def _resolve_policy_bundle(
    *,
    provenance: str,
    dataset: LeRobotDataset,
    device: str,
) -> tuple[Any, Any, Any]:
    if provenance == "human":
        raise ValueError("Cannot shadow replay provenance 'human'.")

    cfg = PreTrainedConfig.from_pretrained(provenance)
    cfg.pretrained_path = Path(provenance) if Path(provenance).exists() else provenance  # type: ignore[assignment]
    cfg.device = device
    policy = make_policy(cfg=cfg, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=cfg.pretrained_path,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": device},
        },
    )
    return policy, preprocessor, postprocessor


def _replay_shadow_policy_episode(
    *,
    dataset: LeRobotDataset,
    episode_positions: list[int],
    provenance: str,
    device: str,
) -> np.ndarray:
    policy, preprocessor, postprocessor = _resolve_policy_bundle(
        provenance=provenance,
        dataset=dataset,
        device=device,
    )
    policy.reset()
    runtime_state = _capture_policy_runtime_state(policy)
    action_dim = dataset.features[ACTION]["shape"][0]
    replay_actions = np.zeros((len(episode_positions), action_dim), dtype=np.float32)

    for local_idx, dataset_pos in enumerate(episode_positions):
        item = dataset[dataset_pos]
        observation = _to_numpy_observation(item)
        task = item["task"] if "task" in item else None
        _restore_policy_runtime_state(policy, runtime_state)
        policy_action = predict_action(
            observation=observation,
            policy=policy,
            device=torch.device(device),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=task,
            robot_type=dataset.meta.robot_type,
        )
        runtime_state.clear()
        runtime_state.update(_capture_policy_runtime_state(policy))
        robot_action = make_robot_action(policy_action, dataset.features)
        replay_actions[local_idx] = np.asarray(
            [robot_action[name] for name in dataset.features[ACTION]["names"]],
            dtype=np.float32,
        )
    return replay_actions


def _build_episode_policy_replay(
    *,
    dataset: LeRobotDataset,
    episode_positions: list[int],
    collector_policy_ids: np.ndarray,
    collector_sources: np.ndarray,
    device: str,
) -> np.ndarray:
    action_dim = dataset.features[ACTION]["shape"][0]
    replay_actions = np.zeros((len(episode_positions), action_dim), dtype=np.float32)
    episode_policy_ids = collector_policy_ids[episode_positions]
    episode_sources = collector_sources[episode_positions]

    segment_start = 0
    while segment_start < len(episode_positions):
        provenance = str(episode_policy_ids[segment_start])
        segment_end = segment_start
        while segment_end + 1 < len(episode_positions) and str(episode_policy_ids[segment_end + 1]) == provenance:
            segment_end += 1

        segment_has_policy = any(str(src) == "policy" for src in episode_sources[segment_start : segment_end + 1])
        if provenance and provenance != "human" and segment_has_policy:
            segment_positions = episode_positions[segment_start : segment_end + 1]
            replay_actions[segment_start : segment_end + 1] = _replay_shadow_policy_episode(
                dataset=dataset,
                episode_positions=segment_positions,
                provenance=provenance,
                device=device,
            )

        segment_start = segment_end + 1

    return replay_actions


def _to_action_array(raw_frames: Any, action_field: str) -> np.ndarray:
    values = raw_frames[action_field]
    if isinstance(values, list):
        return np.asarray(values, dtype=np.float32)
    return np.asarray(values.to_pylist(), dtype=np.float32)


def _build_preference_rows(
    *,
    dataset: LeRobotDataset,
    raw_frames: Any,
    cfg: PreferencePipelineConfig,
) -> pd.DataFrame:
    episode_indices = np.asarray(raw_frames["episode_index"], dtype=np.int64)
    frame_indices = np.asarray(raw_frames["frame_index"], dtype=np.int64)
    absolute_indices = np.asarray(raw_frames["index"], dtype=np.int64)
    task_indices = np.asarray(raw_frames["task_index"], dtype=np.int64)
    executed_actions = _to_action_array(raw_frames, ACTION)
    interventions = np.asarray(raw_frames[cfg.intervention_field], dtype=np.float32)
    collector_policy_ids = np.asarray(raw_frames[cfg.collector_policy_id_field], dtype=object)
    if cfg.collector_source_field in raw_frames.column_names:
        collector_sources = np.asarray(raw_frames[cfg.collector_source_field], dtype=object)
    else:
        collector_sources = np.asarray(["human"] * len(absolute_indices), dtype=object)

    logged_policy_actions = None
    if cfg.policy_action_field in raw_frames.column_names:
        logged_policy_actions = _to_action_array(raw_frames, cfg.policy_action_field)

    episode_to_positions: dict[int, list[int]] = {}
    for pos, episode_index in enumerate(episode_indices.tolist()):
        episode_to_positions.setdefault(int(episode_index), []).append(pos)

    rows_by_index: dict[int, dict[str, Any]] = {}
    for episode_index, episode_positions in episode_to_positions.items():
        episode_interventions = interventions[episode_positions]
        onset_positions = _extract_onset_positions(episode_interventions)
        if not onset_positions:
            continue

        negative_actions = None
        if cfg.negative_source == PREFERENCE_NEGATIVE_SOURCE_SHADOW:
            negative_actions = _build_episode_policy_replay(
                dataset=dataset,
                episode_positions=episode_positions,
                collector_policy_ids=collector_policy_ids,
                collector_sources=collector_sources,
                device=cfg.runtime.device,
            )
        elif logged_policy_actions is not None:
            negative_actions = logged_policy_actions[episode_positions]

        if negative_actions is None:
            continue

        episode_end_pos = len(episode_positions) - 1
        episode_task_lookup = dataset.meta.tasks
        for onset_local_pos in onset_positions:
            for offset in range(cfg.propagation_horizon + 1):
                local_pos = onset_local_pos + offset
                if local_pos > episode_end_pos:
                    break

                abs_pos = episode_positions[local_pos]
                abs_index = int(absolute_indices[abs_pos])
                positive_chunk, positive_pad = _chunk_from_sequence(
                    executed_actions[episode_positions],
                    start_pos=local_pos,
                    horizon=cfg.chunk_horizon,
                    end_pos=episode_end_pos,
                )
                negative_chunk, negative_pad = _chunk_from_sequence(
                    negative_actions,
                    start_pos=local_pos,
                    horizon=cfg.chunk_horizon,
                    end_pos=episode_end_pos,
                )

                row = {
                    "index": abs_index,
                    "episode_index": int(episode_index),
                    "frame_index": int(frame_indices[abs_pos]),
                    "task_index": int(task_indices[abs_pos]),
                    "task": episode_task_lookup.iloc[int(task_indices[abs_pos])].name,
                    "collector_policy_id": str(collector_policy_ids[abs_pos]),
                    "negative_source": cfg.negative_source,
                    "onset_index": int(absolute_indices[episode_positions[onset_local_pos]]),
                    "propagation_offset": int(offset),
                    "propagation_weight": float(cfg.propagation_decay**offset),
                    "positive_action_chunk": positive_chunk.tolist(),
                    "positive_action_pad": positive_pad.astype(bool).tolist(),
                    "negative_action_chunk": negative_chunk.tolist(),
                    "negative_action_pad": negative_pad.astype(bool).tolist(),
                }

                existing = rows_by_index.get(abs_index)
                if existing is None or row["propagation_weight"] > existing["propagation_weight"]:
                    rows_by_index[abs_index] = row

    rows = [rows_by_index[index] for index in sorted(rows_by_index)]
    return pd.DataFrame(rows)


def run_prepare_intervention_preferences(cfg: PreferencePipelineConfig) -> dict[str, Any]:
    cfg.validate()
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    init_logging()
    logging.info(pformat(cfg.to_dict()))

    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=cfg.dataset.episodes,
        revision=cfg.dataset.revision,
        download_videos=cfg.dataset.download_videos,
    )
    raw_frames = dataset.hf_dataset.with_format(None)

    required_columns = {
        ACTION,
        "episode_index",
        "frame_index",
        "index",
        "task_index",
        cfg.intervention_field,
        cfg.collector_policy_id_field,
    }
    missing_columns = sorted(required_columns - set(raw_frames.column_names))
    if missing_columns:
        raise KeyError(f"Dataset is missing required columns: {missing_columns}")

    preferences = _build_preference_rows(dataset=dataset, raw_frames=raw_frames, cfg=cfg)
    preferences.to_parquet(cfg.output_path, index=False)

    result = {
        "output_path": str(cfg.output_path),
        "num_preferences": int(len(preferences)),
        "negative_source": cfg.negative_source,
        "chunk_horizon": int(cfg.chunk_horizon),
        "propagation_horizon": int(cfg.propagation_horizon),
    }
    logging.info("Prepared %d intervention preferences at %s", len(preferences), cfg.output_path)
    return result


@parser.wrap()
def prepare_intervention_preferences(cfg: PreferencePipelineConfig):
    return run_prepare_intervention_preferences(cfg)


def main():
    register_third_party_plugins()
    prepare_intervention_preferences()


if __name__ == "__main__":
    main()
