#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from lerobot.policies.pretrained import ActionSelectKwargs, PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.import_utils import _transformers_available
from lerobot.utils.recording_annotations import EPISODE_SUCCESS, resolve_episode_success_label
from lerobot.values.iccq.configuration_iccq import ICCQConfig
from lerobot.values.iccq.processor_iccq import (
    ICCQ_CURRENT_IMAGE_MASK_KEY,
    ICCQ_CURRENT_IMAGES_KEY,
    ICCQ_CURRENT_STATE_KEY,
    ICCQ_NEXT_IMAGE_MASK_KEY,
    ICCQ_NEXT_IMAGES_KEY,
    ICCQ_NEXT_STATE_KEY,
)
from lerobot.values.pistar06.modeling_pistar06 import (
    EpisodeTargetInfo,
    _extract_hidden_size,
    _extract_vision_feature_size,
    _freeze_module,
    _load_language_model,
    _maybe_enable_gradient_checkpointing,
    _resolve_image_size,
    _resolve_load_dtype,
    _resolve_norm_stats,
    compute_normalized_value_targets,
)

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoImageProcessor, AutoModel
else:
    AutoImageProcessor = None
    AutoModel = None


def _compute_dense_rewards_from_targets(
    targets: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
) -> np.ndarray:
    rewards = np.zeros_like(targets, dtype=np.float32)
    n = targets.shape[0]
    for i in range(n):
        is_next_in_episode = (
            i + 1 < n
            and episode_indices[i + 1] == episode_indices[i]
            and frame_indices[i + 1] == frame_indices[i] + 1
        )
        if is_next_in_episode:
            rewards[i] = float(targets[i] - targets[i + 1])
        else:
            rewards[i] = float(targets[i])
    return rewards


def _expectile_loss(diff: Tensor, tau: float) -> Tensor:
    weight = torch.where(diff > 0, torch.full_like(diff, tau), torch.full_like(diff, 1.0 - tau))
    return weight * diff.square()


class ICCQModel(nn.Module):
    def __init__(self, cfg: ICCQConfig):
        super().__init__()
        if AutoModel is None or AutoImageProcessor is None:
            raise ImportError("transformers is not installed. Install with `pip install 'lerobot[transformers-dep]'`.")

        self.cfg = cfg
        self.model_dtype = _resolve_load_dtype(cfg.dtype)
        self.vision_encoder = AutoModel.from_pretrained(
            cfg.vision_repo_id,
            revision=cfg.vision_revision,
            torch_dtype=self.model_dtype,
        )
        self.language_model = _load_language_model(
            repo_id=cfg.language_repo_id,
            revision=cfg.language_revision,
            dtype=self.model_dtype,
        )

        image_processor = AutoImageProcessor.from_pretrained(
            cfg.vision_repo_id,
            revision=cfg.vision_revision,
            use_fast=True,
        )
        image_height, image_width = _resolve_image_size(image_processor)
        image_mean, image_std = _resolve_norm_stats(image_processor)
        self.image_resolution = (image_height, image_width)
        self.register_buffer(
            "image_mean",
            torch.tensor(image_mean, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(image_std, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )

        vision_feature_size = _extract_vision_feature_size(self.vision_encoder)
        language_hidden_size = _extract_hidden_size(self.language_model)

        self.image_projector = nn.Sequential(
            nn.Linear(vision_feature_size, cfg.image_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.language_projector = nn.Sequential(
            nn.Linear(language_hidden_size, cfg.language_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.state_projector = nn.Sequential(
            nn.Linear(cfg.max_state_dim, cfg.state_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        obs_input_dim = cfg.image_hidden_dim + cfg.language_hidden_dim + cfg.state_hidden_dim
        self.obs_projector = nn.Sequential(
            nn.Linear(obs_input_dim, cfg.obs_hidden_dim),
            nn.LayerNorm(cfg.obs_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        action_input_dim = cfg.chunk_horizon * self._action_dim(cfg)
        self.action_projector = nn.Sequential(
            nn.Linear(action_input_dim, cfg.action_hidden_dim),
            nn.LayerNorm(cfg.action_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        critic_input_dim = cfg.obs_hidden_dim + cfg.action_hidden_dim
        self.q1_head = nn.Sequential(
            nn.Linear(critic_input_dim, cfg.critic_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.critic_hidden_dim, 1),
        )
        self.q2_head = nn.Sequential(
            nn.Linear(critic_input_dim, cfg.critic_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.critic_hidden_dim, 1),
        )
        self.v_head = nn.Sequential(
            nn.Linear(cfg.obs_hidden_dim, cfg.critic_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.critic_hidden_dim, 1),
        )

        if cfg.use_gradient_checkpointing:
            _maybe_enable_gradient_checkpointing(self.language_model)
            _maybe_enable_gradient_checkpointing(self.vision_encoder)
        if cfg.freeze_language_model:
            _freeze_module(self.language_model)
        if cfg.freeze_vision_encoder:
            _freeze_module(self.vision_encoder)

    @staticmethod
    def _action_dim(cfg: ICCQConfig) -> int:
        action_feature = cfg.output_features.get(ACTION) if cfg.output_features else None
        if action_feature is None:
            raise ValueError("ICCQ requires ACTION output feature metadata.")
        return int(action_feature.shape[0])

    def _encode_images(self, flat_images: Tensor) -> Tensor:
        if hasattr(self.vision_encoder, "get_image_features"):
            return self.vision_encoder.get_image_features(pixel_values=flat_images)
        vision_outputs = self.vision_encoder(pixel_values=flat_images, return_dict=True)
        if hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None:
            return vision_outputs.pooler_output
        if hasattr(vision_outputs, "last_hidden_state"):
            return vision_outputs.last_hidden_state.mean(dim=1)
        raise ValueError("Unsupported vision encoder output. Expected pooler_output or last_hidden_state.")

    def _encode_language(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            raise ValueError("Language model output does not contain `last_hidden_state`.")
        token_mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        denom = token_mask.sum(dim=1).clamp_min(1.0)
        return (hidden * token_mask).sum(dim=1) / denom

    def _preprocess_images(self, images: Tensor, image_attention_mask: Tensor) -> Tensor:
        if images.ndim != 5:
            raise ValueError(f"'images' must have shape [B,N,C,H,W], got {tuple(images.shape)}.")
        if image_attention_mask.ndim != 2:
            raise ValueError(f"'image_attention_mask' must have shape [B,N], got {tuple(image_attention_mask.shape)}.")
        bsize, num_cameras = images.shape[:2]
        if image_attention_mask.shape[0] != bsize or image_attention_mask.shape[1] != num_cameras:
            raise ValueError("Batch shape mismatch between images and image_attention_mask.")

        if images.dtype == torch.uint8:
            images = images.to(dtype=torch.float32) / 255.0
        else:
            images = images.to(dtype=torch.float32)
            if bool(torch.max(images) > 1.0) or bool(torch.min(images) < 0.0):
                images = (images / 255.0).clamp(0.0, 1.0)

        flat_images = images.view(bsize * num_cameras, *images.shape[2:])
        if flat_images.shape[-2:] != self.image_resolution:
            flat_images = functional.interpolate(
                flat_images,
                size=self.image_resolution,
                mode="bilinear",
                align_corners=False,
            )

        mean = self.image_mean.to(device=flat_images.device, dtype=flat_images.dtype).view(1, 3, 1, 1)
        std = self.image_std.to(device=flat_images.device, dtype=flat_images.dtype).view(1, 3, 1, 1)
        flat_images = (flat_images - mean) / std
        flat_images = flat_images.view(
            bsize,
            num_cameras,
            flat_images.shape[1],
            flat_images.shape[2],
            flat_images.shape[3],
        )
        camera_mask = image_attention_mask.to(device=flat_images.device, dtype=flat_images.dtype).view(
            bsize, num_cameras, 1, 1, 1
        )
        return flat_images * camera_mask

    def _encode_observation(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        images: Tensor,
        image_attention_mask: Tensor,
        state: Tensor,
    ) -> Tensor:
        processed_images = self._preprocess_images(images, image_attention_mask)
        bsize, num_cameras = processed_images.shape[:2]
        flat_images = processed_images.reshape(bsize * num_cameras, *processed_images.shape[2:])
        flat_images = flat_images.to(dtype=self.model_dtype)

        if self.cfg.freeze_vision_encoder:
            with torch.no_grad():
                image_features = self._encode_images(flat_images)
        else:
            image_features = self._encode_images(flat_images)

        if self.cfg.freeze_language_model:
            with torch.no_grad():
                language_features = self._encode_language(input_ids=input_ids, attention_mask=attention_mask.long())
        else:
            language_features = self._encode_language(input_ids=input_ids, attention_mask=attention_mask.long())

        image_features = image_features.to(dtype=torch.float32)
        language_features = language_features.to(dtype=torch.float32)
        state = state.to(dtype=torch.float32)

        image_tokens = self.image_projector(image_features).view(bsize, num_cameras, -1)
        camera_token_mask = image_attention_mask.unsqueeze(-1).to(dtype=image_tokens.dtype)
        image_tokens = image_tokens * camera_token_mask
        camera_denominator = (
            image_attention_mask.sum(dim=1, keepdim=True).to(dtype=image_tokens.dtype).clamp_min(1.0)
        )
        image_pooled = image_tokens.sum(dim=1) / camera_denominator
        language_token = self.language_projector(language_features)
        state_token = self.state_projector(state)
        return self.obs_projector(torch.cat([image_pooled, language_token, state_token], dim=-1))

    def _encode_action_chunk(self, action_chunk: Tensor, action_pad: Tensor | None) -> Tensor:
        if action_chunk.ndim != 3:
            raise ValueError(f"'action_chunk' must have shape [B,H,A], got {tuple(action_chunk.shape)}.")
        if action_chunk.shape[1] != self.cfg.chunk_horizon:
            raise ValueError(
                f"Expected chunk horizon {self.cfg.chunk_horizon}, got action_chunk shape {tuple(action_chunk.shape)}."
            )
        if action_pad is not None:
            if action_pad.ndim != 2:
                raise ValueError(f"'action_pad' must have shape [B,H], got {tuple(action_pad.shape)}.")
            action_chunk = action_chunk.masked_fill(action_pad.unsqueeze(-1), 0.0)
        return self.action_projector(action_chunk.reshape(action_chunk.shape[0], -1).to(dtype=torch.float32))

    def forward_chunk_values(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        current_images: Tensor,
        current_image_attention_mask: Tensor,
        current_state: Tensor,
        next_images: Tensor,
        next_image_attention_mask: Tensor,
        next_state: Tensor,
        action_chunk: Tensor,
        action_pad: Tensor | None = None,
    ) -> dict[str, Tensor]:
        current_obs = self._encode_observation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=current_images,
            image_attention_mask=current_image_attention_mask,
            state=current_state,
        )
        next_obs = self._encode_observation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=next_images,
            image_attention_mask=next_image_attention_mask,
            state=next_state,
        )
        action_repr = self._encode_action_chunk(action_chunk=action_chunk, action_pad=action_pad)
        critic_input = torch.cat([current_obs, action_repr], dim=-1)
        q1 = self.q1_head(critic_input).squeeze(-1)
        q2 = self.q2_head(critic_input).squeeze(-1)
        v = self.v_head(current_obs).squeeze(-1)
        next_v = self.v_head(next_obs).squeeze(-1)
        return {"q1": q1, "q2": q2, "q_min": torch.minimum(q1, q2), "v": v, "next_v": next_v}


class ICCQPolicy(PreTrainedPolicy):
    config_class = ICCQConfig
    name = "iccq"

    def __init__(self, config: ICCQConfig, dataset_meta=None, **kwargs: Any):
        del dataset_meta, kwargs
        super().__init__(config)
        self.config = config
        self.model = ICCQModel(config)

    def get_optim_params(self):
        return self.parameters()

    def reset(self):
        return

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: ActionSelectKwargs) -> Tensor:
        raise RuntimeError("ICCQPolicy is a critic and does not support action prediction.")

    def select_action(self, batch: dict[str, Tensor], **kwargs: ActionSelectKwargs) -> Tensor:
        raise RuntimeError("ICCQPolicy is a critic and does not support action selection.")

    def predict_chunk_values(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        action_pad = batch.get(f"{ACTION}_is_pad")
        return self.model.forward_chunk_values(
            input_ids=batch[OBS_LANGUAGE_TOKENS],
            attention_mask=batch[OBS_LANGUAGE_ATTENTION_MASK],
            current_images=batch[ICCQ_CURRENT_IMAGES_KEY],
            current_image_attention_mask=batch[ICCQ_CURRENT_IMAGE_MASK_KEY],
            current_state=batch[ICCQ_CURRENT_STATE_KEY],
            next_images=batch[ICCQ_NEXT_IMAGES_KEY],
            next_image_attention_mask=batch[ICCQ_NEXT_IMAGE_MASK_KEY],
            next_state=batch[ICCQ_NEXT_STATE_KEY],
            action_chunk=batch[ACTION],
            action_pad=action_pad,
        )

    def build_training_raw_batch_hook(self, dataset, targets_cfg, pair_path: str | Path | None = None):
        raw_frames = dataset.hf_dataset.with_format(None)
        frame_count = len(raw_frames)
        if frame_count == 0:
            raise ValueError("Dataset has no frames.")

        episode_indices = np.asarray(raw_frames["episode_index"], dtype=np.int64)
        frame_indices = np.asarray(raw_frames["frame_index"], dtype=np.int64)
        absolute_indices = np.asarray(raw_frames["index"], dtype=np.int64)
        abs_index_to_pos = {int(abs_idx): pos for pos, abs_idx in enumerate(absolute_indices.tolist())}

        episodes_ds = dataset.meta.episodes.with_format(None)
        episodes = episodes_ds[:]
        n_episodes = len(episodes_ds)
        has_success = targets_cfg.success_field in episodes_ds.column_names

        episode_info: dict[int, EpisodeTargetInfo] = {}
        task_max_length: dict[int, int] = {}
        episode_end_lookup: dict[int, int] = {}
        for i in range(n_episodes):
            ep_idx = int(episodes["episode_index"][i])
            ep_length = int(episodes["length"][i])
            tasks = episodes["tasks"][i]
            task_name = tasks[0] if isinstance(tasks, list) else tasks
            if task_name not in dataset.meta.tasks.index:
                raise KeyError(f"Episode {ep_idx} references unknown task '{task_name}'.")
            task_index = int(dataset.meta.tasks.loc[task_name].task_index)

            explicit_success = episodes[targets_cfg.success_field][i] if has_success else None
            resolved_success = resolve_episode_success_label(
                explicit_success,
                default_label=targets_cfg.default_success,
                require_label=True,
            )
            ep_success = resolved_success == EPISODE_SUCCESS
            episode_info[ep_idx] = EpisodeTargetInfo(
                episode_index=ep_idx,
                task_index=task_index,
                length=ep_length,
                success=ep_success,
            )
            task_max_length[task_index] = max(task_max_length.get(task_index, 0), ep_length)
            episode_end_lookup[ep_idx] = int(episodes["dataset_to_index"][i]) - 1

        value_targets = compute_normalized_value_targets(
            episode_indices=episode_indices,
            frame_indices=frame_indices,
            episode_info=episode_info,
            task_max_lengths=task_max_length,
            c_fail_coef=targets_cfg.c_fail_coef,
            clip_min=-1.0,
            clip_max=0.0,
        )
        dense_rewards = _compute_dense_rewards_from_targets(value_targets, episode_indices, frame_indices)
        reward_lookup = {int(abs_idx): float(reward) for abs_idx, reward in zip(absolute_indices, dense_rewards, strict=True)}
        calibration_lookup = {
            int(abs_idx): float(target) for abs_idx, target in zip(absolute_indices, value_targets, strict=True)
        }

        preference_lookup: dict[int, dict[str, Any]] = {}
        if pair_path is not None:
            pair_df = pd.read_parquet(pair_path)
            required_columns = {
                "index",
                "propagation_weight",
                "negative_action_chunk",
                "negative_action_pad",
            }
            missing_columns = sorted(required_columns - set(pair_df.columns))
            if missing_columns:
                raise KeyError(f"Preference parquet is missing required columns: {missing_columns}")
            for row in pair_df.to_dict(orient="records"):
                preference_lookup[int(row["index"])] = row

        def q_target_hook(batch: dict[str, Any], step: int) -> dict[str, Any]:
            del step
            batch_indices = batch.get("index")
            if batch_indices is None:
                raise KeyError("Missing 'index' in batch while building ICCQ targets.")
            if not isinstance(batch_indices, Tensor):
                batch_indices = torch.as_tensor(batch_indices)

            batch_indices_np = batch_indices.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
            reward_chunks = np.zeros((len(batch_indices_np), self.config.chunk_horizon), dtype=np.float32)
            reward_pad = np.zeros((len(batch_indices_np), self.config.chunk_horizon), dtype=np.bool_)
            next_obs_is_pad = np.zeros(len(batch_indices_np), dtype=np.bool_)
            calibration_targets = np.zeros(len(batch_indices_np), dtype=np.float32)

            negative_chunk = None
            negative_pad = None
            preference_weight = np.zeros(len(batch_indices_np), dtype=np.float32)
            action_feature = self.config.output_features[ACTION]
            if preference_lookup:
                negative_chunk = np.zeros(
                    (len(batch_indices_np), self.config.chunk_horizon, action_feature.shape[0]),
                    dtype=np.float32,
                )
                negative_pad = np.ones((len(batch_indices_np), self.config.chunk_horizon), dtype=np.bool_)

            for row_idx, abs_index in enumerate(batch_indices_np.tolist()):
                pos = abs_index_to_pos[int(abs_index)]
                episode_index = int(episode_indices[pos])
                episode_end_abs = episode_end_lookup[episode_index]
                for offset in range(self.config.chunk_horizon):
                    reward_abs_index = min(int(abs_index) + offset, episode_end_abs)
                    reward_chunks[row_idx, offset] = reward_lookup[reward_abs_index]
                    reward_pad[row_idx, offset] = int(abs_index) + offset > episode_end_abs
                next_obs_is_pad[row_idx] = int(abs_index) + self.config.chunk_horizon > episode_end_abs
                calibration_targets[row_idx] = calibration_lookup[int(abs_index)]

                if negative_chunk is not None and int(abs_index) in preference_lookup:
                    pref = preference_lookup[int(abs_index)]
                    negative_chunk[row_idx] = np.asarray(pref["negative_action_chunk"], dtype=np.float32)
                    negative_pad[row_idx] = np.asarray(pref["negative_action_pad"], dtype=np.bool_)
                    preference_weight[row_idx] = float(pref["propagation_weight"])

            batch[self.config.reward_chunk_key] = torch.from_numpy(reward_chunks).to(dtype=torch.float32)
            batch[self.config.reward_pad_key] = torch.from_numpy(reward_pad).to(dtype=torch.bool)
            batch[self.config.next_observation_pad_key] = torch.from_numpy(next_obs_is_pad).to(dtype=torch.bool)
            batch[self.config.calibration_target_key] = torch.from_numpy(calibration_targets).to(dtype=torch.float32)
            batch[self.config.preference_weight_key] = torch.from_numpy(preference_weight).to(dtype=torch.float32)
            if negative_chunk is not None and negative_pad is not None:
                batch[self.config.preference_negative_chunk_key] = torch.from_numpy(negative_chunk).to(
                    dtype=torch.float32
                )
                batch[self.config.preference_negative_pad_key] = torch.from_numpy(negative_pad).to(dtype=torch.bool)
            return batch

        return q_target_hook

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        outputs = self.predict_chunk_values(batch)
        q1 = outputs["q1"]
        q2 = outputs["q2"]
        q_min = outputs["q_min"]
        v = outputs["v"]
        next_v = outputs["next_v"].detach()

        if self.config.reward_chunk_key not in batch:
            raise KeyError(
                f"Missing reward chunk key '{self.config.reward_chunk_key}' in batch. "
                "Make sure q_train raw-batch hook is enabled."
            )
        reward_chunk = batch[self.config.reward_chunk_key].to(device=q1.device, dtype=torch.float32)
        reward_pad = batch.get(self.config.reward_pad_key)
        if reward_pad is not None:
            reward_pad = reward_pad.to(device=q1.device, dtype=torch.bool)
        else:
            reward_pad = torch.zeros_like(reward_chunk, dtype=torch.bool)

        if reward_chunk.ndim == 3 and reward_chunk.shape[-1] == 1:
            reward_chunk = reward_chunk.squeeze(-1)
        if reward_chunk.ndim != 2:
            raise ValueError(f"Expected reward chunk shape [B,H], got {tuple(reward_chunk.shape)}.")

        discounts = torch.tensor(
            [self.config.gamma**i for i in range(self.config.chunk_horizon)],
            device=q1.device,
            dtype=torch.float32,
        ).view(1, -1)
        reward_mask = (~reward_pad).to(dtype=torch.float32)
        chunk_return = (reward_chunk * reward_mask * discounts).sum(dim=1)
        done_after_chunk = batch[self.config.next_observation_pad_key].to(device=q1.device, dtype=torch.float32)
        td_target = chunk_return + (self.config.gamma**self.config.chunk_horizon) * (1.0 - done_after_chunk) * next_v

        td_loss = functional.smooth_l1_loss(q1, td_target, reduction="none") + functional.smooth_l1_loss(
            q2, td_target, reduction="none"
        )
        value_loss = _expectile_loss(q_min.detach() - v, self.config.expectile_tau)

        calibration_loss = torch.zeros_like(td_loss)
        if self.config.calibration_target_key in batch:
            calibration_target = batch[self.config.calibration_target_key].to(device=q1.device, dtype=torch.float32)
            calibration_loss = functional.smooth_l1_loss(q_min, calibration_target, reduction="none")

        conservative_loss = torch.zeros_like(td_loss)
        candidate_q_values = [q_min.unsqueeze(1)]
        for _ in range(self.config.conservative_num_perturb_samples):
            noisy_chunk = batch[ACTION] + torch.randn_like(batch[ACTION]) * self.config.conservative_perturb_std
            noisy_values = self.model.forward_chunk_values(
                input_ids=batch[OBS_LANGUAGE_TOKENS],
                attention_mask=batch[OBS_LANGUAGE_ATTENTION_MASK],
                current_images=batch[ICCQ_CURRENT_IMAGES_KEY],
                current_image_attention_mask=batch[ICCQ_CURRENT_IMAGE_MASK_KEY],
                current_state=batch[ICCQ_CURRENT_STATE_KEY],
                next_images=batch[ICCQ_NEXT_IMAGES_KEY],
                next_image_attention_mask=batch[ICCQ_NEXT_IMAGE_MASK_KEY],
                next_state=batch[ICCQ_NEXT_STATE_KEY],
                action_chunk=noisy_chunk,
                action_pad=batch.get(f"{ACTION}_is_pad"),
            )["q_min"]
            candidate_q_values.append(noisy_values.unsqueeze(1))

        pref_loss = torch.zeros_like(td_loss)
        pref_weight = batch.get(self.config.preference_weight_key)
        pref_negative_chunk = batch.get(self.config.preference_negative_chunk_key)
        pref_negative_pad = batch.get(self.config.preference_negative_pad_key)
        if pref_weight is not None and pref_negative_chunk is not None:
            pref_weight = pref_weight.to(device=q1.device, dtype=torch.float32)
            pref_negative_chunk = pref_negative_chunk.to(device=q1.device, dtype=torch.float32)
            if pref_negative_pad is not None:
                pref_negative_pad = pref_negative_pad.to(device=q1.device, dtype=torch.bool)
            negative_values = self.model.forward_chunk_values(
                input_ids=batch[OBS_LANGUAGE_TOKENS],
                attention_mask=batch[OBS_LANGUAGE_ATTENTION_MASK],
                current_images=batch[ICCQ_CURRENT_IMAGES_KEY],
                current_image_attention_mask=batch[ICCQ_CURRENT_IMAGE_MASK_KEY],
                current_state=batch[ICCQ_CURRENT_STATE_KEY],
                next_images=batch[ICCQ_NEXT_IMAGES_KEY],
                next_image_attention_mask=batch[ICCQ_NEXT_IMAGE_MASK_KEY],
                next_state=batch[ICCQ_NEXT_STATE_KEY],
                action_chunk=pref_negative_chunk,
                action_pad=pref_negative_pad,
            )["q_min"]
            pref_margin = q_min - negative_values - self.config.preference_margin
            pref_loss = pref_weight * functional.softplus(-pref_margin)
            candidate_q_values.append(negative_values.unsqueeze(1))
        else:
            pref_weight = torch.zeros_like(td_loss)

        if len(candidate_q_values) > 1:
            candidate_tensor = torch.cat(candidate_q_values, dim=1)
            logsumexp = torch.logsumexp(candidate_tensor / self.config.conservative_alpha, dim=1)
            conservative_loss = self.config.conservative_alpha * (
                logsumexp - np.log(candidate_tensor.shape[1])
            ) - q_min

        total_per_sample = (
            self.config.td_loss_weight * td_loss
            + self.config.calibration_loss_weight * calibration_loss
            + self.config.conservative_loss_weight * conservative_loss
            + self.config.preference_loss_weight * pref_loss
            + self.config.value_loss_weight * value_loss
        )

        loss = total_per_sample if reduction == "none" else total_per_sample.mean()
        loss_dict = {
            "loss": float(loss.mean().detach().item()) if reduction == "none" else float(loss.detach().item()),
            "td_loss": float(td_loss.mean().detach().item()),
            "value_loss": float(value_loss.mean().detach().item()),
            "calibration_loss": float(calibration_loss.mean().detach().item()),
            "conservative_loss": float(conservative_loss.mean().detach().item()),
            "preference_loss": float(pref_loss.mean().detach().item()),
            "q_mean": float(q_min.mean().detach().item()),
            "v_mean": float(v.mean().detach().item()),
            "advantage_mean": float((q_min - v).mean().detach().item()),
            "preference_weight_mean": float(pref_weight.mean().detach().item()),
        }
        return loss, loss_dict
