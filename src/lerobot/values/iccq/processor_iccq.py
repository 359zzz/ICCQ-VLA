#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_IMAGES,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.values.iccq.configuration_iccq import ICCQConfig

ICCQ_CURRENT_IMAGES_KEY = "observation.iccq.current_images"
ICCQ_CURRENT_IMAGE_MASK_KEY = "observation.iccq.current_image_attention_mask"
ICCQ_NEXT_IMAGES_KEY = "observation.iccq.next_images"
ICCQ_NEXT_IMAGE_MASK_KEY = "observation.iccq.next_image_attention_mask"
ICCQ_CURRENT_STATE_KEY = "observation.iccq.current_state"
ICCQ_NEXT_STATE_KEY = "observation.iccq.next_state"


def _pad_last_dim(vector: Tensor, new_dim: int) -> Tensor:
    if vector.shape[-1] >= new_dim:
        return vector[..., :new_dim]
    return functional.pad(vector, (0, new_dim - vector.shape[-1]))


def _ensure_batch_time_lastdim(value: Tensor, expected_rank_without_time: int) -> Tensor:
    if value.ndim == expected_rank_without_time:
        return value.unsqueeze(1)
    if value.ndim == expected_rank_without_time + 1:
        return value
    raise ValueError(f"Unexpected tensor rank {value.ndim} for expected base rank {expected_rank_without_time}.")


@ProcessorStepRegistry.register(name="iccq_prepare_states")
@dataclass
class ICCQPrepareStatesProcessorStep(ProcessorStep):
    state_feature: str
    max_state_dim: int
    next_observation_pad_key: str

    def get_config(self) -> dict[str, Any]:
        return {
            "state_feature": self.state_feature,
            "max_state_dim": self.max_state_dim,
            "next_observation_pad_key": self.next_observation_pad_key,
        }

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})
        complementary_data = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})

        if self.state_feature not in observation:
            raise KeyError(f"Missing state feature '{self.state_feature}'.")

        state = observation[self.state_feature]
        if not isinstance(state, Tensor):
            state = torch.as_tensor(state)
        state = state.detach().to(dtype=torch.float32)
        state = _ensure_batch_time_lastdim(state, expected_rank_without_time=2)
        state = _pad_last_dim(state, self.max_state_dim)

        observation[ICCQ_CURRENT_STATE_KEY] = state[:, 0]
        observation[ICCQ_NEXT_STATE_KEY] = state[:, -1]

        state_pad_key = f"{self.state_feature}_is_pad"
        if state_pad_key in complementary_data:
            pad_mask = complementary_data[state_pad_key]
            if not isinstance(pad_mask, Tensor):
                pad_mask = torch.as_tensor(pad_mask)
            pad_mask = pad_mask.to(dtype=torch.bool)
            pad_mask = _ensure_batch_time_lastdim(pad_mask, expected_rank_without_time=1)
            complementary_data[self.next_observation_pad_key] = pad_mask[:, -1]
        else:
            complementary_data[self.next_observation_pad_key] = torch.zeros(state.shape[0], dtype=torch.bool)

        transition[TransitionKey.OBSERVATION] = observation
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="iccq_prepare_images")
@dataclass
class ICCQPrepareImagesProcessorStep(ProcessorStep):
    camera_features: list[str]

    def get_config(self) -> dict[str, Any]:
        return {"camera_features": self.camera_features}

    @staticmethod
    def _to_bchw_or_btchw(images: Tensor) -> Tensor:
        if images.ndim == 4:
            if images.shape[1] in {1, 3}:
                return images.unsqueeze(1)
            if images.shape[-1] in {1, 3}:
                return images.permute(0, 3, 1, 2).unsqueeze(1)
        if images.ndim == 5:
            if images.shape[2] in {1, 3}:
                return images
            if images.shape[-1] in {1, 3}:
                return images.permute(0, 1, 4, 2, 3)
        raise ValueError(f"Unsupported image tensor shape {tuple(images.shape)}.")

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})

        present_img_keys = [key for key in self.camera_features if key in observation]
        if len(present_img_keys) == 0:
            raise ValueError(
                "All configured cameras are missing in the input batch. "
                f"expected={self.camera_features} batch_keys={list(observation.keys())}"
            )

        reference_img = self._to_bchw_or_btchw(torch.as_tensor(observation[present_img_keys[0]])).to(torch.float32)
        bsize = reference_img.shape[0]
        current_tensors: list[Tensor] = []
        next_tensors: list[Tensor] = []
        current_masks: list[Tensor] = []
        next_masks: list[Tensor] = []

        for key in self.camera_features:
            if key in observation:
                images = self._to_bchw_or_btchw(torch.as_tensor(observation[key])).to(torch.float32)
                current_tensors.append(images[:, 0])
                next_tensors.append(images[:, -1])
                current_masks.append(torch.ones(bsize, dtype=torch.bool))
                next_masks.append(torch.ones(bsize, dtype=torch.bool))
            else:
                zeros = torch.zeros_like(reference_img[:, 0])
                current_tensors.append(zeros)
                next_tensors.append(zeros)
                current_masks.append(torch.zeros(bsize, dtype=torch.bool))
                next_masks.append(torch.zeros(bsize, dtype=torch.bool))

        observation[ICCQ_CURRENT_IMAGES_KEY] = torch.stack(current_tensors, dim=1)
        observation[ICCQ_NEXT_IMAGES_KEY] = torch.stack(next_tensors, dim=1)
        observation[ICCQ_CURRENT_IMAGE_MASK_KEY] = torch.stack(current_masks, dim=1)
        observation[ICCQ_NEXT_IMAGE_MASK_KEY] = torch.stack(next_masks, dim=1)

        transition[TransitionKey.OBSERVATION] = observation
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_iccq_pre_post_processors(
    config: ICCQConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    camera_features = list(config.camera_features)
    if not camera_features:
        camera_features = [k for k in (config.input_features or {}) if k.startswith(OBS_IMAGES)]

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        NormalizerProcessorStep(
            features={**(config.input_features or {}), **(config.output_features or {})},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            normalize_observation_keys={config.state_feature},
        ),
        ICCQPrepareStatesProcessorStep(
            state_feature=config.state_feature,
            max_state_dim=config.max_state_dim,
            next_observation_pad_key=config.next_observation_pad_key,
        ),
        TokenizerProcessorStep(
            tokenizer_name=config.language_repo_id,
            task_key=config.task_field,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
            truncation=True,
        ),
        ICCQPrepareImagesProcessorStep(camera_features=camera_features),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
