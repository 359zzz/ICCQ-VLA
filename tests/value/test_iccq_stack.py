#!/usr/bin/env python

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

import lerobot.processor.tokenizer_processor as tokenizer_processor
import lerobot.values.iccq.modeling_iccq as iccq_modeling
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.configs.value_train import ValueTargetsConfig
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.values.iccq.configuration_iccq import ICCQConfig
from lerobot.values.iccq.modeling_iccq import ICCQPolicy
from lerobot.values.iccq.processor_iccq import (
    ICCQ_CURRENT_IMAGE_MASK_KEY,
    ICCQ_CURRENT_IMAGES_KEY,
    ICCQ_CURRENT_STATE_KEY,
    ICCQ_NEXT_IMAGE_MASK_KEY,
    ICCQ_NEXT_IMAGES_KEY,
    ICCQ_NEXT_STATE_KEY,
    make_iccq_pre_post_processors,
)


class _DummyTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, prompts, max_length, truncation, padding, return_tensors, padding_side="right"):
        del truncation, padding, return_tensors, padding_side
        bsize = len(prompts)
        input_ids = torch.zeros((bsize, max_length), dtype=torch.long)
        attention_mask = torch.zeros((bsize, max_length), dtype=torch.long)
        for i, prompt in enumerate(prompts):
            token_count = min(max_length, max(1, len(prompt.split())))
            input_ids[i, :token_count] = torch.arange(1, token_count + 1, dtype=torch.long)
            attention_mask[i, :token_count] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _DummyImageProcessor:
    size = {"height": 32, "width": 32}
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


class _DummyVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_proj = nn.Linear(3, 16)
        self.config = SimpleNamespace(hidden_size=16)
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        del gradient_checkpointing_kwargs
        self.gradient_checkpointing = True

    def get_image_features(self, pixel_values):
        pooled = pixel_values.mean(dim=(-1, -2))
        return self.image_proj(pooled)


class _DummyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=32)
        self.embed = nn.Embedding(2048, 32)
        self.proj = nn.Linear(32, 32)
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        del gradient_checkpointing_kwargs
        self.gradient_checkpointing = True

    def forward(self, input_ids, attention_mask, return_dict):
        del attention_mask, return_dict
        hidden = self.proj(self.embed(input_ids))
        return SimpleNamespace(last_hidden_state=hidden)


class _DummyAutoModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        del args, kwargs
        return _DummyVisionModel()


class _FakeDatasetTable:
    def __init__(self, data: dict[str, list]):
        self._data = data
        self.column_names = list(data.keys())

    def with_format(self, _):
        return self

    def __len__(self):
        return len(next(iter(self._data.values())))

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._data
        return self._data[key]


class _TaskLocAccessor:
    def __init__(self, mapping: dict[str, int]):
        self._mapping = mapping

    def __getitem__(self, task_name: str):
        return SimpleNamespace(task_index=self._mapping[task_name])


class _TaskIndexTable:
    def __init__(self, mapping: dict[str, int]):
        self.index = mapping.keys()
        self.loc = _TaskLocAccessor(mapping)


@pytest.fixture
def hf_stubs(monkeypatch):
    monkeypatch.setattr(tokenizer_processor, "AutoTokenizer", _DummyTokenizer)
    monkeypatch.setattr(iccq_modeling, "AutoImageProcessor", _DummyImageProcessor)
    monkeypatch.setattr(iccq_modeling, "AutoModel", _DummyAutoModel)
    monkeypatch.setattr(iccq_modeling, "_load_language_model", lambda **kwargs: _DummyLanguageModel())


def _make_cfg(**overrides) -> ICCQConfig:
    return ICCQConfig(
        device="cpu",
        camera_features=["observation.images.front"],
        max_state_dim=16,
        chunk_horizon=3,
        image_hidden_dim=16,
        language_hidden_dim=16,
        state_hidden_dim=8,
        obs_hidden_dim=24,
        action_hidden_dim=12,
        critic_hidden_dim=16,
        dropout=0.0,
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
        },
        **overrides,
    )


def test_iccq_processor_keeps_training_targets_and_splits_current_next(hf_stubs):
    del hf_stubs
    cfg = _make_cfg(camera_features=["observation.images.front", "observation.images.wrist"])
    preprocessor, _ = make_iccq_pre_post_processors(cfg)

    raw_batch = {
        "task": ["pick bottle", "place bottle"],
        OBS_STATE: torch.rand(2, 2, 12),
        "observation.images.front": torch.rand(2, 2, 3, 48, 40),
        ACTION: torch.rand(2, cfg.chunk_horizon, 4),
        cfg.reward_chunk_key: torch.rand(2, cfg.chunk_horizon),
        cfg.reward_pad_key: torch.tensor([[False, False, True], [False, True, True]], dtype=torch.bool),
        cfg.calibration_target_key: torch.rand(2),
        cfg.preference_negative_chunk_key: torch.rand(2, cfg.chunk_horizon, 4),
        cfg.preference_negative_pad_key: torch.tensor(
            [[False, False, True], [False, True, True]], dtype=torch.bool
        ),
        cfg.preference_weight_key: torch.tensor([1.0, 0.5], dtype=torch.float32),
    }
    processed = preprocessor(raw_batch)

    assert processed[OBS_LANGUAGE_TOKENS].shape == (2, cfg.tokenizer_max_length)
    assert processed[OBS_LANGUAGE_ATTENTION_MASK].dtype == torch.bool
    assert processed[ICCQ_CURRENT_IMAGES_KEY].shape == (2, 2, 3, 48, 40)
    assert processed[ICCQ_NEXT_IMAGES_KEY].shape == (2, 2, 3, 48, 40)
    assert processed[ICCQ_CURRENT_STATE_KEY].shape == (2, cfg.max_state_dim)
    assert processed[ICCQ_NEXT_STATE_KEY].shape == (2, cfg.max_state_dim)
    assert processed[cfg.reward_chunk_key].shape == (2, cfg.chunk_horizon)
    assert processed[cfg.preference_negative_chunk_key].shape == (2, cfg.chunk_horizon, 4)
    assert processed[cfg.reward_pad_key].dtype == torch.bool
    assert processed[cfg.preference_negative_pad_key].dtype == torch.bool


def test_iccq_policy_forward_computes_losses(hf_stubs):
    del hf_stubs
    cfg = _make_cfg()
    policy = ICCQPolicy(config=cfg)
    batch = {
        OBS_LANGUAGE_TOKENS: torch.randint(0, 100, (4, 12), dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(4, 12, dtype=torch.bool),
        ICCQ_CURRENT_IMAGES_KEY: torch.rand(4, 1, 3, 32, 32),
        ICCQ_CURRENT_IMAGE_MASK_KEY: torch.ones(4, 1, dtype=torch.bool),
        ICCQ_NEXT_IMAGES_KEY: torch.rand(4, 1, 3, 32, 32),
        ICCQ_NEXT_IMAGE_MASK_KEY: torch.ones(4, 1, dtype=torch.bool),
        ICCQ_CURRENT_STATE_KEY: torch.rand(4, cfg.max_state_dim),
        ICCQ_NEXT_STATE_KEY: torch.rand(4, cfg.max_state_dim),
        ACTION: torch.rand(4, cfg.chunk_horizon, 4),
        "action_is_pad": torch.tensor(
            [[False, False, False], [False, False, True], [False, True, True], [False, False, False]],
            dtype=torch.bool,
        ),
        cfg.reward_chunk_key: torch.rand(4, cfg.chunk_horizon),
        cfg.reward_pad_key: torch.tensor(
            [[False, False, False], [False, False, True], [False, True, True], [False, False, False]],
            dtype=torch.bool,
        ),
        cfg.next_observation_pad_key: torch.tensor([False, False, True, False], dtype=torch.bool),
        cfg.calibration_target_key: torch.rand(4),
        cfg.preference_weight_key: torch.tensor([1.0, 0.5, 0.0, 0.0], dtype=torch.float32),
        cfg.preference_negative_chunk_key: torch.rand(4, cfg.chunk_horizon, 4),
        cfg.preference_negative_pad_key: torch.tensor(
            [[False, False, False], [False, False, True], [False, True, True], [False, False, False]],
            dtype=torch.bool,
        ),
    }

    loss, loss_dict = policy.forward(batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert "td_loss" in loss_dict
    assert "preference_loss" in loss_dict
    assert "advantage_mean" in loss_dict

    loss_none, _ = policy.forward(batch, reduction="none")
    assert loss_none.shape == (4,)


def test_iccq_raw_batch_hook_builds_td_and_preference_targets(hf_stubs, tmp_path):
    del hf_stubs
    cfg = _make_cfg(chunk_horizon=2)
    policy = ICCQPolicy(config=cfg)

    raw_frames = _FakeDatasetTable(
        {
            "episode_index": [0, 0, 0, 0],
            "frame_index": [0, 1, 2, 3],
            "index": [0, 1, 2, 3],
        }
    )
    episodes = _FakeDatasetTable(
        {
            "episode_index": [0],
            "length": [4],
            "tasks": [["pick bottle"]],
            "dataset_to_index": [4],
            "episode_success": ["success"],
        }
    )
    dataset = SimpleNamespace(
        hf_dataset=raw_frames,
        meta=SimpleNamespace(
            episodes=episodes,
            tasks=_TaskIndexTable({"pick bottle": 0}),
        ),
    )

    pair_path = tmp_path / "prefs.parquet"
    pd.DataFrame(
        {
            "index": [1],
            "propagation_weight": [0.5],
            "negative_action_chunk": [[[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]],
            "negative_action_pad": [[False, True]],
        }
    ).to_parquet(pair_path, index=False)

    hook = policy.build_training_raw_batch_hook(
        dataset=dataset,
        targets_cfg=ValueTargetsConfig(),
        pair_path=pair_path,
    )
    batch = {"index": torch.tensor([0, 1, 3], dtype=torch.long)}
    out = hook(batch, step=0)

    assert out[cfg.reward_chunk_key].shape == (3, cfg.chunk_horizon)
    assert out[cfg.reward_pad_key].shape == (3, cfg.chunk_horizon)
    assert out[cfg.calibration_target_key].shape == (3,)
    assert out[cfg.next_observation_pad_key].shape == (3,)
    assert out[cfg.preference_negative_chunk_key].shape == (3, cfg.chunk_horizon, 4)
    assert torch.allclose(out[cfg.preference_weight_key], torch.tensor([0.0, 0.5, 0.0]))
    assert out[cfg.reward_pad_key][2, 1].item() is True
    assert out[cfg.next_observation_pad_key][2].item() is True


def test_coerce_pref_array_handles_object_arrays_with_singleton_dims():
    chunk_value = np.empty((1,), dtype=object)
    chunk_value[0] = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], dtype=np.float32)
    pad_value = np.empty((1,), dtype=object)
    pad_value[0] = np.array([False, True], dtype=np.bool_)

    chunk = iccq_modeling._coerce_pref_array(chunk_value, dtype=np.float32, expected_shape=(2, 4))
    pad = iccq_modeling._coerce_pref_array(pad_value, dtype=np.bool_, expected_shape=(2,))

    assert chunk.shape == (2, 4)
    assert pad.shape == (2,)
    assert np.allclose(chunk[0], np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
    assert pad.tolist() == [False, True]
