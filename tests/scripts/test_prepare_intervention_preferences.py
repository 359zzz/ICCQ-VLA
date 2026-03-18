#!/usr/bin/env python

from types import SimpleNamespace

import numpy as np

from lerobot.scripts.lerobot_prepare_intervention_preferences import (
    PREFERENCE_NEGATIVE_SOURCE_SHADOW,
    PreferenceDatasetConfig,
    PreferencePipelineConfig,
    _build_preference_rows,
    _extract_onset_positions,
    _resolve_policy_bundle,
    _to_action_array,
)
from lerobot.utils.constants import ACTION


class _FakeFrames:
    def __init__(self, data: dict[str, object]):
        self._data = data
        self.column_names = list(data.keys())

    def __getitem__(self, key):
        return self._data[key]


class _TaskIloc:
    def __init__(self, names: list[str]):
        self._names = names

    def __getitem__(self, index: int):
        return SimpleNamespace(name=self._names[index])


class _TaskTable:
    def __init__(self, names: list[str]):
        self.iloc = _TaskIloc(names)


class _ColumnWithToList:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


def test_extract_onset_positions():
    interventions = np.asarray([0.0, 1.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    assert _extract_onset_positions(interventions) == [1, 4]


def test_to_action_array_accepts_column_like_objects():
    raw_frames = _FakeFrames({ACTION: _ColumnWithToList([[1.0, 2.0], [3.0, 4.0]])})

    arr = _to_action_array(raw_frames, ACTION)

    assert arr.shape == (2, 2)
    assert arr.dtype == np.float32
    assert arr.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_resolve_policy_bundle_reuses_cache(monkeypatch):
    dataset = SimpleNamespace(meta=SimpleNamespace(stats={"dummy": 1}))
    fake_cfg = SimpleNamespace(pretrained_path=None, device=None)
    calls = {"make_policy": 0, "make_pre_post_processors": 0}

    monkeypatch.setattr(
        "lerobot.scripts.lerobot_prepare_intervention_preferences.PreTrainedConfig.from_pretrained",
        lambda provenance: fake_cfg,
    )

    def fake_make_policy(*, cfg, ds_meta):
        calls["make_policy"] += 1
        return object()

    def fake_make_pre_post_processors(**kwargs):
        calls["make_pre_post_processors"] += 1
        return object(), object()

    monkeypatch.setattr(
        "lerobot.scripts.lerobot_prepare_intervention_preferences.make_policy",
        fake_make_policy,
    )
    monkeypatch.setattr(
        "lerobot.scripts.lerobot_prepare_intervention_preferences.make_pre_post_processors",
        fake_make_pre_post_processors,
    )

    bundle_cache = {}
    bundle_a = _resolve_policy_bundle(
        provenance="/tmp/fake_ckpt",
        dataset=dataset,
        device="cpu",
        bundle_cache=bundle_cache,
    )
    bundle_b = _resolve_policy_bundle(
        provenance="/tmp/fake_ckpt",
        dataset=dataset,
        device="cpu",
        bundle_cache=bundle_cache,
    )

    assert bundle_a == bundle_b
    assert calls["make_policy"] == 1
    assert calls["make_pre_post_processors"] == 1


def test_build_preference_rows_keeps_strongest_overlap(monkeypatch):
    cfg = PreferencePipelineConfig(
        dataset=PreferenceDatasetConfig(repo_id="dummy/test"),
        chunk_horizon=2,
        propagation_horizon=2,
        propagation_decay=0.5,
        negative_source=PREFERENCE_NEGATIVE_SOURCE_SHADOW,
    )
    cfg.runtime.device = "cpu"

    raw_frames = _FakeFrames(
        {
            ACTION: np.asarray(
                [
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [2.0, 2.0],
                    [3.0, 3.0],
                    [4.0, 4.0],
                ],
                dtype=np.float32,
            ),
            "episode_index": [0, 0, 0, 0, 0],
            "frame_index": [0, 1, 2, 3, 4],
            "index": [0, 1, 2, 3, 4],
            "task_index": [0, 0, 0, 0, 0],
            cfg.intervention_field: [0.0, 1.0, 0.0, 1.0, 0.0],
            cfg.collector_policy_id_field: ["ckptA"] * 5,
            cfg.collector_source_field: ["policy", "human", "human", "human", "policy"],
        }
    )
    dataset = SimpleNamespace(meta=SimpleNamespace(tasks=_TaskTable(["pick bottle"])))

    monkeypatch.setattr(
        "lerobot.scripts.lerobot_prepare_intervention_preferences._build_episode_policy_replay",
        lambda **kwargs: np.asarray(
            [
                [9.0, 9.0],
                [8.0, 8.0],
                [7.0, 7.0],
                [6.0, 6.0],
                [5.0, 5.0],
            ],
            dtype=np.float32,
        ),
    )

    df = _build_preference_rows(dataset=dataset, raw_frames=raw_frames, cfg=cfg)

    assert list(df["index"]) == [1, 2, 3, 4]
    assert set(df["task"]) == {"pick bottle"}
    assert set(df["negative_source"]) == {PREFERENCE_NEGATIVE_SOURCE_SHADOW}

    strongest = df[df["index"] == 3].iloc[0]
    assert strongest["onset_index"] == 3
    assert strongest["propagation_offset"] == 0
    assert strongest["propagation_weight"] == 1.0
    assert len(strongest["positive_action_chunk"]) == cfg.chunk_horizon
    assert len(strongest["negative_action_chunk"]) == cfg.chunk_horizon
