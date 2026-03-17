#!/usr/bin/env python

from types import SimpleNamespace

import numpy as np

from lerobot.scripts.lerobot_prepare_intervention_preferences import (
    PREFERENCE_NEGATIVE_SOURCE_SHADOW,
    PreferenceDatasetConfig,
    PreferencePipelineConfig,
    _build_preference_rows,
    _extract_onset_positions,
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


def test_extract_onset_positions():
    interventions = np.asarray([0.0, 1.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    assert _extract_onset_positions(interventions) == [1, 4]


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
