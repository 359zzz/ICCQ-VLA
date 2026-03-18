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

import json

import pandas as pd

from lerobot.scripts.lerobot_dataset_report import (
    _build_episode_length_histogram,
    _format_ascii_histogram,
    build_report,
)


def test_build_episode_length_histogram_uses_20_bins_and_preserves_counts():
    lengths = [float(v) for v in range(1, 101)]
    histogram = _build_episode_length_histogram(lengths, bins=20)

    assert len(histogram) == 20
    assert sum(bin_info["count"] for bin_info in histogram) == len(lengths)
    assert histogram[0]["start"] == 1.0
    assert histogram[-1]["end"] >= 100


def test_build_episode_length_histogram_handles_constant_length():
    lengths = [42.0, 42.0, 42.0, 42.0]
    histogram = _build_episode_length_histogram(lengths, bins=20)

    assert len(histogram) == 20
    assert sum(bin_info["count"] for bin_info in histogram) == 4
    assert sum(1 for bin_info in histogram if bin_info["count"] > 0) == 1


def test_format_ascii_histogram_renders_one_line_per_bin():
    histogram = _build_episode_length_histogram([10.0, 20.0, 30.0, 40.0], bins=20)
    lines = _format_ascii_histogram(histogram)

    assert len(lines) == 20
    assert all("|" in line for line in lines)
    assert all("s]" in line for line in lines)


def test_build_report_reads_dataset_metadata_from_local_files(tmp_path):
    dataset_root = tmp_path / "demo_dataset"
    (dataset_root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (dataset_root / "data" / "chunk-000").mkdir(parents=True)

    info = {
        "robot_type": "piper_follower",
        "fps": 30,
        "codebase_version": "0.4.4",
        "splits": {"train": "0:2"},
        "total_episodes": 2,
        "total_frames": 5,
        "total_tasks": 1,
        "features": {
            "observation.images.wrist": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
            },
            "action": {"dtype": "float32", "shape": [7], "names": None},
        },
    }
    with open(dataset_root / "meta" / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f)

    pd.DataFrame({"task": ["pick up and put the bottle into the box"]}).to_parquet(
        dataset_root / "meta" / "tasks.parquet"
    )

    pd.DataFrame(
        {
            "episode_index": [0, 1],
            "length": [2, 3],
            "episode_success": ["success", "failure"],
            "tasks": [
                "pick up and put the bottle into the box",
                "pick up and put the bottle into the box",
            ],
            "stats/reward": [1.0, 0.0],
        }
    ).to_parquet(dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    pd.DataFrame(
        {
            "episode_index": [0, 0, 1, 1, 1],
            "complementary_info.is_intervention": [0, 1, 0, 0, 1],
            "complementary_info.collector_policy_id": [
                "human",
                "pi05_ckpt",
                "pi05_ckpt",
                "pi05_ckpt",
                "human",
            ],
            "complementary_info.collector_source": [
                "human",
                "policy",
                "policy",
                "policy",
                "human",
            ],
        }
    ).to_parquet(dataset_root / "data" / "chunk-000" / "file-000.parquet")

    report = build_report(dataset_root)

    assert report["quality"]["actual_episode_count"] == 2
    assert report["quality"]["actual_frame_count"] == 5
    assert report["success_metrics"]["success_count"] == 1
    assert report["success_metrics"]["failure_count"] == 1
    assert report["intervention_metrics"]["intervention_frames"] == 2
    assert report["intervention_metrics"]["episodes_with_intervention"] == 2
    assert report["collector_metrics"]["unique_collector_sources"] == ["human", "policy"]
    assert report["collector_metrics"]["unique_collector_policy_ids"] == ["human", "pi05_ckpt"]
    assert report["structure"]["task_count"] == 1
    assert report["structure"]["features"][0]["name"] == "action"
