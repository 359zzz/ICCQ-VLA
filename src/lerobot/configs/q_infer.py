#!/usr/bin/env python

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus


@dataclass
class QInferenceDatasetConfig:
    repo_id: str
    root: str | None = None
    episodes: list[int] | None = None
    revision: str | None = None
    download_videos: bool = True

    def validate(self) -> None:
        if not self.repo_id:
            raise ValueError("'dataset.repo_id' must be non-empty.")


@dataclass
class QInferenceCheckpointConfig:
    checkpoint_path: str
    checkpoint_ref: str = "last"

    def validate(self) -> None:
        if not self.checkpoint_path:
            raise ValueError("'inference.checkpoint_path' must be non-empty.")
        if not self.checkpoint_ref:
            raise ValueError("'inference.checkpoint_ref' must be non-empty.")


@dataclass
class QInferenceRuntimeConfig:
    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 4

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("'runtime.batch_size' must be > 0.")
        if self.num_workers < 0:
            raise ValueError("'runtime.num_workers' must be >= 0.")


@dataclass
class QInferenceQBCConfig:
    beta: float = 0.2
    clip_min: float = 0.0
    clip_max: float = 10.0
    weight_column: str = "qbc_weight"

    def validate(self) -> None:
        if self.beta <= 0:
            raise ValueError("'qbc.beta' must be > 0.")
        if self.clip_min < 0:
            raise ValueError("'qbc.clip_min' must be >= 0.")
        if self.clip_max < self.clip_min:
            raise ValueError("'qbc.clip_max' must be >= 'qbc.clip_min'.")
        if not self.weight_column:
            raise ValueError("'qbc.weight_column' must be non-empty.")


@dataclass
class QInferencePipelineConfig:
    dataset: QInferenceDatasetConfig
    inference: QInferenceCheckpointConfig = field(
        default_factory=lambda: QInferenceCheckpointConfig(checkpoint_path="")
    )
    runtime: QInferenceRuntimeConfig = field(default_factory=QInferenceRuntimeConfig)
    qbc: QInferenceQBCConfig = field(default_factory=QInferenceQBCConfig)

    output_dir: Path | None = None
    output_filename: str = "q_values.parquet"
    job_name: str | None = None
    seed: int | None = 1000
    rename_map: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        self.dataset.validate()
        self.inference.validate()
        self.runtime.validate()
        self.qbc.validate()

        if not self.job_name:
            repo_tag = self.dataset.repo_id.replace("/", "_")
            self.job_name = f"q_infer_{repo_tag}"

        if self.output_dir is None:
            now = dt.datetime.now()
            out_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/q_infer") / out_dir

    @property
    def output_path(self) -> Path:
        if self.output_dir is None:
            raise ValueError("'output_dir' is not initialized. Call validate() first.")
        return self.output_dir / self.output_filename

    def to_dict(self) -> dict[str, Any]:
        return draccus.encode(self)  # type: ignore[no-any-return]
