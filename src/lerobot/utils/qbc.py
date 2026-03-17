#!/usr/bin/env python

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lerobot.utils.rabc import resolve_hf_path


class QBCWeights:
    """
    Load precomputed ICCQ outputs and produce normalized sample weights for weighted BC.

    The parquet file is expected to contain an `index` column and either:
    - a precomputed weight column (default: `qbc_weight`)
    - an advantage column (default: `advantage`) from which weights can be derived
    """

    log_prefix = "qbc"

    def __init__(
        self,
        weights_path: str | Path,
        *,
        beta: float = 0.2,
        clip_min: float = 0.0,
        clip_max: float = 10.0,
        weight_column: str = "qbc_weight",
        advantage_column: str = "advantage",
        epsilon: float = 1e-6,
        fallback_weight: float = 1.0,
        device: torch.device | None = None,
    ):
        if beta <= 0:
            raise ValueError("'beta' must be > 0.")
        if clip_min < 0:
            raise ValueError("'clip_min' must be >= 0.")
        if clip_max < clip_min:
            raise ValueError("'clip_max' must be >= 'clip_min'.")

        self.weights_path = resolve_hf_path(weights_path)
        self.beta = beta
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.weight_column = weight_column
        self.advantage_column = advantage_column
        self.epsilon = epsilon
        self.fallback_weight = fallback_weight
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info("Loading ICCQ weights from %s", self.weights_path)
        self.df = pd.read_parquet(self.weights_path)
        if "index" not in self.df.columns:
            raise ValueError("QBC parquet must contain an 'index' column.")

        if self.weight_column in self.df.columns:
            raw_weights = self.df[self.weight_column].to_numpy(dtype=np.float32, copy=True)
        elif self.advantage_column in self.df.columns:
            advantages = self.df[self.advantage_column].to_numpy(dtype=np.float32, copy=True)
            raw_weights = self._weights_from_advantage(advantages)
        else:
            raise ValueError(
                f"QBC parquet must contain either '{self.weight_column}' or '{self.advantage_column}'."
            )

        self.weight_lookup: dict[int, float] = {}
        has_advantage = self.advantage_column in self.df.columns
        self.advantage_lookup: dict[int, float] = {}
        for row_idx, global_idx in enumerate(self.df["index"].to_numpy(dtype=np.int64, copy=False)):
            weight = float(raw_weights[row_idx])
            if np.isnan(weight):
                continue
            self.weight_lookup[int(global_idx)] = weight
            if has_advantage:
                advantage = float(self.df.iloc[row_idx][self.advantage_column])
                if not np.isnan(advantage):
                    self.advantage_lookup[int(global_idx)] = advantage

        self.weight_array = np.asarray(list(self.weight_lookup.values()), dtype=np.float32)
        if self.weight_array.size == 0:
            raise ValueError("QBC parquet did not contain any usable weights.")

        self.weight_mean = float(np.mean(self.weight_array))
        self.weight_std = float(np.std(self.weight_array))
        self.weight_min = float(np.min(self.weight_array))
        self.weight_max = float(np.max(self.weight_array))
        logging.info(
            "Loaded %d ICCQ weights | mean=%.6f std=%.6f min=%.6f max=%.6f",
            len(self.weight_lookup),
            self.weight_mean,
            self.weight_std,
            self.weight_min,
            self.weight_max,
        )

    def _weights_from_advantage(self, advantages: np.ndarray) -> np.ndarray:
        scaled = np.clip(advantages / self.beta, -20.0, 20.0)
        weights = np.exp(scaled.astype(np.float64)).astype(np.float32)
        return np.clip(weights, self.clip_min, self.clip_max).astype(np.float32)

    def compute_batch_weights(self, batch: dict) -> tuple[torch.Tensor, dict]:
        indices = batch.get("index")
        if indices is None:
            logging.warning("QBC: Batch missing 'index' key, using uniform weights")
            batch_size = self._get_batch_size(batch)
            return torch.ones(batch_size, device=self.device), {"raw_mean_weight": 1.0}

        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().numpy().tolist()
        elif isinstance(indices, np.ndarray):
            indices = indices.tolist()

        raw_weights = np.asarray(
            [self.weight_lookup.get(int(idx), self.fallback_weight) for idx in indices],
            dtype=np.float32,
        )
        batch_stats = {
            "raw_mean_weight": float(np.mean(raw_weights)),
            "num_fallback_weight": int(np.sum(raw_weights == self.fallback_weight)),
            "num_clipped_max_weight": int(np.sum(raw_weights >= self.clip_max)),
        }

        weights = torch.tensor(raw_weights, device=self.device, dtype=torch.float32)
        batch_size = len(weights)
        weight_sum = weights.sum() + self.epsilon
        weights = weights * batch_size / weight_sum
        return weights, batch_stats

    def _get_batch_size(self, batch: dict) -> int:
        for key in ["action", "index"]:
            if key in batch:
                value = batch[key]
                if isinstance(value, (torch.Tensor, np.ndarray)):
                    return value.shape[0]
        return 1

    def get_stats(self) -> dict:
        return {
            "num_frames": len(self.weight_lookup),
            "weight_mean": self.weight_mean,
            "weight_std": self.weight_std,
            "weight_min": self.weight_min,
            "weight_max": self.weight_max,
            "beta": self.beta,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
        }
