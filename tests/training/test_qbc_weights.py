#!/usr/bin/env python

import pandas as pd
import pytest
import torch

from lerobot.utils.qbc import QBCWeights


def test_qbc_weights_use_precomputed_weight_column(tmp_path):
    weight_path = tmp_path / "q_values.parquet"
    pd.DataFrame(
        {
            "index": [0, 1, 2],
            "qbc_weight": [1.0, 2.0, 0.0],
        }
    ).to_parquet(weight_path, index=False)

    provider = QBCWeights(weight_path, device=torch.device("cpu"))
    weights, stats = provider.compute_batch_weights({"index": torch.tensor([0, 1, 2])})

    assert torch.isclose(weights.sum(), torch.tensor(3.0), atol=1e-6)
    assert weights[1] > weights[0] > weights[2]
    assert stats["raw_mean_weight"] == pytest.approx(1.0)
    assert stats["num_clipped_max_weight"] == 0


def test_qbc_weights_can_compute_from_advantage_and_fallback(tmp_path):
    weight_path = tmp_path / "q_values_from_adv.parquet"
    pd.DataFrame(
        {
            "index": [0, 1],
            "advantage": [0.0, 1.0],
        }
    ).to_parquet(weight_path, index=False)

    provider = QBCWeights(
        weight_path,
        beta=1.0,
        clip_min=0.0,
        clip_max=10.0,
        device=torch.device("cpu"),
    )
    weights, stats = provider.compute_batch_weights({"index": torch.tensor([0, 1, 99])})

    assert weights.shape == (3,)
    assert weights[1] > weights[0]
    assert stats["num_fallback_weight"] == 1
    assert provider.get_stats()["num_frames"] == 2
