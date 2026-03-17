#!/usr/bin/env python

from __future__ import annotations

import logging
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.q_infer import QInferencePipelineConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK, PRETRAINED_MODEL_DIR
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging, inside_slurm
from lerobot.values.iccq.configuration_iccq import ICCQConfig


def _set_infer_logger_levels() -> None:
    for logger_name in ["fsspec", "fsspec.local", "huggingface_hub", "datasets", "torchcodec"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _create_accelerator(cfg: QInferencePipelineConfig, accelerator: Accelerator | None) -> Accelerator:
    if accelerator is not None:
        return accelerator

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    force_cpu = cfg.runtime.device == "cpu"
    return Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs], cpu=force_cpu)


def _resolve_pretrained_model_dir(checkpoint_path: str, checkpoint_ref: str) -> Path:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")

    if (path / SAFETENSORS_SINGLE_FILE).is_file() and (path / "config.json").is_file():
        return path

    if (path / PRETRAINED_MODEL_DIR / SAFETENSORS_SINGLE_FILE).is_file() and (
        path / PRETRAINED_MODEL_DIR / "config.json"
    ).is_file():
        return path / PRETRAINED_MODEL_DIR

    checkpoints_root = path / CHECKPOINTS_DIR if (path / CHECKPOINTS_DIR).is_dir() else path
    step_ref = LAST_CHECKPOINT_LINK if checkpoint_ref == "last" else checkpoint_ref
    step_dir = checkpoints_root / step_ref

    if (step_dir / PRETRAINED_MODEL_DIR / SAFETENSORS_SINGLE_FILE).is_file() and (
        step_dir / PRETRAINED_MODEL_DIR / "config.json"
    ).is_file():
        return step_dir / PRETRAINED_MODEL_DIR

    if (step_dir / SAFETENSORS_SINGLE_FILE).is_file() and (step_dir / "config.json").is_file():
        return step_dir

    raise FileNotFoundError(
        f"Could not resolve pretrained model directory from checkpoint_path={path} checkpoint_ref={checkpoint_ref}."
    )


def _build_dataset(cfg: QInferencePipelineConfig, critic_cfg: ICCQConfig) -> LeRobotDataset:
    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        revision=cfg.dataset.revision,
    )
    delta_timestamps = resolve_delta_timestamps(critic_cfg, ds_meta)
    return LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=cfg.dataset.episodes,
        revision=cfg.dataset.revision,
        download_videos=cfg.dataset.download_videos,
        delta_timestamps=delta_timestamps,
    )


def _load_dataset_distributed(
    cfg: QInferencePipelineConfig,
    accelerator: Accelerator,
    critic_cfg: ICCQConfig,
) -> LeRobotDataset:
    if accelerator.is_main_process:
        dataset = _build_dataset(cfg, critic_cfg)
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        dataset = _build_dataset(cfg, critic_cfg)
    return dataset


def _init_runtime(
    cfg: QInferencePipelineConfig,
    accelerator: Accelerator,
) -> tuple[Path, torch.device]:
    output_dir = cfg.output_dir
    if output_dir is None:
        raise ValueError("'output_dir' is not initialized. Call validate() first.")
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    log_file = output_dir / "q_infer.log" if accelerator.is_main_process else None
    init_logging(log_file=log_file, file_level="INFO", accelerator=accelerator)
    _set_infer_logger_levels()

    if accelerator.is_main_process:
        logging.info(pformat(cfg.to_dict()))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    return output_dir, device


def _compute_qbc_weights(
    advantages: np.ndarray,
    beta: float,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    scaled = np.clip(advantages / beta, -20.0, 20.0)
    weights = np.exp(scaled.astype(np.float64)).astype(np.float32)
    return np.clip(weights, clip_min, clip_max).astype(np.float32)


@parser.wrap()
def q_infer(
    cfg: QInferencePipelineConfig,
    accelerator: Accelerator | None = None,
) -> dict[str, Any]:
    cfg.validate()

    accelerator = _create_accelerator(cfg, accelerator)
    _, device = _init_runtime(cfg, accelerator)

    pretrained_dir = _resolve_pretrained_model_dir(
        checkpoint_path=cfg.inference.checkpoint_path,
        checkpoint_ref=cfg.inference.checkpoint_ref,
    )
    critic_cfg = PreTrainedConfig.from_pretrained(pretrained_dir)
    if not isinstance(critic_cfg, ICCQConfig):
        raise ValueError(
            f"Unsupported critic config type '{type(critic_cfg)}'. lerobot-q-infer currently supports only 'iccq'."
        )

    critic_cfg.pretrained_path = pretrained_dir
    critic_cfg.device = device.type

    dataset = _load_dataset_distributed(cfg, accelerator, critic_cfg)
    raw_frames = dataset.hf_dataset.with_format(None)
    frame_count = len(raw_frames)
    if frame_count == 0:
        raise ValueError("Dataset has no frames.")

    critic_policy = make_policy(
        cfg=critic_cfg,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=critic_cfg,
        pretrained_path=pretrained_dir,
        preprocessor_overrides={"device_processor": {"device": device.type}},
    )

    absolute_indices = np.asarray(raw_frames["index"], dtype=np.int64)
    episode_indices = np.asarray(raw_frames["episode_index"], dtype=np.int64)
    frame_indices = np.asarray(raw_frames["frame_index"], dtype=np.int64)
    task_indices = (
        np.asarray(raw_frames["task_index"], dtype=np.int64)
        if "task_index" in raw_frames.column_names
        else np.full(frame_count, -1, dtype=np.int64)
    )

    eval_loader = DataLoader(
        dataset,
        batch_size=cfg.runtime.batch_size,
        shuffle=False,
        num_workers=cfg.runtime.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    critic_policy = accelerator.prepare(critic_policy)
    eval_loader = accelerator.prepare(eval_loader)

    if accelerator.is_main_process:
        max_abs_index = int(np.max(absolute_indices))
        q1_lookup = np.zeros(max_abs_index + 1, dtype=np.float32)
        q2_lookup = np.zeros(max_abs_index + 1, dtype=np.float32)
        q_lookup = np.zeros(max_abs_index + 1, dtype=np.float32)
        v_lookup = np.zeros(max_abs_index + 1, dtype=np.float32)
        prediction_seen = np.zeros(max_abs_index + 1, dtype=np.bool_)
        logging.info(
            "Start ICCQ inference | world_size=%d batches=%d batch_size=%d checkpoint=%s",
            accelerator.num_processes,
            len(eval_loader),
            cfg.runtime.batch_size,
            pretrained_dir,
        )
    else:
        q1_lookup = None
        q2_lookup = None
        q_lookup = None
        v_lookup = None
        prediction_seen = None

    critic_policy.eval()
    eval_iter = tqdm(
        eval_loader,
        desc="ICCQ inference",
        total=len(eval_loader),
        leave=False,
        disable=(not accelerator.is_main_process) or inside_slurm(),
    )

    with torch.no_grad():
        for raw_batch in eval_iter:
            batch_indices = raw_batch["index"]
            if not isinstance(batch_indices, torch.Tensor):
                batch_indices = torch.as_tensor(batch_indices)
            batch_indices = batch_indices.to(device=device, dtype=torch.long, non_blocking=True)

            processed_batch = preprocessor(raw_batch)
            with accelerator.autocast():
                outputs = accelerator.unwrap_model(critic_policy).predict_chunk_values(processed_batch)

            gathered_idx = accelerator.gather_for_metrics(batch_indices)
            gathered_q1 = accelerator.gather_for_metrics(outputs["q1"])
            gathered_q2 = accelerator.gather_for_metrics(outputs["q2"])
            gathered_q = accelerator.gather_for_metrics(outputs["q_min"])
            gathered_v = accelerator.gather_for_metrics(outputs["v"])

            if accelerator.is_main_process:
                idx_np = gathered_idx.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
                q1_np = gathered_q1.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
                q2_np = gathered_q2.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
                q_np = gathered_q.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
                v_np = gathered_v.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
                q1_lookup[idx_np] = q1_np
                q2_lookup[idx_np] = q2_np
                q_lookup[idx_np] = q_np
                v_lookup[idx_np] = v_np
                prediction_seen[idx_np] = True

    accelerator.wait_for_everyone()

    if not accelerator.is_main_process:
        accelerator.end_training()
        return {
            "main_process": False,
            "world_size": int(accelerator.num_processes),
        }

    if (
        q1_lookup is None
        or q2_lookup is None
        or q_lookup is None
        or v_lookup is None
        or prediction_seen is None
    ):
        raise RuntimeError("Prediction buffers unexpectedly missing on main process.")

    missing_mask = ~prediction_seen[absolute_indices]
    if bool(np.any(missing_mask)):
        missing_count = int(np.sum(missing_mask))
        raise RuntimeError(f"Inference is missing predictions for {missing_count} frames.")

    q1_values = q1_lookup[absolute_indices]
    q2_values = q2_lookup[absolute_indices]
    q_values = q_lookup[absolute_indices]
    v_values = v_lookup[absolute_indices]
    advantages = q_values - v_values
    qbc_weights = _compute_qbc_weights(
        advantages=advantages,
        beta=cfg.qbc.beta,
        clip_min=cfg.qbc.clip_min,
        clip_max=cfg.qbc.clip_max,
    )

    output_df = pd.DataFrame(
        {
            "index": absolute_indices,
            "episode_index": episode_indices,
            "frame_index": frame_indices,
            "task_index": task_indices,
            "q1": q1_values,
            "q2": q2_values,
            "q_value": q_values,
            "v_value": v_values,
            "advantage": advantages,
            cfg.qbc.weight_column: qbc_weights,
        }
    )
    output_df.to_parquet(cfg.output_path, index=False)

    logging.info("Wrote %d ICCQ rows to %s", len(output_df), cfg.output_path)

    result = {
        "main_process": True,
        "world_size": int(accelerator.num_processes),
        "num_frames": int(frame_count),
        "checkpoint": str(pretrained_dir),
        "output_path": str(cfg.output_path),
        "advantage_mean": float(np.mean(advantages)),
        "advantage_std": float(np.std(advantages)),
        "qbc_weight_mean": float(np.mean(qbc_weights)),
        "qbc_weight_max": float(np.max(qbc_weights)),
    }
    accelerator.end_training()
    return result


def main():
    register_third_party_plugins()
    q_infer()


if __name__ == "__main__":
    main()
