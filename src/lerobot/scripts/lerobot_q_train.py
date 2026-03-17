#!/usr/bin/env python

import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.q_train import QTrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.utils import cycle
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import format_big_number, has_method, init_logging


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    policy.train()

    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)

    accelerator.backward(loss)

    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def q_train(
    cfg: QTrainPipelineConfig,
    accelerator: Accelerator | None = None,
):
    cfg.validate()

    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        force_cpu = cfg.critic.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    init_logging(accelerator=accelerator)
    is_main_process = accelerator.is_main_process

    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)

    accelerator.wait_for_everyone()

    if not is_main_process:
        dataset = make_dataset(cfg)

    if is_main_process:
        logging.info("Creating critic model")
    critic_model = make_policy(
        cfg=cfg.critic,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    q_target_raw_batch_hook = None
    if has_method(critic_model, "build_training_raw_batch_hook"):
        q_target_raw_batch_hook = critic_model.build_training_raw_batch_hook(
            dataset=dataset,
            targets_cfg=cfg.targets,
            pair_path=cfg.preferences.pair_path,
        )
        if is_main_process:
            logging.info("Using ICCQ raw-batch hook for TD/calibration/preference targets.")
    elif is_main_process:
        logging.info("Critic model does not define a raw-batch hook; using dataset fields as-is.")

    accelerator.wait_for_everyone()

    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.critic.pretrained_path and not cfg.resume) or not cfg.critic.pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.critic.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**(cfg.critic.input_features or {}), **(cfg.critic.output_features or {})},
                "norm_map": cfg.critic.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": cfg.critic.output_features or {},
                "norm_map": cfg.critic.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.critic,
        pretrained_path=cfg.critic.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, critic_model)

    step = 0
    if cfg.resume:
        if cfg.checkpoint_path is None:
            raise ValueError("'checkpoint_path' is missing while resume=true.")
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in critic_model.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in critic_model.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=not cfg.dataset.streaming,
        sampler=None,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    accelerator.wait_for_everyone()
    critic_model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        critic_model, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

    critic_model.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        logging.info(
            "Start ICCQ critic training on a fixed dataset, with effective batch size: %d",
            effective_batch_size,
        )

    logged_first_prompt = False

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        if q_target_raw_batch_hook is not None:
            batch = q_target_raw_batch_hook(batch, step)
        batch = preprocessor(batch)

        if is_main_process and not logged_first_prompt and cfg.critic.task_field in batch:
            task_batch = batch[cfg.critic.task_field]
            if isinstance(task_batch, str):
                first_prompt = task_batch
            elif len(task_batch) > 0:
                first_prompt = task_batch[0]
            else:
                first_prompt = None
            if first_prompt is not None:
                logging.info("First ICCQ prompt:\n%s", first_prompt)
                logged_first_prompt = True

        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            critic_model,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )

        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint critic after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(critic_model),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

    if is_main_process:
        logging.info("End of ICCQ training")

        if cfg.critic.push_to_hub:
            unwrapped_critic = accelerator.unwrap_model(critic_model)
            if cfg.critic.use_peft:
                unwrapped_critic.push_model_to_hub(cfg, peft_model=unwrapped_critic)
            else:
                unwrapped_critic.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.critic.repo_id)

    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    q_train()


if __name__ == "__main__":
    main()
