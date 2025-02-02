#！/usr/bin/env python

# 版权所有 2024 The HuggingFace， Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版（“许可证”）获得许可;
# 除非遵守许可，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据许可分发的依据是按“原样”分发的，
# 不附带任何明示或暗示的保证或条件。
# 请参阅许可证，了解管理权限的特定语言，以及
# 许可证的限制。
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from pprint import pformat
from threading import Lock

import hydra
import numpy as np
import torch
from deepdiff import DeepDiff
from omegaconf import DictConfig, ListConfig, OmegaConf
from termcolor import colored
from torch import nn
from torch.cuda.amp import GradScaler

from lerobot.common.datasets.factory import make_dataset, resolve_delta_timestamps
from lerobot.common.datasets.lerobot_dataset import MultiLeRobotDataset
from lerobot.common.datasets.online_buffer import OnlineBuffer, compute_sampler_weights
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.logger import Logger, log_output_dir
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.policy_protocol import PolicyWithUpdate
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_hydra_config,
    init_logging,
    set_global_seed,
)
from lerobot.scripts.eval import eval_policy


def make_optimizer_and_scheduler(cfg, policy):
    if cfg.policy.name == "act":
        optimizer_params_dicts = [
            {
                "params": [
                    p
                    for n, p in policy.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in policy.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": cfg.training.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_params_dicts, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
        )
        lr_scheduler = None
    elif cfg.policy.name == "diffusion":
        optimizer = torch.optim.Adam(
            policy.diffusion.parameters(),
            cfg.training.lr,
            cfg.training.adam_betas,
            cfg.training.adam_eps,
            cfg.training.adam_weight_decay,
        )
        from diffusers.optimization import get_scheduler

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=cfg.training.offline_steps,
        )
    elif policy.name == "tdmpc":
        optimizer = torch.optim.Adam(policy.parameters(), cfg.training.lr)
        lr_scheduler = None
    elif cfg.policy.name == "vqbet":
        from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTOptimizer, VQBeTScheduler

        optimizer = VQBeTOptimizer(policy, cfg)
        lr_scheduler = VQBeTScheduler(optimizer, cfg)
    else:
        raise NotImplementedError()

    return optimizer, lr_scheduler


def update_policy(
    policy,
    batch,
    optimizer,
    grad_clip_norm,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
):
    """Returns a dictionary of items for logging."""
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        output_dict = policy.forward(batch)
        # TODO（rcadene）： policy.unnormalize_outputs（out_dict）
        loss = output_dict["loss"]
    grad_scaler.scale(loss).backward()

    # 在梯度裁剪之前就地取消优化器分配的参数的缩放。
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer 的梯度已经未缩放，因此 scaler.step 不会取消缩放它们，
    # 尽管如果梯度包含 infs 或 NaNs，它仍然会跳过 optimizer.step（）。
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # 更新下一次迭代的比例。
    grad_scaler.update()

    optimizer.zero_grad()

    if lr_scheduler is not None:
        lr_scheduler.step()

    if isinstance(policy, PolicyWithUpdate):
        # 可能更新内部缓冲区（例如，像 TDMPC 中的指数移动平均线）。
        policy.update()

    info = {
        "loss": loss.item(),
        "grad_norm": float(grad_norm),
        "lr": optimizer.param_groups[0]["lr"],
        "update_s": time.perf_counter() - start_time,
        **{k: v for k, v in output_dict.items() if k != "loss"},
    }
    info.update({k: v for k, v in output_dict.items() if k not in info})

    return info


def log_train_info(logger: Logger, info, step, cfg, dataset, is_online):
    loss = info["loss"]
    grad_norm = info["grad_norm"]
    lr = info["lr"]
    update_s = info["update_s"]
    dataloading_s = info["dataloading_s"]

    # 样本是 （observation，action） 对，其中 observation 和 action
    # 可以位于多个时间戳上。在一个批次中，我们有 'batch_size' 个样本。
    num_samples = (step + 1) * cfg.training.batch_size
    avg_samples_per_ep = dataset.num_frames / dataset.num_episodes
    num_episodes = num_samples / avg_samples_per_ep
    num_epochs = num_samples / dataset.num_frames
    log_items = [
        f"step:{format_big_number(step)}",
        # 训练期间看到的样本数
        f"smpl:{format_big_number(num_samples)}",
        # 训练期间看到的发作次数
        f"ep:{format_big_number(num_episodes)}",
        # 看到所有唯一样本的次数
        f"epch:{num_epochs:.2f}",
        f"loss:{loss:.3f}",
        f"grdn:{grad_norm:.3f}",
        f"lr:{lr:0.1e}",
        # 以秒为单位
        f"updt_s:{update_s:.3f}",
        f"data_s:{dataloading_s:.3f}",  # 如果不是 ~0，则您受到 CPU 或 IO 的瓶颈
    ]
    logging.info(" ".join(log_items))

    info["step"] = step
    info["num_samples"] = num_samples
    info["num_episodes"] = num_episodes
    info["num_epochs"] = num_epochs
    info["is_online"] = is_online

    logger.log_dict(info, step, mode="train")


def log_eval_info(logger, info, step, cfg, dataset, is_online):
    eval_s = info["eval_s"]
    avg_sum_reward = info["avg_sum_reward"]
    pc_success = info["pc_success"]

    # 样本是 （observation，action） 对，其中 observation 和 action
    # 可以位于多个时间戳上。在一个批次中，我们有 'batch_size' 个样本。
    num_samples = (step + 1) * cfg.training.batch_size
    avg_samples_per_ep = dataset.num_frames / dataset.num_episodes
    num_episodes = num_samples / avg_samples_per_ep
    num_epochs = num_samples / dataset.num_frames
    log_items = [
        f"step:{format_big_number(step)}",
        # 训练期间看到的样本数
        f"smpl:{format_big_number(num_samples)}",
        # 训练期间看到的发作次数
        f"ep:{format_big_number(num_episodes)}",
        # 看到所有唯一样本的次数
        f"epch:{num_epochs:.2f}",
        f"∑rwrd:{avg_sum_reward:.3f}",
        f"success:{pc_success:.1f}%",
        f"eval_s:{eval_s:.3f}",
    ]
    logging.info(" ".join(log_items))

    info["step"] = step
    info["num_samples"] = num_samples
    info["num_episodes"] = num_episodes
    info["num_epochs"] = num_epochs
    info["is_online"] = is_online

    logger.log_dict(info, step, mode="eval")


def train(cfg: DictConfig, out_dir: str | None = None, job_name: str | None = None):
    if out_dir is None:
        raise NotImplementedError()
    if job_name is None:
        raise NotImplementedError()

    init_logging()
    logging.info(pformat(OmegaConf.to_container(cfg)))

    if cfg.training.online_steps > 0 and isinstance(cfg.dataset_repo_id, ListConfig):
        raise NotImplementedError("Online training with LeRobotMultiDataset is not implemented.")

    # 如果我们要恢复运行，我们需要检查日志目录中是否存在检查点，并且我们需要
    # 来检查提供的配置与 checkpoint 的 config 之间是否有任何差异。
    if cfg.resume:
        if not Logger.get_last_checkpoint_dir(out_dir).exists():
            raise RuntimeError(
                "You have set resume=True, but there is no model checkpoint in "
                f"{Logger.get_last_checkpoint_dir(out_dir)}"
            )
        checkpoint_cfg_path = str(Logger.get_last_pretrained_model_dir(out_dir) / "config.yaml")
        logging.info(
            colored(
                "You have set resume=True, indicating that you wish to resume a run",
                color="yellow",
                attrs=["bold"],
            )
        )
        # 从最后一个检查点获取配置文件。
        checkpoint_cfg = init_hydra_config(checkpoint_cfg_path)
        # 检查 checkpoint 配置和提供的配置之间的差异。
        # Hack 提前解决 delta_timestamps 以便正确 diff。
        resolve_delta_timestamps(cfg)
        diff = DeepDiff(OmegaConf.to_container(checkpoint_cfg), OmegaConf.to_container(cfg))
        # 忽略 'resume' 和参数。
        if "values_changed" in diff and "root['resume']" in diff["values_changed"]:
            del diff["values_changed"]["root['resume']"]
        # 记录有关 checkpoint 配置与提供的
        # 配置。
        if len(diff) > 0:
            logging.warning(
                "At least one difference was detected between the checkpoint configuration and "
                f"the provided configuration: \n{pformat(diff)}\nNote that the checkpoint configuration "
                "takes precedence.",
            )
        # 使用 checkpoint 配置而不是提供的配置（但保留 'resume' 参数）。
        cfg = checkpoint_cfg
        cfg.resume = True
    elif Logger.get_last_checkpoint_dir(out_dir).exists():
        raise RuntimeError(
            f"The configured output directory {Logger.get_last_checkpoint_dir(out_dir)} already exists. If "
            "you meant to resume training, please use `resume=true` in your command or yaml configuration."
        )

    if cfg.eval.batch_size > cfg.eval.n_episodes:
        raise ValueError(
            "The eval batch size is greater than the number of eval episodes "
            f"({cfg.eval.batch_size} > {cfg.eval.n_episodes}). As a result, {cfg.eval.batch_size} "
            f"eval environments will be instantiated, but only {cfg.eval.n_episodes} will be used. "
            "This might significantly slow down evaluation. To fix this, you should update your command "
            f"to increase the number of episodes to match the batch size (e.g. `eval.n_episodes={cfg.eval.batch_size}`), "
            f"or lower the batch size (e.g. `eval.batch_size={cfg.eval.n_episodes}`)."
        )

    # 将指标记录到终端和 wandb
    logger = Logger(cfg, out_dir, wandb_job_name=job_name)

    set_global_seed(cfg.seed)

    # 检查设备是否可用
    device = get_safe_torch_device(cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("make_dataset")
    offline_dataset = make_dataset(cfg)
    if isinstance(offline_dataset, MultiLeRobotDataset):
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(offline_dataset.repo_id_to_index , indent=2)}"
        )

    # 创建用于在模拟数据训练期间评估检查点的环境。
    # 对于真实世界的数据，无需创建环境，因为评估是在 train.py 之外完成的。
    # 改用 eval.py，使用 gym_dora 环境和 Dora-RS。
    eval_env = None
    if cfg.training.eval_freq > 0:
        logging.info("make_env")
        eval_env = make_env(cfg)

    logging.info("make_policy")
    policy = make_policy(
        hydra_cfg=cfg,
        dataset_stats=offline_dataset.meta.stats if not cfg.resume else None,
        pretrained_policy_name_or_path=str(logger.last_pretrained_model_dir) if cfg.resume else None,
    )
    assert isinstance(policy, nn.Module)
    # 创建优化器和调度器
    # 将优化器移出策略的临时 hack
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(enabled=cfg.use_amp)

    step = 0  # 策略更新次数（向前 + 向后 + 最佳）

    if cfg.resume:
        step = logger.load_last_training_state(optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    log_output_dir(out_dir)
    logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.training.offline_steps=} ({format_big_number(cfg.training.offline_steps)})")
    logging.info(f"{cfg.training.online_steps=}")
    logging.info(f"{offline_dataset.num_frames=} ({format_big_number(offline_dataset.num_frames)})")
    logging.info(f"{offline_dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # 注意：此帮助程序将用于离线和在线训练循环。
    def evaluate_and_checkpoint_if_needed(step, is_online):
        _num_digits = max(6, len(str(cfg.training.offline_steps + cfg.training.online_steps)))
        step_identifier = f"{step:0{_num_digits}d}"

        if cfg.training.eval_freq > 0 and step % cfg.training.eval_freq == 0:
            logging.info(f"Eval policy at step {step}")
            with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.use_amp else nullcontext():
                assert eval_env is not None
                eval_info = eval_policy(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    videos_dir=Path(out_dir) / "eval" / f"videos_step_{step_identifier}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )
            log_eval_info(logger, eval_info["aggregated"], step, cfg, offline_dataset, is_online=is_online)
            if cfg.wandb.enable:
                logger.log_video(eval_info["video_paths"][0], step, mode="eval")
            logging.info("Resume training")

        if cfg.training.save_checkpoint and (
            step % cfg.training.save_freq == 0
            or step == cfg.training.offline_steps + cfg.training.online_steps
        ):
            logging.info(f"Checkpoint policy after step {step}")
            # 注意：使用 step 作为标识符进行保存，并将其格式设置为至少 6 位数字，但如果
            # 需要（选择 6 作为最小值以保持一致性，而不会矫枉过正）。
            logger.save_checkpoint(
                step,
                policy,
                optimizer,
                lr_scheduler,
                identifier=step_identifier,
            )
            logging.info("Resume training")

    # 创建用于离线训练的 DataLoader
    if cfg.training.get("drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            offline_dataset.episode_data_index,
            drop_n_last_frames=cfg.training.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None
    dataloader = torch.utils.data.DataLoader(
        offline_dataset,
        num_workers=cfg.training.num_workers,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)

    policy.train()
    offline_step = 0
    for _ in range(step, cfg.training.offline_steps):
        if offline_step == 0:
            logging.info("Start offline training on a fixed dataset")

        start_time = time.perf_counter()
        batch = next(dl_iter)
        dataloading_s = time.perf_counter() - start_time

        for key in batch:
            batch[key] = batch[key].to(device, non_blocking=True)

        train_info = update_policy(
            policy,
            batch,
            optimizer,
            cfg.training.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.use_amp,
        )

        train_info["dataloading_s"] = dataloading_s

        if step % cfg.training.log_freq == 0:
            log_train_info(logger, train_info, step, cfg, offline_dataset, is_online=False)

        # 注意：evaluate_and_checkpoint_if_needed发生在“step”训练更新完成后，
        # 所以我们传入步骤 + 1。
        evaluate_and_checkpoint_if_needed(step + 1, is_online=False)

        step += 1
        offline_step += 1  # 型号： SIM113

    if cfg.training.online_steps == 0:
        if eval_env:
            eval_env.close()
        logging.info("End of training")
        return

    # 在线培训。

    # 从策略推出创建专用于在线剧集集合的环境。
    online_env = make_env(cfg, n_envs=cfg.training.online_rollout_batch_size)
    resolve_delta_timestamps(cfg)
    online_buffer_path = logger.log_dir / "online_buffer"
    if cfg.resume and not online_buffer_path.exists():
        # 如果我们要恢复运行，则默认使用保存的在线数据形状和缓冲区容量
        # 缓冲区。
        logging.warning(
            "When online training is resumed, we load the latest online buffer from the prior run, "
            "and this might not coincide with the state of the buffer as it was at the moment the checkpoint "
            "was made. This is because the online buffer is updated on disk during training, independently "
            "of our explicit checkpointing mechanisms."
        )
    online_dataset = OnlineBuffer(
        online_buffer_path,
        data_spec={
            **{k: {"shape": v, "dtype": np.dtype("float32")} for k, v in policy.config.input_shapes.items()},
            **{k: {"shape": v, "dtype": np.dtype("float32")} for k, v in policy.config.output_shapes.items()},
            "next.reward": {"shape": (), "dtype": np.dtype("float32")},
            "next.done": {"shape": (), "dtype": np.dtype("?")},
            "next.success": {"shape": (), "dtype": np.dtype("?")},
        },
        buffer_capacity=cfg.training.online_buffer_capacity,
        fps=online_env.unwrapped.metadata["render_fps"],
        delta_timestamps=cfg.training.delta_timestamps,
    )

    # 如果我们异步执行在线部署，请深复制用于在线部署的策略（此
    # 可以与训练更新并行进行在线部署）。
    online_rollout_policy = deepcopy(policy) if cfg.training.do_online_rollout_async else policy

    # 创建用于在线培训的数据加载器。
    concat_dataset = torch.utils.data.ConcatDataset([offline_dataset, online_dataset])
    sampler_weights = compute_sampler_weights(
        offline_dataset,
        offline_drop_n_last_frames=cfg.training.get("drop_n_last_frames", 0),
        online_dataset=online_dataset,
        # +1，因为联机转出会为“最终观察”返回一个额外的帧。注意：我们没有
        # 这是离线数据集中的最后一个观察结果，但我们可能会在将来添加它们。
        online_drop_n_last_frames=cfg.training.get("drop_n_last_frames", 0) + 1,
        online_sampling_ratio=cfg.training.online_sampling_ratio,
    )
    sampler = torch.utils.data.WeightedRandomSampler(
        sampler_weights,
        num_samples=len(concat_dataset),
        replacement=True,
    )
    dataloader = torch.utils.data.DataLoader(
        concat_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )
    dl_iter = cycle(dataloader)

    # 用于异步在线推出的锁和线程池执行程序。禁用异步模式时，
    # 这些仍在使用，但实际上没有任何作用。
    lock = Lock()
    # 注意：1 个 worker，因为我们一次只想运行一组在线部署。批
    # 转出的并行化在 Job 中处理。
    executor = ThreadPoolExecutor(max_workers=1)

    online_step = 0
    online_rollout_s = 0  # 进行在线部署所需的时间
    update_online_buffer_s = 0  # 使用 Online Rollout 数据更新 Online 缓冲区所花费的时间
    # 等待联机缓冲区完成更新所花费的时间。这在使用 async
    # 在线推出选项。
    await_update_online_buffer_s = 0
    rollout_start_seed = cfg.training.online_env_seed

    while True:
        if online_step == cfg.training.online_steps:
            break

        if online_step == 0:
            logging.info("Start online training by interacting with environment")

        def sample_trajectory_and_update_buffer():
            nonlocal rollout_start_seed
            with lock:
                online_rollout_policy.load_state_dict(policy.state_dict())
            online_rollout_policy.eval()
            start_rollout_time = time.perf_counter()
            with torch.no_grad():
                eval_info = eval_policy(
                    online_env,
                    online_rollout_policy,
                    n_episodes=cfg.training.online_rollout_n_episodes,
                    max_episodes_rendered=min(10, cfg.training.online_rollout_n_episodes),
                    videos_dir=logger.log_dir / "online_rollout_videos",
                    return_episode_data=True,
                    start_seed=(
                        rollout_start_seed := (rollout_start_seed + cfg.training.batch_size) % 1000000
                    ),
                )
            online_rollout_s = time.perf_counter() - start_rollout_time

            with lock:
                start_update_buffer_time = time.perf_counter()
                online_dataset.add_data(eval_info["episodes"])

                # 更新采样期间使用的串联数据集长度。
                concat_dataset.cumulative_sizes = concat_dataset.cumsum(concat_dataset.datasets)

                # 更新采样权重。
                sampler.weights = compute_sampler_weights(
                    offline_dataset,
                    offline_drop_n_last_frames=cfg.training.get("drop_n_last_frames", 0),
                    online_dataset=online_dataset,
                    # +1，因为联机转出会为“最终观察”返回一个额外的帧。注意：我们没有
                    # 这是离线数据集中的最后一个观察结果，但我们可能会在将来添加它们。
                    online_drop_n_last_frames=cfg.training.get("drop_n_last_frames", 0) + 1,
                    online_sampling_ratio=cfg.training.online_sampling_ratio,
                )
                sampler.num_frames = len(concat_dataset)

                update_online_buffer_s = time.perf_counter() - start_update_buffer_time

            return online_rollout_s, update_online_buffer_s

        future = executor.submit(sample_trajectory_and_update_buffer)
        # 如果我们没有进行异步推出，或者我们的缓冲区中还没有获得足够的示例，请等待
        # 直到转出和缓冲区更新完成，然后再继续执行策略更新步骤。
        if (
            not cfg.training.do_online_rollout_async
            or len(online_dataset) <= cfg.training.online_buffer_seed_size
        ):
            online_rollout_s, update_online_buffer_s = future.result()

        if len(online_dataset) <= cfg.training.online_buffer_seed_size:
            logging.info(
                f"Seeding online buffer: {len(online_dataset)}/{cfg.training.online_buffer_seed_size}"
            )
            continue

        policy.train()
        for _ in range(cfg.training.online_steps_between_rollouts):
            with lock:
                start_time = time.perf_counter()
                batch = next(dl_iter)
                dataloading_s = time.perf_counter() - start_time

            for key in batch:
                batch[key] = batch[key].to(cfg.device, non_blocking=True)

            train_info = update_policy(
                policy,
                batch,
                optimizer,
                cfg.training.grad_clip_norm,
                grad_scaler=grad_scaler,
                lr_scheduler=lr_scheduler,
                use_amp=cfg.use_amp,
                lock=lock,
            )

            train_info["dataloading_s"] = dataloading_s
            train_info["online_rollout_s"] = online_rollout_s
            train_info["update_online_buffer_s"] = update_online_buffer_s
            train_info["await_update_online_buffer_s"] = await_update_online_buffer_s
            with lock:
                train_info["online_buffer_size"] = len(online_dataset)

            if step % cfg.training.log_freq == 0:
                log_train_info(logger, train_info, step, cfg, online_dataset, is_online=True)

            # 注意：evaluate_and_checkpoint_if_needed发生在“step”训练更新完成后，
            # 所以我们传入步骤 + 1。
            evaluate_and_checkpoint_if_needed(step + 1, is_online=True)

            step += 1
            online_step += 1

        # 如果我们正在进行异步部署，我们现在应该等到完成异步部署后再继续
        # 以执行下一批转出。
        if future.running():
            start = time.perf_counter()
            online_rollout_s, update_online_buffer_s = future.result()
            await_update_online_buffer_s = time.perf_counter() - start

        if online_step >= cfg.training.online_steps:
            break

    if eval_env:
        eval_env.close()
    logging.info("End of training")


@hydra.main(version_base="1.2", config_name="default", config_path="../configs")
def train_cli(cfg: dict):
    train(
        cfg,
        out_dir=hydra.core.hydra_config.HydraConfig.get().run.dir,
        job_name=hydra.core.hydra_config.HydraConfig.get().job.name,
    )


def train_notebook(out_dir=None, job_name=None, config_name="default", config_path="../configs"):
    from hydra import compose, initialize

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path=config_path)
    cfg = compose(config_name=config_name)
    train(cfg, out_dir=out_dir, job_name=job_name)


if __name__ == "__main__":
    train_cli()
