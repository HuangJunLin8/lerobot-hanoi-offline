import torch
import logging
from pathlib import Path
from typing import List
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_hydra_config, get_safe_torch_device, init_logging, log_say, set_global_seed
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.control_utils import (
    control_loop,
    has_method,
    init_keyboard_listener,
    log_control_info,
    record_episode,
    reset_environment,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
    stop_recording,
    warmup_record,
)

def init_policy(pretrained_policy_paths, policy_overrides):
    """加载多个预训练策略模型，并返回它们的列表"""
    policies = []

    for pretrained_policy_path in pretrained_policy_paths:
        # 加载每个模型的配置
        hydra_cfg = init_hydra_config(pretrained_policy_path + "/config.yaml", policy_overrides)
        
        policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=pretrained_policy_path)
        policies.append(policy)
        
        policy_fps = hydra_cfg.env.fps
        use_amp = hydra_cfg.use_amp

        # 获取设备
        device = get_safe_torch_device(hydra_cfg.device, log=True)

        # 设置模型
        policy.eval()  # 设置为评估模式
        policy.to(device)
    
    print(f"Using device: {device}")
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_global_seed(hydra_cfg.seed)

    return policies, policy_fps, device, use_amp

def execute_model_actions(
        pretrained_model_paths: Path, 
        robot_cfg_path: Path,
        repo_id: str,
        root: Path = None,
        video: bool = True,
        num_image_writer_processes: int = 0,
        num_image_writer_threads_per_camera: int = 4,
        policy_overrides: List[str] | None = None, 
        warmup_time_s: int | float = 8, 
        num_episodes: int = 1,
        episode_time_s: int = 30,
        display_cameras: bool = True,
        ):
    """
    执行多个预训练模型的动作，每个模型顺序执行一次动作。

    :param pretrained_model_paths: 预训练模型的路径列表
    :param robot_cfg_path: 机器人的配置路径
    :param repo_id: 数据集的ID
    :param root: 数据集的根目录
    :param video: 是否使用视频
    :param num_image_writer_processes: 用于写入图像的进程数量
    :param num_image_writer_threads_per_camera: 每个相机用于写入图像的线程数量
    :param policy_overrides: 策略的覆盖参数
    :param warmup_time_s: 预热时间（秒）
    :param num_episodes: 录制的 episode 数
    :param episode_time_s: episode 的时间（秒）
    :param display_cameras: 是否显示相机图像
    """
    robot_cfg = init_hydra_config(robot_cfg_path, policy_overrides)
    robot = make_robot(robot_cfg)

    if not robot.is_connected:
        robot.connect()

    policies, policy_fps, device, use_amp = init_policy(pretrained_model_paths, policy_overrides)

    listener, events = init_keyboard_listener()

    # 建立空的数据集（每个episode存所有7步动作？）
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=policy_fps,
        root=root,
        robot=robot,
        use_videos=video,
        image_writer_processes=num_image_writer_processes,
        image_writer_threads=num_image_writer_threads_per_camera * len(robot.cameras),
    )

    log_say("Warmup record", play_sounds=True)

    # 热身（可摇操摆正机械臂的位置）
    warmup_record(robot, events, 
                  enable_teleoperation=True, 
                  warmup_time_s=warmup_time_s, 
                  display_cameras=display_cameras, 
                  fps=policy_fps)
    
    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()
    
    recorded_episodes = 0
    while True:
        if recorded_episodes >= num_episodes:
            break

        for i in range(7):
            log_say(f"Running step {i}", play_sounds=True)

            record_episode(
                dataset=dataset,
                robot=robot,
                events=events,
                episode_time_s=episode_time_s,
                display_cameras=display_cameras,
                policy=policies[i],
                device=device,
                use_amp=use_amp,
                fps=policy_fps,
            )

        dataset.save_episode("Move 3 disks from A to C")
        recorded_episodes += 1

    log_say("Stop recording", play_sounds=True, blocking=True)
    stop_recording(robot, listener, display_cameras=display_cameras)
    
    if robot.is_connected:
        # 手动断开连接以避免在进程终止时由于相机线程未正确退出
        # 而导致“核心转储” ("Core dump")
        robot.disconnect()


def main():
    # --------------------------初始化---------------------------
    pretrained_model_paths = [
        "outputs/train/Task01_MovA2C_3/checkpoints/last/pretrained_model",
        "outputs/train/Task02_MovA2B_2/checkpoints/last/pretrained_model",
        "outputs/train/Task03_MovC2B_3/checkpoints/last/pretrained_model",
        "outputs/train/Task04_MovA2C_1/checkpoints/last/pretrained_model",
        "outputs/train/Task05_MovB2A_3/checkpoints/last/pretrained_model",
        "outputs/train/Task06_MovB2C_2/checkpoints/last/pretrained_model",
        "outputs/train/Task07_MovA2C_3/checkpoints/last/pretrained_model"
    ]

    policy_overrides = [] 

    robot_cfg_path = "lerobot/configs/robot/so100.yaml"

    init_logging()

    # 执行每个模型的动作
    execute_model_actions( repo_id="ricaal/autoHanoi",
                           pretrained_model_paths=pretrained_model_paths,
                           robot_cfg_path=robot_cfg_path,
                           policy_overrides=policy_overrides, 
                           episode_time_s=25,
                           display_cameras=True)       

if __name__ == "__main__":
    main()



"""
开始测试一次性走完所有动作
python3 test/load_and_execute_models.py



可视化结果
python lerobot/scripts/visualize_dataset_html.py \
--repo-id ricaal/autoHanoi \
--root ~/.cache/huggingface/lerobot/ricaal/autoHanoi \
--local-files-only 1 

"""
