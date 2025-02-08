import torch
import logging
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

        # 获取设备
        device = get_safe_torch_device(hydra_cfg.device, log=True)

        # 设置模型
        policy.eval()  # 设置为评估模式
        policy.to(device)
    
    print(f"Using device: {device}")
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_global_seed(hydra_cfg.seed)

    return policies, policy_fps, device

def execute_model_actions(
        pretrained_model_paths, 
        robot_cfg_path,
        policy_overrides, 
        warmup_time_s=8, 
        num_episodes=10,
        episode_time_s=30,
        display_cameras=False
        ):
    """
    执行多个预训练模型的动作，每个模型顺序执行一次动作。

    :param pretrained_model_paths: 预训练模型的路径列表
    :param robot_cfg_path: 机器人的配置路径
    :param policy_overrides: 策略的覆盖参数列表
    :param warmup_time_s: 热身时间（秒）
    :param num_episodes: 记录的剧集数量
    :param episode_time_s: 剧集时间（秒）
    :param display_cameras: 是否显示相机
    """
    robot_cfg = init_hydra_config(robot_cfg_path, policy_overrides)
    robot = make_robot(robot_cfg)

    if not robot.is_connected:
        robot.connect()

    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    policies, policy_fps, device = init_policy(pretrained_model_paths, policy_overrides)

    listener, events = init_keyboard_listener()

    log_say("Warmup record", play_sounds=True)

    warmup_record(robot, events, 
                  enable_teleoperation=True, 
                  warmup_time_s=warmup_time_s, 
                  display_cameras=display_cameras, 
                  fps=policy_fps)
    
    # recorded_episodes = 0
    # while True:
    #     if recorded_episodes >= num_episodes:
    #         break
        
    #     log_say(f"Recording episode {dataset.num_episodes}", play_sounds=True)
    #     record_episode(
    #         dataset=dataset,
    #         robot=robot,
    #         events=events,
    #         episode_time_s=episode_time_s,
    #         display_cameras=display_cameras,
    #         policy=policy,
    #         device=device,
    #         use_amp=use_amp,
    #         fps=fps,
    #     )

    # stop_recording(robot, listener, display_cameras=display_cameras)
    
    if robot.is_connected:
        # 手动断开连接以避免在进程终止时由于相机线程未正确退出
        # 而导致“核心转储” ("Core dump")
        robot.disconnect()


def main():
    # --------------------------初始化---------------------------
    display_cameras=True # 显示相机

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
    execute_model_actions(pretrained_model_paths=pretrained_model_paths,
                           robot_cfg_path=robot_cfg_path,
                           policy_overrides=policy_overrides, 
                           num_episodes=10,
                           display_cameras=display_cameras)
    

    log_say("Stop recording", play_sounds=True, blocking=True)
    
    

        

if __name__ == "__main__":
    main()
