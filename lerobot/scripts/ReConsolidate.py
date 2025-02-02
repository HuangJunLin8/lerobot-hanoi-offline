import argparse
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def load_and_process_dataset(dataset_repo_id, delta_timestamps=None, video_backend="ffmpeg"):
    """
    加载数据集，并重新计算统计信息。

    参数：
    - dataset_repo_id (str): 数据集的路径（本地）。
    - delta_timestamps (dict | None): 包含 'action' 关键字的时间戳字典。
    - video_backend (str): 处理视频的后端，默认 "ffmpeg"。

    返回：
    - dataset (LeRobotDataset): 处理后的数据集对象。
    """

    # 加载数据集
    dataset = LeRobotDataset(
        dataset_repo_id,
        delta_timestamps=delta_timestamps,  # ✅ 传入字典
        image_transforms=None,  
        video_backend=video_backend,
        local_files_only=True,
    )

    print(f"✅ 数据集 '{dataset_repo_id}' 加载成功！开始重新计算统计信息...")

    # 重新计算统计信息
    dataset.consolidate(run_compute_stats=True)
    print("📊 统计信息更新完成！")

    return dataset  # 返回数据集对象

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="重新计算数据集的统计信息")

    parser.add_argument("--dataset_repo_id", type=str, required=True, help="数据集的本地路径")
    parser.add_argument("--video_backend", type=str, default="pyav", help="视频处理后端 (默认: pyav)")
    parser.add_argument("--fps", type=int, default=30, help="视频的帧率 (默认: 30)")
    parser.add_argument("--chunk_size", type=int, default=1000, help="chunk_size (默认: 1000)")

    args = parser.parse_args()

    # 计算 delta_timestamps，格式为字典
    # delta_timestamps = {"action": [i / args.fps for i in range(args.chunk_size)]}



    # 运行数据处理函数
    dataset = load_and_process_dataset(
        dataset_repo_id=args.dataset_repo_id,
        video_backend=args.video_backend,
    )

    print(f"✅ 处理完成！数据集 {args.dataset_repo_id} 现在可以使用。")
