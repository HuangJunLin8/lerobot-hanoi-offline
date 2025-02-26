import os
import shutil
import json
from collections import defaultdict
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa

def get_total_frames(jsonl_path):
    """
    计算 episodes.jsonl 文件中所有行的 length 值之和。
    """
    total_length = 0

    if not os.path.exists(jsonl_path):
        print(f"文件不存在: {jsonl_path}")
        return total_length

    with open(jsonl_path, "r") as file:
        for line in file:
            episode = json.loads(line)
            total_length += episode["length"]

    return total_length


def modify_episode_index(file_path, episode_index):
    """
    加载 Parquet 文件，修改 episode_index 列，并保存。

    参数:
        file_path (Path): Parquet 文件的路径。
        episode_index (int): 新的 episode_index 值。
    """
    try:
        # 读取 Parquet 文件
        table = pq.read_table(file_path)
        df = table.to_pandas()

        # 检查是否存在 episode_index 列
        if "episode_index" not in df.columns:
            raise ValueError("DataFrame 中没有 'episode_index' 列。")

        # 重新设定 episode_index 列
        df["episode_index"] = episode_index

        # 保存修改后的 DataFrame 为 Parquet 文件
        table = pa.Table.from_pandas(df)
        pq.write_table(table, file_path)

    except Exception as e:
        print(f"操作失败: {e}")


def merge_datasets(dataset_path1, dataset_path2, merged_dir):
    """
    合并两个数据集，将其整合到一个目标目录中，
    并正确更新 meta/episodes.jsonl 和 meta/info.json 文件。

    参数:
        dataset_path1 (str): 第一个数据集的路径。
        dataset_path2 (str): 第二个数据集的路径。
        merged_dir (str): 合并后数据集的目标路径。
    """
    dataset_path1 = Path(dataset_path1)
    dataset_path2 = Path(dataset_path2)

    # 创建合并后的目录结构
    merged_dir = Path(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)

    # 用于跟踪当前递增编号
    file_counters = defaultdict(int)

    # 收集所有 episodes.jsonl 的内容
    combined_episodes = []

    for dataset_dir in [dataset_path1, dataset_path2]:
        # 读取 episodes.jsonl
        meta_path = dataset_dir / 'meta' / 'episodes.jsonl'
        episode_mapping = {}
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                for line in f:
                    episode_data = json.loads(line.strip())
                    old_index = episode_data['episode_index']
                    file_counters['episode'] += 1
                    new_index = file_counters['episode'] - 1
                    episode_data['episode_index'] = new_index

                    # 记录旧索引与新索引的映射
                    episode_mapping[old_index] = new_index
                    combined_episodes.append(episode_data)

        # 遍历文件并重命名
        for root, _, files in os.walk(dataset_dir):
            relative_path = Path(root).relative_to(dataset_dir)
            for file in files:
                if file == 'episodes.jsonl':
                    continue  # 排除原来的 episodes.jsonl 文件

                src_file = Path(root) / file
                dest_dir = merged_dir / relative_path
                dest_dir.mkdir(parents=True, exist_ok=True)

                # 获取文件的扩展名和前缀
                if "episode" in file:
                    file_stem, file_suffix = file.split(".")[0], f".{file.split('.')[-1]}"
                    old_index = int(file_stem.split("_")[1])  # 原始文件索引
                    new_index = episode_mapping.get(old_index, old_index)
                    new_file_name = f"episode_{str(new_index).zfill(6)}{file_suffix}"

                    dest_file = dest_dir / new_file_name
                    shutil.copy(src_file, dest_file)

                    # 修改 Parquet 文件中的 episode_index
                    if file_suffix == ".parquet":
                        modify_episode_index(dest_file, new_index)
                else:
                    # 非 episode 文件直接拷贝
                    dest_file = dest_dir / file
                    shutil.copy(src_file, dest_file)

    # 生成合并后的 episodes.jsonl
    merged_meta_dir = merged_dir / 'meta'
    merged_meta_dir.mkdir(exist_ok=True)
    merged_episodes_path = merged_meta_dir / 'episodes.jsonl'
    with open(merged_episodes_path, 'w') as f:
        for episode in combined_episodes:
            f.write(json.dumps(episode) + '\n')

    # 更新 info.json
    info_path = merged_meta_dir / 'info.json'
    if info_path.exists():
        with open(info_path, 'r') as f:
            info_data = json.load(f)
    else:
        info_data = {}

    info_data.update({
        "total_episodes": len(combined_episodes),
        "total_frames": get_total_frames(merged_episodes_path),
        "total_videos": sum(len(files) for _, _, files in os.walk(merged_dir / 'videos') if files),
    })

    with open(info_path, 'w') as f:
        json.dump(info_data, f, indent=4)

    print("两个数据集合并完成！")


def delete_episode(base_path, episode_index_to_delete):
    """
    删除指定序号的 episode 数据，
    并更新 meta/episodes.jsonl 和 meta/info.json 文件
    """
    base_path = Path(base_path)
    data_path = base_path / 'data'
    videos_path = base_path / 'videos'
    meta_path = base_path / 'meta'
    episodes_file = meta_path / 'episodes.jsonl'
    info_file = meta_path / 'info.json'

    # 删除对应的 episode parquet 文件并重命名后续文件
    for root, _, files in os.walk(data_path):
        files.sort()  # 确保文件按顺序处理
        for file in files:
            if file.startswith("episode_"):
                episode_num = int(file.split("_")[1].split(".")[0])
                src_file = Path(root) / file
                if episode_num == episode_index_to_delete:
                    os.remove(src_file)
                    # print(f"Deleted file: {src_file}")
                elif episode_num > episode_index_to_delete:
                    new_num = episode_num - 1
                    new_file_name = f"episode_{str(new_num).zfill(6)}.parquet"
                    dest_file = Path(root) / new_file_name
                    os.rename(src_file, dest_file)
                    # print(f"Renamed file {src_file} to {dest_file}")
                    
                    # 修改 Parquet 文件中的 episode_index
                    modify_episode_index(dest_file, new_num)

    # 删除对应的视频文件并重命名后续文件
    for root, _, files in os.walk(videos_path):
        files.sort()  # 确保文件按顺序处理
        for file in files:
            if file.startswith("episode_"):
                episode_num = int(file.split("_")[1].split(".")[0])
                src_file = Path(root) / file
                if episode_num == episode_index_to_delete:
                    os.remove(src_file)
                    # print(f"Deleted file: {src_file}")
                elif episode_num > episode_index_to_delete:
                    new_num = episode_num - 1
                    new_file_name = f"episode_{str(new_num).zfill(6)}{Path(file).suffix}"
                    dest_file = Path(root) / new_file_name
                    os.rename(src_file, dest_file)
                    # print(f"Renamed file {src_file} to {dest_file}")

    # 更新 episodes.jsonl
    if episodes_file.exists():
        with open(episodes_file, 'r') as f:
            episodes = [json.loads(line) for line in f]

        # 过滤掉要删除的 episode，并更新索引
        updated_episodes = []
        for episode in episodes:
            if episode['episode_index'] != episode_index_to_delete:
                if episode['episode_index'] > episode_index_to_delete:
                    episode['episode_index'] -= 1
                updated_episodes.append(episode)

        with open(episodes_file, 'w') as f:
            for episode in updated_episodes:
                f.write(json.dumps(episode) + '\n')

    # 更新 info.json
    if info_file.exists():
        with open(info_file, 'r') as f:
            info_data = json.load(f)

        info_data["total_episodes"] = len(updated_episodes)
        info_data["total_frames"] = get_total_frames(episodes_file)
        info_data["total_videos"] = sum(1 for _, _, files in os.walk(videos_path) for file in files)

        with open(info_file, 'w') as f:
            json.dump(info_data, f, indent=4)

    print("删除完成！")

def merge_same_action(base_path, num_items=None):
    """
    合并相同动作形式的数据集，
    将目录按末尾标识进行分组，并对文件进行递增编号命名，
    同时更新 meta/episodes.jsonl 和 meta/info.json 文件。
    
    参数:
        base_path (str): 数据集的基本路径。
        num_items (int, optional): 指定合并每个数据集的前多少项，默认为 None 表示合并所有项。
    """
    base_path = Path(base_path)
    all_dirs = [d for d in base_path.iterdir() if d.is_dir()]

    # 按末尾标识分组
    grouped_dirs = defaultdict(list)
    for d in all_dirs:
        suffix = d.name.split("_")[-1]
        grouped_dirs[suffix].append(d)

    # 创建合并后的目录结构
    for suffix, dirs in grouped_dirs.items():
        merged_dir = base_path / f"{suffix}"
        merged_dir.mkdir(exist_ok=False)

        # 用于跟踪当前递增编号
        file_counters = defaultdict(int)

        # 收集所有 episodes.jsonl 的内容
        combined_episodes = []

        for dataset_dir in dirs:
            # 读取 episodes.jsonl
            meta_path = dataset_dir / 'meta' / 'episodes.jsonl'
            episode_mapping = {}
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    episodes = [json.loads(line.strip()) for line in f]
                    
                    # 如果指定了 num_items，裁剪列表
                    if num_items is not None:
                        episodes = episodes[:num_items]

                    for episode_data in episodes:
                        old_index = episode_data['episode_index']
                        file_counters['episode'] += 1
                        new_index = file_counters['episode'] - 1
                        episode_data['episode_index'] = new_index

                        # 记录旧索引与新索引的映射
                        episode_mapping[old_index] = new_index
                        combined_episodes.append(episode_data)

            # 遍历文件并重命名
            for root, _, files in os.walk(dataset_dir):
                relative_path = Path(root).relative_to(dataset_dir)
                for file in files:
                    if file == 'episodes.jsonl':
                        continue  # 排除原来的 episodes.jsonl 文件

                    src_file = Path(root) / file
                    dest_dir = merged_dir / relative_path
                    dest_dir.mkdir(parents=True, exist_ok=True)

                    # 获取文件的扩展名和前缀
                    if "episode" in file:
                        file_stem, file_suffix = file.split(".")[0], f".{file.split('.')[-1]}"
                        old_index = int(file_stem.split("_")[1])  # 原始文件索引
                        if old_index in episode_mapping:
                            new_index = episode_mapping[old_index]
                            new_file_name = f"episode_{str(new_index).zfill(6)}{file_suffix}"

                            dest_file = dest_dir / new_file_name
                            shutil.copy(src_file, dest_file)

                            # 修改 Parquet 文件中的 episode_index
                            if file_suffix == ".parquet":
                                modify_episode_index(dest_file, new_index)
                    else:
                        # 非 episode 文件直接拷贝
                        dest_file = dest_dir / file
                        shutil.copy(src_file, dest_file)

        # 生成合并后的 episodes.jsonl
        merged_meta_dir = merged_dir / 'meta'
        merged_meta_dir.mkdir(exist_ok=True)
        merged_episodes_path = merged_meta_dir / 'episodes.jsonl'
        with open(merged_episodes_path, 'w') as f:
            for episode in combined_episodes:
                f.write(json.dumps(episode) + '\n')

        # 更新 info.json
        info_path = merged_meta_dir / 'info.json'
        if info_path.exists():
            with open(info_path, 'r') as f:
                info_data = json.load(f)
        else:
            info_data = {}

        info_data.update({
            "total_episodes": len(combined_episodes),
            "total_frames": get_total_frames(merged_episodes_path),
            "total_videos": sum(len(files) for _, _, files in os.walk(merged_dir / 'videos') if files),
        })

        with open(info_path, 'w') as f:
            json.dump(info_data, f, indent=4)

    print("数据集合并完成！")


if __name__ == "__main__":
    # 删除第 38 个episode
    # delete_episode(base_path, 38)

    # 合并两个数据集
    # dataset1 = "/home/rical/.cache/huggingface/lerobot/test/A12-B-C34_mvA2B"
    # dataset2 = "/home/rical/.cache/huggingface/lerobot/test/A1234-B-C_mvA2B"
    # output = "/home/rical/.cache/huggingface/lerobot/test/mvA2B"
    # merge_datasets(dataset1, dataset2, output)

    # 合并目录 test 下，动作后缀相同的数据集（仅合并前20个episode）
    base_path = "/home/rical/.cache/huggingface/lerobot/test"
    merge_same_action(base_path, num_items=20)



