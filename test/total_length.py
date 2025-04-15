import json
import os
from argparse import ArgumentParser

def main():
    # 解析命令行参数
    parser = ArgumentParser(description="统计 episodes.jsonl 文件中所有行的 length 值之和")
    parser.add_argument("--dataset_repo_id", type=str, required=True, help="Hugging Face 数据集仓库 ID，格式为 ${HF_USER}/${TASK_NAME}")
    args = parser.parse_args()

    # 构建文件路径
    hf_user, task_name = args.dataset_repo_id.split("/")
    file_path = os.path.expanduser(f"~/.cache/huggingface/lerobot/{hf_user}/{task_name}/meta/episodes.jsonl")

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return

    # 初始化总和为 0
    total_length = 0

    # 打开文件并逐行读取
    with open(file_path, "r") as file:
        for line in file:
            # 将每一行的 JSON 字符串转换为 Python 字典
            episode = json.loads(line)
            # 累加 length 值
            total_length += episode["length"]

    # 输出总和
    print(f"total_frames: {total_length}")

if __name__ == "__main__":
    main()

