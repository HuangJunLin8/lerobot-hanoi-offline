import pyarrow.parquet as pq
import pandas as pd
import pyarrow as pa

def modify_episode_index(file_path, output_path, episode_index=0):
    """
    加载 Parquet 文件，修改 episode_index 列，并保存为新的 Parquet 文件。

    参数:
        file_path (str): 原始 Parquet 文件的路径。
        output_path (str): 修改后保存的 Parquet 文件路径。
        start_index (int): 新的 episode_index 值，默认为 0。
    """
    try:
        # 1. 读取 Parquet 文件
        table = pq.read_table(file_path)
        df = table.to_pandas()

        # 2. 检查是否存在 episode_index 列
        if "episode_index" not in df.columns:
            raise ValueError("DataFrame 中没有 'episode_index' 列。")

        # 3. 重新设定 episode_index 列
        df["episode_index"] = episode_index

        # 4. 保存修改后的 DataFrame 为 Parquet 文件
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_path)

    except Exception as e:
        print(f"操作失败: {e}")


def parquet_info(file_path):
    """
    加载并显示 Parquet 文件的信息。

    参数:
        file_path (str): Parquet 文件的路径。
    """
    try:
        # 1. 读取 Parquet 文件
        print(f"正在加载文件: {file_path}")
        table = pq.read_table(file_path)

        # 2. 显示文件的 schema
        # print("\n文件 schema:")
        # print(table.schema)

        # 3. 转换为 Pandas DataFrame 并显示前几行数据
        df = table.to_pandas()
        print("\n前 5 行数据:")
        print(df.head())

        # 4. 显示文件的元数据
        # metadata = pq.read_metadata(file_path)
        # print("\n文件元数据:")
        # print(metadata)

        # 5. 显示文件的行数和列数
        print(f"\n文件行数: {len(df)}")
        print(f"文件列数: {len(df.columns)}")

        # 6. 显示每列的数据类型
        print("\n每列的数据类型:")
        print(df.dtypes)

        # 7. 显示索引
        print(f" \n episode_index: \n{df['episode_index']}")

    except Exception as e:
        print(f"加载文件时出错: {e}")

# 文件路径
file_path = "/home/rical/.cache/huggingface/lerobot/test/A12-B-C34_mvA2B/data/chunk-000/episode_000038.parquet"  

# 加载并显示文件信息
parquet_info(file_path)

# 修改 episode_index
# modify_episode_index(file_path, file_path, episode_index=38)

# 加载并显示文件信息
# parquet_info(file_path)