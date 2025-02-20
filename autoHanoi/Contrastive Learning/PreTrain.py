import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# --------------------------------数据集生成---------------------------------
def encode_state(state):
    """将汉诺塔状态转换为一个数值向量"""
    max_disks = 4  # 你的问题中最多 4 个盘子
    vector = np.zeros((3, max_disks))  # 三根柱子，每根柱子最多 4 个盘子

    for i, peg in enumerate(["A", "B", "C"]):
        for j, disk in enumerate(state[peg]):
            vector[i, j] = disk  # 记录盘子编号

    return vector.flatten()  # 转换为 1D 向量


def generate_training_data(moves, initial_state):
    """根据汉诺塔求解过程，生成状态对 (s1, s2)"""
    import copy
    state_sequence = [copy.deepcopy(initial_state)]
    current_state = copy.deepcopy(initial_state)  # 深拷贝，避免修改原始数据

    for move in moves:
        from_peg, to_peg = move["from"], move["to"]
        
        # 确保 from_peg 不为空，否则跳过这个 move
        if not current_state[from_peg]:  
            print(f"错误: 尝试从空柱子 {from_peg} 取盘子")
            continue  # 跳过非法操作
        
        # 从柱子顶部取出盘子（列表末尾元素）
        disk = current_state[from_peg].pop()  
        current_state[to_peg].append(disk)  # 放入目标柱子的顶部（列表末尾）

        # 存储新状态
        state_sequence.append(copy.deepcopy(current_state))

    # 生成状态对
    positive_pairs = []
    negative_pairs = []

    for i in range(len(state_sequence) - 1):
        s1 = encode_state(state_sequence[i])
        s2 = encode_state(state_sequence[i + 1])
        positive_pairs.append((s1, s2, 1))  # 相差一步，标签为 1

        # 选择一个较远的状态作为负样本
        if i + 3 < len(state_sequence):
            s_far = encode_state(state_sequence[i + 3])
            negative_pairs.append((s1, s_far, 0))  # 相差较远，标签为 0

    return positive_pairs + negative_pairs


# -------------------------------汉诺塔算法------------------------------------
def hanoi(n, source, target, auxiliary, initial_state, moves, states):
    if n == 0:
        return

    # 1. 移动 n-1 个盘子到辅助杆
    hanoi(n-1, source, auxiliary, target, initial_state, moves, states)

    # 2. 移动第 n 个盘子到目标杆
    move = {'from': source, 'to': target}
    moves.append(move)
    initial_state[target].append(initial_state[source].pop())
    states.append({k: v.copy() for k, v in initial_state.items()})

    # 3. 移动 n-1 个盘子到目标杆
    hanoi(n-1, auxiliary, target, source, initial_state, moves, states)

def solve_hanoi(initial_state):
    moves = []
    states = []
    states.append({k: v.copy() for k, v in initial_state.items()})  # 初始状态
    n = len(initial_state['A'])  # 假设竿A上有 n 个盘子
    hanoi(n, 'A', 'C', 'B', initial_state, moves, states)
    return moves, states


# -------------------------------模型------------------------------------
class StateEncoder(nn.Module):
    """状态编码器"""
    def __init__(self, input_dim, feature_dim):
        super(StateEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class ContrastiveLoss(nn.Module):
    """对比损失函数"""
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, z1, z2, label):
        distance = torch.norm(z1 - z2, p=2, dim=1)
        loss = (1 - label) * torch.pow(distance, 2) + label * torch.pow(torch.clamp(self.margin - distance, min=0), 2)
        return loss.mean()


def train(initial_state):
    # 初始化模型
    feature_dim = 64
    encoder = StateEncoder(input_dim=12, feature_dim=feature_dim)  # 状态向量长度=12
    loss_fn = ContrastiveLoss()
    optimizer = optim.Adam(encoder.parameters(), lr=0.001)

    # 生成训练数据
    moves, states = solve_hanoi(initial_state)
    training_data = generate_training_data(moves, states[0])

    # 训练
    for epoch in range(100):
        total_loss = 0
        for s1, s2, label in training_data:
            s1 = torch.tensor(s1, dtype=torch.float32).unsqueeze(0)
            s2 = torch.tensor(s2, dtype=torch.float32).unsqueeze(0)
            label = torch.tensor([label], dtype=torch.float32)

            z1, z2 = encoder(s1), encoder(s2)
            loss = loss_fn(z1, z2, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(training_data)}")




if __name__ == "__main__":

    initial_state = {
        "A": [1, 2, 3, 4],
        "B": [],
        "C": []
    }

    train(initial_state)




