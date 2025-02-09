

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

# 测试
initial_state = {
    "A": [1, 2, 3],
    "B": [],
    "C": []
}

"""

# 汉诺塔算法求解
moves, states = solve_hanoi(initial_state)


# # 输出动作列表
# for move in moves:
#     print("Moves:", move)

# # 输出状态列表
# for state in states:
#     print("State:", state)
    
    
# 生成动作模型路径
for i in range(len(moves)):
    path = generate_task_path(states[i], moves[i])
    print(path)

"""