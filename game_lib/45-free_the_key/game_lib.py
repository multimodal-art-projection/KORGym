import copy
import random
from collections import deque
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import numpy as np
import ast
import argparse

def parse_init():
    """
    定义并解析命令行参数，用于服务部署地址与端口的配置。
    """
    parser = argparse.ArgumentParser(description="Data creation utility")
    parser.add_argument('-p', '--port', type=int, default=8775, help='服务部署端口')
    parser.add_argument('-H', '--host', type=str, default="0.0.0.0", help='服务部署地址')
    args = parser.parse_args()
    return args
app = FastAPI()
# 回溯尝试放置砖块
game_prompt="""
You are a good game player, I'll give you a game board and rules.\nYour task is:\n- First, give your answer according to the game board and rules.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g."Answer: 1 right".
The game contains horizontal and vertical blocks (represented by letters), a key (1), and an exit (2). Your goal is to move the blocks and the key to help the key reach the exit. Horizontal blocks can only move horizontally, vertical blocks can only move vertically, and the key can only move horizontally. Control the blocks and the key to make moves so that the key successfully reaches the exit on the right side.
In each turn, output the block/key and its moving direction. For example, "Answer: 1 right" means moving the key one step to the right, and "Answer: F up" means moving the block labeled 'F' one step upward.
Remember,the game will end at 100th epoch.
Current epoch:{epoch}
Board:
{board}
"""
def convert_numpy_types(item):
    if isinstance(item, dict):
        return {k: convert_numpy_types(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [convert_numpy_types(i) for i in item]
    elif isinstance(item, tuple):
        return tuple(convert_numpy_types(i) for i in item)
    elif isinstance(item, np.integer):
        return int(item)
    elif isinstance(item, np.floating):
        return float(item)
    elif isinstance(item, np.ndarray):
        return item.tolist()
    else:
        return item
def backtrack_placement(index, board, bricks, used_h_rows, used_v_cols, num_bricks, max_attempts):
    m, n = len(board), len(board[0])
    # 所有砖块放置完毕后，检查钥匙位置是否可用
    if index == num_bricks:
        if board[2][0] == '0' and board[2][1] == '0':
            board[2][0] = '1'
            board[2][1] = '1'
            key_pos = [(2, 0), (2, 1)]
            board[2][n-1] = '2'  # 出口固定在第三行最右侧
            return board, bricks, key_pos
        else:
            return None

    char = chr(65 + index)  # 用字母表示砖块
    # 对于第0号砖块：强制生成一个纵向砖块，确保覆盖第三行
    if index == 0:
        candidates = []
        # 只有当砖块从第0行开始时，长为3才能覆盖到第三行
        for j in range(1, 5):  # 列在1~4之间
            if board[0][j] == '0' and board[1][j] == '0' and board[2][j] == '0':
                candidates.append((0, j))
        random.shuffle(candidates)
        attempts = 0
        for (i, j) in candidates:
            attempts += 1
            if attempts > max_attempts:
                break
            positions = [(0, j), (1, j), (2, j)]
            for (r, c) in positions:
                board[r][c] = char
            bricks[char] = {'type': 'V', 'positions': positions}
            new_used_v_cols = used_v_cols.copy()
            new_used_v_cols.add(j)
            result = backtrack_placement(index + 1, board, bricks, used_h_rows, new_used_v_cols, num_bricks, max_attempts)
            if result is not None:
                return result
            # 回溯：撤销该砖块
            for (r, c) in positions:
                board[r][c] = '0'
            del bricks[char]
        return None
    else:
        candidates = []
        # 横向候选（只能放在除第三行之外的行，且每行仅允许一个横向砖块）
        for i in [0, 1, 3, 4, 5]:
            if i in used_h_rows:
                continue
            for length in [2, 3]:
                
                for j in range(0, n - length + 1):
                    if all(board[i][j+k] == '0' for k in range(length)):
                        candidates.append(('H', i, j, length))
        # 纵向候选（每列只允许一个纵向砖块，并且除第0号砖块外，不允许覆盖第三行）
        for j in range(n):
            if j in used_v_cols:
                continue
            for length in [2, 3]:
                for i in range(0, m - length + 1):
                    if i <= 2 < i + length:  # 排除覆盖第三行的情况
                        continue
                    if all(board[i+k][j] == '0' for k in range(length)):
                        candidates.append(('V', i, j, length))
        random.shuffle(candidates)
        attempts = 0
        for candidate in candidates:
            attempts += 1
            if attempts > max_attempts:
                # 达到最大尝试次数则直接回溯
                break
            orient, i, j, length = candidate
            positions = [(i, j+k) for k in range(length)] if orient == 'H' else [(i+k, j) for k in range(length)]
            for (r, c) in positions:
                board[r][c] = char
            bricks[char] = {'type': orient, 'positions': positions}
            new_used_h_rows = used_h_rows.copy()
            new_used_v_cols = used_v_cols.copy()
            if orient == 'H':
                new_used_h_rows.add(i)
            else:
                new_used_v_cols.add(j)
            result = backtrack_placement(index + 1, board, bricks, new_used_h_rows, new_used_v_cols, num_bricks, max_attempts)
            if result is not None:
                return result
            # 回溯：撤销该砖块
            for (r, c) in positions:
                board[r][c] = '0'
            del bricks[char]
        return None

# 生成地图、砖块和钥匙（出口固定在第三行最右侧）
def generate(seed):
    random.seed(seed)
    m = 6
    n = 6 
    num_bricks = 6
    max_attempts = 50
    board = [['0' for _ in range(n)] for _ in range(m)]
    bricks = {}
    used_h_rows = set()  # 跟踪已放置横向砖块的行
    used_v_cols = set()  # 跟踪已放置纵向砖块的列
    board, bricks, key_pos = backtrack_placement(0, board, bricks, used_h_rows, used_v_cols, num_bricks, max_attempts)
    if board is None:
        # 若回溯失败，则重新生成整个地图
        return generate(seed)
    item = {
        'score': 1,      # 生成成功
        'is_end': False,
        'response': [],
        'prompt': '',
        'action': "",  # 完整的解答棋盘存入 action 字段
        'epoch': 1,
    }
    item['board'] = board
    item['bricks'] = bricks
    item['key_pos'] = key_pos
    return item

# 以下函数保持不变：利用 BFS 检查地图是否可解、移动砖块/钥匙、交互接口等

def bfs_solve(board, bricks, key_pos):
    m, n = len(board), len(board[0])
    queue = deque([(board, bricks, key_pos, 0)])
    visited = set()
    while queue:
        cur_board, cur_bricks, cur_key, steps = queue.popleft()
        state = ''.join(''.join(row) for row in cur_board) + str(cur_key)
        if state in visited:
            continue
        visited.add(state)
        # 修改胜利条件：检查钥匙中最右侧的1是否与出口（'2'）重叠
        row = cur_key[0][0]  # 假设钥匙总在同一行
        key_right = max(cur_key[0][1], cur_key[1][1])
        if cur_board[row][key_right] == '2':
            return steps
        for char in list(cur_bricks.keys()) + ['1']:
            for direction in ['LEFT', 'RIGHT', 'UP', 'DOWN']:
                new_board = [row[:] for row in cur_board]
                new_bricks = copy.deepcopy(cur_bricks)
                new_key = cur_key[:]
                if move_object(new_board, new_bricks, new_key, char, direction):
                    queue.append((new_board, new_bricks, new_key, steps + 1))
    return -1

def move_object(board, bricks, key_pos, char, direction):
    if char.isalpha() and char in bricks:
        return move_brick(board, bricks, char, direction)
    elif char == '1':
        return move_key(board, key_pos, direction)
    return False

def move_brick(board, bricks, char, direction):
    brick = bricks[char]
    type_ = brick['type']
    positions = brick['positions']
    if type_ == 'H' and direction in ['LEFT', 'RIGHT']:
        if direction == 'LEFT':
            min_j = min(c for (r, c) in positions)
            if min_j - 1 < 0 or board[positions[0][0]][min_j - 1] not in ['0', '2']:
                return False
            max_j = max(c for (r, c) in positions)
            board[positions[0][0]][max_j] = '0'
            new_positions = [(r, c - 1) for (r, c) in positions]
            for (r, c) in new_positions:
                board[r][c] = char
            brick['positions'] = new_positions
            return True
        elif direction == 'RIGHT':
            max_j = max(c for (r, c) in positions)
            if max_j + 1 >= len(board[0]) or board[positions[0][0]][max_j + 1] not in ['0', '2']:
                return False
            min_j = min(c for (r, c) in positions)
            board[positions[0][0]][min_j] = '0'
            new_positions = [(r, c + 1) for (r, c) in positions]
            for (r, c) in new_positions:
                board[r][c] = char
            brick['positions'] = new_positions
            return True
    elif type_ == 'V' and direction in ['UP', 'DOWN']:
        if direction == 'UP':
            min_r = min(r for (r, c) in positions)
            if min_r - 1 < 0 or board[min_r - 1][positions[0][1]] not in ['0', '2']:
                return False
            max_r = max(r for (r, c) in positions)
            board[max_r][positions[0][1]] = '0'
            new_positions = [(r - 1, c) for (r, c) in positions]
            for (r, c) in new_positions:
                board[r][c] = char
            brick['positions'] = new_positions
            return True
        elif direction == 'DOWN':
            max_r = max(r for (r, c) in positions)
            if max_r + 1 >= len(board) or board[max_r + 1][positions[0][1]] not in ['0', '2']:
                return False
            min_r = min(r for (r, c) in positions)
            board[min_r][positions[0][1]] = '0'
            new_positions = [(r + 1, c) for (r, c) in positions]
            for (r, c) in new_positions:
                board[r][c] = char
            brick['positions'] = new_positions
            return True
    return False

def move_key(board, key_pos, direction):
    i, j1 = key_pos[0]
    i, j2 = key_pos[1]
    if j1 > j2:
        j1, j2 = j2, j1
    old_val1 = board[i][j1]
    old_val2 = board[i][j2]
    board[i][j1] = '0'
    board[i][j2] = '0'
    success = False
    new_j1, new_j2 = j1, j2
    if direction == 'LEFT':
        new_j1 = j1 - 1
        new_j2 = j2 - 1
        if new_j1 >= 0 and board[i][new_j1] in ['0', '2'] and board[i][new_j2] in ['0', '2']:
            success = True
    elif direction == 'RIGHT':
        new_j1 = j1 + 1
        new_j2 = j2 + 1
        if new_j2 < len(board[0]) and board[i][new_j1] in ['0', '2'] and board[i][new_j2] in ['0', '2']:
            success = True
    if success:
        # 修改：只检查右侧的钥匙格是否与出口重叠
        exit_reached = board[i][new_j2] == '2'
        board[i][new_j1] = '1'
        board[i][new_j2] = '1'
        key_pos[0] = (i, new_j1)
        key_pos[1] = (i, new_j2)
        if exit_reached:
            return "WIN"
    else:
        board[i][j1] = old_val1
        board[i][j2] = old_val2
    return success

def print_board(item):
    output = ''
    for row in item['board']:
        output = output + ' '.join(row) + '\n'
    return game_prompt.format(board=output,epoch=item['epoch'])

# verify接口，用于根据行动集合更新地图状态
def verify(item):
    action = item['action']
    try:
        action = action.strip().upper().split()
        char, direction = action
    except (ValueError, SyntaxError):
        item['epoch']+=1
        return item
    item['epoch']+=1
    initial_board = item['board']
    bricks = item['bricks']
    key_pos = item['key_pos']
    board = [row[:] for row in initial_board]
    bricks_state = copy.deepcopy(bricks)
    key_state = key_pos[:]
    is_end = 0
    
    res = move_object(board, bricks_state, key_state, char, direction)
    # 修改胜利判断：仅当钥匙中最右侧的1与出口(‘2’)重叠时胜利
    row = key_state[0][0]
    key_right = max(key_state[0][1], key_state[1][1])
    if res and (res == "WIN" or board[row][key_right] == '2'):
        is_end = 1
        score = 1
        item['board'] = board
        item['bricks'] = bricks_state
        item['key_pos'] = key_state
        item['score'] = score
        item['is_end'] = is_end
        return item
    score = 0
    item['bricks'] = bricks_state
    item['key_pos'] = key_state
    item['board'] = board
    item['score'] = score
    item['is_end'] = is_end
    return item

def play_game():
    item = generate(6)
    print("初始地图：")
    print(print_board(item))
    while True:
        if item['action'] == 'QUIT' or item['is_end'] == 1:
            break
        item['action'] = input("请输入移动指令（例如 'A RIGHT' 或 '1 RIGHT'）或 'quit' 退出：").strip().upper()
        
        item = verify(item)
        print(print_board(item))
    return item['score']

class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    bricks: dict
    key_pos: list
    board: list
    score: float
    is_end: bool
    action: str
    response: list
    prompt: str
    epoch: int

# --- API 接口 ---

# 生成游戏板内容
@app.post("/print_board", response_model=BoardRequest)
def api_print_board(request: GameState):
    state = request.dict()
    board_output = print_board(state)
    return {"board": board_output}

# 生成初始游戏状态
@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    game_state = generate(request.seed)
    # 转换 NumPy 数据类型
    game_state = convert_numpy_types(game_state)
    return game_state

# 根据动作更新游戏状态
@app.post("/verify", response_model=GameState)
def api_verify(request: GameState):
    state = request.dict()
    # 转换 endpoints 中的值为元组
    state['key_pos'] = [tuple(coord) for coord in state['key_pos']]
    updated_state = verify(state)
    # 转换 NumPy 数据类型后返回
    updated_state = convert_numpy_types(updated_state)
    return updated_state



if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)
    
    
# if __name__ == "__main__":
#     result = play_game()
#     print(f"结果：{result}")