import random
from collections import deque, defaultdict
import ast
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
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
# 定义方向常量
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
dir_vec = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}

# 游戏提示模板，包含题面说明及待填充的棋盘展示
game_prompt = """
You are a good game player, I'll give you a game board and rules.
Your task is:
- First, give your answer according to the game board and rules.
- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: [[0,1,1,3...],[1,3,2,3...],...]'.

Given three types of pipes with the following initial connections:  
- L connects the top and right sides  
- | connects the top and bottom sides  
- ┏ connects the top, left, and right sides  

You are provided with an n x n grid, where each cell contains one type of pipe. The starting point is to the left of position (0,0), and the goal is to reach the right side of position (n-1,n-1). Players need to rotate the pipes in the grid to ensure a valid connection from the start to the end.

Your task is to output an n x n list in one line, where each element indicates the number of 90° clockwise rotations (0, 1, 2, or 3) applied to the pipe at that position.  
For example:  
'Answer: [[0,1,1,3...],[1,3,2,3...],...]'
Board:
{board}
"""

def generator(n: int):
    """
    根据 n 生成水管迷宫棋盘。
    利用深度优先搜索生成连通图，再根据每个格子与邻居的连通情况确定管件类型。
    返回：
      board: n x n 的二维列表，每个元素为管件字符（如 'L','|','┏'）
      solution: 正确的旋转矩阵（此处均为 0，表示无需旋转）
    """
    start = (0, 0)
    end = (n - 1, n - 1)
    # 初始化连通关系和度数统计
    connected = {(i, j): set() for i in range(n) for j in range(n)}
    deg = {(i, j): 0 for i in range(n) for j in range(n)}
    visited = [[False] * n for _ in range(n)]
    # 防止 DFS 过早连通终点：先标记终点为已访问
    visited[end[0]][end[1]] = True

    def dfs(cell, parent=None):
        i, j = cell
        visited[i][j] = True
        dirs = [UP, RIGHT, DOWN, LEFT]
        random.shuffle(dirs)
        for d in dirs:
            ni, nj = i + dir_vec[d][0], j + dir_vec[d][1]
            if 0 <= ni < n and 0 <= nj < n and not visited[ni][nj]:
                connected[cell].add((ni, nj))
                connected[(ni, nj)].add(cell)
                deg[cell] += 1
                deg[(ni, nj)] += 1
                dfs((ni, nj), cell)
        # 对非起点/终点的叶子节点，尝试额外连通一个已访问的邻居（避免孤岛）
        if cell not in (start, end) and parent is not None and deg[cell] == 1:
            dirs2 = [UP, RIGHT, DOWN, LEFT]
            random.shuffle(dirs2)
            for d in dirs2:
                ni, nj = i + dir_vec[d][0], j + dir_vec[d][1]
                if 0 <= ni < n and 0 <= nj < n and visited[ni][nj] and (ni, nj) != parent:
                    if (ni, nj) not in connected[cell] and deg[(ni, nj)] < 3:
                        connected[cell].add((ni, nj))
                        connected[(ni, nj)].add(cell)
                        deg[cell] += 1
                        deg[(ni, nj)] += 1
                        break

    dfs(start)
    # 解除终点占用，尝试将终点与一个邻居连接
    visited[end[0]][end[1]] = False
    for d in [UP, RIGHT, DOWN, LEFT]:
        ni, nj = end[0] + dir_vec[d][0], end[1] + dir_vec[d][1]
        if 0 <= ni < n and 0 <= nj < n and visited[ni][nj] and deg[(ni, nj)] < 3:
            connected[end].add((ni, nj))
            connected[(ni, nj)].add(end)
            deg[end] += 1
            deg[(ni, nj)] += 1
            break
    visited[end[0]][end[1]] = True

    # 确保所有格子均被访问（连接孤立块）
    for i in range(n):
        for j in range(n):
            if not visited[i][j]:
                for d in [UP, RIGHT, DOWN, LEFT]:
                    ni, nj = i + dir_vec[d][0], j + dir_vec[d][1]
                    if 0 <= ni < n and 0 <= nj < n and visited[ni][nj]:
                        connected[(i, j)].add((ni, nj))
                        connected[(ni, nj)].add((i, j))
                        deg[(i, j)] += 1
                        deg[(ni, nj)] += 1
                        break
                visited[i][j] = True

    # 根据连通情况确定每个格子的管件类型
    board = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            cell = (i, j)
            opens = set()
            # 起点与左侧连通
            if cell == start:
                opens.add(LEFT)
            # 终点与右侧连通
            if cell == end:
                opens.add(RIGHT)
            for (ni, nj) in connected[cell]:
                if ni == i - 1 and nj == j:
                    opens.add(UP)
                elif ni == i + 1 and nj == j:
                    opens.add(DOWN)
                elif ni == i and nj == j - 1:
                    opens.add(LEFT)
                elif ni == i and nj == j + 1:
                    opens.add(RIGHT)
            # 根据连通方向确定管件类型
            if len(opens) == 3:
                piece_type = '┏'
            elif len(opens) == 2:
                if opens == {UP, DOWN} or opens == {LEFT, RIGHT}:
                    piece_type = '|'
                else:
                    piece_type = 'L'
            elif len(opens) == 1:
                # 起点或终点仅连通一个内部格子时仍显示为 'L'
                piece_type = 'L'
                if cell == start and opens == {RIGHT}:
                    piece_type = '|'
                if cell == end and opens == {LEFT}:
                    piece_type = '|'
            else:
                piece_type = ' '  # 理论上不应出现
            board[i][j] = piece_type

    # 模拟随机旋转，正确答案即为全 0 表示无需旋转
    solution = [[0 for _ in range(n)] for _ in range(n)]
    return board, solution

def pipes_verify(board, answer):
    """
    验证水管迷宫是否连通：
    根据每种管件在默认朝向下的连通方向，并根据 answer 中的旋转次数调整方向，
    利用广度优先搜索检查是否能从起点 (0,0) 连通到终点 (n-1,n-1)。
    """
    n = len(board)
    default_opens = {
        '|': {UP, DOWN},
        'L': {UP, RIGHT},
        '┏': {UP, RIGHT, LEFT}
    }
    
    def get_opens(i, j, rot):
        piece = board[i][j]
        if piece not in default_opens:
            return set()
        # 每旋转90°顺时针一次，所有方向均 +1 mod 4
        return {(d + rot) % 4 for d in default_opens[piece]}
    
    queue = deque()
    visited = set()
    # 如果起点管件左侧朝外，则从起点出发
    start_opens = get_opens(0, 0, answer[0][0])
    if LEFT in start_opens:
        queue.append((0, 0, LEFT))
        visited.add((0, 0, LEFT))
    
    while queue:
        i, j, src_dir = queue.popleft()
        opens = get_opens(i, j, answer[i][j])
        # 去除来源方向，防止水流回退
        if src_dir in opens:
            opens = opens - {src_dir}
        for d in opens:
            # 若在终点且管件向右开口，则水流可流出终点
            if i == n - 1 and j == n - 1 and d == RIGHT:
                return True
            # 计算下一格坐标及进入方向
            if d == UP and i - 1 >= 0:
                ni, nj, incoming = i - 1, j, DOWN
            elif d == RIGHT and j + 1 < n:
                ni, nj, incoming = i, j + 1, LEFT
            elif d == DOWN and i + 1 < n:
                ni, nj, incoming = i + 1, j, UP
            elif d == LEFT and j - 1 >= 0:
                ni, nj, incoming = i, j - 1, RIGHT
            else:
                continue
            neighbor_opens = get_opens(ni, nj, answer[ni][nj])
            if incoming in neighbor_opens:
                state = (ni, nj, incoming)
                if state not in visited:
                    visited.add(state)
                    queue.append((ni, nj, incoming))
    return False

# ------------------------ 接口函数 ----------------------------

def generate(seed: int) -> dict:
    """
    根据给定随机种子生成水管迷宫谜题。
    返回的 item 字典中包含：
      - board: 包含题面信息和棋盘展示的文本（用于展示给用户）
      - puzzle_grid: 生成的谜题棋盘（二维列表，用于答案验证）
      - grid_size: 棋盘尺寸 n
      - endpoints: 起点和终点坐标（内部使用）
      - answer: 正确的旋转矩阵（字符串形式，正确答案）
      - score: 初始得分 0
      - is_end: 初始未结束状态 False
      - response: 空列表
      - prompt: 游戏提示文本（题目描述）
      - action: 用户提交的答案（初始为空字符串）
      - epoch: 1
    """
    random.seed(seed)
    n = random.randint(4, 6)
    puzzle_grid, solution = generator(n)
    # 将 puzzle_grid 转换为字符串表示
    board_str = "\n".join(" ".join(str(cell) for cell in row) for row in puzzle_grid)
    
    item = {
        'board': game_prompt.format(board=board_str),
        'puzzle_grid': puzzle_grid,
        'grid_size': n,
        'endpoints': {"start": (0, 0), "end": (n - 1, n - 1)},
        'answer': str(solution),
        'score': 0,
        'is_end': False,
        'response': [],
        'prompt': "",
        'action': "",
        'epoch': 1
    }
    return item

def verify(item: dict) -> dict:
    """
    根据 item 中用户提交的答案（action 字段）验证水管迷宫的连通性，
    若答案正确，则将 score 置为 1，否则置为 0，并将 is_end 置为 True。
    """
    board = item.get('puzzle_grid')
    n = item.get('grid_size')
    # 尝试将 action 转换为二维列表
    if isinstance(item['action'], str):
        try:
            user_answer = ast.literal_eval(item['action'])
        except Exception:
            item['score'] = 0
            item['is_end'] = True
            return item
    else:
        user_answer = item['action']
    # 检查答案格式是否符合 n x n
    if not (isinstance(user_answer, list) and len(user_answer) == n and all(isinstance(row, list) and len(row) == n for row in user_answer)):
        item['score'] = 0
        item['is_end'] = True
        return item

    if pipes_verify(board, user_answer):
        item['score'] = 1
    else:
        item['score'] = 0
    item['is_end'] = True
    return item

def print_board(item: dict) -> str:
    """
    输出包装后的题面文本，将生成的棋盘展示插入到游戏提示模板中。
    """
    return game_prompt.format(board="\n".join(" ".join(str(cell) for cell in row) for row in item.get('puzzle_grid', [])))

# ------------------------ FastAPI 配置 ----------------------------

app = FastAPI()

# 定义请求及状态模型
class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    board: str
    answer: str
    score: int
    is_end: bool
    action: str
    response: List[str]
    prompt: str
    epoch: int
    # 以下为内部使用字段，可选
    puzzle_grid: List[List[str]] = None
    grid_size: int = None
    endpoints: dict = None

class BoardRequest(BaseModel):
    board: str

@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    game_state = generate(request.seed)
    return game_state

@app.post("/verify", response_model=GameState)
def api_verify(request: GameState):
    state = request.dict()
    updated_state = verify(state)
    return updated_state

@app.post("/print_board", response_model=BoardRequest)
def api_print_board(request: GameState):
    state = request.dict()
    board_output = print_board(state)
    return {"board": board_output}

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)
# if __name__ == "__main__":
#     # 生成谜题状态（例如使用 seed=0）
#     for i in range(1):
#         item = generate(i)
        
#         # 调用 print_board 接口，展示谜题棋盘及提示
#         board_output = print_board(item)
#         print("【谜题展示】")
#         print(board_output)
#         print("提示映射:", item["clues"])
        
#         # 展示完整解答棋盘（仅用于测试对比，实际答案对玩家是隐藏的）
#         print("\n【完整解答棋盘】")
#         print(board_to_string(item["sol_board"]))
        
#         # 模拟用户提交答案（此处直接使用正确答案作为示例）
#         # 注意：action 需为字符串形式，可使用 repr() 转换
#         print(repr(item["sol_board"]))
#         item["action"] = repr(item["sol_board"])
        
#         # 调用 verify 接口进行验证
#         item = verify(item)
#         print("\n【验证反馈】")
#         print("反馈信息:", item["response"])
#         print("得分:", item["score"])
