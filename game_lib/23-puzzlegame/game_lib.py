from typing import Dict, List
import random
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import argparse

def parse_init():
    """
    定义并解析eval代码的命令行参数，配置日志记录，并检查输入的数据文件目录和输出的目录是否存在。
    """
    parser = argparse.ArgumentParser(description="Data creation utility")

    # 添加命令行参数
    parser.add_argument('-p', '--port', type=int, default=8775, help='服务部署端口')
    # 添加命令行参数
    parser.add_argument('-H', '--host', type=str, default="0.0.0.0", help='服务部署地址')
    # 解析命令行参数
    args = parser.parse_args()
    return args
# 游戏提示模板，包含拼图规则及目标信息
game_prompt = """
You are a good game player, I'll give you a game board and rules.
Your task is:
- First, give your answer according to the game board and rules.
- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: 15 3 4 2'

The goal of this game is to rearrange tiles into an n×n grid by moving a specified target tile to a target position. For example, if the input target is "4 (2,1)", you must move tile "4" to row 2, column 1 according to specific rules.

Rules:  
- Grid positions are indexed starting from 1, with the top-left coordinate as (1,1).  
- The puzzle consists of an n×n grid: n×n-1 numbered tiles and one empty space (represented by 0).  
- A tile can only be moved into the empty space if it is directly adjacent to it (left, right, above, or below).  
- Tiles cannot move diagonally. For example, in the following state, you cannot move tile "7" into the empty space:  
  5 6 8  
  7 2 3  
  1 0 4  

- Only one tile can be moved at each step.  
- The puzzle is completed when the target tile reaches the target position.  
- You must output your solution as a sequence of numbers separated by spaces. Each number indicates the tile moved into the empty space, following the above rules. For example:  
'Answer: 15 3 4 2'
{board}
"""

def generate(seed: int) -> Dict:
    """
    根据给定种子生成滑动拼图的初始状态，返回的 item 中包含：
      - board: 拼图的字符串表示，每行数字之间用空格分隔，行与行之间用换行符分隔
      - target_num: 目标拼图块数字（需要移动到指定位置）
      - target_grid: 目标位置，以 [行, 列] 表示（1-indexed）
      - answer: 最终答案（验证时更新）
      - score: 得分，正确为 1，错误为 0
      - is_end: 游戏是否结束
      - response: 额外响应信息列表
      - prompt: 游戏提示（可选）
      - action: 用户输入的移动序列（字符串，空格分隔的数字）
      - epoch: 游戏回合数
    """
    random.seed(seed)
    # 随机生成拼图大小 n x n，n 在3到5之间
    n = random.randint(3, 5)
    total_tiles = n * n - 1  # 拼图块数
    arr = list(range(1, total_tiles + 1))
    parity = False
    # 打乱前 total_tiles-2 个元素，并记录置换奇偶性
    for i in range(total_tiles - 2):
        t = random.randint(i, total_tiles - 1)
        if i != t:
            parity = not parity
        arr[i], arr[t] = arr[t], arr[i]
    # 保证可解性，若奇偶性为 True，则交换最后两个元素
    if parity:
        arr[total_tiles - 2], arr[total_tiles - 1] = arr[total_tiles - 1], arr[total_tiles - 2]
    # 添加空白格（用 0 表示）
    arr.append(0)
    # 对空白位置进行随机调整
    blank_index = total_tiles  # 当前空白格位置下标
    d, r = random.randint(0, n - 1), random.randint(0, n - 1)
    for _ in range(d):
        arr[blank_index], arr[blank_index - n] = arr[blank_index - n], arr[blank_index]
        blank_index = blank_index - n
    for _ in range(r):
        arr[blank_index], arr[blank_index - 1] = arr[blank_index - 1], arr[blank_index]
        blank_index = blank_index - 1

    # 生成 board 字符串，每行数字之间用空格分隔
    board_str = "\n".join(" ".join(map(str, arr[i:i+n])) for i in range(0, len(arr), n))
    # 随机生成目标数字与目标位置（行、列均为1-indexed）
    target_num = random.randint(1, 3)
    target_grid = [random.randint(1, n), random.randint(1, n)]
    
    item = {
        "board": board_str,
        "target_num": target_num,
        "target_grid": target_grid,
        "score": 0,
        "is_end": False,
        "response": [],
        "prompt": "",
        "action": "",
        "epoch": 1,
    }
    return item

def verify(item: Dict) -> Dict:
    """
    根据用户输入的移动序列验证拼图状态是否满足目标要求。
    从 item 中提取：
      - board: 初始拼图状态（字符串格式）
      - target_num: 目标拼图块数字
      - target_grid: 目标位置，[行, 列]（1-indexed，目标要求拼图块移动到此处）
      - action: 用户输入的移动序列（空格分隔的数字，每个数字代表一个移动）
    如果用户的移动合法且最终将目标拼图块移至目标位置，则得分置为 1，否则为 0。
    """
    try:
        board_str = item["board"]
        target_num = item["target_num"]
        target_grid = item["target_grid"]
        user_moves = item["action"].strip()
        if not user_moves:
            item["score"] = 0
            return item

        # 将 board 字符串转换为二维列表（数字列表）
        board = [list(map(int, row.split())) for row in board_str.splitlines() if row.strip() != ""]
        n = len(board)  # 根据 board 行数确定 n

        # 目标位置（注意：target_grid 为1-indexed）
        target_row, target_col = target_grid

        # 查找空白（数字0）位置
        found_empty = False
        for r in range(n):
            for c in range(n):
                if board[r][c] == 0:
                    empty_row, empty_col = r, c
                    found_empty = True
                    break
            if found_empty:
                break
        else:
            item["score"] = 0
            return item

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        moves = user_moves.split()
        # 模拟每一步移动
        for move in moves:
            move_num = int(move)
            target_pos = None
            # 查找待移动拼图块的位置
            for r in range(n):
                for c in range(n):
                    if board[r][c] == move_num:
                        target_pos = (r, c)
                        break
                if target_pos is not None:
                    break
            # 如果找不到指定的拼图块，则返回失败
            if target_pos is None:
                item["score"] = 0
                return item
            
            legal_move = False
            # 判断该拼图块是否与空白相邻
            for dr, dc in directions:
                new_r, new_c = target_pos[0] + dr, target_pos[1] + dc
                if 0 <= new_r < n and 0 <= new_c < n and new_r == empty_row and new_c == empty_col:
                    # 合法移动，交换拼图块与空白位置
                    board[empty_row][empty_col], board[target_pos[0]][target_pos[1]] = board[target_pos[0]][target_pos[1]], board[empty_row][empty_col]
                    empty_row, empty_col = target_pos
                    legal_move = True
                    break
            if not legal_move:
                item["score"] = 0
                return item
        
        # 最终检查目标拼图块是否在指定位置（内部下标需减1）
        for r in range(n):
            for c in range(n):
                if board[r][c] == target_num:
                    if (r, c) == (target_row - 1, target_col - 1):
                        item["score"] = 1
                    else:
                        item["score"] = 0
                    return item
        item["score"] = 0
        return item
    except Exception as e:
        # 出现异常时返回得分 0
        print(f"Verification error: {e}")
        item["score"] = 0
        return item

def print_board(item: Dict) -> str:
    """
    根据 item 中的 board 字符串构造显示的棋盘信息，
    并附带目标拼图块和目标位置的提示信息。
    """
    board_str = item.get("board", "")
    target_info = f"Target: move {item.get('target_num')} to {item.get('target_grid')}"
    return game_prompt.format(board=board_str + "\n" + target_info)

# FastAPI 配置及接口定义
app = FastAPI()

class GenerateRequest(BaseModel):
    seed: int

class BoardRequest(BaseModel):
    board: str

class GameState(BaseModel):
    board: str
    target_num: int
    target_grid: List[int]
    score: int
    is_end: bool
    response: List[str]
    prompt: str
    action: str
    epoch: int

@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    game_state = generate(request.seed)
    return game_state

@app.post("/print_board", response_model=BoardRequest)
def api_print_board(state: GameState):
    board_output = print_board(state.dict())
    return {"board": board_output}

@app.post("/verify", response_model=GameState)
def api_verify(state: GameState):
    updated_state = verify(state.dict())
    return updated_state

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)
