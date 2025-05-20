from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
from copy import deepcopy
import uvicorn
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
app = FastAPI()

game_prompt = """
You are a skilled game player. I'll provide you with a game board and rules.

Your tasks are as follows:

- First, provide your solution based on the given game board and rules.
- Second, present your solution strictly in the required format. The final line of your response must follow this format exactly:  
  'Answer: $YOUR_ANSWER'  
(without quotes), where 'YOUR_ANSWER' is your final response. For example, 'Answer: 4 90'.

Next, I will show you a Tetris game board, along with relevant information such as the epoch, score, the current state of the board, and the currently falling block. The game will forcibly end after the 100th round or if a block exceeds the top boundary of the board when placed. If an entire row becomes fully occupied, it will automatically be cleared, increasing the score by 1.The block will continue to fall until it is obstructed by another block or reaches the bottom of the board.

- On the board:
  - '*' represents a square occupied by a block.
  - '.' represents an empty square.

- The current block information includes its shape and rotation state:
  - '*' represents squares occupied by the current block.
  - '0' represents empty spaces.

You must provide your action in the format:  
'Answer: [drop_coordinate] [rotation_angle]' ,e.g. 'Answer: 4 90'.
where:
- '[drop_coordinate]' refers to the leftmost square of the block relative to the board, ranging from 1 (leftmost) to 10 (rightmost).
- '[rotation_angle]' represents the rotation applied to the current falling block, provided explicitly in the game information, where '*' denotes block squares and '0' denotes empty spaces.
For example, if the board is:
..........
..........
..........
..........
..........
..........
..........
......**..
**..****..
**...**...
and the Current Block is:  
0°:
0**
**0
90°:
*0
**
0*
180°:
0**
**0
270°:
*0
**
0*
If the provided action is 'Answer: 3 90', the leftmost square of the block rotated by 90° will land in the third column of the board, and the board will become:
..........
..........
..........
..........
..........
..........
..........
..*...**..
********..
**.*.**...

The current game board is:
{board}

"""

# ----------------- 俄罗斯方块相关辅助函数 -----------------

# 定义所有可能的俄罗斯方块形状
shapes = [
    [[1, 1, 1, 1]],         # I
    [[1, 1], [1, 1]],       # O
    [[0, 1, 1], [1, 1, 0]],  # S
    [[1, 1, 0], [0, 1, 1]],  # Z
    [[1, 0, 0], [1, 1, 1]],  # L
    [[0, 0, 1], [1, 1, 1]],  # J
    [[0, 1, 0], [1, 1, 1]]   # T
]

# 将矩阵转换为字符串（0对应zero_flag，非0对应one_flag）
def get_board_str(matrix, zero_flag, one_flag):
    m, n = len(matrix), len(matrix[0])
    res = [[zero_flag] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if matrix[i][j]:
                res[i][j] = one_flag
    return "\n".join("".join(row) for row in res)

# 显示当前砖块所有旋转角度的字符串表示
def get_shape_input(current_shape):
    res = "0°:\n" + get_board_str(current_shape, zero_flag="0", one_flag="*")
    shape = current_shape
    for i in range(3):
        shape = [list(row) for row in zip(*shape[::-1])]
        res += f"\n{(i+1)*90}°:\n" + get_board_str(shape, zero_flag="0", one_flag="*")
    return res

# 将棋盘中的数字（0和1）转换为对应字符，0显示为'.'，1显示为'*'
def get_board_input(board):
    return get_board_str(board, zero_flag=".", one_flag="*")

# ----------------- 接口函数定义 -----------------

# print_board接口：根据当前item信息生成显示棋盘和砖块状态的字符串
def print_board(item: dict) -> str:
    board_str = get_board_input(item["board"])
    block_str = get_shape_input(item["last_block"])
    board_info = (
        f"Epoch: {item['epoch']}\n"
        f"Score: {item['score']}\n"
        f"Board:\n{board_str}\n"
        f"Current Block (rotations):\n{block_str}\n"
    )
    return game_prompt.format(board=board_info)

# generate接口：根据给定seed（以及可选的宽度和高度）生成初始游戏状态
def generate(seed: int, width: int = 10, height: int = 10) -> dict:
    random.seed(seed)
    board = [[0] * width for _ in range(height)]
    last_block = random.choice(shapes)
    item = {
        'epoch': 1,
        'board': board,
        'last_block': last_block,
        'score': 0,
        'is_end': False,
        'prompt': "",
        'action': "",
        'width': width,
        'height': height
    }
    item['prompt'] = print_board(item)
    return item

# 尝试将砖块放置到棋盘上，location表示砖块最左侧有效部分的落点（从1开始计数）
def place_block(block, board, location):
    block_height, block_width = len(block), len(block[0])
    left_offset = None
    for j in range(block_width):
        for i in range(block_height):
            if block[i][j] == 1:
                left_offset = j
                break
        if left_offset is not None:
            break
    if left_offset is None:
        left_offset = 0

    x = location - 1 - left_offset
    height = len(board)
    width = len(board[0])
    if x < 0:
        x = 0
    if x > width - block_width:
        x = width - block_width

    final_y = None
    for y in range(height - block_height + 1):
        flag = True
        for i in range(block_height):
            for j in range(block_width):
                if block[i][j] == 1 and board[y+i][x+j] == 1:
                    flag = False
                    break
            if not flag:
                break
        if flag:
            final_y = y
        else:
            break
    if final_y is None:
        return False
    for i in range(block_height):
        for j in range(block_width):
            if block[i][j] == 1:
                board[final_y+i][x+j] = 1
    return True

# 消除已填满的行，并返回新棋盘和消除的行数
def clear_lines(board):
    height, width = len(board), len(board[0])
    delete_lines = [i for i in range(height) if sum(board[i]) == width]
    n = len(delete_lines)
    if n == 0:
        return board, 0
    new_board = [[0] * width for _ in range(n)]
    for i, line in enumerate(board):
        if i not in delete_lines:
            new_board.append(line)
    return new_board, n

# 根据用户给定的落点和旋转角度更新棋盘状态
def update(board, last_block, location, direction, score):
    next_block = random.choice(shapes)
    # 每旋转90°一次
    for _ in range(direction // 90):
        last_block = [list(row) for row in zip(*last_block[::-1])]
    can_place = place_block(last_block, board, location)
    if not can_place:
        return score, board, next_block, True
    board, lines_cleared = clear_lines(board)
    return score + lines_cleared, board, next_block, False

# verify接口：根据item中的action更新游戏状态（例如：用户操作 "3 90°"）
def verify(item: dict) -> dict:
    item['epoch'] += 1
    if item.get("is_end", False):
        return item
    try:
        parts = item["action"].split()
        if len(parts) != 2:
            return item
        location_str, direction_str = parts
        location = int(location_str)
        direction = int(direction_str.replace("°", ""))
    except Exception:
        return item
    score, board, next_block, game_over = update(item["board"], item["last_block"], location, direction, item["score"])
    item["score"] = score
    item["board"] = board
    item["last_block"] = next_block
    item["is_end"] = game_over
    return item

# ----------------- FastAPI 请求/响应数据模型 -----------------

class GenerateRequest(BaseModel):
    seed: int


class BoardRequest(BaseModel):
    board: str

class GameState(BaseModel):
    epoch: int
    board: list
    last_block: list
    score: int
    is_end: bool
    prompt: str
    action: str
    width: int
    height: int

# ----------------- FastAPI 接口路由 -----------------

@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    game_state = generate(request.seed)
    return game_state

@app.post("/print_board", response_model=BoardRequest)
def api_print_board(request: GameState):
    board_output = print_board(request.dict())
    return {"board": board_output}

@app.post("/verify", response_model=GameState)
def api_verify(request: GameState):
    state = request.dict()
    updated_state = verify(state)
    return updated_state


if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)