from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import random
import uvicorn
from collections import deque
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

# 全局字典，用于保存各局游戏的完整状态（包括实际棋盘）
GAME_STORE = {}
game_prompt = '''
Minesweeper Game Rules:
1. The board is a 9x9 grid with 10 hidden mines and the coordinate of the top-leftmost grid is (0, 0).
2. Input Format:
   - Uncover a cell: 'uncover (row, col)' e.g., 'uncover (3,4)'
   - Flag a mine: 'flag (row, col)' e.g., 'flag (0,0)'
   - Unflag a cell: 'unflag (row, col)' e.g., 'unflag (0,0)'
3. Win Condition: Correctly flag all mines or uncover all safe cells.
4. The meanings of the blocks are as follows:
    - ?: Unknown block
    - Number: The total number of mines in the eight adjacent cells
    - F: Flagged block
5. The game will end at the 100th epoch or you uncover a mine.
6. The final score is calculated as follows: the mines you flag correctly / total mines.
{board}
Please output your action in the following format: 'Answer: uncover (3,4)'
'''

# --- 核心业务逻辑（Minesweeper版） ---

def generate(seed: int):
    random.seed(seed)
    n = random.randint(10,20)
    mines = random.randint(10,2*n)
    rows = n
    cols = n
    positions = [(i, j) for i in range(rows) for j in range(cols)]
    mine_pos = random.sample(positions, mines)

    actual = [[0 for _ in range(cols)] for _ in range(rows)]
    for i, j in mine_pos:
        actual[i][j] = -1

    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
    for i in range(rows):
        for j in range(cols):
            if actual[i][j] == -1:
                continue
            actual[i][j] = sum(1 for dx, dy in directions if 0 <= i+dx < rows and 0 <= j+dy < cols and actual[i+dx][j+dy] == -1)

    mask = [['?' for _ in range(cols)] for _ in range(rows)]

    state = {
        "actual": actual,
        "mask": mask,
        "score": 0.0,
        "is_end": False,
        "mines": mines,
        "rows": rows,
        "cols": cols,
        'response': [],
        'prompt': "",
        'unflags': 0,
        "epoch": 1,
    }
    uid = str(uuid.uuid4())
    state["uid"] = uid
    GAME_STORE[uid] = state

    item = state.copy()
    del item["actual"]
    item["action"] = ""
    item['response'] = []
    return item


def print_board(item: dict) -> str:
    board_lines = []
    flags = sum(row.count('F') for row in item['mask'])
    unflags = item.get('unflags', 0)
    board_lines.append(f"Score: {item.get('score', 0.0)}, Flags: {flags}/{item.get('mines', 10)}, Unflags: {unflags}")
    board_lines.append("Current Board:")
    for row in item["mask"]:
        board_lines.append(" ".join(row))
    return game_prompt.format(board="\n".join(board_lines))


def reveal_empty(actual: list, mask: list, start_r: int, start_c: int):
    rows, cols = len(actual), len(actual[0])
    visited = set()
    queue = deque([(start_r, start_c)])
    if actual[start_r][start_c] != 0:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = start_r + dr, start_c + dc
                if 0 <= nr < rows and 0 <= nc < cols and actual[nr][nc] == 0:
                    queue.append((nr, nc))

    while queue:
        r, c = queue.popleft()
        if (r, c) in visited:
            continue
        visited.add((r, c))
        if mask[r][c] != '?':
            continue
        mask[r][c] = str(actual[r][c]) if actual[r][c] > 0 else '0'
        if actual[r][c] == 0:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                        queue.append((nr, nc))


def check_victory(state: dict) -> bool:
    mask, actual = state['mask'], state['actual']
    total_flags = sum(row.count('F') for row in mask)
    correct_flags = sum(1 for i in range(state['rows']) for j in range(state['cols']) if mask[i][j] == 'F' and actual[i][j] == -1)
    return total_flags == state['mines'] and correct_flags == state['mines']


def verify(item: dict) -> dict:
    uid = item.get('uid')
    if not uid or uid not in GAME_STORE:
        raise ValueError('Invalid game id')
    state = GAME_STORE[uid]

    action = item.get('action', '').strip().lower()
    if state['is_end']:
        return item

    try:
        cmd, pos = action.split(' ', 1)
        if pos.startswith('(') and pos.endswith(')'):
            pos = pos[1:-1]
        row, col = map(int, pos.split(','))
    except:
        return item

    if not (0 <= row < state['rows'] and 0 <= col < state['cols']):
        return item

    mask = state['mask']
    actual = state['actual']
    current_flags = sum(r.count('F') for r in mask)

    if cmd == 'uncover':
        if mask[row][col] != '?':
            return item
        if actual[row][col] == -1:
            state['is_end'] = True
            for i in range(state['rows']):
                for j in range(state['cols']):
                    if actual[i][j] == -1:
                        mask[i][j] = 'X'
            # state['score'] = 0.0
        else:
            reveal_empty(actual, mask, row, col)
            safe_cells = state['rows'] * state['cols'] - state['mines']
            revealed = sum(1 for i in range(state['rows']) for j in range(state['cols']) if mask[i][j] not in ['?', 'F'])
            if revealed == safe_cells:
                state['is_end'] = True
                state['score'] = 1.0

    elif cmd == 'flag':
        if mask[row][col] == '?' and current_flags < state['mines']:
            mask[row][col] = 'F'
            if actual[row][col] == -1:
                state['score'] += 1.0 / state['mines']
        else:
            return item

    elif cmd == 'unflag':
        if mask[row][col] == 'F':
            mask[row][col] = '?'
            state['unflags'] += 1
            if actual[row][col] == -1:
                state['score'] -= 1.0 / state['mines']
        else:
            return item

    if check_victory(state):
        state['is_end'] = True
        state['score'] = 1.0

    state['epoch'] += 1
    GAME_STORE[uid] = state

    item.update({
        'mask': mask,
        'score': state['score'],
        'is_end': state['is_end'],
        'epoch': state['epoch'],
        'unflags': state['unflags'],
        'prompt': ''
    })
    item.setdefault('response', []).append(action)
    return item

# --- 定义请求和响应数据模型 ---

class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    uid: str = None
    rows: int
    cols: int
    mines: int
    score: float
    is_end: bool
    epoch: int
    mask: list
    prompt: str
    action: str
    response: list
    unflags: int

# --- API 接口 ---

@app.post('/print_board', response_model=BoardRequest)
def api_print_board(request: GameState):
    state = request.dict()
    board_output = print_board(state)
    return {'board': board_output}

@app.post('/generate', response_model=GameState)
def api_generate(request: GenerateRequest):
    return generate(request.seed)

@app.post('/verify', response_model=GameState)
def api_verify(request: GameState):
    state = request.dict()
    uid = state.get('uid')
    if not uid or uid not in GAME_STORE:
        raise HTTPException(status_code=400, detail='Invalid or expired game id')
    state['actual'] = GAME_STORE[uid]['actual']
    updated = verify(state)
    updated.pop('actual', None)
    return updated


if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)