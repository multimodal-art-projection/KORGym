from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import random
from typing import Dict
import ast

app = FastAPI()

game_prompt = '''
Minesweeper Game Rules:
1. The board is a 9x9 grid with 10 hidden mines.
2. Input Format:
   - Uncover a cell: `[uncover(row, col)]` e.g., `[uncover(3,4)]`
   - Flag a mine: `[flag(row, col)]` e.g., `[flag(0,0)]`
3. Win Condition: Correctly flag all mines or uncover all safe cells.
4. Current Board State:
{board}
Please output your action in the following format: `Answer: [uncover(3,4)]`
'''

# 全局游戏状态存储
games: Dict[str, dict] = {}

def print_board(state: dict) -> str:
    board = state["current_state"]
    rows = len(board)
    cols = len(board[0]) if rows > 0 else 0
    
    # 构建棋盘字符串
    board_str = "   " + " ".join(f"{i:2}" for i in range(cols)) + "\n"
    for i in range(rows):
        row = [f"{i:2}"] + [cell if cell != '?' else '■' for cell in board[i]]
        board_str += " ".join(row) + "\n"
    
    return game_prompt.format(board=board_str)

def generate(seed: int) -> dict:
    random.seed(seed)
    game_id = str(random.randint(100000, 999999))
    
    # 初始化棋盘
    rows, cols = 9, 9
    mines = 10
    
    positions = [(i, j) for i in range(rows) for j in range(cols)]
    mine_pos = random.sample(positions, mines)
    actual = [[0]*cols for _ in range(rows)]
    for i, j in mine_pos:
        actual[i][j] = -1
    
    # 计算数字提示
    directions = [(-1,-1), (-1,0), (-1,1),
                  (0,-1),         (0,1),
                  (1,-1),  (1,0),  (1,1)]
    
    for i in range(rows):
        for j in range(cols):
            if actual[i][j] == -1:
                continue
            count = 0
            for dx, dy in directions:
                x, y = i+dx, j+dy
                if 0 <= x < rows and 0 <= y < cols:
                    if actual[x][y] == -1:
                        count += 1
            actual[i][j] = count
    
    games[game_id] = {
        "actual": actual,
        "mask": [["?" for _ in range(cols)] for _ in range(rows)],
        "mines": set(mine_pos),
        "score": 0.0,
        "is_end": False,
    }
    item = {
        "current_state": games[game_id]["mask"],
        "score": 0.0,
        "is_end": False,
        "game_id": game_id,
        "epoch": 1,
    }
    return item

def reveal_empty(actual: list, mask: list, row: int, col: int):
    """递归揭开空白区域"""
    rows, cols = len(actual), len(actual[0])
    stack = [(row, col)]
    
    while stack:
        r, c = stack.pop()
        if mask[r][c] != "?":
            continue
        
        val = actual[r][c]
        mask[r][c] = str(val) if val > 0 else " "
        
        if val == 0:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nr, nc = r+dx, c+dy
                    if 0 <= nr < rows and 0 <= nc < cols:
                        stack.append((nr, nc))

def verify(state: dict) -> dict:
    """验证用户操作并更新游戏状态"""
    game_id = state["game_id"]
    action = state["action"]
    
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = games[game_id]
    if game["is_end"]:
        return state
    
    try:
        # 解析操作指令
        action_type, pos = action.strip().split(" ", 1)
        action_type = action_type.lower()
        row, col = ast.literal_eval(pos)
    except:
        raise HTTPException(status_code=400, detail="Invalid action format")
    
    # 验证坐标有效性
    if not (0 <= row < 9 and 0 <= col < 9):
        raise HTTPException(status_code=400, detail="Invalid coordinates")
    
    mask = game["mask"]
    actual = game["actual"]
    
    try:
        if action_type == "uncover":
            if mask[row][col] != "?":
                raise HTTPException(status_code=400, detail="Cell already revealed")
            
            if actual[row][col] == -1:  # 踩雷
                game["is_end"] = True
                # 计算得分（正确标记的地雷数）
                correct_flags = sum(
                    1 for i in range(9) for j in range(9)
                    if mask[i][j] == "F" and (i,j) in game["mines"]
                )
                game["score"] = correct_flags / len(game["mines"])
                # 显示所有地雷
                for i, j in game["mines"]:
                    mask[i][j] = "X"
            else:
                reveal_empty(actual, mask, row, col)
                # 检查胜利条件
                revealed = sum(
                    1 for i in range(9) for j in range(9)
                    if mask[i][j] != "?" and (i,j) not in game["mines"]
                )
                if revealed == 81 - len(game["mines"]):
                    game["is_end"] = True
                    game["score"] = 1.0
        
        elif action_type == "flag":
            if mask[row][col] == "?":
                mask[row][col] = "F"
                # 更新得分
                if (row, col) in game["mines"]:
                    game["score"] += 1/len(game["mines"])
            elif mask[row][col] == "F":
                mask[row][col] = "?"
                if (row, col) in game["mines"]:
                    game["score"] -= 1/len(game["mines"])
            
            # 检查胜利条件（所有地雷正确标记）
            correct_flags = sum(
                1 for i,j in game["mines"] if mask[i][j] == "F"
            )
            if correct_flags == len(game["mines"]):
                game["is_end"] = True
                game["score"] = 1.0
        else:
            raise HTTPException(status_code=400, detail="Invalid action type")
        
        # 更新返回状态
        return {
            "current_state": mask,
            "score": game["score"],
            "is_end": game["is_end"],
            "game_id": game_id,
            "epoch": state["epoch"] + 1,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    current_state: list
    game_id: str
    score: float
    is_end: bool
    action: str
    epoch: int

@app.post("/print_board", response_model=BoardRequest)
def api_print_board(request: GameState):
    state = request.dict()
    return {"board": print_board(state)}

@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    return generate(request.seed)

@app.post("/verify", response_model=GameState)
def api_verify(request: GameState):
    return verify(request.dict())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8775)