from spider_solitaire import SpiderSolitaire
import copy
import random
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
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
game_prompt = '''
You are a Spider Solitaire expert. After I show you the current board, choose the best next action and reply with:

  1. (Optional) Your reasoning.
  2. A final line in exactly this format:  
     Answer: $YOUR_ANSWER  

Where '$YOUR_ANSWER' is one of:  
- A move '(FromColumn,StartIndex,ToColumn)', e.g. '(A,4,B)'  
- 'hit' to deal new cards  

### Rules
- **Goal**: Build 8 complete K→A sequences.  
- **Move**: You may relocate any descending, same‐suit run onto a column whose top card is exactly one rank higher (or onto an empty column).  
- **Deal**: If no legal moves remain and the deck has ≥10 cards and every column is non‑empty, use 'hit' (deals one card to each of the 10 columns; max ⌊deck_size/10⌋ hits).  
- **Score**: Start at 0; +1 for each K→A sequence removed; no penalties for moves or hits.  
- **Turn Limit**: The game also ends after 100 epochs (moves or hits).

### Visibility & Completion
1. Only the bottom card of each column is face‑up; hidden cards are shown as 'XX'.  
2. After you move a face‑up run away, the new bottom card flips face‑up automatically.  
3. Completing a full K→A same‑suit sequence in any column removes those 13 cards immediately and awards +1 point.  

### Columns & Format
- Columns are labeled A–J (indices 0–9).  
- Always output exactly:  'Answer: $YOUR_ANSWER',e.g.'Answer: (A,4,B)' means move cards from column A starting at index 4 to column B.


Current Game Board:
{board}
'''

from inspect import getmodule

def print_board(state: dict) -> str:
    """
    Generate the prompt with the current game board and remaining hit count
    """
    board = state["board"]
    column_labels = "ABCDEFGHIJ"
    # Format the board for display
    board_str = "  " + " ".join(column_labels[:len(board)]) + "\n"
    board_str += "  " + "-" * (2 * len(board) - 1) + "\n"
    max_length = max(len(col) for col in board)
    for i in range(max_length):
        row = []
        for j, column in enumerate(board):
            if i < len(column):
                card = column[i]
                if card[0] == 'unknown':
                    row.append("XX")
                else:
                    row.append(f"{card[1]}{card[0][0]}")
            else:
                row.append("  ")
        board_str += f"{i} {' '.join(row)}\n"
    # 计算并显示剩余hit次数（每次发牌消耗10张）
    try:
        remaining_hits = len(print_board.__globals__['verify'].game.deck) // 10
    except Exception:
        remaining_hits = 0
    board_str += f"Epoch: {state['epoch']}/100\n"
    board_str += f"The remaining chances of 'hit': {remaining_hits}\n"
    return game_prompt.format(board=board_str)


def generate(seed: int) -> dict:
    """
    Generate initial game state using provided seed
    """
    # 随机种子（如用户未指定可直接使用随机）
    seed = random.randint(1, 100000)
    game = SpiderSolitaire()
    board = game.setup_game(seed)
    # 同步初始化 verify.game 以便打印剩余hit次数
    generate.__globals__['verify'].seed = seed
    generate.__globals__['verify'].game = game

    return {
        "board": board,
        "score": 0.0,
        "epoch": 1,
        "is_end": False,
        "action": "",
        "response": [],
        "prompt": "",
    }

def verify(state: dict) -> dict:
    """
    Verify and apply an action to the current game state
    
    Args:
        state: Current game state
        
    Returns:
        dict: Updated game state
    """
    seed = getattr(verify, 'seed', None)
    if seed is None:
        seed = random.randint(1, 100000)
        verify.seed = seed
        verify.game = SpiderSolitaire()
        verify.game.setup_game(seed)
    
    game = verify.game
    action = state["action"]
    state["epoch"] += 1
    if action.strip().lower() == "hit":
        success = game.deal_cards()
        if not success:
            return state
    else:
        try:
            parts = action.strip("()").split(",")
            if len(parts) != 3:
                return state
            
            from_col = ord(parts[0].strip().upper()) - ord('A')
            start_idx = int(parts[1].strip())
            to_col = ord(parts[2].strip().upper()) - ord('A')
            
            success = game.move_cards(from_col, start_idx, to_col)
            if not success:
                return state
        except Exception as e:
            return state
    
    state["board"] = game.get_visible_board()
    state["score"] = game.score
    
    state["is_end"] = game.completed_sets == 8
    return state

class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    board: list
    score: int
    epoch: int
    is_end: bool
    action: str
    response: list
    prompt: str

@app.post("/print_board", response_model=BoardRequest)
def api_print_board(request: GameState):
    state = request.model_dump()
    board_output = print_board(state)
    return {"board": board_output}

@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    return generate(request.seed)

@app.post("/verify", response_model=GameState)
def api_verify(request: GameState):
    return verify(request.model_dump())

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)