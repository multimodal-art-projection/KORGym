# game_lib/7-black_white_copy/game_lib.py

#Standard libraries
import random
import ast
import argparse

#Commonly used open-source libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

def parse_init():
    """
    Parses command-line arguments for launching the FastAPI server.

    Returns:
        argparse.Namespace: Parsed host and port configuration.
    """
    parser = argparse.ArgumentParser(description="Data creation utility")
    parser.add_argument('-p', '--port', type=int, default=8775, help='服务部署端口')
    parser.add_argument('-H', '--host', type=str, default="0.0.0.0", help='服务部署地址')
    args = parser.parse_args()
    return args

app = FastAPI()
game_prompt="""
You are a good game player, I'll give you a game board and rules.\nYour task is:\n- First, give your answer according to the game board and rules.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: [['row', 3], ['line', 0], ['diagonal_black', 6], ...]'

Given an  n * n  chessboard, each cell can contain either a black (B) or white (W) piece. Initially, all cells contain white pieces. You can perform the following operations:

1. Row operation (row): Turns all pieces in the selected row to white.
2. Column operation ('line'): Turns all pieces in the selected column to black.
3. Diagonal operation ('diagonal_black') (from bottom-left to top-right): Turns all pieces on the selected diagonal to black.
4. Diagonal operation ('diagonal_white') (from top-left to bottom-right): Turns all pieces on the selected diagonal to white.

Given a target pattern and a limited number of operations, your task is to achieve the target pattern starting from an all-white board.  
Output your solution as a list in the format '[[operation_name, position], ...]',e.g.'Answer: [['row', 3], ['line', 0], ['diagonal_black', 6], ...]'
Target Board:
{board}
Limited Number:
{num}
"""


def create_board(n):
    return [['W' for _ in range(n)] for _ in range(n)]

def apply_operation(board, op):
    """
    Applies a specified operation to the board.

    Supported operations:
        - 'row': set entire row to 'W'
        - 'line': set entire column to 'B'
        - 'diagonal_black': set anti-diagonal to 'B'
        - 'diagonal_white': set main-diagonal to 'W'

    Args:
        board (List[List[str]]): The current board state.
        op (Tuple[str, int]): Operation name and index.

    Returns:
        List[List[str]]: Modified board.
    """
    n = len(board)
    op_name, idx = op
    if op_name == "row":
        for j in range(n):
            board[idx][j] = 'W'

    elif op_name == "line":
        for i in range(n):
            board[i][idx] = 'B'

    elif op_name == "diagonal_black":
        for i in range(n):
            for j in range(n):
                if i + j == idx:
                    board[i][j] = 'B'

    elif op_name == "diagonal_white":
        target_diff = idx - (n - 1)
        for i in range(n):
            for j in range(n):
                if i - j == target_diff:
                    board[i][j] = 'W'
    return board

def simulate_ops(ops, n):
    board = create_board(n)
    for op in ops:
        board = apply_operation(board, op)
    return board

def boards_equal(board1, board2):
    return all(''.join(row1) == ''.join(row2) for row1, row2 in zip(board1, board2))

def optimize_ops(ops, n):
    """
    Greedily removes redundant operations from the sequence.

    Args:
        ops (List[Tuple[str, int]]): Original operation sequence.
        n (int): Board size.

    Returns:
        List[Tuple[str, int]]: Optimized operation list.
    """
    target_board = simulate_ops(ops, n)
    i = 0
    while i < len(ops):
        candidate = ops[:i] + ops[i+1:]
        if boards_equal(simulate_ops(candidate, n), target_board):
            ops = candidate 
            i = 0         
        else:
            i += 1
    return ops

def generate(seed):
    """
    Generates a target board configuration and the minimum required number of operations.

    Args:
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[List[str], int, int]: Target board (as list of strings), allowed steps, board size.
    """
    random.seed(seed)
    n = 6  #board_size
    board = create_board(n)
    
    operations = []
    for r in range(n):
        operations.append(("row", r))
    for c in range(n):
        operations.append(("line", c))
    for d in range(2 * n - 1):
        operations.append(("diagonal_black", d))
    for d in range(2 * n - 1):
        operations.append(("diagonal_white", d))
    
    m = random.randint(5, min(10, len(operations)))
    random.shuffle(operations)
    chosen_ops = operations[:m]
    
    optimized_ops = optimize_ops(chosen_ops, n)
    
    final_board = simulate_ops(optimized_ops, n)
    target_map = [''.join(row) for row in final_board]
    num = len(optimized_ops)
    return target_map, num, n

def verify_ops(action, target_map, num):
    """
    Verifies whether the given action sequence matches the target board
    within the allowed number of operations.

    Args:
        action (List[List[str, int]]): Sequence of operations to verify.
        target_map (List[str]): Target board configuration.
        num (int): Maximum allowed operations.

    Returns:
        int: 1 if successful match, else 0.
    """

    if len(action) > num:
        return 0
    
    n = len(target_map)
    board = create_board(n)
    performed_ops = set()
    
    for op in action:
        if not isinstance(op, list) or len(op) != 2:
            return 0
        op_name, op_index = op
        
        if op_name not in ["row", "line", "diagonal_black", "diagonal_white"]:
            return 0
        
        if op_name in ["row", "line"]:
            if not (0 <= op_index < n):
                return 0
        elif op_name in ["diagonal_black", "diagonal_white"]:
            if not (0 <= op_index < 2 * n - 1):
                return 0
        
        if (op_name, op_index) in performed_ops:
            return 0
        performed_ops.add((op_name, op_index))
        
        board = apply_operation(board, (op_name, op_index))
    
    final_map = [''.join(row) for row in board]
    return 1 if final_map == target_map else 0

def board_to_str(target_map):
    return "\n".join(target_map)

def generate_game_state(seed: int) -> dict:
    """
    Generates a full game state from a seed, including the board and prompt.

    Args:
        seed (int): Random seed.

    Returns:
        dict: Game state dictionary with target board, num steps, prompt, etc.
    """
    target_map, num, n = generate(seed)
    prompt = (
        "Target:\n" +
        "\n".join(target_map) +
        f"\nPlease give your actions in {num} moves，e.g. [action_name, num]，"
        "Actions include：row, line, diagonal_black, diagonal_white。"
    )
    return {
        "target_map": target_map,
        "num": num,
        "n": n,
        "score": 0,
        "is_end": False,
        "action": "",
        "response": [],
        "prompt": prompt,
        "epoch": 1
    }

def verify_game_state(state: dict) -> dict:
    """
    Verifies the submitted action sequence and updates the score in the game state.

    Args:
        state (dict): Full game state including action and target map.

    Returns:
        dict: Updated game state with score.
    """
    try:
        action = state.get('action')
        if isinstance(action, str) and action:
            action = ast.literal_eval(action)
        elif action is None:
            state['score'] = 0
            return state
    except Exception as e:
        state['score'] = 0
        return state
    
    state['score'] = verify_ops(action, state['target_map'], state['num'])
    return state


def get_board_str(state: dict) -> str:
    return game_prompt.format(board=board_to_str(state['target_map']),num=state['num'])

# --------------------------
# FastAPI
# --------------------------

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    target_map: list
    num: int
    n: int
    score: int
    is_end: bool
    action: str
    response: list
    prompt: str
    epoch: int

class BoardRequest(BaseModel):
    board: str

@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    return generate_game_state(request.seed)

@app.post("/verify", response_model=GameState)
def api_verify(request: GameState):
    state = request.dict()
    return verify_game_state(state)

@app.post("/print_board", response_model=BoardRequest)
def api_print_board(request: GameState):
    return {"board": get_board_str(request.dict())}

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)