# game_lib/6-LongCat/game_lib.py

#Standard libraries
import random
from collections import deque
import copy
import argparse
import ast

#Commonly used open-source libraries
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

def parse_init():
    """
    Parses command-line arguments to configure the host and port of the FastAPI server.

    Returns:
        argparse.Namespace: Parsed host and port parameters.
    """
    parser = argparse.ArgumentParser(description="Data creation utility")
    parser.add_argument('-p', '--port', type=int, default=8775, help='服务部署端口')
    parser.add_argument('-H', '--host', type=str, default="0.0.0.0", help='服务部署地址')
    args = parser.parse_args()
    return args
app = FastAPI()
game_prompt = """
You are a good game player, I'll give you a game board and rules.
Your task is:
- First, give your answer according to the game board and rules.
- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: ['left', 'down', 'right', 'up', 'left']'.

Next, I will provide an n × n board containing a cat ('C'), empty spaces ('E'), and walls ('X'). You need to control the cat's movement by entering directions: up, down, left, or right. The cat moves from its initial position, sliding continuously in the chosen direction until hitting a wall. All empty spaces ('E') traversed along the path will turn into walls ('X'). The game is won when all empty spaces have been filled. Please output your solution as a list containing directions ('up', 'left', 'right', 'down'), for example:  
'Answer: ['left', 'down', 'right', 'up', 'left']'
Board:
{board}
"""


def is_solvable(game_map):
    """
    Determines whether a LongCat board is solvable via DFS.

    Rules: The cat starts at 'C', slides in one direction until hitting a wall ('X'),
    turning all traversed 'E' cells into walls. The goal is to cover all 'E' cells.

    Args:
        game_map (List[List[str]]): 2D board with 'E', 'X', and 'C'.

    Returns:
        bool: True if the board is solvable, else False.
    """
    rows = len(game_map)
    cols = len(game_map[0])
    cat_pos = None
    board = [row.copy() for row in game_map]
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == 'C':
                cat_pos = (r, c)
                board[r][c] = 'X'
                break
        if cat_pos:
            break
    if not cat_pos:
        return False

    directions = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1)
    }

    def board_to_key(board, cat_pos):
        return (cat_pos, tuple(tuple(row) for row in board))

    visited = {}

    def dfs(board, cat_pos):
        key = board_to_key(board, cat_pos)
        if key in visited:
            return visited[key]
        if all(cell != 'E' for row in board for cell in row):
            visited[key] = True
            return True
        for dr, dc in directions.values():
            r, c = cat_pos
            path = []
            while True:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    break
                if board[nr][nc] == 'X':
                    break
                if board[nr][nc] == 'E':
                    path.append((nr, nc))
                r, c = nr, nc
            if not path:
                continue
            new_board = [list(row) for row in board]
            for pr, pc in path:
                new_board[pr][pc] = 'X'
            new_cat_pos = path[-1]
            if dfs(new_board, new_cat_pos):
                visited[key] = True
                return True
        visited[key] = False
        return False

    return dfs(board, cat_pos)

def init_map(rows, cols):
    return [['X' if r == 0 or r == rows - 1 or c == 0 or c == cols - 1 else 'E'
             for c in range(cols)] for r in range(rows)]

def is_valid(r, c, rows, cols):
    return 0 <= r < rows and 0 <= c < cols

def get_neighbors(game_map, r, c, cell_type):
    """
    Gets all 4-directional neighbors of a given cell that match a specific type.

    Returns:
        List[Tuple[int, int]]: List of neighbor coordinates.
    """
    rows = len(game_map)
    cols = len(game_map[0])
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if is_valid(nr, nc, rows, cols) and game_map[nr][nc] == cell_type:
            neighbors.append((nr, nc))
    return neighbors

def bfs_connectivity(game_map, start):
    """
    Uses BFS to determine the connected 'E' area from a starting cell.

    Args:
        start (Tuple[int, int]): Starting cell.

    Returns:
        Set[Tuple[int, int]]: All reachable 'E' cells.
    """
    rows = len(game_map)
    cols = len(game_map[0])
    visited = set()
    q = deque([start])
    visited.add(start)

    while q:
        r, c = q.popleft()
        for nr, nc in get_neighbors(game_map, r, c, 'E'):
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc))
    return visited

def add_random_walls(game_map, rows, cols):
    """
    Randomly adds a few internal walls while maintaining connectivity among empty cells.

    Modifies:
        game_map (List[List[str]]): In-place modification.
    """

    internal_cells = [(r, c) for r in range(1, rows-1)
                      for c in range(1, cols-1) if game_map[r][c] == 'E']
    num_walls = len(internal_cells) // 5

    for _ in range(num_walls):
        r, c = random.choice(internal_cells)
        original = game_map[r][c]
        game_map[r][c] = 'X'
        e_cells = [(r, c) for r in range(rows) for c in range(cols) if game_map[r][c] == 'E']
        if e_cells:
            visited = bfs_connectivity(game_map, e_cells[0])
            if len(visited) != len(e_cells):
                game_map[r][c] = original

def place_cat(game_map, rows, cols):
    """
    Places the cat ('C') on a cell, preferably one with only one adjacent 'E'.

    Raises:
        RuntimeError: If no empty cell is available.
    """

    leaves = []
    all_e = []
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            if game_map[r][c] == 'E':
                all_e.append((r, c))
                if len(get_neighbors(game_map, r, c, 'E')) == 1:
                    leaves.append((r, c))
    if leaves:
        cat_r, cat_c = random.choice(leaves)
    else:
        if not all_e:
            raise RuntimeError("没有剩余的空格用于放置猫咪")
        cat_r, cat_c = random.choice(all_e)
    game_map[cat_r][cat_c] = 'C'

def generate_map(seed: int):
    """
    Generates a valid and solvable LongCat board with random size and layout.

    Args:
        seed (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary representing the initial game state including the map.
    """
    random.seed(seed)
    item = {
        "score": 0,
        "is_end": False,
        "action": "",       # list,e.g. ['right', 'down', ...]
        "response": [],
        "prompt": "",
        "epoch": 1,
    }
    rows = random.randint(5, 10)
    cols = random.randint(5, 10)
    item['row_num'] = rows
    item['col_num'] = cols 
    
    attempt = 0
    while True:
        attempt += 1
        game_map = init_map(rows, cols)
        add_random_walls(game_map, rows, cols)
        place_cat(game_map, rows, cols)
        if is_solvable(game_map):
            item['game_map'] = game_map
            break
    return item

def verify(actions, game_map):
    """
    Verifies whether the given sequence of moves successfully fills all 'E' cells.

    Args:
        actions (List[str]): Sequence of moves ('up', 'down', etc.).
        game_map (List[List[str]]): The initial map with 'C', 'E', and 'X'.

    Returns:
        int: 1 if successful, 0 otherwise.
    """
    current_map = [row.copy() for row in game_map]
    rows = len(current_map)
    cols = len(current_map[0]) if rows > 0 else 0

    # Cat initial place
    cat_pos = None
    for r in range(rows):
        for c in range(cols):
            if current_map[r][c] == 'C':
                cat_pos = (r, c)
                current_map[r][c] = 'X'  
                break
        if cat_pos:
            break

    if not cat_pos:
        return 0

    directions = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1)
    }

    for move in actions:
        move = move.lower()
        if move not in directions:
            continue
        dr, dc = directions[move]
        current_r, current_c = cat_pos
        path = []
        while True:
            next_r = current_r + dr
            next_c = current_c + dc
            if not (0 <= next_r < rows and 0 <= next_c < cols):
                break
            if current_map[next_r][next_c] == 'X':
                break
            if current_map[next_r][next_c] == 'E':
                path.append((next_r, next_c))
            current_r, current_c = next_r, next_c
        if not path:
            continue  # 无效移动
        for r, c in path:
            current_map[r][c] = 'X'
        cat_pos = path[-1]
    for row in current_map:
        if 'E' in row:
            return 0
    return 1

def verify_game(item):
    """
    Wrapper function that verifies the player's move sequence and updates score.

    Args:
        item (dict): Game state including 'action' and 'game_map'.

    Returns:
        dict: Updated game state with score field modified.
    """
    actions = item.get("action")
    if isinstance(actions, str):
        try:
            actions = ast.literal_eval(actions)
        except Exception as e:
            item["score"] = 0
            return item
    game_map = item.get("game_map")
    score = verify(actions, game_map)
    item["score"] = score
    return item

def print_board(item):
    """
    Converts the game map into a textual string embedded into the prompt template.

    Args:
        item (dict): Game state containing 'game_map'.

    Returns:
        str: Final prompt string with rules and board.
    """
    game_map = item.get("game_map", [])
    board_str = "\n".join([" ".join(row) for row in game_map])
    return game_prompt.format(board=board_str)

# ================================
# FastAPI API
# ================================

class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    game_map: list
    row_num: int
    col_num: int
    score: int
    is_end: bool
    action: str
    response: list
    prompt: str
    epoch: int

@app.post("/print_board", response_model=BoardRequest)
def api_print_board(request: GameState):
    state = request.dict()
    board_output = print_board(state)
    return {"board": board_output}

@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    game_state = generate_map(request.seed)
    return game_state

@app.post("/verify", response_model=GameState)
def api_verify(request: GameState):
    state = request.dict()
    updated_state = verify_game(state)
    return updated_state

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)