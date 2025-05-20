# game_lib/11-maze/game_lib.py

#Standard libraries
import random
import time
import ast
import argparse

#Commonly used open-source libraries
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

def parse_init():
    """
    Parses command-line arguments for FastAPI server deployment.

    Returns:
        argparse.Namespace: Parsed arguments with 'host' and 'port'.
    """
    parser = argparse.ArgumentParser(description="Data creation utility")
    parser.add_argument('-p', '--port', type=int, default=8775, help='服务部署端口')
    parser.add_argument('-H', '--host', type=str, default="0.0.0.0", help='服务部署地址')
    args = parser.parse_args()
    return args
app = FastAPI()

game_prompt = """
You need to provide a path from the start point to the end point based on an n*n maze map that I provide. Output your answer in the form of a list, where:
'I' represents the starting point
'X' represents the destination point
'o' represents empty space (passable)
'*' represents a wall (impassable)

Your available moves are:
'up': move one cell upwards
'down': move one cell downwards
'left': move one cell to the left
'right': move one cell to the right
You need to output your answer as a list of these strings，e.g."Answer: ['up','down','down',...]"
Maze Board:
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

def print_board(item):
    """
    Generate a formatted maze game prompt from the character matrix.
    
    Args:
        item (dict): Contains 'char_maze' as a list of character rows.

    Returns:
        str: The complete game prompt with rules and visual maze layout.
    """
    output=""
    for line in item['char_maze']:
        output=output+"".join(line)+'\n'
    return game_prompt.format(board=output)

def generate(seed, generate_method='PRIM'):
    """
    Generate a new maze game state based on the given seed and generation method.

    Args:
        seed (int): Random seed for reproducibility.
        generate_method (str): Algorithm to use, either 'PRIM' or 'DFS'.

    Returns:
        dict: A dictionary representing the game state including maze, start, end, and metadata.
    """
    n=random.randint(10,30)
    scale = (n,n)
    random.seed(seed)
    # 计算基础迷宫尺寸（注意：对于奇数尺寸迷宫，通常要求基础尺寸为 ( (n+1)//2, (m+1)//2 )）
    base_size = ((scale[0] + 1) // 2, (scale[1] + 1) // 2)
    numeric_maze = generate_maze_map(generate_method, base_size)
    start, end = init_maze(numeric_maze)
    char_maze = convert_to_char_matrix(numeric_maze, start, end)
    item = {
        'score': 0,
        'is_end': False,
        'response': [],
        'prompt': '',
        'action': '',
        'epoch': 1,
    }
    item['scale'] = n
    item['char_maze'] = char_maze
    item['start'] = start
    item['end'] = end
    return item

def generate_maze_map(method, size):
    """
    Dispatch to the selected maze generation algorithm and log the generation time.

    Args:
        method (str): 'PRIM' or 'DFS'.
        size (tuple): Base size of the maze before expansion.

    Returns:
        np.ndarray: A binary maze map (0 = path, 1 = wall).
    """
    start_time = time.time()
    if method == "DFS":
        maze_map = _dfs_maze(size)
    elif method == "PRIM":
        maze_map = _prim_maze(size)
    else:
        maze_map = _prim_maze(size)
    generation_time = time.time() - start_time
    print(generation_time)
    return maze_map

def _prim_maze(size):
    """
    Generate a maze using the Prim algorithm with compressed representation.

    Args:
        size (tuple): Base maze size.

    Returns:
        np.ndarray: Binary maze map.
    """
    # 此处将基础尺寸再缩小一半
    size = (size[0] // 2, size[1] // 2)
    maze = np.empty((size[0], size[1], 5), dtype=np.uint8)
    maze[:, :, 0] = 0
    maze[:, :, 1:] = 1
    maze[0, 0, 0] = 1
    memory = [[0, 0]]
    while memory:
        prim_det(maze, memory, size)
    return prim2map(maze)

def prim_det(maze, memory, size):
    """
    Core step in Prim's algorithm to expand the maze from a random frontier cell.

    Args:
        maze (np.ndarray): 3D array tracking visited state and connections.
        memory (list): Frontier list for candidate expansion cells.
        size (tuple): Dimensions of the maze.
    """
    index = np.array(random.choice(memory))
    direction = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
    legal_direction = []
    for i, d in enumerate(direction):
        new_index = index + d
        if not (0 <= new_index[0] < size[0] and 0 <= new_index[1] < size[1]):
            continue
        if maze[new_index[0], new_index[1], 0] == 1:
            continue
        legal_direction.append(i)
    if legal_direction:
        dire = random.choice(legal_direction)
        new_index = index + direction[dire]
        # 若新位置不在内存中，则添加；否则删除当前选中的节点
        if 0 != np.min(np.sum(np.abs(np.array(memory) - new_index), axis=1)):
            memory.append(list(new_index))
            maze[index[0], index[1], dire + 1] = 0
            maze[new_index[0], new_index[1], (dire + 2) % 4 + 1] = 0
            maze[new_index[0], new_index[1], 0] = 1
        else:
            memory.remove(list(index))
    else:
        memory.remove(list(index))

def prim2map(maze):
    """
    Convert the 3D Prim maze into a 2D binary matrix for gameplay.

    Args:
        maze (np.ndarray): 3D maze representation.

    Returns:
        np.ndarray: 2D maze grid (0 = path, 1 = wall).
    """
    shape = maze.shape[:2]
    maze_map = np.ones((shape[0] * 2 - 1, shape[1] * 2 - 1), dtype=np.uint8)
    for i in range(maze_map.shape[0]):
        for j in range(maze_map.shape[1]):
            if i % 2 == 0 and j % 2 == 0:
                maze_map[i, j] = 0
            elif i % 2 == 0 and j % 2 == 1:
                maze_map[i, j] = maze[i // 2, j // 2, 1] + maze[i // 2, j // 2 + 1, 3]
            elif i % 2 == 1 and j % 2 == 0:
                maze_map[i, j] = maze[i // 2, j // 2, 2] + maze[i // 2 + 1, j // 2, 4]
            # 若(i,j)均为奇数，则保持为1（墙）
    return maze_map

def _dfs_maze(size):
    """
    Generate a maze using depth-first search (DFS) with backtracking.

    Args:
        size (tuple): Maze dimensions.

    Returns:
        np.ndarray: Binary maze (0 = path, 1 = wall).
    """
    maze = np.empty((size[0], size[1], 2), dtype=np.uint8)
    maze[:, :, 0] = 1
    maze[:, :, 1] = 0
    maze[0, 0, 0] = 0
    maze[0, 0, 1] = 1
    memory = [np.array([0, 0])]
    while memory:
        legal_direction = judge_direction(maze, memory[-1], size)
        if not legal_direction:
            memory.pop()
        else:
            new_index = random.choice(legal_direction)
            memory.append(new_index)
            maze[new_index[0], new_index[1]] = np.array([0, 1])
    return maze[:, :, 0]

def judge_direction(maze, index, size):
    """
    Determine legal directions to expand during DFS, avoiding dense path areas.

    Args:
        maze (np.ndarray): 3D maze representation.
        index (np.ndarray): Current DFS position.
        size (tuple): Maze size.

    Returns:
        list: Valid next positions.
    """
    direction = np.array([[0, 1], [1, 0], [-1, 0], [0, -1]])
    legal_direction = []
    for d in direction:
        new_index = index + d
        if not (0 <= new_index[0] < size[0] and 0 <= new_index[1] < size[1]):
            continue
        if maze[new_index[0], new_index[1], 1] == 1:
            continue
        pass_value = 0
        for dire in direction:
            temp_index = new_index + dire
            if 0 <= temp_index[0] < size[0] and 0 <= temp_index[1] < size[1]:
                pass_value += maze[temp_index[0], temp_index[1], 0]
            else:
                pass_value += 1
        if pass_value < 3:
            maze[new_index[0], new_index[1], 1] = 1
            continue
        legal_direction.append(new_index)
    return legal_direction

def init_maze(numeric_maze):
    """
    Select the start and end positions for the maze.

    Args:
        numeric_maze (np.ndarray): Binary maze matrix.

    Returns:
        tuple: (start_position, end_position)
    """
    start = (0, 0)
    road = np.argwhere(numeric_maze == 0)
    end = tuple(road[np.argmax(np.sum(road * 2, axis=1))])
    return start, end

def convert_to_char_matrix(numeric_maze, start, end):
    """
    Convert binary maze to a character matrix with symbols:
    - 'I': start
    - 'X': end
    - 'o': path
    - '*': wall

    Args:
        numeric_maze (np.ndarray): Binary maze map.
        start (tuple): Start position.
        end (tuple): End position.

    Returns:
        list: Character matrix representing the maze.
    """
    char_maze = []
    for i in range(numeric_maze.shape[0]):
        row = []
        for j in range(numeric_maze.shape[1]):
            if (i, j) == start:
                row.append('I')
            elif (i, j) == end:
                row.append('X')
            else:
                row.append('o' if numeric_maze[i, j] == 0 else '*')
        char_maze.append(row)
    return char_maze

def verify(item):
    """
    Verify whether a given sequence of moves leads from start to end.

    Args:
        item (dict): Contains char_maze, action list, start, and end positions.

    Returns:
        dict: Updated item with score = 1 (success) or 0 (failure).
    """
    dir_map = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1)
    }
    try:
        # 如果item['action']为字符串，尝试转换为列表
        if isinstance(item['action'], str):
            actions = ast.literal_eval(item['action'])
        else:
            actions = item['action']
    except Exception as e:
        # 转换失败
        item['score'] = 0
        return item
    maze = item['char_maze']
    start = item['start']
    end = item['end']
    rows, cols = len(maze), len(maze[0])
    current = start
    for move in actions:
        dr, dc = dir_map.get(move, (0, 0))
        if (dr, dc) == (0, 0):
            item['score'] = 0  # 非法指令
            return item
        nr, nc = current[0] + dr, current[1] + dc
        if not (0 <= nr < rows and 0 <= nc < cols):
            item['score'] = 0 
            return item  # 越界
        target_cell = maze[nr][nc]
        if target_cell == '*':
            item['score'] = 0 
            return item  # 撞墙
        if (nr, nc) == end and move != actions[-1]:
            item['score'] = 0 
            return item  # 中途到达终点
        current = (nr, nc)
    item['score'] = current == end
    return item

def print_maze(maze):
    print("\n".join([" ".join(row) for row in maze]))

class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    char_maze: list
    start: tuple
    end: tuple
    scale: int
    score: int
    is_end: bool
    action: str
    response: list
    prompt: str
    epoch: int
# 生成初始游戏状态
@app.post("/print_board", response_model=BoardRequest)
def api_print_board(request: GameState):
    state = request.dict()
    board_output = print_board(state)
    return {"board": board_output}


# 生成初始游戏状态
@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    game_state = generate(request.seed)
    game_state = convert_numpy_types(game_state)
    return game_state

# 根据动作更新游戏状态
@app.post("/verify", response_model=GameState)
def api_verify(request: GameState):
    # 从请求中获取游戏状态，并设置新的动作
    state = request.dict()
    state['start'] = tuple(state['start'])
    state['end'] = tuple(state['end'])
    updated_state = verify(state)
    updated_state = convert_numpy_types(updated_state)
    return updated_state

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)

# # 示例用法
# if __name__ == "__main__":
#     # 根据种子和尺寸生成迷宫（推荐使用奇数尺寸）
#     item = generate(seed=442)
    
#     print("生成的迷宫：")
#     print_maze(item['char_maze'])
    
#     # 示例动作序列（实际验证时需根据生成迷宫设计合理路径）
#     item['action'] = ['right'] * 8 + ['down'] * 2 + ['right'] * 4 + ['down'] * 2 + ['right'] * 2 + ['down'] * 10
#     print(f"\n验证结果： {verify(item)['score']}")
