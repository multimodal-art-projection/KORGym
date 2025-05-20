# game_lib/13-play_lines/game_lib.py

#Standard libraries
import random
from collections import deque
import copy
import uuid  # 用于生成唯一标识符
from typing import Optional
import ast
import argparse

#Commonly used open-source libraries
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

def parse_init():
    """
    Define and parse command-line arguments for server deployment settings.

    Returns:
        argparse.Namespace: Parsed arguments including host and port.
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
- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question, e.g., "Answer: [['E','X','E',...],['E','1','1',...]...]".

Next, I will provide an n*n chessboard. On the chessboard, 'E' indicates that the element is an empty space, 'X' indicates a node that cannot be passed through, and numbers indicate nodes that need to be connected. You need to fill in the numbers on the empty spaces of the chessboard so that all identical numbers on the chessboard are connected.Moreover, the final chessboard must not have any empty spaces; every cell must be filled with a number (or remain 'X' if it's an impassable cell). Importantly, the connection for each color must form a single continuous line without branching For example, if the initial chessboard is:
E E E E E
E X E 3 E
E 3 E 1 E
E 2 E E E
1 E E 2 E
The filled chessboard could be:
2 2 2 2 2
2 X 3 3 2
2 3 3 1 2
2 2 1 1 2
1 1 1 2 2
When all the numbers on the chessboard are connected, it is considered a game victory, and you score 1 point; otherwise, if any number does not meet the connection requirement, the score will be 0.

Board:
{board}
Please output the answer in the form of a list within one line and do not break lines when outputting Answer, e.g., "Answer: [['E','X','E',...],['E','1','1',...]...]".
"""

def convert_numpy_types(item):
    """
    Recursively convert NumPy data types to native Python types.

    This function is useful for ensuring JSON-serializable output by converting
    numpy.int64, numpy.float64, numpy.ndarray, etc., to their Python equivalents.

    Args:
        item (Any): A nested structure of lists, tuples, dicts, or NumPy types.

    Returns:
        Any: The same structure with all NumPy-specific types converted to native Python types.
    """
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
    Generate a formatted Sokoban Path Connect game prompt from the puzzle grid.

    Args:
        item (dict): A game state dictionary containing the 'puzzle_grid' key.

    Returns:
        str: A game prompt with rules and a visual board formatted for model input.
    """
    grid = item['puzzle_grid']
    output = ""
    for row in grid:
        output=output+''.join(str(cell) for cell in row)+'\n'
    return game_prompt.format(board = output)

def generate_endpoints(grid_size, num_colors, num_x):
    """
    Randomly generate puzzle endpoints and wall placements on an empty grid.

    For each color, two non-adjacent endpoints are placed. Additional impassable
    'X' cells are added at random empty locations.

    Args:
        grid_size (int): The size of the square grid.
        num_colors (int): Number of unique color pairs to place.
        num_x (int): Number of wall cells ('X') to add.

    Returns:
        tuple[list[list[str]], dict[int, tuple]]: The puzzle grid and a dictionary of endpoints.
    """
    grid = [['E' for _ in range(grid_size)] for _ in range(grid_size)]
    endpoints = {}
    # 随机排列所有位置
    all_positions = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    random.shuffle(all_positions)
    # 对每个颜色选取两个位置
    for color in range(1, num_colors + 1):
        if len(all_positions) < 2:
            return None, None
        pos1 = all_positions.pop()
        # 选择第二个位置时，确保它不与第一个位置相邻
        pos2 = None
        for i in range(len(all_positions)):
            candidate = all_positions[i]
            # 检查是否与 pos1 相邻
            if abs(candidate[0] - pos1[0]) + abs(candidate[1] - pos1[1]) > 1:
                pos2 = candidate
                all_positions.pop(i)
                break
        if pos2 is None:
            return None, None
        grid[pos1[0]][pos1[1]] = color
        grid[pos2[0]][pos2[1]] = color
        endpoints[color] = (pos1, pos2)
    
    # 随机放置 num_x 个 'X'
    x_positions = random.sample(all_positions, min(num_x, len(all_positions)))
    for (x, y) in x_positions:
        grid[x][y] = 'X'
    
    return grid, endpoints

def bfs_path_solution(sol_grid, start, end, color, grid_size, allow_empty=True):
    """
    Perform BFS to find a valid path between two endpoints of a given color.

    Args:
        sol_grid (list[list[Any]]): The current grid (may contain partial paths).
        start (tuple): Starting coordinate.
        end (tuple): Ending coordinate.
        color (int): The color number to connect.
        grid_size (int): The size of the grid.
        allow_empty (bool): Whether traversal through empty cells is allowed.

    Returns:
        list[tuple] or None: A list of coordinates forming the path, or None if no path exists.
    """
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    start = tuple(start) if isinstance(start, list) else start
    end = tuple(end) if isinstance(end, list) else end
    
    queue = deque([(start, [start])])
    visited = set([start])
    
    while queue:
        current, path = queue.popleft()
        if current == end:
            return path
        x, y = current
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                neighbor = (nx, ny)
                if neighbor not in visited:
                    if nx < len(sol_grid) and ny < len(sol_grid[nx]):
                        cell = sol_grid[nx][ny]
                        # 若不允许空格，则只能走与color相同的格子（同时排除'X'）
                        if cell != 'X' and (cell == color or (allow_empty and cell == 'E')):
                            visited.add(neighbor)
                            queue.append((neighbor, path + [neighbor]))
    return None

def compute_solution_paths(puzzle_grid, endpoints, grid_size):
    """
    Compute paths for each color pair in the puzzle using BFS.

    Prioritizes longer Manhattan distances first to reduce conflict.
    Updates the solution grid in-place.

    Args:
        puzzle_grid (list[list[str]]): The original puzzle grid with only endpoints.
        endpoints (dict[int, tuple[tuple, tuple]]): Dictionary of color to endpoint pairs.
        grid_size (int): Size of the square puzzle grid.

    Returns:
        tuple[list[list[Any]], dict[int, list[tuple]]]: The updated solution grid and paths per color.
    """
    # 复制谜题网格作为初始解答网格
    sol_grid = copy.deepcopy(puzzle_grid)
    solution_paths = {}
    # 计算每个颜色端点的曼哈顿距离
    def manhattan(color):
        (x1, y1), (x2, y2) = endpoints[color]
        return abs(x1 - x2) + abs(y1 - y2)
    # 按距离降序排序（距离大的先连，因为较远的更难连通）
    colors = sorted(endpoints.keys(), key=lambda c: manhattan(c), reverse=True)

    for color in colors:
        start, end = endpoints[color]
        path = bfs_path_solution(sol_grid, start, end, color, grid_size)
        if path is None:
            return None, None
        solution_paths[color] = path
        # 将路径上的所有格子标记为当前颜色（注意不要覆盖其他颜色的端点）
        for (x, y) in path:
            sol_grid[x][y] = color
    return sol_grid, solution_paths

def extend_paths(sol_grid, endpoints):
    """
    Extend color paths from current endpoints into empty cells where extension is deterministic.

    For each color, identifies endpoints (cells with exactly one adjacent cell of the same color),
    and attempts to extend the path into empty neighboring cells that would maintain a single-line structure.

    Args:
        sol_grid (list[list[Any]]): The partially filled solution grid.
        endpoints (dict[int, tuple[tuple, tuple]]): The original endpoint pairs for each color.

    Returns:
        list[list[Any]]: The updated solution grid after forced extensions.
    """
    grid_size = len(sol_grid)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    changed = True
    while changed:
        changed = False
        # 重新计算各颜色当前的端点（要求同色邻居数为 1）
        current_endpoints = {}
        for color in endpoints.keys():
            current_endpoints[color] = []
            for i in range(grid_size):
                for j in range(grid_size):
                    if sol_grid[i][j] == color:
                        count = 0
                        for dx, dy in directions:
                            ni, nj = i + dx, j + dy
                            if 0 <= ni < grid_size and 0 <= nj < grid_size:
                                if sol_grid[ni][nj] == color:
                                    count += 1
                        if count == 1:
                            current_endpoints[color].append((i, j))
        # 对每个端点尝试强制延伸
        for color, eps in current_endpoints.items():
            for ep in eps:
                i, j = ep
                candidates = []
                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < grid_size and 0 <= nj < grid_size:
                        if sol_grid[ni][nj] == 'E':
                            # 检查：若填入当前颜色，候选格的同色邻居数应恰为 1（仅与当前端点相连）
                            count_same = 0
                            for ddx, ddy in directions:
                                nni, nnj = ni + ddx, nj + ddy
                                if 0 <= nni < grid_size and 0 <= nnj < grid_size:
                                    if sol_grid[nni][nnj] == color:
                                        count_same += 1
                            if count_same == 1:
                                candidates.append((ni, nj))
                # 仅当候选唯一时，执行延伸
                if len(candidates) == 1:
                    sol_grid[candidates[0][0]][candidates[0][1]] = color
                    changed = True
    return sol_grid

def generate(seed):
    """
    Generate a valid Sokoban Path Connect puzzle with a complete solution.

    This function iteratively tries to:
    1. Randomly place color endpoints and walls.
    2. Connect color pairs using BFS.
    3. Extend paths to fill the grid.
    4. Validate that the result is fully filled and non-branching.

    Args:
        seed (int): The random seed for reproducibility.

    Returns:
        dict: The complete puzzle state with metadata including 'puzzle_grid', 'endpoints', and 'grid_size'.
    """
    random.seed(seed)
    count=0
    while True:  # 外层循环，确保在生成失败时重新生成参数
        grid_size = random.randint(5, 10)      # 棋盘尺寸 5x5 到 10x10
        num_colors = random.randint(3,5)        # 颜色数量 5 到 8
        num_x = random.randint(1, 3)             # 为提高延伸成功率，适当减少障碍数量
        max_attempts = random.randint(5000, 10000)  # 最大尝试次数

        attempts = 0
        while attempts < max_attempts:
            puzzle_grid, endpoints = generate_endpoints(grid_size, num_colors, num_x)
            if puzzle_grid is None:
                attempts += 1
                continue
            sol_grid, solution_paths = compute_solution_paths(puzzle_grid, endpoints, grid_size)
            if sol_grid is None:
                attempts += 1
                continue
            # 利用强制延伸填充剩余空格
            sol_grid = extend_paths(sol_grid, endpoints)
            # 如果延伸后仍有空格，则视为失败
            if any('E' in row for row in sol_grid):
                attempts += 1
                continue
            # 整体检查：确保每种颜色的路径为单线（端点同色邻居为1，其余为2）
            if not check_no_branching(sol_grid, endpoints):
                attempts += 1
                continue

            item = {
                'puzzle_grid': puzzle_grid,
                'endpoints': endpoints,
                'score': 1,      # 生成成功
                'is_end': False,
                'response': [],
                'prompt': '',
                'action': "",  # 完整的解答棋盘存入 action 字段
                'epoch': 1,
                'grid_size': grid_size,
            }
            return item
        print(f"Retrying {count}")
        count+=1






# 新增辅助函数：检查每个颜色的连接是否为单一不分叉的路径
def check_no_branching(sol_grid, endpoints):
    """
    Check that all color paths in the solution form non-branching single lines.

    Each color path must satisfy:
    - Exactly two endpoints, each with one adjacent same-color cell.
    - All internal cells must have exactly two adjacent same-color neighbors.

    Args:
        sol_grid (list[list[Any]]): The final solution grid to validate.
        endpoints (dict[int, tuple[tuple, tuple]]): Mapping of colors to their original endpoint pairs.

    Returns:
        bool: True if all paths are valid and non-branching, False otherwise.
    """
    grid_size = len(sol_grid)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for color, (start, end) in endpoints.items():
        # 将起点、终点转换为元组（若有需要）
        if isinstance(start, list):
            start = tuple(start)
        if isinstance(end, list):
            end = tuple(end)
        # 收集该颜色在棋盘中的所有格子位置
        cells = [(i, j) for i in range(grid_size) for j in range(grid_size) if sol_grid[i][j] == color]
        # 对每个格子统计相邻同色个数
        endpoint_count = 0
        for i, j in cells:
            count = 0
            for dx, dy in directions:
                ni, nj = i + dx, j + dy
                if 0 <= ni < grid_size and 0 <= nj < grid_size:
                    if sol_grid[ni][nj] == color:
                        count += 1
            # 如果该位置是起点或终点，要求仅有1个同色邻居
            if (i, j) == start or (i, j) == end:
                if count != 1:
                    return False
                endpoint_count += 1
            else:
                # 中间的格子必须正好有2个同色邻居
                if count != 2:
                    return False
        # 检查该颜色区域中必须正好有2个端点（起点与终点）
        if endpoint_count != 2:
            return False
    return True

# 修改后的 verify 函数，增加了对不分叉要求的验证
def verify(item):
    """
    Verify whether a proposed solution satisfies all Sokoban Path Connect game rules.

    The solution must:
    - Connect each color's endpoints with a valid path.
    - Fill the entire board without any 'E' cells remaining.
    - Maintain non-branching path structures for each color.

    Args:
        item (dict): Game state dictionary containing the submitted 'action' as a solution grid.

    Returns:
        dict: The updated game state including the 'score' and 'is_end' fields.
    """
    # 确保 action 是列表
    if isinstance(item['action'], str):
        try:
            sol_grid = ast.literal_eval(item['action'])
        except (ValueError, SyntaxError):
            item['score'] = 0
            item['is_end'] = True
            return item
    else:
        sol_grid = item['action']
    
    if not isinstance(sol_grid, list) or not all(isinstance(row, list) for row in sol_grid):
        item['score'] = 0
        item['is_end'] = True
        return item
    
    endpoints = item['endpoints']
    grid_size = item['grid_size']
    
    for color in endpoints:
        if isinstance(endpoints[color], list):
            endpoints[color] = tuple(map(tuple, endpoints[color]))
    
    # 先验证各颜色连通
    for color, (start, end) in endpoints.items():
        # 验证时不允许走空格
        path = bfs_path_solution(sol_grid, start, end, color, grid_size, allow_empty=False)
        if path is None:
            item['score'] = 0
            item['is_end'] = True
            return item

    # 验证棋盘内不允许有空格
    for row in sol_grid:
        if 'E' in row:
            item['score'] = 0
            item['is_end'] = True
            return item

    # 新增验证：不允许分叉，每个颜色必须是一条单线连接
    if not check_no_branching(sol_grid, endpoints):
        item['score'] = 0
        item['is_end'] = True
        return item

    item['score'] = 1
    item['is_end'] = True
    return item




def print_grid(grid):
    """ 打印二维网格 """
    for row in grid:
        print(' '.join(str(cell) for cell in row))

def print_solution(sol_grid):
    """ 打印解答网格 """
    print("Solution Grid:")
    print_grid(sol_grid)

def main():
    count=0
    for i in range(2):
        item = generate(i)
        if item['puzzle_grid'] is None:
            count+=1
            print("Failed to generate a solvable puzzle after multiple attempts.")
        else:
            print("Puzzle Grid (Endpoints Only):")
            print_grid(item['puzzle_grid'])
            print()
            print_solution(item['action'])
            print("\nTesting solution...")
            item = verify(item)
            if item['score'] == 1:
                print("Test passed: All paths are valid.")
            else:
                print("Test failed: Some paths are invalid.")
            print(print_board(item))
    print(count)
# if __name__ == "__main__":
#     main()
# --- 定义请求和响应数据模型 ---

class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    puzzle_grid: list
    endpoints: dict
    grid_size: int
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
    if 'endpoints' in state:
        endpoints = state['endpoints']
        for color in endpoints:
            if isinstance(endpoints[color], list):  # 如果值是列表
                endpoints[color] = tuple(map(tuple, endpoints[color]))  # 将列表转换为元组
        state['endpoints'] = endpoints
    updated_state = verify(state)
    # 转换 NumPy 数据类型后返回
    updated_state = convert_numpy_types(updated_state)
    return updated_state

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)

    # main()