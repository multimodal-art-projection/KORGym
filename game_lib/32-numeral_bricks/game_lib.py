import random
import ast
from typing import List, Dict, Any, Set
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
game_prompt="""
You are a good game player, I'll give you a game board and rules.\nYour task is:\n- First, give your answer according to the game board and rules.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g."Answer: [['c','a','a',...],['c','a','b',...]...]"

You need to fill the grid by coloring cells based on the numeric mappings. The letter indicates the color of the cell, and each cell containing a digit indicates the total number of adjacent cells (including diagonal, horizontal, or vertical directions) that should be colored starting from that cell. For example, if a cell contains the number 5, you must color exactly 5 cells adjacent to it, forming a path in any combination of directions. For instance, you might move 3 steps to the right, 1 step up, and 1 step down from the starting cell, totaling 5 colored cells. You are free to choose the direction of each segment, but the total number of colored cells must exactly match the number in the starting cell. Each segment must be a straight line — once a direction is chosen, you must continue in that direction without turning.The number specified in the mapping refers to the number of additional cells to be colored, excluding the starting cell itself.
For example,if the board is 
0a000
00b00
c0000
00de0
0f00g
and a:4 b:2 c:3 d:3 e:1 f:1 g:5
The game answer can be
caaaa
cabbb
ccdeg
dddeg
ffggg
{board}

"""
# ------------------------------
# 以下为辅助函数（与原始源代码基本一致）
# ------------------------------

def generate_candidates(rows: int, cols: int, max_region_size: int = None) -> List[Dict]:
    """
    生成所有候选“加号区域”。
    每个候选区域由一个提示格 (r,c) 及扩展参数 a, b, u, d 决定，
    区域覆盖：
      - 水平方向：所在行从 (r, c-a) 到 (r, c+b)
      - 垂直方向：所在列从 (r-u, c) 到 (r+d, c)
    区域大小 = a+b+u+d+1，对应提示数字 = 区域大小 - 1
    要求区域大小至少为2。
    """
    candidates = []
    for r in range(rows):
        for c in range(cols):
            max_left = c
            max_right = cols - 1 - c
            max_up = r
            max_down = rows - 1 - r
            for a in range(max_left + 1):      # 向左扩展 a 格
                for b in range(max_right + 1): # 向右扩展 b 格
                    for u in range(max_up + 1):    # 向上扩展 u 格
                        for d in range(max_down + 1):  # 向下扩展 d 格
                            if a == 0 and b == 0 and u == 0 and d == 0:
                                continue  # 区域大小为1则跳过
                            cells = set()
                            # 水平方向（所在行）
                            for j in range(c - a, c + b + 1):
                                cells.add((r, j))
                            # 垂直方向（所在列）
                            for i in range(r - u, r + d + 1):
                                cells.add((i, c))
                            region_size = len(cells)  # 应等于 a+b+u+d+1
                            if max_region_size is not None and region_size > max_region_size:
                                continue
                            candidate = {
                                'clue': (r, c),
                                'left': a,
                                'right': b,
                                'up': u,
                                'down': d,
                                'cells': cells,
                                'digit': region_size - 1  # 提示数字（>=1）
                            }
                            candidates.append(candidate)
    return candidates

def solve_tiling(rows: int, cols: int, candidates: List[Dict]) -> List[Dict]:
    """
    在棋盘 (rows x cols) 上，从候选区域中选择一组互不重叠的区域，
    使得这些区域的并集正好覆盖整个棋盘。
    采用 MRV 启发式（选取候选区域数最少的未覆盖单元格）进行回溯求解。
    """
    board_cells = {(i, j) for i in range(rows) for j in range(cols)}
    # 构建每个单元格到候选区域的映射
    cell_to_candidates: Dict[tuple, List[Dict]] = { (i, j): [] for i in range(rows) for j in range(cols) }
    for cand in candidates:
        for cell in cand['cells']:
            if cell in cell_to_candidates:
                cell_to_candidates[cell].append(cand)

    def backtrack(covered: Set[tuple], solution: List[Dict]) -> List[Dict]:
        if covered == board_cells:
            return solution
        # 从未覆盖的单元格中，选择候选区域数量最少的那个（MRV启发式）
        uncovered = board_cells - covered
        best_cell = None
        best_cands = None
        best_count = float('inf')
        for cell in uncovered:
            valid_cands = [cand for cand in cell_to_candidates[cell] if cand['cells'].isdisjoint(covered)]
            if len(valid_cands) < best_count:
                best_count = len(valid_cands)
                best_cell = cell
                best_cands = valid_cands
            if best_count == 0:
                return None  # 当前cell无候选区域，剪枝返回
        random.shuffle(best_cands)
        for cand in best_cands:
            new_covered = covered.union(cand['cells'])
            solution.append(cand)
            res = backtrack(new_covered, solution)
            if res is not None:
                return res
            solution.pop()
        return None

    return backtrack(set(), [])

def solve_tiling_with_retries(seed: int, rows: int, cols: int, max_total_attempts: int = 50, max_attempts_per_candidate: int = 1) -> List[Dict]:
    """
    尝试求解铺板问题。如果连续多次使用同一候选列表未能求解，
    则重新生成候选区域，最多尝试 max_total_attempts 次。
    """
    random.seed(seed)
    total_attempts = 0
    while total_attempts < max_total_attempts:
        candidates = generate_candidates(rows, cols)
        for _ in range(max_attempts_per_candidate):
            solution = solve_tiling(rows, cols, candidates)
            total_attempts += 1
            if solution is not None:
                return solution
    return None

def create_boards(solution: List[Dict], rows: int, cols: int):
    """
    根据铺板解答生成：
      - 完整解答棋盘：每个单元格填入该区域分配的字母；
      - 谜题棋盘：仅在每个区域的提示格位置显示字母，其余单元格置 '0'；
      - 提示映射：字母 -> digit（digit = 区域大小 - 1）。
    """
    sol_board = [['' for _ in range(cols)] for _ in range(rows)]
    puzzle_board = [['0' for _ in range(cols)] for _ in range(rows)]
    clues = {}
    # 按顺序为每个区域分配一个字母（a, b, c, ...）
    for index, region in enumerate(solution):
        letter = chr(97 + index)
        r, c = region['clue']
        clues[letter] = region['digit']
        for (i, j) in region['cells']:
            sol_board[i][j] = letter
        # 谜题棋盘仅在提示格位置显示字母
        puzzle_board[r][c] = letter
    return sol_board, puzzle_board, clues

def board_to_string(board: List[List[str]]) -> str:
    """将二维棋盘转换为字符串，每行连接后以换行符分隔。"""
    return "\n".join("".join(row) for row in board)

# ------------------------------
# 以下为对外接口，均以 item 字典作为信息传递媒介
# ------------------------------

def generate(seed: int, rows: int = 5, cols: int = 6) -> Dict[str, Any]:
    """
    生成谜题状态：
      - 调用 tiling 求解器生成铺板方案，
      - 通过 create_boards 得到完整解答棋盘（sol_board）、谜题棋盘（puzzle_board）及提示映射（clues）。
    返回的 item 中包含：
      - sol_board: 完整解答棋盘（二维列表）
      - puzzle_board: 谜题棋盘（二维列表，除提示格外均为 '0'）
      - clues: 提示映射（字母 -> 数字）
      - prompt: 谜题棋盘字符串（用于展示）
      - score: 初始得分 0.0
      - is_end: 游戏未结束（False）
      - response: 空列表（用于存储验证反馈）
      - action: 空字符串（待用户填写答案）
      - epoch: 1
    """
    solution = solve_tiling_with_retries(seed, rows, cols)
    if solution is None:
        raise RuntimeError("未能找到合法的铺板方案，请重试或调整参数。")
    sol_board, puzzle_board, clues = create_boards(solution, rows, cols)
    item = {
        "sol_board": sol_board,
        "puzzle_board": puzzle_board,
        "clues": clues,
        "prompt": board_to_string(puzzle_board),
        "score": 0.0,
        "is_end": False,
        "response": [],
        "action": "",
        "epoch": 1,
    }
    return item

def print_board(item: Dict[str, Any]) -> Dict[str, str]:
    """
    根据 item 中的 puzzle_board 和 clues，
    生成展示谜题棋盘和提示映射的字符串。
    返回格式： {"board": <字符串>}
    """
    board_str = "Board:\n" + board_to_string(item.get('puzzle_board', [])) + "\n\n"
    clues = item.get('clues', {})
    if clues:
        board_str += " Mapping (Letter : Number, where Number = Expansion Grid Count)：\n"
        for letter in sorted(clues.keys()):
            board_str += f"{letter} : {clues[letter]}\n"
    return game_prompt.format(board=board_str)

def verify(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    验证用户答案：
      - 要求用户在 item['action'] 中提供完整解答棋盘，格式为二维列表（每个元素为单个字母）。
      - 对照谜题规则验证答案是否合法（见下方 verify 函数）。
    验证结果写入 item['response']，若答案正确，则 score 置为 1.0，否则为 0.0。
    """
    puzzle_board = item.get("puzzle_board")
    clues = item.get("clues")
    
    # 尝试将 item['action'] 解析为二维列表（使用 ast.literal_eval）
    try:
        if isinstance(item.get("action"), str):
            generated_answer = ast.literal_eval(item["action"])
        else:
            generated_answer = item["action"]
    except Exception as e:
        item["score"] = 0.0
        # item["response"] = [f"答案解析错误: {e}"]
        return item

    # 检查答案格式：必须为二维列表且尺寸与 puzzle_board 相同
    rows = len(puzzle_board)
    cols = len(puzzle_board[0]) if rows > 0 else 0
    if not (isinstance(generated_answer, list) and len(generated_answer) == rows and
            all(isinstance(row, list) and len(row) == cols for row in generated_answer)):
        item["score"] = 0.0
        # item["response"] = ["答案尺寸不匹配"]
        return item

    # 调用下方定义的 verify 函数进行验证
    valid = _verify(puzzle_board, clues, generated_answer)
    if valid:
        item["score"] = 1.0
        # item["response"] = ["答案正确"]
    else:
        item["score"] = 0.0
        # item["response"] = ["答案错误"]
    return item

# ------------------------------
# 原始源代码中的验证逻辑（未对外接口，内部调用）
# ------------------------------

def _verify(puzzle_board: List[List[str]], clues: Dict[str, int], generated_answer: List[List[str]]) -> bool:
    rows = len(puzzle_board)
    if rows == 0:
        return False
    cols = len(puzzle_board[0])
    if len(generated_answer) != rows or any(len(row) != cols for row in generated_answer):
        return False

    valid_letters = set(clues.keys())
    # 检查 generated_answer 每个格子均为有效字母
    for r in range(rows):
        for c in range(cols):
            if generated_answer[r][c] not in valid_letters:
                return False

    for letter, clue_val in clues.items():
        # 找到 puzzle_board 中 letter 的提示格，必须恰好1个
        clue_positions = [(r, c) for r in range(rows) for c in range(cols) if puzzle_board[r][c] == letter]
        if len(clue_positions) != 1:
            return False
        clue_r, clue_c = clue_positions[0]
        # 检查 generated_answer 中该位置为 letter
        if generated_answer[clue_r][clue_c] != letter:
            return False

        # region_cells：generated_answer 中所有 letter 格子
        region_cells = {(r, c) for r in range(rows) for c in range(cols) if generated_answer[r][c] == letter}
        # 每个 cell 必须与 (clue_r, clue_c) 在同一行或同一列
        for (r, c) in region_cells:
            if r != clue_r and c != clue_c:
                return False

        # 在 clue 所在行扫描连续 letter 区域
        row_segment = set()
        c_left = clue_c
        while c_left >= 0 and generated_answer[clue_r][c_left] == letter:
            row_segment.add((clue_r, c_left))
            c_left -= 1
        c_right = clue_c + 1
        while c_right < cols and generated_answer[clue_r][c_right] == letter:
            row_segment.add((clue_r, c_right))
            c_right += 1

        # 在 clue 所在列扫描连续 letter 区域
        col_segment = set()
        r_up = clue_r - 1
        while r_up >= 0 and generated_answer[r_up][clue_c] == letter:
            col_segment.add((r_up, clue_c))
            r_up -= 1
        r_down = clue_r + 1
        while r_down < rows and generated_answer[r_down][clue_c] == letter:
            col_segment.add((r_down, clue_c))
            r_down += 1

        expected_region = row_segment | col_segment
        if expected_region != region_cells:
            return False

        horizontal_extension = len(row_segment) - 1
        vertical_extension = len(col_segment) - 1
        expected_clue = horizontal_extension + vertical_extension
        if abs(expected_clue - clue_val) > 1:
            return False

    return True



class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    sol_board: list
    puzzle_board: list
    clues:dict
    score: int
    is_end: bool
    action: str
    response: list
    prompt: str
    epoch: int

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
# ------------------------------
# 测试示例
# ------------------------------

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
