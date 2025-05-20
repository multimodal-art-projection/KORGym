import math
import random
import time
import ast
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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

# ================================
# 游戏核心逻辑函数（改写自源代码）
# ================================
game_prompt = """
You are a good game player, I'll give you a game board and rules.
Your task is:
- First, give your answer according to the game board and rules.
- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: (3, 12)'.

You will be given an n*n map containing the following elements:
  - Player (P)
  - Empty cell (E)
  - Portal (paired with matching numbers): Represented by numbers and appear in pairs (1,1; 2,2; etc.). Stepping onto one portal will teleport the player to the other portal with the same number. For example, stepping onto portal 1 will teleport the player to the other portal 1.
  - Jumper (J): Stepping onto a jumper will cause the player to leap two steps in the current direction, skipping the cell in between. For example, if the player is at (1,1) and the jumper is at (1,2), and the move is UP, the player will land at (1,4), and the element at (1,3) will not be triggered.
  - Wall (W): A wall blocks the player's movement, causing them to stay in the original position.
  - Reverser (A): The direction of movement will be reversed when passing through a reverser. For example, if the player is at (3,3), the reverser is at (3,4), and the intended move is UP, the actual movement will be DOWN, landing at (3,2).
  - Trap (T): Stepping into a trap will trap the player for one turn, making the next move ineffective. For example, if the player is at (3,3), the trap is at (3,4), and the move sequence is UP, UP, LEFT, then the first UP puts the player into the trap, the next UP is canceled, and the player ends up performing LEFT next.
  - Repeater (R): Stepping onto a repeater causes the player to move an extra step in the same direction. For example, if the player is at (1,1), and the repeater is at (1,2), and the move is UP, the player will end up at (1,3).

Additional Rules:
  - Map elements can be combined. For example, a jumper may cause the player to land on a trap two cells away.
  - Elements that have already been triggered during the current turn will not trigger again (except for walls), to prevent infinite loops.
  - The map boundaries are all walls to prevent going out of bounds.
  - Map coordinates start from (0,0), i.e., the top-left corner is (0,0).

You will see a generated sequence of moves. Based on the given map and the move sequence, determine the player's final position after executing all moves.

Please output the final player coordinate in the following format:'Answer: (row, col)',e.g.'Answer: (3, 12)'

Map:
{board_str}

Move sequence:
{task_str}
"""

def generate(seed: int):
    """
    按给定种子、地图尺寸（rows, cols）和步数生成一个合法的游戏地图和行动序列。
    若生成的地图或模拟失败，则调整种子后重试。
    """
    random.seed(seed)
    while True:
        scale = (random.randint(10,50),random.randint(10,50))
        num_step = random.randint(10,50)
        game_map, task = generate_core(seed, scale, num_step)
        if game_map is None or task is None:
            seed += 1
            continue
        sim_result = simulate(game_map, task)
        if sim_result is None:
            seed += 1
            continue
        # 利用模拟验证地图和行动序列能正确执行（返回值应稳定）
        if verify_answer(game_map, task, sim_result) == 1:
            return game_map, task
        else:
            seed += 1

def generate_core(seed: int, scale: tuple, num_step: int):
    """
    核心生成逻辑：
      - 初始化地图，边界为墙(W)，内部为空地(E)
      - 随机放置玩家(P)
      - 随机放置传送门（成对出现的数字）
      - 随机放置其他元素：跳板(J)、反向器(A)、陷阱(T)、重复器(R)
      - 生成随机行动序列（UP, DOWN, LEFT, RIGHT）
    """
    random.seed(seed)
    rows, cols = scale
    area = (rows - 2) * (cols - 2)
    portal_num_max = math.ceil(area * 0.05)
    jatr_num_max = math.ceil(area * 0.4) // 4
    if area <= 1 + portal_num_max * 2 + jatr_num_max:
        return None, None

    # 初始化地图：内部为'E'，边界为'W'
    game_map = [['E' for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if i == 0 or i == rows - 1 or j == 0 or j == cols - 1:
                game_map[i][j] = 'W'

    # 放置玩家'P'
    possible_positions = [(i, j) for i in range(1, rows - 1) for j in range(1, cols - 1)]
    if not possible_positions:
        raise ValueError("没有可用位置放置玩家")
    p_pos = random.choice(possible_positions)
    possible_positions.remove(p_pos)
    game_map[p_pos[0]][p_pos[1]] = 'P'

    # 放置传送门：成对出现，使用数字标识
    portal_num = random.randint(1, portal_num_max) if portal_num_max >= 1 else 1
    portal_id = 1
    for _ in range(portal_num):
        if len(possible_positions) >= 2:
            pos1 = random.choice(possible_positions)
            possible_positions.remove(pos1)
            pos2 = random.choice(possible_positions)
            possible_positions.remove(pos2)
            game_map[pos1[0]][pos1[1]] = str(portal_id)
            game_map[pos2[0]][pos2[1]] = str(portal_id)
            portal_id += 1

    # 放置其他元素：跳板(J)、反向器(A)、陷阱(T)、重复器(R)
    elements = ['J', 'A', 'T', 'R']
    for elem in elements:
        count = random.randint(0, jatr_num_max)
        for _ in range(count):
            if possible_positions:
                pos = random.choice(possible_positions)
                possible_positions.remove(pos)
                game_map[pos[0]][pos[1]] = elem

    # 生成行动序列（大小写均为大写）
    directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    task = [random.choice(directions) for _ in range(num_step)]
    return game_map, task

def simulate(game_map, task):
    """
    模拟玩家在游戏地图上根据行动序列移动后的最终坐标。
    规则说明：
      - 玩家初始位置为 'P'
      - 遇到墙(W)则停在原地
      - 数字代表传送门，进入后传送至相同数字的另一位置
      - 跳板(J)：玩家沿当前方向飞跃一步（两步移动，跳过中间的格子）
      - 反向器(A)：将行动方向反转
      - 陷阱(T)：进入后使下一步行动失效（停顿一回合）
      - 重复器(R)：触发后当前方向额外执行一次移动
    """
    rows = len(game_map)
    cols = len(game_map[0]) if rows > 0 else 0
    # 寻找玩家初始位置
    start_pos = None
    for i in range(rows):
        for j in range(cols):
            if game_map[i][j] == 'P':
                start_pos = (i, j)
                break
        if start_pos:
            break
    if not start_pos:
        return None

    current_pos = start_pos
    action_idx = 0
    trapped = 0
    repeated_action = None
    outer_loop_count = 0
    while action_idx < len(task):
        outer_loop_count += 1
        if outer_loop_count > 200:
            # 防止无限循环
            return None
        if trapped > 0:
            trapped -= 1
            action_idx += 1
            continue
        if repeated_action is not None:
            current_action = repeated_action
            repeated_action = None
        else:
            current_action = task[action_idx]
            action_idx += 1

        dx, dy = 0, 0
        if current_action == 'UP':
            dx = -1
        elif current_action == 'DOWN':
            dx = 1
        elif current_action == 'LEFT':
            dy = -1
        elif current_action == 'RIGHT':
            dy = 1

        new_x = current_pos[0] + dx
        new_y = current_pos[1] + dy
        if not (0 <= new_x < rows and 0 <= new_y < cols):
            new_x, new_y = current_pos
            element = 'W'
        else:
            element = game_map[new_x][new_y]

        inner_loop_count = 0
        while True:
            inner_loop_count += 1
            if inner_loop_count > 200:
                return None
            if element == 'W':
                new_x, new_y = current_pos
                break
            if element.isdigit():
                # 传送门：寻找同样数字的另一位置
                other = None
                for i in range(rows):
                    for j in range(cols):
                        if game_map[i][j] == element and (i, j) != (new_x, new_y):
                            other = (i, j)
                            break
                    if other:
                        break
                if other:
                    new_x, new_y = other
                break
            elif element == 'J':
                # 跳板：沿当前方向跳跃一步（共两步），中间格子不触发
                jump_x = new_x + dx*2
                jump_y = new_y + dy*2
                if 0 <= jump_x < rows and 0 <= jump_y < cols and game_map[jump_x][jump_y] != 'W':
                    new_x, new_y = jump_x, jump_y
                    element = game_map[new_x][new_y]
                else:
                    element = 'E'
                    break
            elif element == 'A':
                # 反向器：将方向反转
                dx, dy = -dx, -dy
                rev_x = current_pos[0] + dx
                rev_y = current_pos[1] + dy
                if 0 <= rev_x < rows and 0 <= rev_y < cols and game_map[rev_x][rev_y] != 'W':
                    new_x, new_y = rev_x, rev_y
                    element = game_map[new_x][new_y]
                else:
                    new_x, new_y = current_pos
                    element = 'E'
                    break
            elif element == 'T':
                # 陷阱：触发后本回合后续移动无效，下一步失效
                trapped = 1
                break
            elif element == 'R':
                # 重复器：同一方向额外执行一次移动
                repeated_action = current_action
                break
            else:
                break
        current_pos = (new_x, new_y)
    return current_pos

def verify_answer(game_map, task, user_pred):
    """
    验证用户提交的答案（最终坐标）是否正确。
    计算模拟后的正确坐标，并与用户答案进行比较，相同返回1，否则返回0。
    """
    correct_pos = simulate(game_map, task)
    return 1 if correct_pos == user_pred else 0

def print_board(item):
    board_str = "\n".join([" ".join(row) for row in item['game_map']])
    task_str = ", ".join(item['task'])
    return game_prompt.format(board_str=board_str,task_str=task_str)
# ================================
# FastAPI 接口及数据模型
# ================================

class GenerateRequest(BaseModel):
    seed: int

class BoardRequest(BaseModel):
    board: str

class GameState(BaseModel):
    game_map: list
    task: list
    # 用户提交的最终坐标答案，格式例如 "(2, 3)" 或 [2, 3]
    action: str 
    score: int 
    is_end: bool 
    prompt: str 
    epoch: int 
    row_num: int
    col_num: int
    seed: int 

@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    game_map, task = generate(request.seed)
    if game_map is None or task is None:
        raise HTTPException(status_code=400, detail="生成游戏失败，请调整参数")
    item = {
        "game_map": game_map,
        "task": task,
        "action": "",
        "score": 0,
        "is_end": False,
        "prompt": "",  # 可由 print_board 接口更新
        "epoch": 1,
        "row_num": len(game_map),
        "col_num": len(game_map[0]),
        "seed": request.seed
    }
    return item

@app.post("/verify", response_model=GameState)
def api_verify(state: GameState):
    """
    接收包含 game_map、task 及用户答案（answer）的 item，
    利用 simulate 计算正确坐标，并更新 score 字段（正确返回1，否则返回0）。
    """
    try:
        user_answer = ast.literal_eval(state.action)
        if not (isinstance(user_answer, tuple) or isinstance(user_answer, list)):
            raise ValueError("答案格式错误")
        user_answer = tuple(user_answer)
    except Exception as e:
        state.score = 0
        return state
    result = verify_answer(state.game_map, state.task, user_answer)
    state.score = result
    return state

@app.post("/print_board", response_model=BoardRequest)
def api_print_board(state: GameState):
    """
    将 game_map 转换为字符串格式，并生成包含游戏规则、地图及行动序列的提示信息。
    """
    board_output = print_board(state.dict())
    return {"board": board_output}

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)