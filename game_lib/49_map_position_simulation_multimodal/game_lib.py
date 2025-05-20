from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
import os
import base64
import math
import time
import uvicorn
from PIL import Image, ImageDraw, ImageFont
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
# -------------------- 全局参数与颜色映射 --------------------
CELL_SIZE = 50   # 单元格像素大小
BORDER = 2       # 单元格边框宽度
game_prompt = """
You are a good game player, I'll give you a game board and rules.
Your task is:
- First, give your answer according to the game board(picture) and rules.
- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: (3, 12)'.

You will be given an n*n map(picture) containing the following elements:
  - Player (P):Red
  - Empty cell (E):Whtie
  - Portal (paired with matching numbers): Blue,Represented by numbers and appear in pairs (1,1; 2,2; etc.). Stepping onto one portal will teleport the player to the other portal with the same number. For example, stepping onto portal 1 will teleport the player to the other portal 1.
  - Jumper (J): Green,stepping onto a jumper will cause the player to leap two steps in the current direction, skipping the cell in between. For example, if the player is at (1,1) and the jumper is at (1,2), and the move is UP, the player will land at (1,4), and the element at (1,3) will not be triggered.
  - Wall (W): Grey,a wall blocks the player's movement, causing them to stay in the original position.
  - Reverser (A): Orange,the direction of movement will be reversed when passing through a reverser. For example, if the player is at (3,3), the reverser is at (3,4), and the intended move is UP, the actual movement will be DOWN, landing at (3,2).
  - Trap (T): Purple,stepping into a trap will trap the player for one turn, making the next move ineffective. For example, if the player is at (3,3), the trap is at (3,4), and the move sequence is UP, UP, LEFT, then the first UP puts the player into the trap, the next UP is canceled, and the player ends up performing LEFT next.
  - Repeater (R): Yellow,stepping onto a repeater causes the player to move an extra step in the same direction. For example, if the player is at (1,1), and the repeater is at (1,2), and the move is UP, the player will end up at (1,3).

Additional Rules:
  - Map elements can be combined. For example, a jumper may cause the player to land on a trap two cells away.
  - Elements that have already been triggered during the current turn will not trigger again (except for walls), to prevent infinite loops.
  - The map boundaries are all walls to prevent going out of bounds.
  - Map coordinates start from (0,0), i.e., the top-left corner is (0,0).

You will see a generated sequence of moves. Based on the given map and the move sequence, determine the player's final position after executing all moves.

Please output the final player coordinate in the following format:'Answer: (row, col)',e.g.'Answer: (3, 12)'

{board}
"""
# 定义地图元素颜色映射
COLOR_MAP = {
    "P": "#FF0000",   # 玩家：红色
    "W": "#808080",   # 墙：灰色
    "E": "#FFFFFF",   # 空地：白色
    "J": "#00FF00",   # 跳板：绿色
    "A": "#FFA500",   # 反向器：橙色
    "T": "#800080",   # 陷阱：紫色
    "R": "#FFFF00",   # 重复器：黄色
    "default": "#0000FF"  # 对于传送门（数字）及其他未定义元素使用蓝色
}

# -------------------- 工具函数 --------------------
def encode_image(image_path: str) -> str:
    """将图片文件转为 base64 编码字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def draw_map(game_map, current_pos, save_filename, step=0):
    """
    根据 game_map 和当前玩家位置 current_pos 绘制地图图片，
    并将图片保存在 cache 目录下，返回 base64 编码后的图片字符串。
    """
    rows = len(game_map)
    cols = len(game_map[0]) if rows > 0 else 0
    img_width = cols * CELL_SIZE + (cols + 1) * BORDER
    img_height = rows * CELL_SIZE + (rows + 1) * BORDER
    img = Image.new("RGB", (img_width, img_height), color="#000000")
    draw = ImageDraw.Draw(img)
    # 尝试加载字体
    font_size = 24
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    for i in range(rows):
        for j in range(cols):
            x1 = j * (CELL_SIZE + BORDER) + BORDER
            y1 = i * (CELL_SIZE + BORDER) + BORDER
            x2 = x1 + CELL_SIZE
            y2 = y1 + CELL_SIZE
            # 如果该格为当前玩家位置，则用玩家颜色绘制
            if (i, j) == current_pos:
                color = COLOR_MAP.get("P")
                cell_text = "P"
            else:
                cell = game_map[i][j]
                color = COLOR_MAP.get(cell, COLOR_MAP["default"])
                cell_text = cell
            draw.rectangle([x1, y1, x2, y2], fill=color)
            # 非空地的格子显示元素文字
            if cell_text != "E":
                text = cell_text
                left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
                text_width = right - left
                text_height = bottom - top
                draw.text(((x1 + (CELL_SIZE - text_width) / 2), (y1 + (CELL_SIZE - text_height) / 2)),
                          text, fill="#000000", font=font)
    # 确保 cache 目录存在
    os.makedirs("cache", exist_ok=True)
    file_path = os.path.join("cache", save_filename)
    img.save(file_path)
    return encode_image(file_path)

def find_player_position(game_map):
    """在 game_map 中查找玩家 'P' 的位置，返回 (row, col)"""
    rows = len(game_map)
    cols = len(game_map[0]) if rows > 0 else 0
    for i in range(rows):
        for j in range(cols):
            if game_map[i][j] == 'P':
                return (i, j)
    return (-1, -1)

# -------------------- 地图生成与模拟逻辑 --------------------
def generate_core(seed, scale, num_step):
    """
    根据给定 seed、地图规模 scale（(rows, cols)）和步数 num_step 生成地图与移动序列。
    地图外围均为墙 'W'，内部初始为空地 'E'。
    玩家 'P'随机放置在内部，传送门、跳板、反向器、陷阱和重复器分别随机分布。
    """
    random.seed(seed)
    rows, cols = scale
    area = (rows - 2) * (cols - 2)
    portal_num_max = math.ceil(area * 0.05)
    jatr_num_max = math.ceil(area * 0.4) // 4

    # 初始化地图，内部为空地，边界为墙
    game_map = [['E' for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if i == 0 or i == rows - 1 or j == 0 or j == cols - 1:
                game_map[i][j] = 'W'

    # 放置玩家 'P'
    possible_positions = [(i, j) for i in range(1, rows - 1) for j in range(1, cols - 1)]
    if not possible_positions:
        raise ValueError("没有可用位置放置玩家")
    p_pos = random.choice(possible_positions)
    possible_positions.remove(p_pos)
    game_map[p_pos[0]][p_pos[1]] = 'P'

    # 放置传送门（成对出现，用数字标识）
    portal_num = random.randint(1, portal_num_max) if portal_num_max > 0 else 0
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

    # 放置其他元素：跳板 (J)、反向器 (A)、陷阱 (T) 和重复器 (R)
    elements = ['J', 'A', 'T', 'R']
    for elem in elements:
        count = random.randint(0, jatr_num_max)
        for _ in range(count):
            if possible_positions:
                pos = random.choice(possible_positions)
                possible_positions.remove(pos)
                game_map[pos[0]][pos[1]] = elem

    # 生成移动序列（任务），每一步随机选择方向
    directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    task = [random.choice(directions) for _ in range(num_step)]
    return game_map, task

def simulate_no_draw(game_map, task):
    """
    根据 game_map 与移动序列 task 模拟玩家移动过程，不进行图片绘制，
    返回最终玩家坐标 (row, col)。
    """
    rows = len(game_map)
    cols = len(game_map[0]) if rows > 0 else 0
    # 查找初始玩家位置
    start_pos = None
    for i in range(rows):
        for j in range(cols):
            if game_map[i][j] == 'P':
                start_pos = (i, j)
                break
        if start_pos:
            break
    if start_pos is None:
        return None
    current_pos = start_pos
    action_idx = 0
    trapped = 0
    repeated_action = None

    while action_idx < len(task):
        if trapped > 0:
            trapped -= 1
            action_idx += 1
            continue
        current_action = repeated_action if repeated_action is not None else task[action_idx]
        if repeated_action is not None:
            repeated_action = None
        else:
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
        # 越界时视为碰墙，不移动
        if new_x < 0 or new_x >= rows or new_y < 0 or new_y >= cols:
            new_x, new_y = current_pos
            current_pos = (new_x, new_y)
            continue
        element = game_map[new_x][new_y]
        inner_loop_count = 0
        while True:
            inner_loop_count += 1
            if inner_loop_count > 200:
                return current_pos
            if element == 'W':
                new_x, new_y = current_pos
                break
            if element.isdigit():
                # 传送门：寻找相同数字的另一端
                target = None
                for i in range(rows):
                    for j in range(cols):
                        if game_map[i][j] == element and (i, j) != (new_x, new_y):
                            target = (i, j)
                if target:
                    new_x, new_y = target
                break
            elif element == 'J':
                # 跳板：沿当前方向跳过一个单元格
                nx = new_x + dx * 2
                ny = new_y + dy * 2
                if 0 <= nx < rows and 0 <= ny < cols and game_map[nx][ny] != 'W':
                    new_x, new_y = nx, ny
                    element = game_map[new_x][new_y]
                    continue
                else:
                    element = 'E'
                    break
            elif element == 'A':
                # 反向器：方向取反
                dx, dy = -dx, -dy
                nx = current_pos[0] + dx
                ny = current_pos[1] + dy
                if 0 <= nx < rows and 0 <= ny < cols and game_map[nx][ny] != 'W':
                    new_x, new_y = nx, ny
                    element = game_map[new_x][new_y]
                    continue
                else:
                    new_x, new_y = current_pos
                    element = 'E'
                    break
            elif element == 'T':
                # 陷阱：本次移动后下一步无效
                trapped = 1
                break
            elif element == 'R':
                # 重复器：本次动作会重复执行
                repeated_action = current_action
                break
            else:
                break
        current_pos = (new_x, new_y)
    return current_pos

def render_state_to_image(item):
    """
    根据 item 中的地图状态生成当前局面的图片：
      - 若游戏已结束，则先更新地图，将原来的玩家位置清除，并在最终位置上标记玩家。
      - 图片保存在 cache 目录中，并返回图片的 base64 编码字符串。
    """
    if item.get("is_end", False):
        final_pos = simulate_no_draw(item["game_map"], item["task"])
        # 清除原来 'P' 标记
        for i in range(len(item["game_map"])):
            for j in range(len(item["game_map"][0])):
                if item["game_map"][i][j] == 'P':
                    item["game_map"][i][j] = 'E'
        if final_pos is not None:
            item["game_map"][final_pos[0]][final_pos[1]] = 'P'
            item["current_pos"] = final_pos
        else:
            item["current_pos"] = (-1, -1)
    current_pos = item.get("current_pos", (-1, -1))
    file_name = f"board_{item['seed']}_{item['epoch']}_{random.randint(0,100000)}.png"
    return draw_map(item["game_map"], current_pos, file_name, step=item["epoch"])

# -------------------- 接口封装 --------------------
def generate(seed: int) -> dict:
    """
    根据给定 seed 生成初始游戏状态：
      - 地图规模默认为 10×10，移动步数默认为 10 步。
      - 返回的 item 包含地图、任务、初始玩家位置、状态标志、图片等信息。
    """
    random.seed(seed)
    n=random.randint(10,20)
    scale = (n, n)
    num_step = n
    game_map, task = generate_core(seed, scale, num_step)
    current_pos = find_player_position(game_map)
    item = {
        "seed": seed,
        "epoch": 1,
        "game_map": game_map,
        "task": task,
        "current_pos": current_pos,
        "is_end": False,
        "prompt": "",
        "action": "",
        "score": 0,
        "base64_image": ""
    }
    item["base64_image"] = render_state_to_image(item)
    return item

def print_board(item: dict) -> str:
    """
    返回当前游戏状态的文本描述，包括回合数、地图每行内容以及移动任务序列。
    """
    board_lines = [f"Epoch: {item['epoch']}"]
    board_lines.append("Task: " + " ".join(item["task"]))
    return game_prompt.format(board="\n".join(board_lines))

def verify(item: dict) -> dict:
    """
    根据 item 中的 action 更新游戏状态：
      - 预期 action 格式为形如 "(row, col)" 的字符串（表示玩家对最终位置的猜测）。
      - 模拟移动过程计算正确的最终位置，与用户猜测进行比对，更新分数、状态与提示。
      - 同时更新最终局面图片。
    """
    guess_str = item.get("action", "").strip()
    try:
        # 解析形如 "(row, col)" 的输入
        guess_str = guess_str.strip("()")
        parts = guess_str.split(",")
        guess = tuple(int(x.strip()) for x in parts)
        if len(guess) != 2:
            raise ValueError
    except Exception:
        return item  # 格式错误时不做更新

    correct_pos = simulate_no_draw(item["game_map"], item["task"])
    item["epoch"] += 1
    item["is_end"] = True
    if guess == correct_pos:
        item["score"] = 1
    else:
        item["score"] = 0
    # 更新地图中玩家标记：先清除原 'P'，再将最终位置标记为 'P'
    for i in range(len(item["game_map"])):
        for j in range(len(item["game_map"][0])):
            if item["game_map"][i][j] == 'P':
                item["game_map"][i][j] = 'E'
    if correct_pos is not None:
        item["game_map"][correct_pos[0]][correct_pos[1]] = 'P'
        item["current_pos"] = correct_pos
    else:
        item["current_pos"] = (-1, -1)
    item["base64_image"] = render_state_to_image(item)
    return item

# -------------------- FastAPI 接口定义 --------------------
app = FastAPI()

class GenerateRequest(BaseModel):
    seed: int

class BoardRequest(BaseModel):
    board: str

class GameState(BaseModel):
    seed: int
    epoch: int
    game_map: list
    task: list
    current_pos: tuple
    is_end: bool
    prompt: str
    action: str
    score: int
    base64_image: str

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
    state=request.dict()
    state['current_pos']=tuple(state['current_pos'])
    updated_state = verify(state)
    return updated_state

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)