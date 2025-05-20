from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
import copy
import uvicorn
from PIL import Image, ImageDraw, ImageFont
import os
import base64
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
game_prompt = """
You are a good game player, I'll give you a game board and rules.
Your task is:
- First, give your answer according to the game board and rules.
- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question, e.g., 'Answer: DB'
Next, I'll provide a game board(a picture) which contains several tubes and balls of several colors inside the tubes. Your goal is to move the balls among the tubes so that three tubes each contain exactly four balls of the same color.Additionally, the ball being moved must either match the color of the ball at the top of the target tube or the target tube must be empty. You need to provide two letters to indicate moving the top ball from one tube onto the top of another tube. For example, 'Answer: DC' means moving the top ball from tube D onto the top of tube C.
{board}
"""
app = FastAPI()

# -------------------- Ball Sort Puzzle 逻辑 --------------------
CAPACITY = 4  # 每个管子最多容纳 4 个球
# Function to encode the image
def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
        
def num_colors_for_level(level: int) -> int:
    """
    根据难度等级返回颜色数（例如：level=1 返回 3）
    """
    return level + 2

def generate_puzzle(level: int, seed: int = None) -> list:
    """
    生成初始球排列：
      - 颜色数 = level + 2
      - 管子总数 = 颜色数 + 2（前 n 个管子填满球，后 2 个管子为空）
    """
    if seed is not None:
        random.seed(seed)
    n = num_colors_for_level(level)
    total_tubes = n + 2

    # 生成每种颜色各4个球
    balls = []
    for color in range(1, n + 1):
        balls.extend([color] * CAPACITY)
    random.shuffle(balls)

    state = []
    idx = 0
    # 前 n 个管子填满球
    for _ in range(n):
        tube = balls[idx: idx + CAPACITY]
        idx += CAPACITY
        state.append(tube)
    # 后 2 个管子为空
    for _ in range(2):
        state.append([0] * CAPACITY)
    return state

def move_ball(state: list, src, dst) -> bool:
    """
    尝试将 src 管顶端的球移动到 dst 管：
      - src 和 dst 可为字母（例如 'A', 'B'）或索引
      - 移动要求：目标管若非空，其顶球颜色必须与待移动球相同；且目标管未满
    """
    label_map = {chr(65 + i): i for i in range(len(state))}
    if isinstance(src, str):
        src = label_map.get(src.upper(), -1)
    if isinstance(dst, str):
        dst = label_map.get(dst.upper(), -1)
    if not (0 <= src < len(state) and 0 <= dst < len(state)):
        return False

    src_tube = state[src]
    dst_tube = state[dst]

    # 找到 src 管的顶端球
    src_top = -1
    for i in range(CAPACITY - 1, -1, -1):
        if src_tube[i] != 0:
            src_top = i
            break
    if src_top == -1:
        return False  # 源管为空

    ball = src_tube[src_top]
    dst_count = sum(1 for x in dst_tube if x != 0)
    if dst_count >= CAPACITY:
        return False  # 目标管已满

    # 如果目标管非空，其顶端球颜色必须与 ball 相同
    if dst_count > 0:
        dst_top = -1
        for i in range(CAPACITY - 1, -1, -1):
            if dst_tube[i] != 0:
                dst_top = i
                break
        if dst_top == -1 or dst_tube[dst_top] != ball:
            return False
        place_index = dst_top + 1
    else:
        place_index = 0

    # 执行移动
    src_tube[src_top] = 0
    dst_tube[place_index] = ball
    return True

def is_solved(state: list) -> bool:
    """
    如果每个非空管子均被同一种颜色填满，则认为谜题已解决
    """
    for tube in state:
        if all(x == 0 for x in tube):
            continue
        if any(x == 0 for x in tube):
            return False
        if len(set(tube)) != 1:
            return False
    return True

def is_stuck(state: list) -> bool:
    """
    如果没有任何合法移动，则返回 True
    """
    for i, tube in enumerate(state):
        top_idx = -1
        for j in range(CAPACITY - 1, -1, -1):
            if tube[j] != 0:
                top_idx = j
                break
        if top_idx == -1:
            continue  # 空管
        ball = tube[top_idx]
        for k, dst_tube in enumerate(state):
            if k == i:
                continue
            dst_count = sum(1 for x in dst_tube if x != 0)
            if dst_count < CAPACITY:
                if dst_count == 0:
                    return False
                dst_top_idx = -1
                for z in range(CAPACITY - 1, -1, -1):
                    if dst_tube[z] != 0:
                        dst_top_idx = z
                        break
                if dst_top_idx != -1 and dst_tube[dst_top_idx] == ball:
                    return False
    return True

def print_board(item: dict) -> str:
    """
    根据 item 中的状态返回当前游戏的文本描述：
      显示内容包括：难度等级、回合数、移动次数以及各管子的状态（下标 0 为底部，最高位为顶部）
    """
    state = item.get("state", [])
    level = item.get("level", 1)
    epoch = item.get("epoch", 1)

    return game_prompt.format(board=f"Epoch: {epoch} ")

def generate(seed: int) -> dict:
    """
    根据给定的 seed 和 level 生成初始游戏状态，返回 item 字典
    """
    random.seed(seed)
    level=random.randint(1,4)
    state = generate_puzzle(level, seed)
    item = {
        "seed": seed,
        "epoch": 1,
        "level": level,
        "state": state,
        "is_end": False,
        "prompt": "",
        "action": "",
        "score" : 0
    }
    item['base64_image']=render_state_to_image(item)
    return item

def render_state_to_image(item):
    state = item['state']
    margin = 50
    spacing = 100
    tube_width = 30
    tube_height = 120
    circle_r = 12

    total_tubes = len(state)
    width = margin * 2 + (total_tubes - 1) * spacing + tube_width
    height = 300

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for i, tube in enumerate(state):
        x_left = margin + i * spacing
        y_top = (height - tube_height) // 2
        x_right = x_left + tube_width
        y_bottom = y_top + tube_height
        draw.rectangle([x_left, y_top, x_right, y_bottom], outline=(0,0,0), width=2)
        label = chr(65 + i)
        draw.text((x_left, y_top - 15), label, fill=(0,0,0), font=font)

        slot_h = tube_height / CAPACITY
        for slot_i in range(CAPACITY):
            ball_color_id = tube[slot_i]
            if ball_color_id != 0:
                color_map = {
                    1: (255,0,0),
                    2: (0,255,0),
                    3: (0,0,255),
                    4: (255,255,0),
                    5: (255,0,255),
                    6: (0,255,255),
                }
                fill_color = color_map.get(ball_color_id, (128,128,128))
                cx = x_left + tube_width / 2
                cy = y_bottom - slot_h * (slot_i + 0.5)
                draw.ellipse([(cx - circle_r, cy - circle_r), (cx + circle_r, cy + circle_r)], fill=fill_color, outline=(0,0,0))

    os.makedirs("cache", exist_ok=True)
    file_path = f"cache/board_{item['seed']}_{item['epoch']}_{random.randint(0,100000)}.png"

    img.save(file_path)
    return encode_image(file_path)
    # print(f"Saved board image to {file_path}")

def verify(item: dict) -> dict:
    """
    根据 item 中的 action 更新游戏状态：
      - action 格式要求为形如 "AD" 或 "A D" 的字符串（表示将 A 管顶端的球移动到 D 管）
      - 更新后检查是否已解决谜题或无合法移动
    """
    item["epoch"] += 1
    # 解析并规范用户输入的移动指令
    action = item.get("action", "").replace(" ", "").upper()
    if len(action) != 2:
        return item

    src, dst = action[0], action[1]
    if not move_ball(item["state"], src, dst):
        return item

    # 检查是否已解决谜题或没有更多合法移动
    if is_solved(item["state"]):
        item["is_end"] = True
        item['score'] = 1
    elif is_stuck(item["state"]):
        item["is_end"] = True
        item['score'] = 0
    item['base64_image']=render_state_to_image(item)
    
    return item

# -------------------- FastAPI 数据模型 --------------------
class GenerateRequest(BaseModel):
    seed: int

class BoardRequest(BaseModel):
    board: str

class GameState(BaseModel):
    seed: int
    epoch: int
    level: int
    state: list
    is_end: bool
    prompt: str
    action: str
    score: int
    base64_image: str
# -------------------- FastAPI 接口路由 --------------------
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
    state = request.dict()
    updated_state = verify(state)
    return updated_state

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)
# if __name__ == "__main__":
#     print("=== Ball Sort Puzzle 游戏 ===")
#     try:
#         seed_input = int(input("请输入种子 (seed): "))
#     except ValueError:
#         seed_input = 42
#         print("输入无效，使用默认种子 42。")
    
#     item = generate(seed_input)
#     render_state_to_image(item['state'])
#     print(print_board(item))
    
#     while not item["is_end"]:
#         cmd = input("请输入移动指令 (例如 'AD' 表示将 A 管的顶球移动到 D 管): ")
#         item["action"] = cmd
#         item = verify(item)
#         render_state_to_image(item['state'])
#         print(print_board(item))
    
#     print("游戏结束。")