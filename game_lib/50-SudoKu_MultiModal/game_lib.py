import random
import numpy as np
from collections import deque
from PIL import Image, ImageDraw, ImageFont
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ast
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

You are a good game player, I'll give you a game board which is a picture and rules.\nYour task is:\n- First, give your answer according to the game board and rules.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g."Answer: ['up','down','down',...]"
Next, I will provide you with a Sokoban puzzle in a picture. You need to give a solution (push all boxes into the target areas) as a list. The meaning of grid in the picture are:

- The green grid indicates the player's position.
- The red grid indicates boxes.
- The blue grid indicates target areas where boxes should be moved.
- The black grid indicates walls.

You can choose from the following movements:

- 'up' indicates moving one step upward.
- 'down' indicates moving one step downward.
- 'left' indicates moving one step to the left.
- 'right' indicates moving one step to the right.

Important rules to remember:

- The player can push only one box at a time. If two boxes are aligned in the moving direction, the player cannot push them.
- Walls block the player's movement.
- The game is won only when all boxes are pushed into target areas.
- You must output the solution as a list of strings with each move in lowercase, e.g., "Answer: ['up','down','down',...]".

"""
app = FastAPI()
# Function to encode the image
def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
# 辅助函数：递归转换 NumPy 类型到原生 Python 类型
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
# 判断 (x,y) 是否为合法移动位置（在地图范围内且不是墙）
def is_valid_move(map_data, width, height, x, y):
    return 0 <= x < width and 0 <= y < height and map_data[y][x] != "X"

# 利用 BFS 判断从 start 到 end 是否存在路径（只避开墙壁 X）
def is_path_exists(map_data, width, height, start, end):
    visited = [[False] * width for _ in range(height)]
    queue = deque([start])
    visited[start[1]][start[0]] = True
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        x, y = queue.popleft()
        if (x, y) == end:
            return True
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and not visited[ny][nx]:
                if map_data[ny][nx] != "X":
                    visited[ny][nx] = True
                    queue.append((nx, ny))
    return False

# 检查箱子是否被墙壁“夹住”
def is_box_in_corner(map_data, width, height, box_pos):
    x, y = box_pos
    # 这里简单检查左右或上下组合
    right_bottom_blocked = (map_data[y + 1][x] == 'X' and map_data[y][x - 1] == 'X') if (y + 1 < height and x - 1 >= 0) else False
    left_bottom_blocked = (map_data[y + 1][x] == 'X' and map_data[y][x + 1] == 'X') if (y + 1 < height and x + 1 < width) else False
    left_top_blocked = (map_data[y - 1][x] == 'X' and map_data[y][x - 1] == 'X') if (y - 1 >= 0 and x - 1 >= 0) else False
    right_top_blocked = (map_data[y - 1][x] == 'X' and map_data[y][x + 1] == 'X') if (y - 1 >= 0 and x + 1 < width) else False
    if right_bottom_blocked or left_bottom_blocked or left_top_blocked or right_top_blocked:
        return False
    return True

# 简单判断箱子是否可向 target 方向移动
def is_box_movable(map_data, width, height, box_pos, target_pos):
    x, y = box_pos
    tx, ty = target_pos
    directions = []
    if tx > x:
        directions.append((1, 0))
        if not is_valid_move(map_data, width, height, x - 1, y):
            return False
    elif tx < x:
        directions.append((-1, 0))
        if not is_valid_move(map_data, width, height, x + 1, y):
            return False
    if ty > y:
        directions.append((0, 1))
        if not is_valid_move(map_data, width, height, x, y - 1):
            return False
    elif ty < y:
        directions.append((0, -1))
        if not is_valid_move(map_data, width, height, x, y + 1):
            return False
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if not is_valid_move(map_data, width, height, nx, ny):
            return False
    return True

# 计算在给定箱子分布下，玩家从 start 可到达的所有位置
def get_reachable_positions(map_data, width, height, start, boxes):
    reachable = set()
    queue = deque([start])
    obstacles = set(boxes)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        x, y = queue.popleft()
        if (x, y) in reachable:
            continue
        reachable.add((x, y))
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                if map_data[ny][nx] != "X" and (nx, ny) not in obstacles:
                    queue.append((nx, ny))
    return reachable

# 利用 BFS 在状态空间中搜索：状态由 (player, tuple(sorted(boxes))) 表示，
# 判断是否可以将所有箱子推到目标位置上
def is_solvable_state(map_data, width, height, target_positions, player, boxes):
    initial_state = (player, tuple(sorted(boxes)))
    visited = set([initial_state])
    queue = deque([initial_state])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        cur_player, cur_boxes = queue.popleft()
        if set(cur_boxes) == set(target_positions):
            return True
        reachable = get_reachable_positions(map_data, width, height, cur_player, cur_boxes)
        for box in cur_boxes:
            bx, by = box
            for dx, dy in directions:
                push_from = (bx - dx, by - dy)
                if push_from in reachable:
                    new_box = (bx + dx, by + dy)
                    if (0 <= new_box[0] < width and 0 <= new_box[1] < height and
                        map_data[new_box[1]][new_box[0]] != "X" and new_box not in cur_boxes):
                        new_player = box  # 玩家推完箱子后来到箱子原处
                        new_boxes = list(cur_boxes)
                        new_boxes.remove(box)
                        new_boxes.append(new_box)
                        new_state = (new_player, tuple(sorted(new_boxes)))
                        if new_state not in visited:
                            visited.add(new_state)
                            queue.append(new_state)
    return False

# 生成地图，并保证生成的关卡有解
def generate(seed, width=8, height=8):
    random.seed(seed)
    n = random.randint(1,3)
    while True:
        # 初始化地图：全部置为 'E'
        map_data = [['E' for _ in range(width)] for _ in range(height)]
        positions = set()
        # 随机生成玩家位置（不在边界上）
        player_pos = (random.randint(1, width - 2), random.randint(1, height - 2))
        positions.add(player_pos)
        # 生成 n 个箱子的位置
        boxes = []
        for _ in range(n):
            pos = (random.randint(1, width - 2), random.randint(1, height - 2))
            while pos in positions:
                pos = (random.randint(1, width - 2), random.randint(1, height - 2))
            positions.add(pos)
            boxes.append(pos)
        # 生成 n 个目标位置
        targets = []
        for _ in range(n):
            pos = (random.randint(1, width - 2), random.randint(1, height - 2))
            while pos in positions:
                pos = (random.randint(1, width - 2), random.randint(1, height - 2))
            positions.add(pos)
            targets.append(pos)
        # 将玩家、箱子和目标放入地图
        map_data[player_pos[1]][player_pos[0]] = 'I'
        for box in boxes:
            map_data[box[1]][box[0]] = 'B'
        for target in targets:
            map_data[target[1]][target[0]] = 'T'
        # 随机生成墙壁：空地以 20% 的概率变成墙壁
        for y in range(height):
            for x in range(width):
                if map_data[y][x] == 'E' and random.random() < 0.2:
                    map_data[y][x] = 'X'
        # 保证边界全部为墙
        for x in range(width):
            map_data[0][x] = 'X'
            map_data[height - 1][x] = 'X'
        for y in range(height):
            map_data[y][0] = 'X'
            map_data[y][width - 1] = 'X'
        # 检查连通性：玩家必须能到达所有箱子；每个箱子至少能通往某个目标且箱子不被卡住
        if not all(is_path_exists(map_data, width, height, player_pos, box) for box in boxes):
            continue
        valid = True
        for box in boxes:
            if not any(is_path_exists(map_data, width, height, box, target) and is_box_movable(map_data, width, height, box, target)
                       for target in targets):
                valid = False
                break
            if not is_box_in_corner(map_data, width, height, box):
                valid = False
                break
        if not valid:
            continue
        # 调用求解器判断关卡是否有解
        if is_solvable_state(map_data, width, height, targets, player_pos, boxes):
            # 重新将箱子标记到地图上
            for box in boxes:
                map_data[box[1]][box[0]] = 'B'

            # 返回状态字典
            state = {
                'score': 0,
                'is_end': False,
                'response': [],
                'prompt': '',
                'action': '',
                'epoch': 1,
                "width": width,
                "height": height,
                "n": n,
                "map_data": map_data,
                "target_positions": targets,
                "player_pos": player_pos,
                "box_positions": boxes
            }
            os.makedirs('cache', exist_ok=True)
            output_img_path = f"cache/game_map_{random.randint(0,9999)}.png"
            save_map_as_image(state, output_img_path)
            state['image_path'] = output_img_path
            state['base64_image'] = encode_image(output_img_path)
            print(state['base64_image'])
            return state

# 打印地图，叠加显示目标、箱子和玩家（防止目标被覆盖）
def print_board(state):
    return game_prompt
# 将地图保存为图片
def save_map_as_image(state, filename="game_map.png"):
    width = state["width"]
    height = state["height"]
    map_data = state["map_data"]
    player_pos = state["player_pos"]
    box_positions = state["box_positions"]
    target_positions = state["target_positions"]
    cell_size = 50
    img_width = width * cell_size
    img_height = height * cell_size
    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except IOError:
        font = ImageFont.load_default()
    color_map = {
        'I': (0, 255, 0),
        'B': (255, 0, 0),
        'T': (0, 0, 255),
        'X': (0, 0, 0),
        'E': (255, 255, 255)
    }
    display_map = [row[:] for row in map_data]
    for t in target_positions:
        x, y = t
        if (x, y) != player_pos and (x, y) not in box_positions:
            display_map[y][x] = 'T'
    for bx, by in box_positions:
        display_map[by][bx] = 'B'
    px, py = player_pos
    display_map[py][px] = 'I'
    for y in range(height):
        for x in range(width):
            cell = display_map[y][x]
            color = color_map.get(cell, (255, 255, 255))
            draw.rectangle([x * cell_size, y * cell_size, (x + 1) * cell_size, (y + 1) * cell_size], fill=color)
            draw.rectangle([x * cell_size, y * cell_size, (x + 1) * cell_size, (y + 1) * cell_size], outline="black", width=2)
            if cell in "IBT":
                text = cell
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                text_x = x * cell_size + (cell_size - text_width) / 2
                text_y = y * cell_size + (cell_size - text_height) / 2
                draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))
    img.save(filename)
    print(f"地图已经保存为图片：{filename}")

# 玩家移动：若目标位置为箱子，则尝试推动箱子
def move_player(state, direction):
    map_data = state["map_data"]
    width = state["width"]
    height = state["height"]
    player_pos = state["player_pos"]
    box_positions = state["box_positions"]
    dx, dy = direction
    old_px, old_py = player_pos
    new_px = old_px + dx
    new_py = old_py + dy
    if is_valid_move(map_data, width, height, new_px, new_py):
        if (new_px, new_py) in box_positions:
            box_index = box_positions.index((new_px, new_py))
            bx, by = box_positions[box_index]
            new_bx = bx + dx
            new_by = by + dy
            if is_valid_move(map_data, width, height, new_bx, new_by) and (new_bx, new_by) not in box_positions:
                map_data[by][bx] = 'T' if (bx, by) in state["target_positions"] else 'E'
                box_positions[box_index] = (new_bx, new_by)
                map_data[new_by][new_bx] = 'B'
                map_data[old_py][old_px] = 'T' if (old_px, old_py) in state["target_positions"] else 'E'
                state["player_pos"] = (new_px, new_py)
                map_data[new_py][new_px] = 'I'
        else:
            map_data[old_py][old_px] = 'T' if (old_px, old_py) in state["target_positions"] else 'E'
            state["player_pos"] = (new_px, new_py)
            map_data[new_py][new_px] = 'I'

# 判断是否胜利：所有箱子都在目标上
def check_victory(state):
    return set(state["box_positions"]) == set(state["target_positions"])

# 根据动作序列移动，途中若胜利则返回 True
def verify(state):
    try:
        # 如果item['action']为字符串，尝试转换为列表
        if isinstance(state['action'], str):
            move_sequence = ast.literal_eval(state['action'])
        else:
            move_sequence = state['action']
    except Exception as e:
        # 转换失败
        state['score'] = 0
        return state
    for move in move_sequence:
        if move == "up":
            direction = (0, -1)
        elif move == "down":
            direction = (0, 1)
        elif move == "left":
            direction = (-1, 0)
        elif move == "right":
            direction = (1, 0)
        else:
            continue
        move_player(state, direction)
        if check_victory(state):
            state['score'] = 1
            return state
    state['score'] = 0
    return state

# def main():
#     n = 3  # 箱子数量（同时目标数量为 n）
#     width, height = 7, 7
#     state = generate(2)
#     print(state)
#     save_map_as_image(state, "game_map.png")
#     while True:
#         print(print_board(state))
#         move_sequence = input("请输入完整的动作序列 (up, down, left, right)，用空格分隔：").strip().split()
#         valid_moves = {"up", "down", "left", "right"}
#         if all(move in valid_moves for move in move_sequence):
#             if verify(state):
#                 print("恭喜！你成功将所有箱子推到目标位置。")
#                 print("动作序列:", move_sequence)
#                 break
#             else:
#                 print("箱子未能全部推到目标位置，请继续尝试。")
#         else:
#             print("无效的动作序列，请确保输入的是 up, down, left, right。")
class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    width: int
    height: int
    n: int
    map_data: list
    target_positions: list
    image_path: str
    base64_image: str
    player_pos: tuple
    box_positions: list
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
    state['player_pos'] = tuple(state['player_pos'])
    state['target_positions'] = [tuple(coord) for coord in state['target_positions']]
    state['box_positions'] = [tuple(coord) for coord in state['box_positions']]
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
    state['player_pos'] = tuple(state['player_pos'])
    state['target_positions'] = [tuple(coord) for coord in state['target_positions']]
    state['box_positions'] = [tuple(coord) for coord in state['box_positions']]
    updated_state = verify(state)
    updated_state = convert_numpy_types(updated_state)
    return updated_state

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)