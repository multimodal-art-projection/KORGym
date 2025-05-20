import random
import os
import math
import uuid
from copy import deepcopy
from typing import List, Tuple, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import argparse
import uvicorn
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
game_prompt="""
In the following, you are required to participate in a simplified version of the Plants vs. Zombies game. The game is played on a 5×7 board, where zombies spawn from the far right side and move one step to the left each turn. The types of plants and zombies are as follows:

Plants  
- Sunflower (X): Costs 50 sun, has 2 HP, and generates an extra 10 sun each turn.  
- Peashooter (W): Costs 100 sun, has 2 HP, and deals 1 damage each turn to the first zombie in its current row.  
- Three-Line Shooter (S): Costs 325 sun, has 2 HP, and deals 1 damage each turn to the first zombie in its current row as well as the first zombie in each of the adjacent rows.  
- Wall-nut (J): Costs 50 sun and has 10 HP.  
- Torch Stump (H): Costs 125 sun, has 2 HP; it increases the damage of the plant to its left in the same row (applied directly to the plant rather than to a projectile) by +1, and this effect can only be applied once.  
- Fire Chili (F): Costs 300 sun and eliminates all zombies in its row.

Zombies  
- Regular Zombie (N): Has 4 HP and deals 1 damage each turn to the plant that blocks its path.  
- Roadblock Zombie (R): Has 8 HP and deals 1 damage each turn to the plant that blocks its path.  
- Bucket Zombie (B): Has 12 HP and deals 1 damage each turn to the plant that blocks its path.  
- High-Attack Zombie (I): Has 6 HP and deals 3 damage each turn to the plant that blocks its path.

Rules  
- At least 25 sun is gained each turn.
- A new zombie is spawned every 5 turns.
- After every 10 turns, newly spawned zombies have their HP increased by 4, and the number of zombies spawned increases by 1.
- Your score increases by 1 each turn.
- The game lasts for a maximum of 100 turns.
- Plants cannot be placed on the same grid cell, but zombies can coexist in the same cell.
- There are no lawn mowers.
- Roadblock Zombies only spawn after turn 10, and Bucket Zombies and High-Attack Zombies only spawn after turn 20.

Please input in the format "PlantType Row Column". If multiple plants need to be planted, separate them using a semicolon (`;`).  
Example: "Answer: X 2 0;W 1 1"
{board}

"""
# 全局环境存储，用于保存各个 uid 对应的游戏实例
ENV_STORE = {}

# 常量定义
ROWS = 5
COLS = 7
PLANT_COST = {
    'X': 50,    # 向日葵
    'W': 100,   # 豌豆
    'S': 325,   # 三线豌豆
    'J': 50,    # 坚果
    'H': 125,   # 火炬
    'F': 300    # 辣椒（火爆辣椒）
}
PLANT_HEALTH = {
    'X': 2,
    'W': 2,
    'S': 2,
    'J': 10,
    'H': 2,
    'F': float("inf")
}
ZOMBIE_HEALTH_AND_ATTACK = {
    'normal': {"health": 4, "attack": 1}, 
    'roadblock': {"health": 8, "attack": 1}, 
    'barrel': {"health": 12, "attack": 1}, 
    'high': {"health": 6, "attack": 3}
}
ZOMBIE_RENDER = {
    'normal': "N",
    'roadblock': "R", 
    'barrel': "B", 
    'high': "I"
}

# PVZ 游戏类
class PVZGame:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.board = {
            "plants": {},  # {(row, col): {"type": str, "health": int}}
            "zombies": [],  # [{'type': str, 'row': int, 'col': int, 'health': int, 'attack': int}]
            "sun": 50,
            "score": 0,
            "game_over": 0,
            "total_rounds": 0
        }
        
    def generate(self) -> Dict:
        # 初始化时返回当前棋盘状态
        return self.board.copy()
    
    def verify(self, action: List[Tuple[str, int, int]]) -> Tuple[Dict, int, int]:
        # 根据玩家操作执行回合逻辑：先种植，再更新回合
        self.process_action(action)
        self.update_round()
        return self.board.copy(), self.check_game_over(), self.board["score"]
    
    def process_action(self, action: List[Tuple[str, int, int]]):
        # 每个操作均为 (plant_type, row, col)
        for plant in action:
            plant_type, row, col = plant
            # 如果该位置已有植物，则跳过
            if (row, col) in self.board["plants"]:
                # 提示：该位置已有植物
                print(f"位置 ({row}, {col}) 已有植物，请选择其他位置。")
                continue
            # 检查阳光是否足够
            if self.board["sun"] >= PLANT_COST.get(plant_type, float("inf")):
                self.board["sun"] -= PLANT_COST[plant_type]
                self.board["plants"][(row, col)] = {
                    "type": plant_type,
                    "health": PLANT_HEALTH[plant_type]
                }
            else:
                print("阳光不足！")
    
    def update_round(self):
        # 每个回合依次：清理死亡单位 -> 植物行动 -> 僵尸行动 -> 新僵尸生成 -> 基础阳光奖励
        self.cleanup()
        self.plants_action()
        self.zombies_action()
        self.generate_new_zombies()
        self.board["sun"] += 25
        self.board['total_rounds'] += 1
        self.board['score'] += 1
    
    def cleanup(self):
        self.board["zombies"] = [zombie for zombie in self.board["zombies"] if zombie["health"] > 0]
        self.board["plants"] = {pos: data for pos, data in self.board["plants"].items() if data["health"] > 0}
    
    def plants_action(self):
        self.chilli_action()
        self.sun_flower_action()
        self.peas_action()
    
    def chilli_action(self):
        # 火爆辣椒：将所在行所有僵尸清除，同时自身消失
        for pos in list(self.board["plants"].keys()):
            plant_type = self.board["plants"][pos]["type"]
            if plant_type == "F":
                row, _ = pos
                self.board["zombies"] = [zombie for zombie in self.board["zombies"] if zombie["row"] != row]
                del self.board["plants"][pos]
    
    def sun_flower_action(self):
        # 向日葵：每回合获得额外阳光
        for pos in list(self.board["plants"].keys()):
            plant_type = self.board["plants"][pos]["type"]
            if plant_type == "X":
                self.board["sun"] += 10
    
    def peas_action(self):
        # 豌豆类植物攻击：W 为单线攻击，S 为三线攻击
        peas = [(pos, data) for pos, data in self.board['plants'].items() if data["type"] in ('W', 'S')]
        for pos, data in peas:
            row, col = pos
            if data["type"] == "W":
                lines = [row]
            elif data["type"] == "S":
                lines = [max(0, row - 1), row, min(ROWS - 1, row + 1)]
            else:
                lines = []
            
            for line in lines:
                zombies_in_line = [zombie for zombie in self.board["zombies"] if zombie["row"] == line]
                if not zombies_in_line:
                    continue
                # 选择离植物最近的僵尸（即列值最小的僵尸）
                target_zombie = min(zombies_in_line, key=lambda x: x["col"])
                dmg = self.calculate_damage(line, col)
                idx = self.board["zombies"].index(target_zombie)
                target_zombie["health"] -= dmg
                if target_zombie["health"] <= 0:
                    del self.board["zombies"][idx]
                else:
                    self.board["zombies"][idx] = target_zombie
    
    def calculate_damage(self, row, col):
        base_damage = 1
        # 若在植物和僵尸之间存在火炬植物，则增加额外伤害
        for c in range(col, COLS):
            if (row, c) in self.board["plants"]:
                if self.board["plants"][(row, c)]["type"] == "H":
                    base_damage += 1
                    break
        return base_damage
    
    def zombies_action(self):
        new_zombies = []
        for zombie in self.board["zombies"]:
            row = zombie["row"]
            col = zombie["col"]
            attack = zombie["attack"]
            # 如果当前位置有植物，则攻击植物
            if (row, col) in self.board["plants"]:
                plant = self.board["plants"][(row, col)]
                new_health = plant["health"] - attack
                if new_health <= 0:
                    del self.board["plants"][(row, col)]
                else:
                    self.board["plants"][(row, col)]["health"] = new_health
                new_zombies.append(zombie)
            else:
                new_col = col - 1
                if new_col < 0:
                    # 僵尸越界，游戏结束
                    self.board["score"] = self.board["total_rounds"]
                    self.board["game_over"] = 1
                    return
                zombie["col"] = new_col
                new_zombies.append(zombie)
        self.board["zombies"] = new_zombies
    
    def generate_new_zombies(self):
        # 每 5 回合生成僵尸，每 10 回合增加一个僵尸
        round_num = self.board["total_rounds"]
        if round_num % 5 != 0:
            return
        base_count = 1 + math.floor(round_num / 10)
        for _ in range(base_count):
            row = random.randint(0, ROWS - 1)
            zombie_type = self.select_zombie_type(round_num)
            health = ZOMBIE_HEALTH_AND_ATTACK[zombie_type]["health"] + (round_num // 10) * 4
            self.board["zombies"].append({
                "type": zombie_type,
                "row": row,
                "col": COLS - 1,
                "health": health,
                "attack": ZOMBIE_HEALTH_AND_ATTACK[zombie_type]["attack"]
            })
    
    def select_zombie_type(self, round_num):
        if round_num < 10:
            return "normal"
        elif round_num < 20:
            return random.choice(["normal", "roadblock"])
        else:
            return random.choice(["normal", "roadblock", "barrel", "high"])
    
    def check_game_over(self):
        if self.board["game_over"] == 1:
            return 1
        if self.board["total_rounds"] >= 100:
            return 1
        return 0

# 将 board 数据转换为字符串表示（用于前端展示）
def board_to_string(board: dict) -> str:
    header = f"Turn:{board['total_rounds']} | Sun:{board['sun']} | Score: {board['score']}"
    grid = [['0'] * COLS for _ in range(ROWS)]
    # 放置植物
    for (r, c), plant in board["plants"].items():
        grid[r][c] = plant["type"]
    # 放置僵尸
    for zombie in board["zombies"]:
        r, c = zombie["row"], zombie["col"]
        z_char = ZOMBIE_RENDER.get(zombie["type"], "?")
        if grid[r][c] == '0':
            grid[r][c] = z_char
        else:
            grid[r][c] += z_char
    grid_str = "\n".join([f"Line{i}|" + "|".join(f"{cell:3}" for cell in row) for i, row in enumerate(grid)])
    return header + "\n\nCurrent Battlefield (X: Sunflower, W: Peashooter, S: Three-Line Shooter, J: Wall-nut, H: Torch Stump, F: Fire Chili, N: Zombie, R: Roadblock Zombie,B: Bucket Zombie,I: High-Attack Zombie)\n" + grid_str

# 接口封装

def generate(seed: int) -> dict:
    """
    生成初始游戏状态，将 PVZGame 实例存入全局 ENV_STORE 中，
    返回包含 uid、棋盘显示、得分、是否结束等信息的 item 字典。
    """
    game = PVZGame(seed)
    board = game.generate()
    uid = str(uuid.uuid4())
    item = {
        "uid": uid,
        "board": board_to_string(board),
        "score": board["score"],
        "is_end": bool(board["game_over"]) or (board["total_rounds"] >= 100),
        "action": "",
        "response": [],
        "prompt": board_to_string(board),
        "epoch": board["total_rounds"] + 1,
    }
    ENV_STORE[uid] = game
    return item

def verify(item: dict) -> dict:
    """
    根据 item 中的 action（格式如："F 3 4; X 0 2"）更新游戏状态，
    并返回更新后的 item。
    """
    uid = item.get("uid")
    if uid not in ENV_STORE:
        item["response"].append("无效的 UID")
        return item
    game = ENV_STORE[uid]
    
    action_str = item.get("action", "").strip()
    actions = []
    if action_str:
        for cmd in action_str.split(';'):
            cmd = cmd.strip()
            if not cmd:
                continue
            parts = cmd.split()
            if len(parts) != 3:
                item["response"].append("操作格式错误，应为：植物类型 行 列")
                continue
            p_type = parts[0].upper()
            try:
                row = int(parts[1])
                col = int(parts[2])
            except ValueError:
                item["response"].append("行和列必须为整数。")
                continue
            if p_type not in PLANT_COST:
                item["response"].append(f"无效的植物类型：{p_type}")
                continue
            if not (0 <= row < ROWS) or not (0 <= col < COLS):
                item["response"].append("行或列超出范围。")
                continue
            actions.append((p_type, row, col))
    
    board, game_over_flag, score = game.verify(actions)
    item["board"] = board_to_string(board)
    item["score"] = score
    item["is_end"] = bool(board["game_over"]) or (board["total_rounds"] >= 100)
    item["prompt"] = board_to_string(board)
    item["epoch"] = board["total_rounds"] + 1
    item["action"] = ""  # 执行后清空 action
    return item

def print_board(item: dict) -> dict:
    """
    返回当前棋盘的字符串表示，键名为 board。
    """
    
    return game_prompt.format(board=item.get("board", ""))



class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    board: str
    uid: str
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


# if __name__ == "__main__":
#     item = generate(1123)
#     print(print_board(item))
#     while item['is_end'] == False:
#         item['action']=input("请输入:")
#         item=verify(item)
#         print(print_board(item))


if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)