import random
import os
from typing import List, Tuple, Dict
import math
ROWS = 5
COLS = 7
PLANT_COST = {
    'X': 50, 'W': 100, 'S': 325, 'J': 50, 'H': 125, 'F': 300
}
PLANT_HEALTH = {
    'X': 2, 'W': 2, 'S': 2, 'J': 10, 'H': 2, 'F': float("inf")
}
ZOMBIE_HEALTH_AND_ATTACK = {
    'normal': {"health":4, "attack":1}, 
    'roadblock': {"health":8, "attack":1}, 
    'barrel': {"health":12,"attack":1}, 
    'high': {"health":6,"attack":3}
}
ZOMBIE_RENDER = {
    'normal': "N",
    'roadblock': "R", 
    'barrel': "B", 
    'high': "I"
}

class PVZGame:
    def __init__(self, seed=None):
        if seed != None:
            random.seed(seed)
        self.board = {
            "plants": {}, #{(row,col):{"type":str, "health":int }}
            "zombies": [], #[{'type':str, 'row':int, 'col':int, 'health':int, 'attack':int}]
            "sun": 50,
            "score": 0,
            "game_over": 0,
            "total_rounds": 0
        }
    def generate(self) -> Dict:
        return self.board.copy()
    
    def verify(self, action: List[Tuple[str, int, int]]) -> Tuple[Dict, int, int]:
        self.process_action(action)
        self.update_round()
        return self.board.copy(), self.check_game_over(), self.board["score"]
    
    def process_action(self, action: List[Tuple[str, int, int]]):
        # 种植物
        for plant in action:
            plant_type, row, col = plant
            
            if (row,col) in self.board["plants"]:
                print("This place got a plant! Change another pos!!!!")
                continue
            
            if self.board["sun"] >= PLANT_COST.get(plant_type, float("inf")):
                self.board["sun"] -= PLANT_COST[plant_type]
                self.board["plants"][(row, col)] = {
                    "type": plant_type,
                    "health": PLANT_HEALTH[plant_type]
                }
            else:
                print("No sun left!")
    def update_round(self):
        # 流程：清除死亡单位 -> 植物行动（向日葵 -> 火爆辣椒 -> 豌豆） -> 僵尸移动与攻击 -> 僵尸生成  -> 基础阳光奖励
        self.cleanup()
        
        self.plants_action()
        
        self.zombies_action()
        
        self.generate_new_zombies()
        
        self.board["sun"] += 25
        
        self.board['total_rounds'] += 1
        self.board['score'] += 1
        
        return
        
        
    def cleanup(self):
        self.board["zombies"] = [zombie for zombie in self.board["zombies"] if zombie["health"] > 0]
        self.board["plants"] = {pos: data for pos, data in self.board["plants"].items() if data["health"] > 0}
    
    def plants_action(self):
        self.chilli_action()
        self.sun_flower_action()
        self.peas_action()
    
    def chilli_action(self):
        for pos in list(self.board["plants"].keys()):
            plant_type = self.board["plants"][pos]["type"]
            if plant_type == "F":
                row, col = pos
                self.board["zombies"] = [zombie for zombie in self.board["zombies"] if zombie["row"] != row]
                del self.board["plants"][pos]
    
    def sun_flower_action(self):
        for pos in list(self.board["plants"].keys()):
            plant_type = self.board["plants"][pos]["type"]
            if plant_type == "X":
                self.board["sun"] += 25
    
    def peas_action(self):
        peas = [(pos, data) for pos, data in self.board['plants'].items() if data["type"] in ('W', 'S')]
        for pea in peas:
            # 攻击路线
            row, col = pea[0]
            if pea[1]["type"] == "W":
                lines = [row]
            elif pea[1]["type"] == "S":
                lines = [max(0, row-1), row, min(ROWS-1, row+1)]
            
            for line in lines:
                zombies_in_line = [zombie for zombie in self.board["zombies"] if zombie["row"] == line]
                
                if len(zombies_in_line) == 0:
                    continue
                
                target_zombie = min(zombies_in_line, key=lambda x: x["col"])
                dmg = self.calculate_damage(line, col)
                idx = self.board["zombies"].index(target_zombie)
                new_health = target_zombie["health"] - dmg
                
                target_zombie["health"] = new_health
                
                if new_health <= 0:
                    del self.board["zombies"][idx]
                else:
                    self.board["zombies"][idx] = {
                        "type": target_zombie["type"], 
                        "row": target_zombie["row"], 
                        "col": target_zombie["col"], 
                        "health": target_zombie["health"],
                        "attack": target_zombie["attack"]
                    }
    def calculate_damage(self, row, col):
        base_damage = 1
        for col in range(col,COLS):
            if (row, col) in self.board["plants"]:
                if self.board["plants"][(row,col)]["type"] == "H":
                    base_damage += 1
                    break
        return base_damage
    
    def zombies_action(self):
        
        new_zombies = []
        for zombie in self.board["zombies"]:
            zombie_type = zombie["type"]
            row = zombie["row"]
            col = zombie["col"]
            health = zombie["health"]
            attack = zombie["attack"]
            
            # 首先看现在僵尸所在的位置有没有植物，如果有那就在这吃
            if (row, col) in self.board["plants"]:
                plant = self.board["plants"][(row,col)]
                new_health = plant["health"] - attack
                if new_health <= 0:
                    del self.board["plants"][(row,col)]
                else:
                    self.board["plants"][(row,col)] = {
                        "type": plant["type"],
                        "health": new_health
                    }
                new_zombies.append({
                    "type": zombie_type,
                    "row": row,
                    "col": col,
                    "health": health,
                    "attack": attack
                    })
            #没有，就往前走一步
            else:
                new_col = col - 1
                if new_col < 0:
                    self.board["score"] = self.board["total_rounds"]
                    self.board["game_over"] = 1
                    return
                new_zombies.append({
                    "type": zombie_type,
                    "row": row,
                    "col": new_col,
                    "health": health,
                    "attack": attack
                    })
        self.board["zombies"] = new_zombies
        
        
    def generate_new_zombies(self):
        # 规则：每五回合生成一只，生成数量每十回合加一只
        round_num = self.board["total_rounds"]
        need_generate = round_num % 5
        print(need_generate)
        if need_generate == 0:
            base_count = 1 + math.floor(round_num / 10)
        else:
            return
        for _ in range(base_count):
            row = random.randint(0, ROWS - 1)
            zombie_type = self.select_zombie_type(round_num)
            health = ZOMBIE_HEALTH_AND_ATTACK[zombie_type]["health"] + (round_num//10) * 4
            self.board["zombies"].append({
                "type": zombie_type,
                "row": row,
                "col": COLS -1,
                "health": health,
                "attack": ZOMBIE_HEALTH_AND_ATTACK[zombie_type]["attack"]
            })
    def select_zombie_type(self, round_num):
        if round_num < 10:
            return "normal"
        elif round_num < 20:
            return random.choice(["normal","roadblock"])
        else:
            return random.choice(["normal","roadblock","barrel","high"])
    def check_game_over(self):
        if self.board["game_over"] == 1:
            return 1
        if self.board["total_rounds"] == 100:
            return 1
        return 0



# 初始化PVZGame之后，generate会先初始化一个棋盘，然后Verify可以每轮更新状态；与文档要求一致
# 下面是一个简单的前端，喂给模型的输出格式可能需要在Generate中做自行修改
# 有bug请联系：zjtpku(juntingzhou@stu.pku.edu.cn)

if __name__ == "__main__":
    
    game = PVZGame()
    board = game.generate()

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"回合：{board['total_rounds']} | 阳光：{board['sun']} | 得分：{board['score']}")
        grid = [['0']*COLS for _ in range(ROWS)]
        for (r,c), plant in board['plants'].items():
            grid[r][c] = plant['type']
        for z in board['zombies']:
            r, c = z['row'], z['col']
            zombie_type = ZOMBIE_RENDER[z["type"]]
            if grid[r][c] == '0':
                grid[r][c] = zombie_type
            else:
                grid[r][c] += zombie_type
        
        print("\n当前战场（X:向日葵 W:豌豆 S:三线 J:坚果 H:火炬 F:辣椒 Z:僵尸）")
        for i, row in enumerate(grid):
            print(f"行{i}|" + "|".join(f"{cell:3}" for cell in row))
        
        if board['game_over'] or board['total_rounds'] >= 100:
            print("\n游戏结束！最终得分：", board['score'])
            break
            
        print("\n输入要种植的植物（格式：植物类型 行 列，多个用分号分隔）")
        action_str = input("你的操作：")
        
        # 解析操作
        actions = []
        for cmd in action_str.split(';'):
            if not cmd.strip():
                continue
            parts = cmd.upper().split()
            if len(parts) != 3:
                raise ValueError
            p_type, row, col = parts[0], int(parts[1]), int(parts[2])
            if p_type not in PLANT_COST:
                raise ValueError
            if not (0 <= row < ROWS) or not (0 <= col < COLS):
                raise ValueError
            actions.append((p_type, row, col))

        # 执行回合
        new_board, is_end, score = game.verify(actions)
        board = new_board
