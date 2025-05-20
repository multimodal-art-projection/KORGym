import sys
import random 
from copy import deepcopy
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Circle
import math
from datetime import datetime
import os
import base64
import numpy as np
import random
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid  # 用于生成唯一标识符
from typing import Optional
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
game_prompt = """
You are a good game player, I'll give you a game board which is a picture and rules.\nYour task is:\n- First, give your answer according to the game board and rules.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g."Answer: [(1,'A'),(2,'D'),...]"
In the board shown, the red solid circles represent cats, the white circles represent empty spaces, the black solid circles represent walls, and the blue hollow circles represent exits. The cat's goal is to run to the exit and escape, while your goal is to trap the cat with walls. The cat and you take turns; on each turn, the cat can move one cell to an adjacent empty space, and on your turn you can create a wall on an empty space, but not on an exit. If the cat reaches the exit, you lose; if you successfully trap the cat, you win. You need to output a set of coordinates each turn to indicate where you build a wall, e.g. "Answer: 7 6".
"""
# 全局字典，用于存储 env 对象，避免通过 JSON 传输
ENV_STORE = {}
# Function to encode the image
def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
class HexBoardGame:
    def __init__(self, size=11):
        self.size = size
        self.cat = None
        self.walls = set()
        self.board = None
        self.session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"./cache/game_session_{self.session_time}"
        self.round_counter = 0
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(self, seed=None):
        
        random.seed(seed)
        self.board = [['0' for _ in range(self.size)] for _ in range(self.size)]
        center = self.size // 2
        self.cat = (center, center)
        self.board[center][center] = 'C'
        
        # Generate random walls (15% density)
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) == self.cat:
                    continue
                if random.random() < 0.15:
                    self.board[i][j] = '1'
                    self.walls.add((i, j))
        return self.board
    
    def get_neighbors(self, pos):
        x, y = pos
        neighbors = []
        if x % 2 == 1:  # Odd row
            offsets = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
        else:  # Even row
            offsets = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]
        
        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                neighbors.append((nx, ny))
        return neighbors
    
    def is_boundary(self, pos):
        x, y = pos
        return x == 0 or x == self.size-1 or y == 0 or y == self.size-1
    
    def calculate_distance(self, pos, temp_walls):
        visited = set()
        q = deque([(pos[0], pos[1], 0)])
        visited.add(pos)
        
        while q:
            x, y, dist = q.popleft()
            if self.is_boundary((x, y)):
                return dist+1
            for (nx, ny) in self.get_neighbors((x, y)):
                if (nx, ny) not in visited and (nx, ny) not in temp_walls and (nx, ny) != self.cat:
                    visited.add((nx, ny))
                    q.append((nx, ny, dist + 1))
        return float('inf')
    
    def find_best_cat_move(self):
        current_pos = self.cat
        best_move = None
        min_max_distance = float('inf')
        is_possible = False
        
        neighbors = self.get_neighbors(current_pos)
        valid_moves = []
        for n in neighbors:
            if self.board[n[0]][n[1]] == '0':
                valid_moves.append(n)
        if not valid_moves:
            return None  # Cat is trapped
        
        for move in valid_moves:
            is_possible_move = False
            # Simulate cat move
            temp_walls = set(self.walls)
            temp_cat = move
            if self.is_boundary(move):
                return move  # Immediate escape
            
            # Evaluate all possible player responses
            max_distance = 0
            # Get all possible wall placements
            possible_walls = [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i][j] == '0' and (i, j) != temp_cat]
            
            if not possible_walls:
                # No walls can be placed, use current state
                curr_dist = self.calculate_distance(temp_cat, temp_walls)
                if curr_dist < min_max_distance:
                    min_max_distance = curr_dist
                    best_move = move
                continue
            
            for wall in possible_walls:
                new_walls = temp_walls | {wall}
                dist = self.calculate_distance(temp_cat, new_walls)
                if dist > max_distance:
                    max_distance = dist
                if dist < max_distance:
                    is_possible_move = True
            if is_possible_move == True:
                is_possible = True
            if max_distance < min_max_distance:
                min_max_distance = max_distance
                best_move = move
        if best_move != None:
            return best_move
        elif best_move == None and is_possible == True:
            return valid_moves[0]
        else:
            return best_move    
    
    def verify(self, board, action):
        #return: board, score(0 if player loss, 1 if player win , -1 if not the end of game), is_end
        try:
            x, y = map(int, action.strip().split())
        except:
            return board, 0, 0  # Invalid action
        if not (0 <= x < self.size and 0 <= y < self.size):
            return board, 0, 0
        
        if board[x][y] != '0':
            return board, 0, 0
        # Update board with wall
        new_board = deepcopy(board)
        new_board[x][y] = '1'
        self.walls.add((x, y))
        self.board = new_board
        # Cat's turn
        cat_move = self.find_best_cat_move()
        if cat_move is None:
            self.board = new_board
            return new_board, 1, 1
        if self.is_boundary(cat_move):
            self.board = new_board
            return new_board, 0, 1
        
        cx, cy = self.cat
        new_board[cx][cy] = '0'
        new_board[cat_move[0]][cat_move[1]] = 'C'
        self.cat = cat_move
        
        if self.is_boundary(cat_move):
            self.board = new_board
            return new_board, 0, 1
        
        neighbors = self.get_neighbors(cat_move)
        trapped = all(new_board[nx][ny] == '1' for (nx, ny) in neighbors)
        if trapped:
            return new_board, 1, 1
        self.board = new_board
        return new_board, 0, 0

    def render(self):
        plt.close()
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')
        hex_radius = 0.4 
        circle_radius = 0.30  
        hex_orientation = math.radians(30)
        max_x = (self.size - 0.5) * math.sqrt(3) * hex_radius
        max_y = (self.size - 1) * 1.5 * hex_radius
        plt.xlim(-hex_radius*2, max_x + hex_radius*2)
        plt.ylim(-hex_radius*2, max_y + hex_radius*2)
        ax.axis('off')

        for i in range(self.size):
            for j in range(self.size):
                x = j * math.sqrt(3) * hex_radius
                if i % 2 == 1:
                    x -= math.sqrt(3) * hex_radius / 2
                y = i * 1.5 * hex_radius
                cell = self.board[i][j]
                is_boundary = self.is_boundary((i, j))

                facecolor = 'white'
                edgecolor = 'blue' if is_boundary else 'black'
                textcolor = 'black'
                
                if cell == '1':
                    facecolor = '#808080'  # 墙壁
                elif cell == 'C':
                    facecolor = '#ff0000'  # 猫的位置
                    textcolor = 'white'
                circle = Circle(
                    (x, y),
                    radius=circle_radius,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=1.8 if is_boundary else 0.8
                )
                ax.add_patch(circle)
                ax.text(
                    x, y,
                    f'{i},{j}',
                    ha='center',
                    va='center',
                    color=textcolor,
                    fontsize=8,
                    weight='bold' if is_boundary else 'normal'
                )
                if cell == 'C':
                    ax.plot(x, y, 'wo', markersize=6, alpha=0.8)
        plt.title('Hex Cat Game (Circular Cells)', pad=20)
        plt.tight_layout()
        plt.draw()
        filename = os.path.join(self.output_dir, f"round_{self.round_counter:03d}.png")
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        self.round_counter += 1
        plt.show(block=False)  # 非阻塞显示
        plt.pause(0.1)  # 允许图像更新
        return filename

def generate(seed):
    item = {
        'score': 0,
        'is_end': False,
        'response': [],
        'prompt': '',
        'action': '',
        'epoch': 1,
    }
    uid = str(uuid.uuid4())
    item["uid"] = uid
    game = HexBoardGame()
    ENV_STORE[uid] = game
    item['board'] = game.generate(seed)
    item['base64_image']=encode_image(game.render())
    return item

def verify(item):
    game = ENV_STORE[item["uid"]]
    new_board, score, is_end = game.verify(item['board'], item['action'])
    item['board'] = new_board
    item['score'] = score
    item['is_end'] = is_end
    item['epoch'] += 1
    item['base64_image']=encode_image(game.render())
    return item
def print_board(item):
    return game_prompt
class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    board: list
    uid: str
    score: int
    is_end: bool
    action: str
    base64_image : str
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
    return game_state

# 根据动作更新游戏状态
@app.post("/verify", response_model=GameState)
def api_verify(request: GameState):
    # 从请求中获取游戏状态，并设置新的动作
    state = request.dict()
    updated_state = verify(state)
    return updated_state

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)
# if __name__ == "__main__":
#     item = generate(42)
    
#     while True:
#         item['action'] = input("Enter wall position (x y): ")
#         item = verify(item)
        
#         if item['is_end']:
#             print(item['board'])
#             print("Game Over! Score:", item['score'])
#             plt.close()
#             break
