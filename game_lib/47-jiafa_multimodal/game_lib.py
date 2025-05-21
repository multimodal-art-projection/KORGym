import random
from PIL import Image, ImageDraw, ImageFont
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import ast
import argparse
import base64
import os
import colorsys

def parse_init():
    parser = argparse.ArgumentParser(description="Data creation utility")
    parser.add_argument('-p', '--port', type=int, default=8775, help='服务部署端口')
    parser.add_argument('-H', '--host', type=str, default="0.0.0.0", help='服务部署地址')
    return parser.parse_args()

app = FastAPI()

game_prompt = """
You are a good game player, I'll give you a game board which is a picture and rules.\nYour task is:\n- First, analyze the color blocks and calculate column sums.\n- Second, output the answer in the required format. The last line of your response should be: 'Answer: $YOUR_ANSWER' (e.g. "Answer: [12,3,12,-15]").

The image shows:
- Each color block represents a numerical value
- Right-side numbers are row sums
- Your task is to calculate column sums

Rules:
1. Same color represents same value
2. Different colors represent different values
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

def generate_distinct_colors(num_colors):
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        lightness = 0.5 + random.uniform(-0.1, 0.1)
        saturation = 0.7 + random.uniform(-0.1, 0.1)
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors

def save_color_board(image_path, grid, row_sums, cell_size=50):
    rows = len(grid)
    cols = len(grid[0])
    img_width = cols * cell_size + 200  # 右边留白显示行和
    img_height = rows * cell_size + 20  # 底部留白
    
    img = Image.new('RGB', (img_width, img_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # 绘制颜色块
    for y in range(rows):
        for x in range(cols):
            left = x * cell_size
            upper = y * cell_size
            right = (x+1) * cell_size
            lower = (y+1) * cell_size
            draw.rectangle([left, upper, right, lower], fill=grid[y][x])
    
    # 绘制行和
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for y in range(rows):
        text = str(row_sums[y])
        draw.text(
            (cols * cell_size + 10, y * cell_size + 15),
            text,
            font=font,
            fill=(0, 0, 0)
        )
    
    img.save(image_path)

def generate(seed):
    random.seed(seed)
    while True:
        row_count = random.randint(5, 8)
        col_count = random.randint(5, 8)
        num_colors = min(row_count, col_count) - 1
        
        # 生成唯一颜色和对应数值
        colors = generate_distinct_colors(num_colors)
        color_values = {color: random.randint(-100, 100) for color in colors}
        
        # 生成颜色矩阵
        grid = []
        for _ in range(row_count):
            row = [random.choice(colors) for _ in range(col_count)]
            grid.append(row)
        
        # 确保所有颜色都被使用
        used_colors = {c for row in grid for c in row}
        if len(used_colors) < num_colors:
            positions = [(i,j) for i in range(row_count) for j in range(col_count)]
            random.shuffle(positions)
            for color in set(colors) - used_colors:
                if positions:
                    i, j = positions.pop()
                    grid[i][j] = color
        
        # 构建系数矩阵
        A = np.array([[row.count(color) for color in colors] for row in grid])
        if np.linalg.matrix_rank(A) != num_colors:
            continue
        
        # 计算行和与列和
        row_sums = [sum(color_values[c] for c in row) for row in grid]
        col_sums = [sum(color_values[grid[i][j]] for i in range(row_count)) 
                   for j in range(col_count)]
        
        # 保存图片
        os.makedirs("cache", exist_ok=True)
        image_path = f"cache/color_board_{seed}_{random.randint(0,9999)}.png"
        save_color_board(image_path, grid, row_sums)
        
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode()
        
        return {
            "score": 1,
            "is_end": False,
            "grid": grid,
            "row_sums": row_sums,
            "col_sums": col_sums,
            "col_count": col_count,
            "color_values": color_values,
            "image_path": image_path,
            "base64_image": base64_image,
            "response": [],
            "action": "",
            "epoch": 1,
            "prompt": ""
        }

def verify(item):
    try:
        user_answer = ast.literal_eval(item["action"]) if isinstance(item["action"], str) else item["action"]
        if not isinstance(user_answer, list) or len(user_answer) != item["col_count"]:
            item["score"] = 0
        else:
            item["score"] = 1 if user_answer == item["col_sums"] else 0
    except:
        item["score"] = 0
    return item

class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    grid: list
    row_sums: list
    col_sums: list
    col_count: int
    color_values: dict
    score: float
    is_end: bool
    action: str
    response: list
    prompt: str
    epoch: int
    image_path: str
    base64_image: str

@app.post("/print_board", response_model=BoardRequest)
def api_print_board(request: GameState):
    return {"board": game_prompt}

@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    state = generate(request.seed)
    return convert_numpy_types(state)

@app.post("/verify", response_model=GameState)
def api_verify(request: GameState):
    state = request.dict()
    return convert_numpy_types(verify(state))

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)