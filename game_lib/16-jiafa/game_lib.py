import random
from PIL import Image, ImageDraw, ImageFont
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid  # 用于生成唯一标识符
from typing import Optional
import numpy as np
import ast
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
You are a good game player, I'll give you a game board and rules.\nYour task is:\n- First, give your answer according to the game board and rules.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g."Answer: [12,3,12,-15]".
Given a rectangular grid that contains several symbols, where each symbol represents a numerical value, and provided with the sum of the elements in each row, you need to compute the sum of the elements in each column and output the result as a list, e.g., "Answer: [12,3,12,-15]".
Grid:
{board}
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
def print_board(item):
    """打印生成的矩阵、每行的和以及符号对应的数值"""
    grid = item['grid']
    row_sums = item['row_sums']
    col_sums = item ['col_sums'] 
    symbol_values = item ['symbol_values']
    output = ""
    for i, row in enumerate(grid):
        output = output + ''.join(row) + f" {row_sums[i]}" + '\n'
    return game_prompt.format(board = output)
    
    # print("\nSymbol Values:")
    # for symbol, value in symbol_values.items():
    #     print(f"{symbol}: {value}")

    # print("\nColumn sums:", ' '.join(map(str, col_sums)))
    
    
def generate_symbols(num_symbols):
    """生成符号及其对应的随机数字"""
    symbols = [
        '*', '@', '√', 'x', '+', '-', '*', '/', '%', '^', '&', '#', '$', '!', '?', 
        '∑', '∆', '∏', '∫', '≈', '≠', '≥', '≤', '⊕', '⊗', '⊙', '∩', '∈', '∉', 
        '⇒', '⇔', '←', '→', '↑', '↓', '∇', '∞', '∂', '∃', '∀', '¬', '∝', '⊥', '∥', 
        '∅', '∴', '∵', '♠', '♣', '♥', '♦'
    ]
    chosen_symbols = random.sample(symbols, num_symbols)
    symbol_values = {symbol: random.randint(-100, 100) for symbol in chosen_symbols}
    return symbol_values

def generate(seed):
    random.seed(seed)
    while True:  # 循环直到生成一个满秩的棋盘
        row_count = random.randint(5, 10)
        col_count = random.randint(5, 10)
        # 符号数量设为 min(row_count, col_count)-1
        symbol_num = min(row_count, col_count) - 1
        symbol_values = generate_symbols(symbol_num)
    
        # 生成完整棋盘，每个格子随机选取一个符号
        grid = []
        for i in range(row_count):
            row = [random.choice(list(symbol_values.keys())) for _ in range(col_count)]
            grid.append(row)
    
        # 确保每个选定符号至少出现一次
        used_symbols = {symbol for row in grid for symbol in row}
        missing_symbols = set(symbol_values.keys()) - used_symbols
        if missing_symbols:
            positions = [(i, j) for i in range(row_count) for j in range(col_count)]
            random.shuffle(positions)
            for symbol in missing_symbols:
                if positions:
                    i, j = positions.pop()
                    grid[i][j] = symbol

        # 计算系数矩阵：统计每一行中各符号的出现次数
        symbols = list(symbol_values.keys())
        A = np.array([[row.count(sym) for sym in symbols] for row in grid])
        # 检查矩阵的列秩是否等于符号数量
        if np.linalg.matrix_rank(A) == len(symbols):
            break  # 满足条件，退出循环
    
    # 计算行和与列和
    row_sums = []
    col_sums = [0] * col_count
    for i, row in enumerate(grid):
        current_row_sum = 0
        for j, symbol in enumerate(row):
            current_row_sum += symbol_values[symbol]
            col_sums[j] += symbol_values[symbol]
        row_sums.append(current_row_sum)
    
    item = {
        'score': 1,   # 生成成功
        'is_end': False,
        'response': [],
        'prompt': '',
        'action': '',
        'epoch': 1,
        'grid': grid,
        'row_sums': row_sums,
        'col_sums': col_sums,
        'col_count': col_count,
        'symbol_values': symbol_values,
    }
    return item




def verify(item):
    """验证用户输入的列和是否与计算的结果匹配"""
    
    user_sums = item['action'] 
    col_sums = item['col_sums']
    col_count = item['col_count']
    if isinstance(item['action'], str):
        try:
            user_sums = ast.literal_eval(item['action'])
        except (ValueError, SyntaxError):
            item['score'] = 0
            item['is_end'] = True
            return item
    if not isinstance(user_sums, list) or any(not isinstance(x, int) for x in user_sums):
        item['score'] = 0
        return item

    if len(user_sums) != col_count:
        item['score'] = 0
        return item

    if user_sums == col_sums:
        item['score'] = 1
        return item
    else:
        item['score'] = 0
        return item

def save_as_image(filename, grid, row_count, col_count, symbol_values):
    """将矩阵和每行和保存为图片"""
    cell_size = 50      # 每个单元格的大小
    extra_width = 200   # 额外的宽度用于显示行和
    img_width = col_count * cell_size + extra_width
    img_height = row_count * cell_size + 100  # 额外高度用于显示列和
    image = Image.new('RGB', (img_width, img_height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # 尝试加载支持 Unicode 的字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except IOError:
        print("\n⚠️ Warning: Unicode font not found, using default font (may cause errors).")
        font = ImageFont.load_default()

    start_x, start_y = 50, 50
    # 绘制矩阵和行和
    for i, row in enumerate(grid):
        # 计算每行的和（也可直接使用 generate 返回的 row_sums）
        row_sum = sum(symbol_values[symbol] for symbol in row)
        for j, symbol in enumerate(row):
            draw.text((start_x + j * cell_size, start_y + i * cell_size), symbol, font=font, fill="black")
        # 绘制每行的和，位于矩阵右侧
        draw.text((start_x + col_count * cell_size + 10, start_y + i * cell_size), str(row_sum), font=font, fill="red")

    # 保存图片
    image.save(filename)
    print(f"✅ Image saved as {filename}")


class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    grid: list
    row_sums: list
    col_sums: list
    col_count: int
    symbol_values: dict
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
    updated_state = verify(state)
    # 转换 NumPy 数据类型后返回
    updated_state = convert_numpy_types(updated_state)
    return updated_state



if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)
# if __name__ == "__main__":
#     item = generate(213)
#     # 显示结果
#     print(print_board(item))

#     # 获取用户输入并验证列和
#     try:
#         user_input = input("\nEnter column sums (comma-separated, e.g., 12, -5, 8, 20):\n")
#         user_sums = list(map(int, user_input.split(',')))  # 转换为整数列表
#         item['action'] = user_sums
#         print(verify(item))
        
#     except ValueError:
#         print("\n❌ Input Error! Please enter a comma-separated list of integers, e.g., `12, -5, 8, 20`.")

    # 保存矩阵为图片
    # save_as_image("game_output.png", grid, row_count, col_count, symbol_values)
