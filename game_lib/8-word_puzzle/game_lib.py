# game_lib/8-word_puzzle/game_lib.py

#Standard libraries
import math
import ast
import argparse
import random
import pandas as pd
from typing import Tuple, List, Dict
import os
import base64

#Commonly used open-source libraries
from PIL import Image, ImageDraw, ImageFont
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


def parse_init():
    """
    Parses command-line arguments to configure the FastAPI server host and port.

    Returns:
        argparse.Namespace: Parsed argument object with 'host' and 'port'.
    """
    parser = argparse.ArgumentParser(description="Data creation utility")
    parser.add_argument('-p', '--port', type=int, default=8775, help='服务部署端口')
    parser.add_argument('-H', '--host', type=str, default="0.0.0.0", help='服务部署地址')
    args = parser.parse_args()
    return args
game_prompt='''
You are a good game player, I'll give you a game board which is a picture and rules.\nYour task is:\n- First, give your answer according to the game board and rules.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: ["happy","person"]'
You need to complete a crossword puzzle that consists of grey and white squares. The solver guesses the words corresponding to the horizontal and vertical directions based on a set of clues. During the decryption process, every white square must be filled with a letter, and the red number in the first white square of each entry corresponds to its clue number. You need to provide all the words in order in your answer as a list, e.g. 'Answer: ["happy", "sad", ...]'.

Clues:
{clues}
'''
app = FastAPI()

def print_board(item):
    output=""
    for i, clue in enumerate(item['clues'], 1):
        output+=(f"{i}. {clue}\n")
    return game_prompt.format(clues=output)

def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_word_bank(word_clues_path: str = "high_quality_word_clues.csv"):
    """
    Loads word and clue pairs from CSV and filters invalid entries.

    Args:
        word_clues_path (str): Path to the CSV file.

    Returns:
        Dict[str, str]: Dictionary mapping words to clues.
    """
    word_bank = pd.read_csv(word_clues_path)
    word_bank['word'] = word_bank['word'].str.strip().str.lower()
    word_bank = word_bank.drop_duplicates('word').set_index('word')['clue'].to_dict()
    valid_words = {w: c for w, c in word_bank.items() if 3 <= len(w) <= 12 and w.isalpha()}
    return valid_words

def select_words(valid_words: Dict[str, str], num: int, seed: int):
    """
    Selects a set of words with bias toward mid-length words and retrieves their clues.

    Args:
        valid_words (dict): Word-to-clue mapping.
        num (int): Number of words to select.
        seed (int): Random seed.

    Returns:
        Tuple[List[str], List[str]]: Selected words and corresponding clues.
    """
    random.seed(seed)
    words = list(valid_words.keys())
    weights = [3 if 5 <= len(w) <= 8 else 1 for w in words]
    selected = []
    while len(selected) < num and words:
        chosen = random.choices(words, weights=weights, k=1)[0]
        if chosen not in selected:
            selected.append(chosen)
            idx = words.index(chosen)
            words.pop(idx)
            weights.pop(idx)
    descriptions = [valid_words[w] for w in selected]
    return selected, descriptions

def place_words(words: List[str], grid_size: int = 20):
    """
    Places a list of words into a grid while ensuring valid crossword constraints.

    Args:
        words (List[str]): Words to be placed.
        grid_size (int): Dimension of the square grid.

    Returns:
        Tuple[List[List[str]], List[Dict]]: Final grid and metadata for each placed word.
    """
    grid = [[None] * grid_size for _ in range(grid_size)]
    placed_info = []
    
    sorted_words = sorted(enumerate(words), key=lambda x: -len(x[1]))
    
    for original_idx, word in sorted_words:
        word = word.upper()
        placed = False
        max_attempts = 200 if original_idx == 0 else 500
        
        for _ in range(max_attempts):
            direction = random.choice(['across', 'down'])
            word_len = len(word)
            
            if direction == 'across':
                max_col = grid_size - word_len
                max_row = grid_size - 1
            else:
                max_row = grid_size - word_len
                max_col = grid_size - 1
            
            if max_row < 0 or max_col < 0:
                continue
            
            start_row = random.randint(0, max_row)
            start_col = random.randint(0, max_col)
            
            valid = True
            overlaps = 0
            temp_grid = [row[:] for row in grid]
            
            for i in range(word_len):
                r = start_row + (i if direction == 'down' else 0)
                c = start_col + (i if direction == 'across' else 0)
                
                if temp_grid[r][c]:
                    if temp_grid[r][c] != word[i]:
                        valid = False
                        break
                    overlaps += 1
                else:
                    temp_grid[r][c] = word[i]
            
            if valid and (overlaps > 0 or original_idx == 0):
                for i in range(word_len):
                    r = start_row + (i if direction == 'down' else 0)
                    c = start_col + (i if direction == 'across' else 0)
                    grid[r][c] = temp_grid[r][c]
                
                placed_info.append({
                    'number': original_idx + 1,
                    'row': start_row,
                    'col': start_col,
                    'direction': direction,
                    'word': word
                })
                placed = True
                break
        
        if not placed:
            return grid, placed_info
                    
    return grid, placed_info

def render_image(char_grid: List[List[str]], placed_info: List[Dict], difficulty: float):
    """
    Renders the crossword puzzle image with letters partially masked.

    Args:
        char_grid (List[List[str]]): Final letter grid.
        placed_info (List[Dict]): Metadata about placed words.
        difficulty (float): Value in [0, 1] representing clue masking rate.

    Returns:
        str: Path to the rendered image.
    """
    cell_size = 35

    min_row = min(info['row'] for info in placed_info)
    max_row = max((info['row'] + len(info['word']) - 1) if info['direction'] == 'down' else info['row'] for info in placed_info)
    min_col = min(info['col'] for info in placed_info)
    max_col = max((info['col'] + len(info['word']) - 1) if info['direction'] == 'across' else info['col'] for info in placed_info)

    padding = 25
    img = Image.new('RGB', 
                    ((max_col - min_col + 1) * cell_size + 2 * padding, 
                     (max_row - min_row + 1) * cell_size + 2 * padding),
                    color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    masked_positions = set()

    for info in placed_info:
        word = info['word']
        word_len = len(word)
        k = max(1, math.ceil(difficulty * word_len))
        k = min(k, word_len)
        indices = random.sample(range(word_len), k)
        start_row, start_col = info['row'], info['col']
        direction = info['direction']
        for i in indices:
            if direction == 'across':
                r = start_row
                c = start_col + i
            else:
                r = start_row + i
                c = start_col
            masked_positions.add((r, c))

    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            x0 = padding + (c - min_col) * cell_size
            y0 = padding + (r - min_row) * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size

            if r >= len(char_grid) or c >= len(char_grid[0]):
                continue
            is_active = char_grid[r][c] is not None
            
            if is_active:
                draw.rectangle([x0, y0, x1, y1], fill='white', outline='#CCCCCC')
                for info in placed_info:
                    if info['row'] == r and info['col'] == c:
                        draw.text((x0 + 2, y0 + 2), str(info['number']), fill='#FF4444', font=ImageFont.truetype("arial.ttf", 12) if os.path.exists("arial.ttf") else ImageFont.load_default())

                if (r, c) not in masked_positions:
                    char = char_grid[r][c]
                    bbox = draw.textbbox((0, 0), char, font=ImageFont.truetype("arial.ttf", 14) if os.path.exists("arial.ttf") else ImageFont.load_default())
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    tx = x0 + (cell_size - text_width) / 2
                    ty = y0 + (cell_size - text_height) / 2
                    draw.text((tx, ty), char, fill='black', font=ImageFont.truetype("arial.ttf", 14) if os.path.exists("arial.ttf") else ImageFont.load_default())
                else:
                    underline_y = y0 + cell_size - 4
                    draw.line([x0 + 3, underline_y, x1 - 3, underline_y], fill='#666666', width=2)
            else:
                draw.rectangle([x0, y0, x1, y1], fill='#DDDDDD', outline='#CCCCCC')
    
    os.makedirs('cache', exist_ok=True)
    img_path = f'cache/crossword_{random.randint(1000, 9999)}.png'
    img.save(img_path)
    return img_path

def generate(seed: int, word_clues_path: str = "high_quality_word_clues.csv"):
    """
    Generates a complete crossword puzzle including image, words, and clues.

    It retries word placement with increasing grid size until success.

    Args:
        seed (int): Random seed for reproducibility.
        word_clues_path (str): Path to the word-clue CSV file.

    Returns:
        dict: Game state including words, clues, image path, base64-encoded image, etc.
    """
    valid_words = load_word_bank(word_clues_path)
    if len(valid_words) < 5:
        raise ValueError("可用单词不足")
    
    grid_size = 20              # 初始网格尺寸
    overall_attempts = 0        # 全局尝试次数计数器
    max_overall_attempts = 1000 # 全局最大尝试次数
    
    while overall_attempts < max_overall_attempts:
        # 随机选择单词数量和难度
        num = random.randint(5, 15)
        difficulty = random.randint(5, 9) / 10
        if num > len(valid_words):
            raise ValueError("可用单词不足")
        
        max_retries = 20
        for retry in range(max_retries):
            overall_attempts += 1
            selected, descs = select_words(valid_words, num, seed + overall_attempts)
            grid, placed = place_words(selected, grid_size=grid_size)
            
            if len(placed) == num:
                item = {
                    'score': 0,
                    'is_end': False,
                    'response': [],
                    'prompt': '',
                    'action': '',
                    'epoch': 1,
                }
                placed_sorted = sorted(placed, key=lambda x: x['number'])
                final_words = [p['word'].lower() for p in placed_sorted]
                final_descs = [descs[p['number'] - 1] for p in placed_sorted]
                img_path = render_image(grid, placed, difficulty)
                item['answer'] = final_words
                item['clues'] = final_descs
                item['image_path'] = img_path
                item['base64_image'] = encode_image(img_path)
                return item
        # 如果内层 max_retries 次尝试均未成功，则增加网格尺寸再重试
        grid_size += 5
        # 如果网格尺寸超过上限，则报错退出
        if grid_size > 40:
            raise RuntimeError("无法生成有效的填字游戏，已达到最大网格尺寸")
    
    raise RuntimeError("生成填字游戏尝试次数过多")


def verify(item):
    """
    Verifies the user's submitted answers against the correct crossword solution.

    Compares each answer individually and returns partial credit.

    Args:
        item (dict): Game state containing 'answer' and user 'action'.

    Returns:
        dict: Updated game state with detailed response and final score.
    """
    correct = item['answer']
    
    # 尝试将 item['action'] 转换为列表结构
    try:
        if isinstance(item['action'], str):
            answers = ast.literal_eval(item['action'])
        else:
            answers = item['action']
        if not isinstance(answers, list):
            item['score'] = 0
            return item
    except Exception as e:
        item['score'] = 0
        return item

    details = []  # 存储逐项比较结果
    correct_count = 0
    total = len(correct)
    # 如果用户答案数量与正确答案数量不匹配，也记录到 response 中
    if len(answers) != total:
        details.append({
            "warning": f"用户答案数量({len(answers)})与正确答案数量({total})不匹配。"
        })
        
    # 逐项比较（按索引比较，缺失部分视为错误）
    for i, corr in enumerate(correct):
        user_ans = answers[i].strip() if i < len(answers) else ""
        is_correct = user_ans.lower() == corr.strip().lower()
        details.append({
            "index": i+1,
            "user_answer": user_ans,
            "correct_answer": corr,
            "is_correct": is_correct
        })
        if is_correct:
            correct_count += 1

    item['score'] = correct_count / total if total > 0 else 0
    return item


class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    answer : list
    clues : list
    image_path : str
    base64_image : str
    score: float
    is_end: bool
    action: str
    response: list
    prompt: str
    epoch: int


@app.post("/print_board", response_model=BoardRequest)
def api_print_board(request: GameState):
    state = request.dict()
    board_output = print_board(state)
    return {"board": board_output}

@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    game_state = generate(request.seed)
    return game_state

@app.post("/verify", response_model=GameState)
def api_verify(request: GameState):
    # 从请求中获取游戏状态，并设置新的动作
    state = request.dict()
    updated_state = verify(state)
    return updated_state

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)
# 使用示例
# if __name__ == "__main__":
#     item = generate(seed=42)
#     print(f"成功生成填字游戏：{item['image_path']}")
#     print("答案：", item['answer'])
#     print("提示：")
#     for i, clue in enumerate(item['clues'], 1):
#         print(f"{i}. {clue}")
    
#     # 模拟用户答案
#     item['action'] = [f"{i + 1}. {w}" for i, w in enumerate(item['answer'])]
#     score = verify(item)['score']
#     print(f"验证结果：{score}")
    
