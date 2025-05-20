import random
import base64
import os
from PIL import Image, ImageDraw, ImageFont
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
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

# ============ 全局配置 ============
game_prompt = """
You are a good game player, I'll give you a game board and rules.
Your task is:
- First, give your answer according to the game board and rules.
- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question, e.g., 'Answer: happy'

Welcome to Wordle!
Rules:
1. You have a limited number of attempts (max_attempts={MAX_ATTEMPTS}) to guess the secret word.
2. Each guess must match the word_length={WORD_LENGTH} exactly.
3. After each guess, you get an updated grid image:
   - Green: correct letter in the correct position
   - Yellow: letter exists but in the wrong position
   - Gray: letter does not exist in the secret word
Current attempts:{current_attempts}
Good luck!
"""

# 绘图相关常量
CELL_SIZE = 60
PADDING = 5
FONT_SIZE = 32
FONT_PATH = "arial.ttf"  # 如有需要，请换成系统中存在的字体文件

# 颜色映射
COLOR_MAP = {
    "GREEN":  (106, 170, 100),
    "YELLOW": (201, 180, 88),
    "GRAY":   (120, 124, 126),
    "WHITE":  (255, 255, 255)
}

# ============ 辅助函数 ============

def get_word_bank(path="words.txt"):
    """
    从文件加载词库，返回一个 {word_length: [words]} 的字典；
    若文件不存在，则采用内置的默认词库。
    """
    word_bank = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if not word:
                    continue
                length = len(word)
                word_bank.setdefault(length, []).append(word)
    except Exception:
        word_bank = {
            5: ["apple", "berry", "charm", "delta", "eagle"],
            6: ["orange", "banana", "tomato", "potato", "papaya"]
        }
    return word_bank

def generate_secret(seed: int, level: int, word_bank: dict) -> str:
    """
    根据种子和目标单词长度 (level) 从词库中随机选取一个秘密单词
    """
    possible = word_bank.get(level, [])
    if not possible:
        possible = [w for lst in word_bank.values() for w in lst]
    random.seed(seed)
    return random.choice(possible)

def get_feedback(secret: str, guess: str) -> List[str]:
    """
    对比 guess 与 secret，返回与 guess 长度相同的反馈列表：
      - "GREEN": 正确位置的正确字母
      - "YELLOW": 存在于单词中但位置错误
      - "GRAY": 字母不存在于秘密单词中
    """
    feedback = []
    for i, g in enumerate(guess):
        if i < len(secret) and g == secret[i]:
            feedback.append("GREEN")
        elif g in secret:
            feedback.append("YELLOW")
        else:
            feedback.append("GRAY")
    return feedback

def draw_board(guesses: List[str], feedbacks: List[List[str]], word_length: int, max_attempts: int, save_path: str):
    """
    根据猜测记录绘制网格图片，每个格子只显示一个字母。
    """
    width = word_length * CELL_SIZE + (word_length + 1) * PADDING
    height = max_attempts * CELL_SIZE + (max_attempts + 1) * PADDING
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()
    
    for row in range(max_attempts):
        for col in range(word_length):
            x1 = col * CELL_SIZE + (col + 1) * PADDING
            y1 = row * CELL_SIZE + (row + 1) * PADDING
            x2 = x1 + CELL_SIZE
            y2 = y1 + CELL_SIZE
            if row < len(guesses):
                # 修改处：每个格子只显示对应字母，而不是整个单词
                letter = guesses[row][col].upper()
                color_name = feedbacks[row][col]
                color = COLOR_MAP[color_name]
            else:
                letter = ""
                color = COLOR_MAP["WHITE"]
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0))
            if letter:
                bbox = font.getbbox(letter)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                text_x = x1 + (CELL_SIZE - w) / 2
                text_y = y1 + (CELL_SIZE - h) / 2
                draw.text((text_x, text_y), letter, fill=(0, 0, 0), font=font)
    image.save(save_path)

def encode_image(image_path: str) -> str:
    """
    将图片文件转换为 base64 编码的字符串
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def render_board_to_image(item: dict) -> str:
    """
    根据 item 中的猜测记录生成当前游戏的网格图片，并返回其 base64 编码字符串
    """
    os.makedirs(f"cache/{item['seed']}", exist_ok=True)
    file_path = f"cache/{item['seed']}/wordle_{item['attempt']}.png"
    draw_board(item["guesses"], item["feedbacks"], item["level"], item["max_attempts"], file_path)
    return encode_image(file_path)

# ============ 接口抽象函数 ============

def generate_item(seed: int) -> dict:
    """
    根据种子生成 Wordle 游戏初始状态，并返回 item 字典
    """
    random.seed(seed)
    level = random.randint(5,12)           # 秘密单词长度
    max_attempts = 10    # 最大尝试次数
    word_bank = get_word_bank()
    secret = generate_secret(seed, level, word_bank)
    item = {
        "seed": seed,
        "level": level,
        "max_attempts": max_attempts,
        "secret": secret,
        "attempt": 0,
        "guesses": [],
        "feedbacks": [],
        "is_end": False,
        "score": 0,
        "prompt": "",
        "action": ""
    }
    item["base64_image"] = render_board_to_image(item)
    return item

def verify_item(item: dict) -> dict:
    """
    根据 item 中的 action（玩家猜测）更新游戏状态，
    检查猜词是否合法，计算反馈，更新记录，并判断游戏是否结束
    """
    guess = item.get("action", "").strip().lower()
    if len(guess) != item["level"]:
        # 猜测长度不正确时不做处理直接返回当前状态
        return item
    
    feedback = get_feedback(item["secret"], guess)
    item["guesses"].append(guess)
    item["feedbacks"].append(feedback)
    item["attempt"] += 1
    
    if guess == item["secret"]:
        item["is_end"] = True
        item["score"] = 1
    elif item["attempt"] >= item["max_attempts"]:
        item["is_end"] = True
        item["score"] = 0
    
    item["base64_image"] = render_board_to_image(item)
    return item

def print_board_item(item: dict) -> str:
    """
    根据 item 中的猜测记录返回当前游戏的文本描述
    """
    output = f"Attempt: {item['attempt']}/{item['max_attempts']}\n"
    # for idx, (guess, fb) in enumerate(zip(item["guesses"], item["feedbacks"]), 1):
    #     fb_str = "".join(['G' if c == "GREEN" else 'Y' if c == "YELLOW" else 'X' for c in fb])
    #     output += f"{idx}: {guess.upper()}  {fb_str}\n"
    return game_prompt.format(MAX_ATTEMPTS=item['max_attempts'],WORD_LENGTH=item['level'],current_attempts=item['attempt'])

# ============ FastAPI 数据模型 ============
class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    seed: int
    level: int
    max_attempts: int
    secret: str
    attempt: int
    guesses: List[str]
    feedbacks: List[List[str]]
    is_end: bool
    score: int
    prompt: str
    action: str
    base64_image: str

class BoardRequest(BaseModel):
    board: str

# ============ FastAPI 接口 ============
@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    item = generate_item(request.seed)
    return item

@app.post("/verify", response_model=GameState)
def api_verify(state: GameState):
    item = state.dict()
    updated_item = verify_item(item)
    return updated_item

@app.post("/print_board", response_model=BoardRequest)
def api_print_board(state: GameState):
    board_text = print_board_item(state.dict())
    return {"board": board_text}

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)
