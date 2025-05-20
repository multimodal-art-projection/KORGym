from PIL import Image, ImageDraw, ImageFont, ImageOps
import os
import random
import math
import base64
import ast
import uvicorn
from fastapi import FastAPI, HTTPException
import argparse
from pydantic import BaseModel
def parse_init():
    parser = argparse.ArgumentParser(description="Data creation utility")
    parser.add_argument('-p', '--port', type=int, default=8775, help='服务部署端口')
    parser.add_argument('-H', '--host', type=str, default="0.0.0.0", help='服务部署地址')
    return parser.parse_args()

app = FastAPI()
# -----------------------------
# 常量设置与辅助函数
# -----------------------------
FONT_PATH = "arialbd.ttf"  
TARGET_SIZE = (200, 200)    # 主图尺寸
OPTION_SIZE = (70, 70)      # 选项块尺寸
CACHE_DIR = "cache"
game_prompt = """
You are a good game player, I'll give you a game board which is a picture and rules.\nYour task is:\n- First, give your answer according to the game board and rules.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g."Answer: F"

I will provide an image in which a rectangular piece is missing from a larger pattern. Below the pattern, there are several options for pieces that could fit into the blank space. Please choose the most suitable piece to fill the blank and output the letter corresponding to that option, e.g., 'Answer: G'.
"""
# 确保 cache 文件夹存在
os.makedirs(CACHE_DIR, exist_ok=True)

def encode_image(image_path: str):
    """将图片转换为 Base64 字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# -----------------------------
# 业务逻辑接口
# -----------------------------
def generate(seed: int):
    """
    根据 seed 生成拼图题目：
    - 从 pictures 目录随机选取一个子文件夹及图片
    - 将图片统一 resize 到 TARGET_SIZE
    - 从右侧中部挖取一块区域作为缺口，并生成正确答案拼图块
    - 随机生成干扰选项，并将正确答案混排后确定正确选项字母
    - 排版生成包含标题、缺口拼图以及选项的完整图片
    返回的 item 包含：
        - answer: 正确选项字母（例如 'A'）
        - image_path: 生成图片的存储路径
        - base64_image: 图片的 Base64 编码
        - prompt: 游戏提示文字
        - 其它状态信息（score、is_end、response、action、epoch）
    """
    item = {
        'score': 0,
        'is_end': False,
        'response': [],
        'prompt': '',
        'action': '',
        'epoch': 1,
    }
    random.seed(seed)
    base_directory = "pictures"
    sub_folders = os.listdir(base_directory)
    if not sub_folders:
        raise FileNotFoundError("未找到 pictures 文件夹中的子目录，请检查图片资源。")
    # 随机选取一个类别文件夹
    category = random.choice(sub_folders)
    img_dir = os.path.join(base_directory, category)
    all_imgs = [f for f in os.listdir(img_dir) if f.lower().endswith('.png')]
    if not all_imgs:
        raise FileNotFoundError("图片目录为空，请放置至少一张 PNG 图片。")
    
    # 选取主图，并统一 resize 到 TARGET_SIZE
    main_img_name = random.choice(all_imgs)
    main_img_path = os.path.join(img_dir, main_img_name)
    original_img = Image.open(main_img_path).convert("RGBA")
    original_img = original_img.resize(TARGET_SIZE, Image.LANCZOS)
    width, height = TARGET_SIZE

    # 挖取正确拼图块区域（右侧中部，距离右边20px）
    shape_mask = Image.new("L", TARGET_SIZE, 0)
    draw_mask = ImageDraw.Draw(shape_mask)
    shape_w, shape_h = OPTION_SIZE
    offset_x = width - shape_w - 20
    offset_y = (height - shape_h) // 2
    draw_mask.rectangle([offset_x, offset_y, offset_x + shape_w, offset_y + shape_h], fill=255)

    # 裁剪正确的拼图块
    correct_piece = Image.new("RGBA", TARGET_SIZE, (0, 0, 0, 0))
    correct_piece.paste(original_img, mask=shape_mask)
    correct_piece_cropped = correct_piece.crop((offset_x, offset_y, offset_x + shape_w, offset_y + shape_h))
    correct_piece_cropped = correct_piece_cropped.resize(OPTION_SIZE, Image.LANCZOS)

    # 在主图上挖空对应区域生成缺口
    puzzle_img = original_img.copy()
    inv_mask = Image.new("L", TARGET_SIZE, 255)
    inv_draw = ImageDraw.Draw(inv_mask)
    inv_draw.rectangle([offset_x, offset_y, offset_x + shape_w, offset_y + shape_h], fill=0)
    puzzle_array = puzzle_img.load()
    inv_array = inv_mask.load()
    for y in range(height):
        for x in range(width):
            if inv_array[x, y] == 0:
                puzzle_array[x, y] = (255, 255, 255, 0)  # 挖空

    # 生成干扰选项（共 nums 个选项，包含正确答案）
    nums = random.randint(8, 20)
    distractor_pieces = []
    while len(distractor_pieces) < (nums - 1):
        distract_img_name = random.choice(all_imgs)
        distract_img_path = os.path.join(img_dir, distract_img_name)
        d_img = Image.open(distract_img_path).convert("RGBA").resize(TARGET_SIZE, Image.LANCZOS)
        rand_x = random.randint(0, width - shape_w)
        rand_y = random.randint(0, height - shape_h)
        piece = d_img.crop((rand_x, rand_y, rand_x + shape_w, rand_y + shape_h))
        piece = piece.resize(OPTION_SIZE, Image.LANCZOS)
        if random.random() < 0.3:
            piece = piece.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.3:
            piece = piece.transpose(Image.ROTATE_90)
        distractor_pieces.append(piece)
    # 补足干扰选项（极端情况下不够数量时复制正确块）
    while len(distractor_pieces) < (nums - 1):
        distractor_pieces.append(correct_piece_cropped.copy())

    # 正确拼图块与干扰块混排
    all_options = distractor_pieces + [correct_piece_cropped]
    random.shuffle(all_options)
    correct_idx = all_options.index(correct_piece_cropped)
    answer_options = [chr(ord('A') + i) for i in range(nums)]
    correct_answer = answer_options[correct_idx]

    # 排版最终图片：包括标题、拼图（缺口图）以及选项块
    title_text = "1. Please choose the shape that best fits the missing piece"
    try:
        font_title = ImageFont.truetype(FONT_PATH, 40)
    except IOError:
        font_title = ImageFont.load_default()
    left, top, right, bottom = font_title.getbbox(title_text)
    title_w = right - left
    title_h = bottom - top
    margin = 20
    columns = 4
    rows = math.ceil(nums / columns)
    options_total_width = (OPTION_SIZE[0] + margin) * columns - margin
    options_total_height = (OPTION_SIZE[1] + margin) * rows - margin
    final_width = max(width, options_total_width, title_w) + margin * 2
    final_height = margin + title_h + margin + height + margin + options_total_height + margin

    final_image = Image.new("RGBA", (final_width, final_height), (255, 255, 255, 255))
    draw_final = ImageDraw.Draw(final_image)
    # 绘制标题（居中）
    title_x = (final_width - title_w) // 2
    title_y = margin
    draw_final.text((title_x, title_y), title_text, fill=(0, 0, 0), font=font_title)
    # 贴上缺口拼图
    puzzle_x = (final_width - width) // 2
    puzzle_y = title_y + title_h + margin
    final_image.paste(puzzle_img, (puzzle_x, puzzle_y), puzzle_img)
    # 绘制选项块及标注字母
    try:
        font_option = ImageFont.truetype(FONT_PATH, 24)
    except IOError:
        font_option = ImageFont.load_default()
    options_start_x = (final_width - options_total_width) // 2
    options_start_y = puzzle_y + height + margin
    current_x = options_start_x
    current_y = options_start_y
    for i, piece_img in enumerate(all_options):
        final_image.paste(piece_img, (current_x, current_y), piece_img)
        draw_final.text((current_x + 5, current_y + 5),
                        answer_options[i], fill=(255, 0, 0), font=font_option)
        current_x += OPTION_SIZE[0] + margin
        if (i + 1) % columns == 0:
            current_x = options_start_x
            current_y += OPTION_SIZE[1] + margin

    # 保存最终图片到 cache 文件夹
    output_img_path = os.path.join(CACHE_DIR, f"puzzle_{seed}_{random.randint(0,9999)}.png")
    final_image.save(output_img_path)

    # 更新 item 信息
    item['answer'] = correct_answer
    item['image_path'] = output_img_path
    item['base64_image'] = encode_image(output_img_path)
    return item

def verify(item: dict):
    """
    校验用户提交的答案：
    - 尝试解析 item['action']（形如 "Answer: A"），
    - 与 item 中存储的正确答案比对，正确得分 1.0，否则得 0.0
    """
    correct_answer = item.get('answer')
    try:
        generated_answer = item.get('action')
        if isinstance(generated_answer, str) and generated_answer.startswith("Answer:"):
            generated_answer = generated_answer.replace("Answer:", "").strip()
    except Exception:
        item['score'] = 0
        return item

    item['score'] = 1.0 if generated_answer == correct_answer else 0.0
    return item

def print_board(item: dict):
    """
    返回拼图题目的提示文字，可用于展示游戏板说明
    """
    return game_prompt

# -----------------------------
# FastAPI 配置与接口定义
# -----------------------------
app = FastAPI()

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    answer: str
    image_path: str
    base64_image: str
    score: float
    is_end: bool
    action: str
    response: list
    prompt: str
    epoch: int

class BoardRequest(BaseModel):
    board: str

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
    board_text = print_board(state)
    return {"board": board_text}

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)