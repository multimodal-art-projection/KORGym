# game_lib/9-Jigsaw_puzzle/game_lib.py

#Standard libraries
import os
import random
import string
import base64
import ast
import argparse

#Commonly used open-source libraries
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont

def parse_init():
    """
    Parses command-line arguments for server deployment.

    Returns:
        argparse.Namespace: Object containing parsed host and port parameters.
    """
    parser = argparse.ArgumentParser(description="Data creation utility")
    parser.add_argument('-p', '--port', type=int, default=8775, help='服务部署端口')
    parser.add_argument('-H', '--host', type=str, default="0.0.0.0", help='服务部署地址')
    args = parser.parse_args()
    return args
app = FastAPI()
game_prompt = """
You are a good game player, I'll give you a game board which is a picture and rules.\nYour task is:\n- First, give your answer according to the game board and rules.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g."Answer: [(1,'A'),(2,'D'),...]"
As shown in the picture, you need to solve the puzzle game. The top of the image is the puzzle board (represented by numbers) where pieces need to be filled in, and the bottom of the image shows the puzzle pieces (represented by letters). You need to place the puzzle pieces from the bottom into the blank spaces at the top to restore the original picture. The output should be in the format: "Answer: [(1,'A'),(2,'D'),...]"

"""

def print_board(item):
    return game_prompt

def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
        
def generate(seed):
    """
    Generates a jigsaw puzzle task by slicing an image into pieces, assigning random labels,
    rendering the puzzle board with blank positions and options, and encoding it into base64 format.

    Args:
        seed (int): Random seed to ensure reproducibility.

    Returns:
        dict: A dictionary containing the game state, including:
            - answer: the correct mapping as a list of (position, label)
            - image_path: local file path of the rendered puzzle
            - base64_image: base64-encoded string of the image
            - other default state fields (score, is_end, etc.)
    """
    item = {
        'score': 0,
        'is_end': False,
        'response': [],
        'prompt': '',
        'action': '',
        'epoch': 1,
    }
    input_dir = 'pictures'
    output_dir = 'cache'
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    LETTER_HEIGHT = 60  # 字母区域高度
    SPACING = 10  # 选项图片间距
    LINE_WIDTH = 2  # 分割线宽度
    # 随机选择输入图片
    png_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
    if not png_files:
        raise FileNotFoundError("No PNG found.")
    img_name = random.choice(png_files)
    img_path = os.path.join(input_dir, img_name)

    # 打开并预处理图片
    img = Image.open(img_path)
    width, height = img.size
    short = random.randint(2, 4) # 短边被切成2-4块
    long = random.randint(short, min(2*short, 26//short)) # 长边大于短边，同时整体数量不超过两倍短边并且总块数不超过26
    if width >= height:
        w = long
        h = short
    else:
        w = short
        h = long
    block_width = width // w
    block_height = height // h
    img = img.resize((block_width * w, block_height * h))  # 确保能被整除

    # 切割图片碎片
    pieces = []
    for y in range(h):
        for x in range(w):
            left = x * block_width
            upper = y * block_height
            pieces.append(img.crop((left, upper, left + block_width, upper + block_height)))

    # 生成随机映射关系
    numbers = list(range(1, h*w+1))
    letters = list(string.ascii_uppercase[:h*w])
    random.shuffle(letters)
    mapping = {n: l for n, l in zip(numbers, letters)}
    correct_answer = list(zip(numbers, letters))

######################################################### 创建空白图片
    board = Image.new('RGB', img.size, 'white')
    draw = ImageDraw.Draw(board)

    # 绘制分割线
    for x in range(1, w):
        draw.line([(x * block_width, 0), (x * block_width, img.size[1])], fill='black', width=LINE_WIDTH)
    for y in range(1, h):
        draw.line([(0, y * block_height), (img.size[0], y * block_height)], fill='black', width=LINE_WIDTH)

    # 添加数字编号
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 40)
    except:
        font = ImageFont.load_default()

    for idx in range(w*h):
        y_pos = idx // w
        x_pos = idx % w
        draw.text((x_pos * block_width + block_width/2, y_pos * block_height + block_height/2), str(idx + 1), fill='black', font=font)
    # board.save(os.path.join(output_dir, "puzzle_template.png"))

##################################################### 创建拼图选项图
    # 创建带字母的拼图块
    labeled_pieces = []
    for idx, piece in enumerate(pieces):
        new_img = Image.new('RGB', (block_width, block_height + LETTER_HEIGHT), 'white')
        new_img.paste(piece, (0, 0))

        draw = ImageDraw.Draw(new_img)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 40)
        except:
            font = ImageFont.load_default()

        letter = mapping[idx + 1]
        text_width = draw.textlength(letter, font=font)
        draw.text(((block_width - text_width) // 2, block_height + 5), letter, fill='black', font=font)
        labeled_pieces.append(new_img)

    # 随机打乱顺序
    random.shuffle(labeled_pieces)

    # 创建选项图片
    option_width = w * block_width + 3 * SPACING
    option_height = h * (block_height + LETTER_HEIGHT) + SPACING
    fragments = Image.new('RGB', (option_width, option_height), 'white')

    for i, piece in enumerate(labeled_pieces):
        row = i // w
        col = i % w
        x = col * (block_width + SPACING)
        y = row * (block_height + LETTER_HEIGHT + SPACING)
        fragments.paste(piece, (x, y))

    combined_width = max(board.width, fragments.width)
    combined_height = board.height + fragments.height
    combined_image = Image.new('RGB', (combined_width, combined_height), 'white')
    combined_image.paste(board, (0, 0))
    combined_image.paste(fragments, (0, board.height))
    output_img_path=os.path.join(output_dir, f"puzzle_{img_name}_{random.randint(0,9999)}.png")
    combined_image.save(output_img_path)
    item['answer'] = correct_answer
    item['image_path'] = output_img_path
    item['base64_image'] = encode_image(output_img_path)
    return item


def verify(item):
    """
    Compares the user-submitted answer with the correct answer using set intersection.

    Args:
        item (dict): Game state, including the user's `action` and the correct `answer`.

    Returns:
        dict: Updated item with computed score (fraction of correct mappings).
    """
    correct_answer = item['answer']
    # 尝试将 item['action'] 转换为列表结构
    try:
        if isinstance(item['action'], str):
            generated_answer = ast.literal_eval(item['action'])
        else:
            generated_answer = item['action']
        if not isinstance(generated_answer, list):
            item['score'] = 0
            return item
    except Exception as e:
        item['score'] = 0
        return item
    # generated_answer = item['action']
    # 将两个列表转换为集合
    set1 = set(correct_answer)
    set2 = set(generated_answer)

    # 求两个集合的交集
    intersection = set1 & set2
    score = len(intersection)/len(correct_answer)
    item['score'] = score
    return item


class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    answer : list
    image_path : str
    base64_image : str
    score: float
    is_end: bool
    action: str
    response: list
    prompt: str
    epoch: int
# 生成初始游戏状态
@app.post("/print_board", response_model=BoardRequest)
def api_print_board(request: GameState):
    state = request.dict()
    state['answer'] = [tuple(coord) for coord in state['answer']]
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
    state['answer'] = [tuple(coord) for coord in state['answer']]
    updated_state = verify(state)
    return updated_state

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)


# if __name__ == "__main__":
#     seed = 6841
#     item = generate(seed)
#     item['action'] = item['answer']
#     print(item['action'])
#     score = verify(item)['score']
#     print(score)
