import random
from PIL import Image, ImageDraw, ImageFont

# ============ 1. 全局配置 ============

GAME_RULES = """\
Welcome to Wordle!
Rules:
1. You have a limited number of attempts (max_attempts=$MAX_ATTEMPTS$) to guess the secret word.
2. Each guess must match the word_length=$WORD_LENGTH$ exactly.
3. After each guess, you get an updated grid image:
   - Green: correct letter in the correct position
   - Yellow: letter exists but in the wrong position
   - Gray: letter does not exist in the secret word
4. If you guess correctly at any point, you see "Success !".
   Otherwise, if you run out of attempts, you see "Fail !" and the correct word.
Good luck!
"""

# 绘图相关常量
CELL_SIZE = 60
PADDING = 5
FONT_SIZE = 32
FONT_PATH = "arial.ttf"  # 请替换为自己系统有的字体文件，若出错可用默认字体

# 颜色映射
COLOR_MAP = {
    "GREEN":  (106, 170, 100),
    "YELLOW": (201, 180, 88),
    "GRAY":   (120, 124, 126),
    "WHITE":  (255, 255, 255)
}


# ============ 2. 构建词库函数 ============

def get_word_bank(path: str = "words.txt"):
    """
    从文件加载词库，返回一个 {length: [words]} 的字典
    """
    word_bank = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if not word:
                continue
            length = len(word)
            word_bank.setdefault(length, []).append(word)
    
    def _getter():
        return word_bank
    
    return _getter


# ============ 3. 随机生成答案 ============

def generate(seed: int, level: int, bank_getter=None) -> str:
    """
    指定随机种子和单词长度，返回一个秘密单词。
    如果给定长度在词库里没有，就从所有单词中随机选一个。
    """
    if bank_getter is None:
        bank_getter = get_word_bank("words.txt")
    word_bank = bank_getter()
    
    possible = word_bank.get(level, [])
    if not possible:
        possible = [w for v in word_bank.values() for w in v]
    
    random.seed(seed)
    return random.choice(possible)


# ============ 4. 验证并给出颜色反馈 ============

def verify(secret: str, guess: str):
    """
    对比 guess 与 secret，返回和 guess 等长的颜色列表：
    - 'GREEN': 该位置字母完全正确
    - 'YELLOW': 字母存在但位置错误
    - 'GRAY': 字母不存在
    （不做严格次数匹配，只要字母在 secret 中就判定 YELLOW）
    """
    colors = []
    for i, g_char in enumerate(guess):
        if i < len(secret) and g_char == secret[i]:
            colors.append("GREEN")
        elif g_char in secret:
            colors.append("YELLOW")
        else:
            colors.append("GRAY")
    return colors


# ============ 5. 绘制并输出网格图片 ============

def draw_board(guesses, feedbacks, word_length, max_attempts, save_path):
    width  = word_length * CELL_SIZE + (word_length + 1) * PADDING
    height = max_attempts * CELL_SIZE + (max_attempts + 1) * PADDING
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except:
        font = ImageFont.load_default()
    
    for row in range(max_attempts):
        for col in range(word_length):
            x1 = col * CELL_SIZE + (col + 1) * PADDING
            y1 = row * CELL_SIZE + (row + 1) * PADDING
            x2 = x1 + CELL_SIZE
            y2 = y1 + CELL_SIZE
            
            if row < len(guesses):
                # 已经有猜测的行
                letter = guesses[row][col].upper()
                color_name = feedbacks[row][col]
                color = COLOR_MAP[color_name]
            else:
                # 尚未猜测的行
                letter = ""
                color = COLOR_MAP["WHITE"]
            
            # 注意这里添加了 outline 参数
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0))

            if letter:
                bbox = font.getbbox(letter)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]

                text_x = x1 + (CELL_SIZE - w) / 2
                text_y = y1 + (CELL_SIZE - h) / 2
                draw.text((text_x, text_y), letter, fill=(0, 0, 0), font=font)
    
    image.save(save_path)



# ============ 6. 核心游戏循环（多模态版） ============

def play_wordle_multimodal():
    
    # 1) 参数可自行修改
    seed = 42
    level = 6
    max_attempts = 6
    
    # 2) 获取词库并生成秘密单词
    bank_getter = get_word_bank("words.txt")
    secret_word = generate(seed, level, bank_getter)
    # secret_word = "strawberry"; level = 10; max_attempts = 10  # 测试用
    
    # 3) 打印英文规则
    print(GAME_RULES.replace("$MAX_ATTEMPTS$", str(max_attempts)).replace("$WORD_LENGTH$", str(level)))
    
    # 4) 第一次输入前，返回空网格图片
    guesses = []
    feedbacks = []
    draw_board(
        guesses, feedbacks, word_length=level, max_attempts=max_attempts,
        save_path="wordle_attempt_0.png"
    )
    
    # 5) 开始循环猜词
    for attempt in range(1, max_attempts + 1):
        # 不打印文本提示，直接读输入
        print(">>> ", end="")
        guess = input().strip().lower()
        
        # 若用户输入不是正确长度，则忽略并继续让其输入（不扣次数、不提示）
        if len(guess) != level:
            print(f"Please enter a word with exactly {level} letters!")
            # 若短于level，则拼横杠；若长于level，则截断
            if len(guess) < level:
                guess += "-" * (level - len(guess))
            else:
                guess = guess[:level]
        
        # 获取颜色反馈
        color_result = verify(secret_word, guess)
        
        # 更新数据
        guesses.append(guess)
        feedbacks.append(color_result)
        
        # 输出新的网格图片
        draw_board(
            guesses, feedbacks, word_length=level, max_attempts=max_attempts,
            save_path=f"wordle_attempt_{attempt}.png"
        )
        
        # 判断是否全对
        if guess == secret_word:
            print("Success !")
            return
    
    # 如果循环结束仍未猜对
    print("Fail !")
    print(f"The correct answer was: {secret_word}")


if __name__ == "__main__":
    play_wordle_multimodal()
