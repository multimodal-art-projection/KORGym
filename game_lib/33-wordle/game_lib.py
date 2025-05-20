import random
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
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
You are a good game player, I'll give you a game board and rules.
Your task is:
- First, give your answer according to the game board and rules.
- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question, e.g., 'Answer: happy'
You need to guess a specific location-based word according to the information provided below. You have several attempts, and each guess result will be recorded in the History for future reference. Please provide your guess for this round based on the following information, e.g., 'Answer: happy'.
{board}
"""
# --------------------- Wordle 游戏逻辑 ---------------------

def get_word_bank(path: str = "words.txt"):
    """
    从文件中构建单词库：
      key = 单词长度
      value = 该长度的单词列表
    """
    word_bank = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if not word:
                    continue
                length = len(word)
                if length not in word_bank:
                    word_bank[length] = []
                word_bank[length].append(word)
    except Exception as e:
        raise Exception(f"读取单词文件失败：{e}")
    def _getter():
        return word_bank
    return _getter

def generate_secret_word(seed: int, level: int, bank_getter=None) -> str:
    """
    根据给定 seed 与 level（单词长度）生成随机 secret word。
    如果 bank_getter 为 None，则从 "words.txt" 中构建单词库。
    """
    if bank_getter is None:
        bank_getter = get_word_bank("words.txt")
    word_bank = bank_getter()
    possible_words = word_bank.get(level, [])
    if not possible_words:
        # 若没有指定长度的单词，则将所有单词合并
        possible_words = [w for w_list in word_bank.values() for w in w_list]
    random.seed(seed)
    secret_word = random.choice(possible_words)
    return secret_word

def verify_guess(word: str, guess: str) -> str:
    """
    对比用户猜测 guess 与 secret word，返回每个字母的反馈信息：
      - "The letter X located at idx=i is in the word and in the correct spot,"
      - "The letter X located at idx=i is in the word but in the wrong spot,"
      - "The letter X located at idx=i is not in the word in any spot,"
    """
    feedback_lines = []
    for i, g_char in enumerate(guess):
        if i < len(word) and g_char == word[i]:
            feedback_lines.append(
                f"The letter {g_char} located at idx={i} is in the word and in the correct spot,"
            )
        elif g_char in word:
            feedback_lines.append(
                f"The letter {g_char} located at idx={i} is in the word but in the wrong spot,"
            )
        else:
            feedback_lines.append(
                f"The letter {g_char} located at idx={i} is not in the word in any spot,"
            )
    return "\n".join(feedback_lines)

def generate(seed: int, bank_getter=None) -> dict:
    """
    生成 Wordle 游戏的初始状态，并返回一个 item 字典，包含：
      - secret_word：待猜单词（实际游戏中应隐藏）
      - epoch：当前回合（从 1 开始）
      - attempts：最大允许尝试次数（固定为 6）
      - history：猜测历史记录（列表，每条记录包含猜测与反馈）
      - is_end：游戏结束标志
      - prompt：当前游戏状态显示文本
      - action：用户最新的猜测（初始为空）
      - level：单词长度
      - seed：随机种子
      - score：当前得分（猜中为1，否则为0；未结束时默认为0）
    """
    random.seed(seed)
    level = random.randint(4, 12)
    secret_word = generate_secret_word(seed, level, bank_getter)
    item = {
        "epoch": 1,
        "secret_word": secret_word,
        "attempts": 10,
        "history": [],
        "is_end": False,
        "prompt": "",
        "action": "",
        "level": level,
        "seed": seed,
        "score": 0
    }
    return item

def verify(item: dict) -> dict:
    """
    根据 item 中的 action（用户的猜测）更新游戏状态：
      - 对比猜测与 secret word，生成反馈信息
      - 将猜测与反馈记录追加到 history 中
      - 若猜中或达到最大回合数，则设置 is_end 与 score
      - 增加 epoch，更新 prompt 字段
    返回更新后的 item。
    """
    guess = item.get("action", "").strip().lower()
    secret = item["secret_word"]
    
    # 如果猜测长度不符，进行补全或截断
    if len(guess) != len(secret):
        if len(guess) < len(secret):
            guess = guess + "-" * (len(secret) - len(guess))
        else:
            guess = guess[:len(secret)]
    
    feedback = verify_guess(secret, guess)
    item["history"].append({"guess": guess, "feedback": feedback})
    
    if guess == secret:
        item["score"] = 1
        item["is_end"] = True
    elif item["epoch"] >= item["attempts"]:
        item["score"] = 0
        item["is_end"] = True
    else:
        item["score"] = 0  # 游戏未结束时默认得分为0
    
    item["epoch"] += 1
    item["prompt"] = print_board(item)
    return item

def print_board(item: dict) -> str:
    """
    根据当前 item 状态生成游戏界面文本，
    显示游戏标题、当前回合、单词长度以及历史猜测与反馈。
    """
    lines = []
    lines.append("Wordle Game")
    current_attempt = min(item["epoch"], item["attempts"])
    lines.append(f"Attempt: {current_attempt} of {item['attempts']}")
    lines.append(f"Word length: {len(item['secret_word'])}")
    lines.append("History:")
    if item["history"]:
        for idx, entry in enumerate(item["history"], start=1):
            lines.append(f"{idx}. Guess: {entry['guess']}")
            lines.append("Feedback:")
            lines.append(entry["feedback"])
    return game_prompt.format(board="\n".join(lines))

# --------------------- FastAPI 接口及数据模型 ---------------------

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    epoch: int
    secret_word: str
    attempts: int
    history: List[dict]
    is_end: bool
    prompt: str
    action: str
    level: int
    seed: int
    score: Optional[int] = 0

class BoardRequest(BaseModel):
    board: str

@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    """
    根据传入的 seed 生成游戏初始状态
    """
    try:
        game_state = generate(request.seed)
        return game_state
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/print_board", response_model=BoardRequest)
def api_print_board(state: GameState):
    """
    根据当前游戏状态返回游戏界面文本
    """
    board_output = print_board(state.dict())
    return {"board": board_output}

@app.post("/verify", response_model=GameState)
def api_verify(state: GameState):
    """
    根据用户提交的猜测（action）更新游戏状态
    """
    updated_state = verify(state.dict())
    return updated_state

# --------------------- 程序入口 ---------------------

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)