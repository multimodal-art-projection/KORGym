import random
import string
import os
import collections
from typing import Optional, List, Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
game_prompt = """
You are a good game player, I'll give you a game board and rules.
Your task is:
- First, give your answer according to the game board and rules.
- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: 😀=c,😂=d...'.

Next, I will provide a sentence encoded by replacing each letter with a unique emoji. Then, I will reveal the letter corresponding to the most frequently occurring emoji. You'll have several attempts to guess the words, and each guess will be recorded in History for future reference. Based on the provided information, please submit your guesses for this round in the format 'emoji=word', separated by commas, e.g., 'Answer: 😀=c,😂=d...'.
Note that the emoji provided as a hint, as well as previously correctly answered emojis, must also be included in your answer.
{board}
"""

# --------------------- CryptoWord 游戏逻辑 ---------------------
class CryptoWord:
    def __init__(self, sentences_file: str = "sentences.txt"):
        """
        初始化 CryptoWord 游戏
        Args:
            sentences_file (str): 包含句子的文件路径
        """
        self.sentences_file = sentences_file
        self.sentences = self._load_sentences()
        self.emojis = [
            "😀", "😂", "😍", "🤔", "😎", "🥳", "😴", "🤩", "🥺", "😱",
            "🙄", "😇", "🤗", "🤫", "🤭", "🤥", "🤮", "🤧", "🥶", "🥵",
            "🤠", "🥴", "🤑", "🤓", "🧐", "😈", "👻", "👽", "🤖", "💩",
            "🐶", "🐱", "🐭", "🐹", "🐰", "🦊", "🐻", "🐼", "🐨", "🐯",
            "🦁", "🐮", "🐷", "🐸", "🐵", "🐔", "🐧", "🐦", "🦆", "🦉"
        ]
        
    def _load_sentences(self) -> List[str]:
        """从文件中加载句子；如果文件不存在则使用默认句子"""
        if not os.path.exists(self.sentences_file):
            return [
                "The quick brown fox jumps over the lazy dog.",
                "Pack my box with five dozen liquor jugs.",
                "How vexingly quick daft zebras jump!",
                "Sphinx of black quartz, judge my vow."
            ]
        with open(self.sentences_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    def generate(self, seed: Optional[int] = None, encoding_table: Optional[Dict[str, str]] = None, replacement_ratio: float = 0.5) -> Dict[str, Any]:
        """
        生成编码后的句子
        Args:
            seed (int, optional): 随机种子
            encoding_table (dict, optional): 自定义编码表
            replacement_ratio (float): 替换字母的比例（0.0-1.0）
        Returns:
            dict: 包含 encoded_sentence, hint（最高频 emoji 提示）及 answer（完整映射）
        """
        if seed is not None:
            random.seed(seed)
        
        # 选择一个随机句子
        original_sentence = random.choice(self.sentences)
        original_sentence_lower = original_sentence.lower()
        
        # 统计句子中字母频率
        letter_counts = collections.Counter([c for c in original_sentence_lower if c in string.ascii_lowercase])
        unique_letters = list(letter_counts.keys())
        
        # 确定替换字母数量
        num_unique_letters = len(unique_letters)
        num_to_replace = max(1, min(int(num_unique_letters * replacement_ratio), num_unique_letters))
        
        # 随机选择要替换的字母
        random.shuffle(unique_letters)
        letters_to_replace = unique_letters[:num_to_replace]
        
        # 准备 emoji 列表
        available_emojis = self.emojis.copy()
        random.shuffle(available_emojis)
        
        # 构造编码表
        if encoding_table is None:
            encoding_table = {}
            for i, letter in enumerate(letters_to_replace):
                if i < len(available_emojis):
                    encoding_table[letter] = available_emojis[i]
        
        # 构造反向映射，用于生成答案提示
        reverse_mapping = {v: k for k, v in encoding_table.items()}
        
        # 对句子进行编码
        encoded_sentence = ""
        for char in original_sentence_lower:
            if char in encoding_table:
                encoded_sentence += encoding_table[char]
            else:
                encoded_sentence += char
        
        # 找出句子中出现次数最多的 emoji
        emoji_counts = {}
        for char in encoded_sentence:
            if char in reverse_mapping:
                emoji_counts[char] = emoji_counts.get(char, 0) + 1
        
        most_frequent_emoji = None
        if emoji_counts:
            most_frequent_emoji = max(emoji_counts.items(), key=lambda x: x[1])[0]
        
        hint_answer = ""
        if most_frequent_emoji:
            hint_answer = f"{most_frequent_emoji}={reverse_mapping[most_frequent_emoji]}"
        
        return {
            "encoded_sentence": encoded_sentence,
            "hint": hint_answer,
            "answer": reverse_mapping  # 完整 emoji -> letter 映射
        }
    
    def verify(self, answer: Dict[str, str], generated_answer: str) -> Dict[str, Dict[str, bool]]:
        """
        验证玩家的答案
        Args:
            answer (dict): 正确的 emoji-to-letter 映射
            generated_answer (str): 玩家提交的答案，格式为 "emoji=letter,emoji=letter,..."
        Returns:
            dict: 每个 emoji 的验证反馈，True 表示回答正确
        """
        feedback = {}
        # 解析玩家答案，要求格式：emoji=letter, emoji=letter, ...
        for pair in generated_answer.split(','):
            pair = pair.strip()
            if '=' in pair:
                emoji_guess, letter_guess = pair.split('=', 1)
                emoji_guess = emoji_guess.strip()
                letter_guess = letter_guess.strip().lower()
                if emoji_guess in answer:
                    feedback[emoji_guess] = (letter_guess == answer[emoji_guess])
                else:
                    feedback[emoji_guess] = False
        return {"feedback": feedback}

# --------------------- 接口封装 ---------------------
# 以下三个接口均使用 item 字典作为状态载体

def generate(seed: int) :
    """
    生成 CryptoWord 游戏的初始状态
    返回的 item 包含：
      - encoded_sentence: 编码后的句子
      - hint: 最高频 emoji 的提示
      - answer: 完整的 emoji-to-letter 映射（实际游戏中应隐藏）
      - epoch: 当前尝试轮数（从 1 开始）
      - max_attempts: 最大尝试次数（固定为 10）
      - history: 猜测历史记录（列表，每条记录包含猜测与反馈）
      - correct_guesses: 正确识别的 emoji 映射（字典）
      - action: 用户最新提交的答案（初始为空）
      - is_end: 游戏结束标志
      - seed: 随机种子
      - replacement_ratio: 替换比例
      - score: 当前得分（猜中为 1，否则为 0）
      - prompt: 当前游戏状态文本（由 print_board 生成）
    """
    random.seed(seed)
    game = CryptoWord()
    replacement_ratio = random.randint(3,10)/10
    result = game.generate(seed=seed, replacement_ratio=replacement_ratio)
    item = {
        "epoch": 1,
        "max_attempts": 10,
        "encoded_sentence": result["encoded_sentence"],
        "hint": result["hint"],
        "answer": result["answer"],
        "history": [],
        "correct_guesses": {},
        "action": "",
        "is_end": False,
        "seed": seed,
        "replacement_ratio": replacement_ratio,
        "score": 0,
        "prompt": ""
    }
    return item

def verify(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据 item 中的 action（玩家答案）更新游戏状态：
      - 调用 CryptoWord.verify 进行验证，生成反馈
      - 将猜测与反馈追加到 history 中
      - 更新 correct_guesses（仅记录回答正确的 emoji）
      - 判断是否全部答对或达到最大尝试次数，从而设置 is_end 与 score
      - 增加 epoch，并更新 prompt
    返回更新后的 item。
    """
    guess = item.get("action", "").strip()
    if not guess:
        return item  # 如果没有输入答案，不做处理

    game = CryptoWord()
    verification_result = game.verify(item["answer"], guess)
    feedback = verification_result.get("feedback", {})

    # 记录本次猜测及反馈
    item["history"].append({"guess": guess, "feedback": feedback})

    # 解析玩家输入，更新正确猜测记录
    pairs = [pair.strip() for pair in guess.split(',') if '=' in pair]
    for pair in pairs:
        emoji, letter = [p.strip() for p in pair.split('=', 1)]
        if emoji in feedback and feedback[emoji]:
            item["correct_guesses"][emoji] = letter.lower()

    # 判断游戏是否全部答对
    if len(item["correct_guesses"]) == len(item["answer"]):
        item["is_end"] = True
        item["score"] = 1
    elif item["epoch"] >= item["max_attempts"]:
        item["is_end"] = True
        item["score"] = 0
    else:
        item["score"] = 0

    item["epoch"] += 1
    return item

def print_board(item: Dict[str, Any]) -> str:
    """
    根据当前 item 状态生成游戏界面文本，
    显示游戏标题、当前尝试轮数、编码句子、提示以及历史猜测与反馈。
    """
    lines = []
    lines.append("Crypto Word Game")
    current_attempt = min(item["epoch"], item["max_attempts"])
    lines.append(f"Attempt: {current_attempt} of {item['max_attempts']}")
    lines.append(f"Encoded Sentence: {item['encoded_sentence']}")
    lines.append(f"Hint: {item['hint']}")
    lines.append("History:")
    if item["history"]:
        for idx, entry in enumerate(item["history"], start=1):
            lines.append(f"{idx}. Guess: {entry['guess']}")
            lines.append(f"   Feedback: {entry['feedback']}")
    else:
        lines.append("No guesses yet.")
    return game_prompt.format(board="\n".join(lines))

# --------------------- FastAPI 接口及数据模型 ---------------------
app = FastAPI()

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    epoch: int
    max_attempts: int
    encoded_sentence: str
    hint: str
    answer: Dict[str, str]
    history: List[Dict[str, Any]]
    correct_guesses: Dict[str, str]
    is_end: bool
    action: str
    seed: int
    replacement_ratio: float
    score: Optional[int] = 0
    prompt: str

class BoardRequest(BaseModel):
    board: str

@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    """
    根据传入的 seed 及替换比例生成游戏初始状态
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
    根据玩家提交的答案（action）更新游戏状态
    """
    updated_state = verify(state.dict())
    return updated_state

# --------------------- 程序入口 ---------------------
if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)