from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
import uvicorn
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
# 游戏提示（可选），格式化输出题目内容
game_prompt = '''
You are a good game problem-solver, I'll give you a question.
Your task is:
- First, answer the question.
- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question, e.g. 'Answer: happy'
{board}
'''
# 加载单词列表，文件名为 words.txt，要求文件编码为 utf-8
def load_words():
    words = []
    try:
        with open("words.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # 过滤掉长度小于等于4的单词
                if len(line) <= 4:
                    continue
                words.append(line)
    except Exception as e:
        print(f"Error loading words.txt: {e}")
    return words

WORDS = load_words()



# generate 接口：根据种子生成游戏状态，生成一个 anagram 题目
def generate(seed: int) -> dict:
    random.seed(seed)
    if not WORDS:
        raise ValueError("No words available. Please check words.txt.")
    # 随机选择一个单词作为正确答案
    correct_word = random.choice(WORDS)
    # 固定第一个字母，其余字母随机打乱
    chars = list(correct_word[1:])
    random.shuffle(chars)
    anagram = " ".join([correct_word[0]] + chars)
    # 构造题目文本
    question = (
        "Please rearrange the letters to form the original word for this anagram. "
        "The first letter is already in the correct position.\n" + anagram
    )
    # 用 item 作为游戏状态的载体
    item = {
        "correct_word": correct_word,
        "board": question,
        "action": "",
        "score": 0
    }
    return item

# print_board 接口：根据 item 返回题目的文本
def print_board(item: dict) -> str:
    return game_prompt.format(board=item.get("board", ""))

# verify 接口：验证用户输入是否正确，并更新 item 中的 score
def verify(item: dict) -> dict:
    correct_word = item.get("correct_word", "")
    user_answer = item.get("action", "").strip().lower()
    if not correct_word:
        item["score"] = 0
        return item
    # 简单比较（均转为小写进行比较）
    if user_answer == correct_word.lower():
        item["score"] = 1
    else:
        item["score"] = 0
    return item

# 定义请求和响应的数据模型
class GenerateRequest(BaseModel):
    seed: int

class BoardRequest(BaseModel):
    board: str

class GameState(BaseModel):
    board: str
    correct_word: str = ""
    action: str = ""
    score: int = 0

# API 接口：生成初始游戏状态
@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    state = generate(request.seed)
    return state

# API 接口：输出题面文本
@app.post("/print_board", response_model=BoardRequest)
def api_print_board(state: GameState):
    board_text = print_board(state.dict())
    return {"board": board_text}

# API 接口：根据用户提交的答案进行验证，并更新状态
@app.post("/verify", response_model=GameState)
def api_verify(state: GameState):
    updated_state = verify(state.dict())
    return updated_state

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)

