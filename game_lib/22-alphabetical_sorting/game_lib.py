import os
import random
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
game_prompt = '''
You are a good game problem-solver, I'll give you a question.
Your task is:
- First, answer the question.
- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question, e.g. 'Answer: happy'
{board}
'''
# ------------------------------
# DFS及路径生成（3x3网格所有可能路径）
# ------------------------------
n = 3

def dfs(k, visited, n, i, j, path, ans):
    if i < 0 or i >= n or j < 0 or j >= n:
        return
    if visited[i][j]:
        return
    visited[i][j] = True
    path = path + [[i, j]]
    if k == n * n:
        ans.append(path[:])
        visited[i][j] = False
        return
    dfs(k + 1, visited, n, i + 1, j, path, ans)
    dfs(k + 1, visited, n, i - 1, j, path, ans)
    dfs(k + 1, visited, n, i, j + 1, path, ans)
    dfs(k + 1, visited, n, i, j - 1, path, ans)
    visited[i][j] = False

def generate_all_paths(n):
    all_paths = []
    for i in range(n):
        for j in range(n):
            ans = []
            visited = [[False] * n for _ in range(n)]
            dfs(1, visited, n, i, j, [], ans)
            all_paths.extend(ans)
    return all_paths

# 预先计算3x3棋盘中所有可能的路径
all_paths = generate_all_paths(n)

# ------------------------------
# 接口1：generate
# ------------------------------
def generate(seed: int) -> dict:
    """
    根据给定种子生成游戏状态。
    1. 从 words.txt 中读取所有长度为 9 的单词；
    2. 随机选择一个单词作为正确答案；
    3. 初始化 3x3 棋盘，并随机选取一条 DFS 路径，将正确答案的字母依次填入对应位置；
    4. 构造显示用的提示信息 prompt。
    返回的 item 包含：board, prompt, correct_word（隐藏答案，用于调试或验证）及其他控制字段。
    """
    words = []
    try:
        with open("words.txt", "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if len(word) == 9:
                    words.append(word)
    except Exception as e:
        raise Exception("读取 words.txt 出错: " + str(e))
    
    if not words:
        raise Exception("words.txt 中未找到长度为9的单词。")
    
    random.seed(seed)
    correct_word = random.choice(words)
    
    # 初始化3x3棋盘（空字符串表示未填充的格子）
    board = [[''] * n for _ in range(n)]
    # 从所有路径中随机选取一条路径，将 correct_word 的字母按顺序填入棋盘
    path = random.choice(all_paths)
    for pos, char in zip(path, correct_word):
        i, j = pos
        board[i][j] = char

    # 构造棋盘显示字符串，每行用"|"分隔
    board_str = "\n".join(["|".join(row) for row in board])
    prompt_text = (
        "Game rules: A word with a length of 9, randomly select a starting point in a 3x3 square, and fill in the letters in the order they appear in the word, selecting consecutive positions to place them in the grid. Please identify the word in the square.\n"
        "board:\n" + board_str
    )
    
    # 构造 item 信息
    item = {
        'answer': "",
        'question': prompt_text,
        'score': 0,
        'is_end': False,
        'response': [],
        'prompt': "",
        'action': "",
        'epoch': 1,
        'board': board,
        'correct_word': correct_word,  # 隐藏答案，仅供调试或验证用
    }
    return item

# ------------------------------
# 接口2：print_board
# ------------------------------
def print_board(item: dict) -> str:
    """
    根据 item 中的信息返回显示的棋盘提示信息。
    直接返回 generate 接口构造的 prompt 字符串。
    """
    return game_prompt.format(board=item.get('question', ''))

# ------------------------------
# 接口3：verify
# ------------------------------
def verify(item: dict) -> dict:
    """
    根据 item 中的 action（玩家输入答案）校验答案是否正确：
    1. 从 words.txt 中读取所有合法的 9 字母单词；
    2. 遍历所有可能的 DFS 路径，从 board 上构造单词，若单词在合法集合中则记录；
    3. 若玩家答案（忽略大小写）在可能答案中，则 score 置为 1，否则为 0。
    """
    try:
        words = []
        with open("words.txt", "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if len(word) == 9:
                    words.append(word)
        words_set = set(words)
        
        user_answer = item.get('action', '').strip().lower()
        board = item.get('board', [])
        possible_answers = set()
        for path in all_paths:
            word = "".join([board[i][j] for i, j in path])
            if word in words_set:
                possible_answers.add(word.lower())
        if user_answer in possible_answers:
            item['score'] = 1
        else:
            item['score'] = 0
    except Exception as e:
        print(f"验证过程中出现错误: {e}")
        item['score'] = 0
    return item

# ------------------------------
# FastAPI请求数据模型
# ------------------------------
class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    question: str
    board: list
    answer: str
    score: int
    is_end: bool
    action: str
    response: list
    prompt: str
    epoch: int
    correct_word: str = None

# ------------------------------
# FastAPI接口
# ------------------------------

@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    item = generate(request.seed)
    return item

@app.post("/print_board", response_model=BoardRequest)
def api_print_board(request: GameState):
    board_output = print_board(request.dict())
    return {"board": board_output}

@app.post("/verify", response_model=GameState)
def api_verify(request: GameState):
    state = request.dict()
    updated_state = verify(state)
    return updated_state

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)
