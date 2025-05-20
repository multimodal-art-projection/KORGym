# game_lib/15-emoji_connect/game_lib.py

#Standard libraries
from typing import List
import random
import time
import ast
import argparse

#Commonly used open-source libraries
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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
game_prompt='''
You are a good game problem-solver, I'll give you a question.\nYour task is:\n- First, answer the question.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: 192'
There is a rectangular board made up of emojis. Your task is to count the number of horizontal or vertical lines formed by the same emoji, with a length of 2 or more. Output the total count of such lines as the answer.
board:
{board}
Please provide the number as your answer,e.g.'Answer: 192'
'''
def print_board(item):
    output = ""
    for line in item['board']:
        output += "".join(line)
        output += '\n'
    return game_prompt.format(board=output)
def generate(seed: int):
    random.seed(seed)
    emoji_num = random.randint(3,10)
    if emoji_num<=3:
        scale=[5,5]
    elif emoji_num>3 and emoji_num<=5:
        scale=[6,6]
    elif emoji_num>5 and emoji_num<=7:
        scale=[7,7]
    else:
        scale=[10,10]
    random.seed(seed)
    # 预定义的emoji列表，足够多的常见emoji
    all_emojis = [
        "😀", "😃", "😄", "😁", "😆", "🥰", "🏄", "🦭", "🧽", "🤚", "🚀", "🎁",
        "🐶", "🐱", "🐭", "🐹", "🐰", "🦊", "🐻", "🐼", "🐨", "🐯", "🦁", "🐮",
        "🐷", "🐸", "🐵", "🐔", "🐧", "🐦", "🐤", "🐣", "🐥", "🦆", "🦅", "🦉",
        "🦇", "🐺", "🐗", "🐴", "🦄", "🐝", "🐛", "🦋", "🐌", "🐞", "🐜", "🦟",
        "🦗", "🕷", "🦂", "🐢", "🐍", "🦎", "🦖", "🦕", "🐙", "🦑", "🦐", "🦞",
        "🦀", "🐡", "🐠", "🐟", "🐬", "🐳", "🐋", "🦈", "🐊", "🐅", "🐆", "🦓",
        "🦍", "🐘", "🦏", "🦛", "🐪", "🐫", "🦒", "🦘", "🐃", "🐂", "🐄", "🐎",
        "🐖", "🐏", "🐑", "🦙", "🐐", "🐕", "🐩", "🦮", "🐈", "🐓", "🦃", "🦚",
        "🦜", "🦢", "🦩", "🦨", "🦦", "🦥", "🐿", "🦔", "🌵", "🎄", "🌲", "🌳",
        "🌴", "🌱", "🌿", "☘️", "🍀", "🎍", "🎋", "🍃", "🍂", "🍁", "🌾", "🌺",
        "🌻", "🌹", "🥀", "🌷", "🌼", "🌸", "💐", "🍄", "🌰", "🎃", "🐚", "🪐",
        "🌎", "🌍", "🌏", "🌕", "🌖", "🌗", "🌘", "🌑", "🌒", "🌓", "🌔", "🌚",
        "🌝", "🌞", "🌙", "⭐️", "🌟", "💫", "✨", "☄️", "🔥", "💥", "🌈", "☀️",
        "⛅️", "☁️", "❄️", "💧", "💦", "🌊"
    ]
    
    # 确保不重复选择emojis
    selected_emojis = random.sample(all_emojis, emoji_num)
    
    rows, cols = scale[0], scale[1]
    board = []
    for _ in range(rows):
        row = [random.choice(selected_emojis) for _ in range(cols)]
        board.append(row)
    item = {
        'answer': 0,
        'score': 0,
        'is_end': False,
        'response': [],
        'prompt': '',
        'action': '',
        'epoch': 1,
    }
    item['board'] = board
    return item

def calculate_lines(board: List[List[str]]) -> int:
    if not board:
        return 0
    rows = len(board)
    cols = len(board[0]) if rows > 0 else 0
    total = 0
    
    # 检查行
    for row in board:
        current_len = 1
        current_emoji = row[0]
        for emoji in row[1:]:
            if emoji == current_emoji:
                current_len += 1
            else:
                if current_len >= 2:
                    total += 1 
                current_emoji = emoji
                current_len = 1
        if current_len >= 2:
            total += 1 
    # 检查列
    for c in range(cols):
        current_len = 1
        current_emoji = board[0][c]
        for r in range(1, rows):
            emoji = board[r][c]
            if emoji == current_emoji:
                current_len += 1
            else:
                if current_len >= 2:
                    total += 1 
                current_emoji = emoji
                current_len = 1
        if current_len >= 2:
            total += 1 
    
    return total

def verify(item):
    try:
        board = item['board']
        correct = calculate_lines(board)
        item['answer'] = correct
        
        # 检查 action 是否为空
        if item['action'].strip().lower() == "":
            item['score'] = 0
            return item
        
        # 尝试将 action 转换为整数
        answer = int(item['action'].strip().lower())
        item['answer'] = correct
        # 如果答案正确，得分为 1，否则为 0
        if answer == correct:
            item['score'] = 1
        else:
            item['score'] = 0
            
    except (ValueError, TypeError) as e:
        # 如果 action 无法转换为整数，或 item['action'] 为空/无效，设置 score 为 0
        print(f"Error in converting action: {e}")
        item['score'] = 0
    except KeyError as e:
        # 如果 item 字典中缺少键，设置 score 为 0
        print(f"KeyError: Missing key in the item: {e}")
        item['score'] = 0
    except Exception as e:
        # 捕获其他所有异常并设置 score 为 0
        print(f"An unexpected error occurred: {e}")
        item['score'] = 0
    
    return item


def test():
    # 测试样例
    board = [
        ['🏄', '🏄', '🥰', '🏄', '🦭'],
        ['🥰', '🥰', '🥰', '🥰', '🥰'],
        ['🦭', '🦭', '🥰', '🧽', '🤚']
    ]
    print(calculate_lines(board))
    assert calculate_lines(board) == 4
    
    # 修改此处：原预期6改为3
    board1 = [
        ['A', 'A', 'A'],
        ['B', 'B', 'B'],
        ['C', 'C', 'C']
    ]
    assert calculate_lines(board1) == 3  # 3行，列无
    
    # 测试部分行和列
    board2 = [
        ['A', 'B', 'A'],
        ['B', 'B', 'B'],
        ['C', 'C', 'D']
    ]
    assert calculate_lines(board2) == 3  # 2行+1列
    
    # 测试单行单列
    board3 = [
        ['A', 'A']
    ]
    assert calculate_lines(board3) == 1  # 1行
    
    board4 = [
        ['A'],
        ['A']
    ]
    assert calculate_lines(board4) == 1  # 1列
    
    print("All test cases passed!")

# if __name__ == "__main__":
#     item = generate(1223)
#     print(print_board(item))
#     item['action'] = "21"
#     print('score:',verify(item)['score'])

class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    board: list
    answer: int
    score: int
    is_end: bool
    action: str
    response: list
    prompt: str
    epoch: int
# 生成初始游戏状态
@app.post("/print_board", response_model=BoardRequest)
def api_print_board(request: GameState):
    state = request.dict()
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
    updated_state = verify(state)
    return updated_state

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)