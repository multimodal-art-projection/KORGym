# game_lib/5-light_out_game/game_lib.py

#Standard libraries
import random
import re
import argparse

#Commonly used open-source libraries
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

def parse_init():
    """
    Parses command-line arguments for launching the FastAPI server.

    Returns:
        argparse.Namespace: Object containing host and port settings.
    """

    parser = argparse.ArgumentParser(description="Data creation utility")

    parser.add_argument('-p', '--port', type=int, default=8775, help='服务部署端口')
    parser.add_argument('-H', '--host', type=str, default="0.0.0.0", help='服务部署地址')
    args = parser.parse_args()
    return args

app = FastAPI()
light_out_game_prompt='''
You are a good game problem-solver, I'll give you a game board and rules.\nYour task is:\n- First, give your answer according to the game board and rules.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: (0,2), (2,1)'
The game consists of a 3 by 3 grid of lights at (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1) and (2,2). '1' means the light at that position is on and '0' means the light at that position is off. When the game starts, a random number or a stored pattern of these lights is switched on. Pressing any of the lights will toggle it and the adjacent lights(up, left, right and down).For example, if the board is
000
000
000
you press the button at (1,1), the board will be
010
111
010
If the light is at the boundary of the board, it will only affect its adjacent lights. For example, if the board is
000
000
000
you press the button at (2,1), the board will be
000
010
111
The goal of this game is to switch all the lights off, preferably in as few button presses as possible. You should give you answer by a series of (a,b), which means press the light at row a and column b.You should give a series of (a,b) split by ',' to switch all the lights off.If the answer is not unique, just provide one correct answer.
Example 1:
If the board is 
000
010
111
We press the button (2,1),  which will toggle the light at (2,1) and toggle the adjacent lights (1,1), (2,0) and (2,2). The game board is
000
000
000
All the lights have been switched off. So, your answer can be 'Answer: (2,1)'.
Example 2:
If the board is 
100
011
010
First,  we press the button (0,0), which will toggle the light at (0,0) and toggle the adjacent lights (0,1) and (1,0). The game board is
010
111
010
Then, we press the button (1,1), which will toggle the light at (1,1) and toggle the adjacent lights (0,1),(1,0), (1,2) and (2,1) .The game board is
000
000
000
All the lights have been switched off. So, your answer can be 'Answer: (0,0), (1,1)'.
Example 3:
If the board is 
011
000
011
We press the button (2,2),  which will toggle the light at (2,2) and toggle the adjacent lights (2,1) and (1,2). The game board is
011
001
000
We press the button (0,2),  which will toggle the light at (0,2) and toggle the adjacent lights (0,1) and (1,2). The game board is
000
000
000
All the lights have been switched off. So, your answer can be 'Answer: (2,2) ,(0,2)'.
Board:
{board}
'''

def toggle(board, i, j):
    """
    Toggles the light at (i, j) and its orthogonal neighbors (up, down, left, right).

    Args:
        board (List[List[int]]): The game board.
        i (int): Row index.
        j (int): Column index.
    """
    n = len(board)
    board[i][j] ^= 1  
    if i > 0:
        board[i-1][j] ^= 1
    if i < n - 1:
        board[i+1][j] ^= 1
    if j > 0:
        board[i][j-1] ^= 1
    if j < n - 1:
        board[i][j+1] ^= 1

def print_board(item):
    """
    Converts the current game board into a formatted prompt string for the model.

    Args:
        item (dict): The game state containing the current board.

    Returns:
        str: A natural language prompt with the game rules and current board.
    """
    board=item['board']
    board_size=len(board)
    output=""
    for i in range(board_size):
        for j in range(board_size):
            output+=str(board[i][j])
            if j == board_size-1:
                output+='\n'
    return light_out_game_prompt.format(board=output)

def generate(seed):
    """
    Generates a new solvable 'Lights Out' puzzle instance based on a seed.

    Args:
        seed (int): Random seed to initialize puzzle.

    Returns:
        dict: A dictionary representing the initial game state.
    """
    random.seed(seed)
    level = random.randint(1,15)
    if level <= 5:
        n = 3
        k = level
    else:
        n = 4
        k = level-4
    board = [[0 for _ in range(n)] for _ in range(n)]
    all_positions = [(i, j) for i in range(n) for j in range(n)]
    selected_positions = random.sample(all_positions, k)
    for i, j in selected_positions:
        toggle(board, i, j)
    item = {
        'score': 0,
        'is_end': False,
        'response': [],
        'prompt': '',
        'action': '',
        'epoch': 1,
    }
    item['board'] = board
    item['level'] = level
    return item

def verify(item):
    """
    Verifies whether a sequence of light presses results in turning off all lights.

    Args:
        item (dict): The game state, including initial board and user action string.

    Returns:
        dict: Updated game state with `score = 1` if the solution is correct, otherwise 0.
    """

    board=item['board']
    action_str=item['action']
    answer = [
            tuple(map(int, re.findall(r'\d+', item.strip())))  
            for item in action_str.split('),') if item.strip() 
        ]
    
    if not board:
        item['score']=0  
    n = len(board)
    current = [row.copy() for row in board]
    for step in answer:
        if len(step) != 2:
            item['score']=0
            return item
        i, j = step
        if i < 0 or i >= n or j < 0 or j >= n:
            item['score']=0
            return item
        toggle(current, i, j)
    for row in current:
        if any(row):
            item['score']=0
            return item
    item['score']=1
    return item

def test():
    item1={}
    item2={}
    item3={}
    item1['board'] = [
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ]
    item1['action'] = [(0, 0)]
    assert verify(item1)['score'] == 1, "测试用例1失败：正确解未被接受"

    item2['board'] = [
        [1, 0],
        [0, 1]
    ]
    item2['action'] = [(2, 2)]  # 2x2 棋盘的合法坐标为 (0,0)-(1,1)
    assert verify(item2)['score'] == 0, "测试用例2失败：非法坐标未被检测"

    item3['board'] = [
		[1, 1, 1],
		[0, 1, 0],
		[0, 0, 0]
	]
    item3['action'] = [(1, 2)]
    assert verify(item3)['score'] == 0, "测试用例3失败：错误解未被检测"

    print("✅ 所有测试通过！")

class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    level : int
    board : list
    score: int
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
    state = request.dict()
    updated_state = verify(state)
    return updated_state

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)
