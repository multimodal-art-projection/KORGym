# game_lib/1-DateCount/game_lib.py

#Standard libraries
import random
from datetime import datetime, timedelta
import json
import uvicorn

#Commonly used open-source libraries
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

def parse_init():
    """
    Parses command-line arguments for launching the FastAPI server.

    Returns:
        argparse.Namespace: Contains host and port settings for the server.
    """
    parser = argparse.ArgumentParser(description="Data creation utility")

    parser.add_argument('-p', '--port', type=int, default=8775, help='服务部署端口')
    parser.add_argument('-H', '--host', type=str, default="0.0.0.0", help='服务部署地址')
    args = parser.parse_args()
    return args
app = FastAPI()
game_prompt='''
You are a good game player, I'll give you a game board and rules.\nYour task is:\n- First, give your answer according to the game board and rules.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: 1992/05/18'

{question}
'''
def print_board(item):
    """
    Formats the game state into a natural language prompt for the model.

    Args:
        item (dict): A dictionary containing the current game state including the question.

    Returns:
        str: A formatted string prompt for the model, including game rules and the question.
    """
    prompt = game_prompt.format(question=item['current_problem'])
    return prompt
        
def generate(seed=None):
    """
    Generates a new game instance consisting of a date offset question.

    Args:
        seed (int, optional): Random seed for reproducibility.

    Returns:
        dict: A game state dictionary including the question, answer, and metadata.
    """
    item = {
        'score': 0,
        'is_end': False,
        'response': [],
        'prompt': '',
        'action': '',
        'epoch' : 1,
    }
    if seed is not None:
        random.seed(seed)
        
    year = random.randint(500, 1525)
    month = random.randint(1, 12)
    day = random.randint(1, 28)  
    offset = random.randint(-100000, 100000)
    base_date = datetime(year, month, day)
    target_date = base_date + timedelta(days=offset)  # 目标日期

    item['correct_answer'] = target_date.strftime("%Y/%m/%d")
    direction = "ago" if offset > 0 else "later"
    abs_offset = abs(offset)
    item['current_problem'] = f"The date {abs_offset} days {direction} is {base_date.year}/{base_date.month}/{base_date.day}, what is the date today? (The output should be in the format: 'Answer: year/month/date')"
    return item 
    
def verify(item):
    """
    Checks whether the model's action matches the correct answer.

    Args:
        item (dict): The game state containing the model's response and correct answer.

    Returns:
        dict: The updated game state with score set to 1 if correct, else 0.
    """

    answer = str(item['action']).strip()
    correct_answer = str(item['correct_answer']).strip()
    item['score']= 1 if answer == correct_answer else 0
    return item

class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    correct_answer: str
    current_problem: str
    score: int
    is_end: bool
    action: str
    response: list
    prompt: str
    epoch: int

@app.post("/print_board", response_model=BoardRequest)
def api_print_board(request: GameState):
    """
    FastAPI endpoint to render the current game state into a model-friendly prompt.

    Args:
        request (GameState): The input game state.

    Returns:
        dict: Dictionary with a single key "board" containing the formatted prompt.
    """
    state = request.dict()
    board_output = print_board(state)
    return {"board": board_output}


# 生成初始游戏状态
@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    """
    FastAPI endpoint to generate a new game instance based on the input seed.

    Args:
        request (GenerateRequest): Contains a seed value for reproducibility.

    Returns:
        GameState: The initial game state.
    """
    game_state = generate(request.seed)
    return game_state

# 根据动作更新游戏状态
@app.post("/verify", response_model=GameState)
def api_verify(request: GameState):
    """
    FastAPI endpoint to verify the model's action and return the updated game state.

    Args:
        request (GameState): Contains the game state and action to verify.

    Returns:
        GameState: Updated game state with score set.
    """
    state = request.dict()
    updated_state = verify(state)
    return updated_state

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)
