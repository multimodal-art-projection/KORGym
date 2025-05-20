# game_lib/2-GuessWord/game_lib.py

#Standard libraries
import random
import string
import uvicorn
import argparse

#Commonly used open-source libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

def parse_init():
    """
    Parses command-line arguments for launching the FastAPI game server.

    Returns:
        argparse.Namespace: Contains the host and port configuration.
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
You are a good game player, I'll give you a game board and rules.\nYour task is:\n- First, give your answer according to the game board and rules.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: 1992/05/18'

{question}
'''
def print_board(item):
    """
    Constructs a natural language prompt for the model using the current game state.

    Args:
        item (dict): The game state, including the word length and letter-position rules.

    Returns:
        str: A formatted string prompt including rules and instructions.
    """
    rules_desc = []
    for pos, letter in item['rules']:
        rules_desc.append(f"the letter at position {pos+1} is '{letter}'")
    
    rules_text = " and ".join(rules_desc)
    board=f"Please provide an English word that meets the following requirements:\n1. The word must be {item['length']} letters long\n2. {rules_text}\n"
    prompt = game_prompt.format(question=board)
    return prompt

def get_valid_words(length, rules, word_list):
    """
    Filters a word list to obtain valid words satisfying the specified length and letter-position rules.

    Args:
        length (int): The target length of the word.
        rules (List[Tuple[int, str]]): A list of rules specifying required letters at specific positions.
        word_list (Set[str]): A list or set of candidate words.

    Returns:
        Set[str]: A set of valid words that meet the specified constraints.
    """
    valid_words = set()
    for word in word_list:
        if len(word.lower()) == length:
            match = True
            for pos, letter in rules:
                if pos >= length or word[pos].lower() != letter.lower():
                    match = False
                    break
            if match:
                valid_words.add(word.lower())
    return valid_words

def generate(seed):
    """
    Verifies whether the provided word (action) is correct based on the rule set and dictionary.

    Args:
        item (dict): A game state containing the model's proposed answer and game constraints.

    Returns:
        dict: The updated game state with a score (1 for correct, 0 for incorrect).
    """
    words = []
    with open("words.txt", "r") as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) <= 4:
                continue
            words.append(line)
    word_list = set(words)

    if seed is not None:
        random.seed(seed)

    while True:
        length = random.randint(5, 10)
        rules_num = random.randint(3, 4)
        rules = []
        attempts = 0
        max_attempts = 100

        while len(rules) < rules_num and attempts < max_attempts:
            pos = random.randint(0, length - 1)
            letter = random.choice(string.ascii_lowercase)

            if not any(pos == p for p, _ in rules):
                temp_rules = rules + [(pos, letter)]
                valid_words = get_valid_words(length, temp_rules, word_list)
                if valid_words:
                    rules = temp_rules
            attempts += 1

        if len(rules) == rules_num:
            break

    item = {
        'score': 0,
        'is_end': False,
        'response': [],
        'prompt': '',
        'action': '',
        'epoch': 1,
    }
    item['current_valid_words'] = get_valid_words(length, rules, word_list)
    item['length'] = length
    item['rules'] = rules
    return item


def verify(item):
    """
    Response model for the /print_board endpoint.

    Attributes:
        board (str): The prompt text shown to the model.
    """
    words = []
    with open("verify_words.txt", "r") as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) <= 4:
                continue
            words.append(line)
    word_list = set(words)
    answer=item['action']
    
    if len(answer) != item['length']:
        item['score'] = 0
        return item
        
    if answer.lower() not in word_list:
        item['score'] = 0
        return item
        
    for pos, letter in item['rules']:
        if answer[pos].lower() != letter.lower():
            item['score'] = 0
            return item
    item['score'] = 1
    return item

class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    current_valid_words: list
    length: int
    rules: list
    score: int
    is_end: bool
    action: str
    response: list
    prompt: str
    epoch: int

@app.post("/print_board", response_model=BoardRequest)
def api_print_board(request: GameState):
    state = request.dict()
    state['rules'] = [tuple(rule) for rule in state['rules']]
    board_output = print_board(state)
    return {"board": board_output}


@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    game_state = generate(request.seed)
    return game_state

@app.post("/verify", response_model=GameState)
def api_verify(request: GameState):
    state = request.dict()
    state['rules'] = [tuple(rule) for rule in state['rules']]
    updated_state = verify(state)
    return updated_state

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)

