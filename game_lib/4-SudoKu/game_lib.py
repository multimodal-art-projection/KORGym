# game_lib/4-SudoKu/game_lib.py

#Standard libraries
import math
import ast
import random
import copy
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
size = 9
sqrt_size = int(math.sqrt(size))
difficulty_levels = {
            "easy": ((int(size * size/2) - int(size * size/10)), 0),
            "moderate": (int(size * size), int(size * size/15)),
            "difficult": (int(size * size), int(size * size/10))
        }

game_prompt='''
You are a good game player, I'll give you a game board and rules.\nYour task is:\n- First, give your answer according to the game board and rules.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: happy'

{board}
'''

def print_board(item):
    """
    Generates a natural language prompt describing Sudoku rules and the current board.

    Args:
        item (dict): The current game state containing the Sudoku board.

    Returns:
        str: Formatted prompt string for the LLM.
    """
    prompt = "Please solve this Sudoku puzzle. Fill in the empty cells (marked as 0) "
    prompt += "with numbers 1-9 so that each row, column, and 3x3 block contains "
    prompt += "all numbers from 1 to 9 without repetition.\n\n"
    prompt += "Please provide your solution in exactly the same format as above, "
    prompt += "i.e., a 9x9 grid where each row is a list of numbers. "
    prompt += "Example format: Answer: [[1, 2, 3, 4, 5, 6, 7, 8, 9],[4, 5, 6, 7, 8, 9, 1, 2, 3],...]\n"
    prompt += "Current Sudoku board:\n"
    for row in item['current_board']:
        prompt += str(row) + "\n"
    return game_prompt.format(board=prompt)

def generate(seed=None):
    """
    Generates a new Sudoku puzzle with a specific difficulty.

    Args:
        seed (int, optional): Random seed for reproducibility.

    Returns:
        dict: Game state including the full solution, initial puzzle, and metadata.
    """
    item = {
        'score': 0,
        'is_end': False,
        'response': [],
        'prompt': '',
        'action': '',
        'epoch': 1,
    }

    if seed is not None:
        random.seed(seed)
        
    difficulty = random.choice(["easy", "moderate", "difficult"])
    item['difficulty'] = difficulty
    print("cur difficulty: ", difficulty)
    item['solution'] = make_board()
    print("cur solution: ", item['solution'])
    item['current_board'] = [row[:] for row in item['solution']]
    item['current_board']=make_puzzle(item['current_board'],difficulty)
    return item

def make_board():
    """
    Constructs a complete, valid Sudoku solution.

    Returns:
        List[List[int]]: A fully-filled 9x9 Sudoku board.
    """
    board = [[0 for _ in range(size)] for _ in range(size)]
    
    for i in range(size):
        for j in range(size):
            board[i][j] = int((i * sqrt_size + i//sqrt_size + j) % size) + 1
            
    for _ in range(random.randint(8, 15)):
        board = shuffle_board(board)
        
    return board

def shuffle_board(board):
    """
    Shuffles a complete board by swapping values and rows within blocks.

    Args:
        board (List[List[int]]): Sudoku solution board.

    Returns:
        List[List[int]]: Shuffled board maintaining Sudoku validity.
    """
    num1 = random.randint(1, size)
    num2 = random.randint(1, size)
    while num2 == num1:
        num2 = random.randint(1, size)
        
    for i in range(size):
        for j in range(size):
            if board[i][j] == num1:
                board[i][j] = num2
            elif board[i][j] == num2:
                board[i][j] = num1
                
    if sqrt_size > 1:
        block = random.randint(0, sqrt_size-1)
        row1 = random.randint(0, sqrt_size-1) + block * sqrt_size
        row2 = random.randint(0, sqrt_size-1) + block * sqrt_size
        while row2 == row1:
            row2 = random.randint(0, sqrt_size-1) + block * sqrt_size
        board[row1], board[row2] = board[row2], board[row1]
    return board
        
def remove_numbers_logically(current_board, cutoff):
    """
    Removes numbers from a board using logical constraint checks.

    Args:
        current_board (List[List[int]]): The filled Sudoku board.
        cutoff (int): Number of digits to attempt removing logically.

    Returns:
        List[List[int]]: Partially filled board (the puzzle).
    """
    removed_items = 0
    for _ in range(size * 500):
        if removed_items >= cutoff:
            break
            
        i = random.randint(0, size-1)
        j = random.randint(0, size-1)
        
        if current_board[i][j] == 0:
            continue
            
        temp = current_board[i][j]
        current_board[i][j] = 0
        
        if len(get_possible_numbers(current_board,i, j)) == 1:
            removed_items += 1
        else:
            current_board[i][j] = temp
    return current_board

def remove_numbers_randomly(current_board, cutoff):
    """
    Removes numbers from the board randomly while ensuring the puzzle remains solvable.

    Args:
        current_board (List[List[int]]): Board to modify.
        cutoff (int): Number of removals.

    Returns:
        List[List[int]]: The updated puzzle board.
    """
    removed_items = 0
    for i in range(size):
        for j in range(size):
            if removed_items >= cutoff:
                return current_board  
            if current_board[i][j] == 0:
                continue
            temp = current_board[i][j]
            current_board[i][j] = 0
            test_board = [row[:] for row in current_board]
            
            if solve_board(test_board):
                removed_items += 1
            else:
                current_board[i][j] = temp
    return current_board


def get_possible_numbers(current_board, row, col):
    """
    Returns all valid numbers for a given cell in the Sudoku grid.

    Args:
        current_board (List[List[int]]): The current board state.
        row (int): Row index.
        col (int): Column index.

    Returns:
        List[int]: Valid numbers that can go into the specified cell.
    """
    possible = set(range(1, size + 1))
    possible -= set(current_board[row])
    possible -= set(current_board[i][col] for i in range(size))
    
    start_row = (row // sqrt_size) * sqrt_size
    start_col = (col // sqrt_size) * sqrt_size
    for i in range(start_row, start_row + sqrt_size):
        for j in range(start_col, start_col + sqrt_size):
            possible.discard(current_board[i][j])
            
    return list(possible)

def solve_board( board):
    """
    Solves a Sudoku board using backtracking.

    Args:
        board (List[List[int]]): The Sudoku board to solve.

    Returns:
        bool: True if the board is solvable, False otherwise.
    """
    empty = find_empty(board)
    if not empty:
        return True
        
    row, col = empty
    
    for num in range(1, size + 1):
        if is_safe(board, row, col, num):
            board[row][col] = num
            if solve_board(board):
                return True
            board[row][col] = 0
            
    return False

def find_empty(board):
    """
    Finds the first empty cell (with value 0).

    Args:
        board (List[List[int]]): The board to search.

    Returns:
        Tuple[int, int] or None: Row and column of the empty cell or None if full.
    """
    for i in range(size):
        for j in range(size):
            if board[i][j] == 0:
                return (i, j)
    return None

def is_safe( board, row, col, num):
    """
    Checks whether placing a number at a specific position is valid.

    Args:
        board (List[List[int]]): Current board.
        row (int): Row index.
        col (int): Column index.
        num (int): Number to check.

    Returns:
        bool: True if the number can be safely placed, False otherwise.
    """
    if num in board[row]:
        return False
        
    if num in [board[i][col] for i in range(size)]:
        return False
        
    start_row = (row // sqrt_size) * sqrt_size
    start_col = (col // sqrt_size) * sqrt_size
    
    for i in range(start_row, start_row + sqrt_size):
        for j in range(start_col, start_col + sqrt_size):
            if board[i][j] == num:
                return False
                
    return True


def make_puzzle(current_board,difficulty):
    """
    Removes numbers from the full board based on difficulty level.

    Args:
        current_board (List[List[int]]): Full solution board.
        difficulty (str): Difficulty level ('easy', 'moderate', 'difficult').

    Returns:
        List[List[int]]: Puzzle board with removed numbers.
    """
    logical_cutoff, random_cutoff = difficulty_levels[difficulty]
    current_board = remove_numbers_logically(current_board,logical_cutoff)
    if random_cutoff > 0:
        current_board = remove_numbers_randomly(current_board,random_cutoff)
    return current_board

def verify(item):
    """
    Verifies the model's proposed Sudoku solution.

    Checks:
    - If filled values match original clues.
    - If each row, column, and block is valid.

    Args:
        item (dict): Game state containing 'action' (the answer board) and 'current_board'.

    Returns:
        dict: Updated game state with score set to 1 (correct) or 0 (incorrect).
    """
    try:
        if isinstance(item['action'], str):
            board = ast.literal_eval(item['action'])
        else:
            board = item['action']
    except Exception as e:
        item['score'] = 0
        return item

    if not isinstance(board, list) or len(board) != size:
        item['score'] = 0
        return item
    for row in board:
        if not isinstance(row, list) or len(row) != size:
            item['score'] = 0
            return item

    current_board = item['current_board']
    
    for i in range(size):
        for j in range(size):
            if current_board[i][j] != 0 and current_board[i][j] != board[i][j]:
                item['score'] = 0
                return item

    for row in board:
        if not is_valid_unit(row):
            item['score'] = 0
            return item
            
    for j in range(size):
        column = [board[i][j] for i in range(size)]
        if not is_valid_unit(column):
            item['score'] = 0
            return item
            
    for block_row in range(0, size, sqrt_size):
        for block_col in range(0, size, sqrt_size):
            block = []
            for i in range(block_row, block_row + sqrt_size):
                for j in range(block_col, block_col + sqrt_size):
                    block.append(board[i][j])
            if not is_valid_unit(block):
                item['score'] = 0
                return item
    item['score'] = 1            
    return item


def is_valid_unit(unit):
    return sorted(unit) == list(range(1, size + 1))



def get_answer(self, modelname):
    try:
        prompt = self.get_prompt()
        print("prompt: ", prompt)
        messages = [{"role": "user", "content": prompt}]
        response, _ = call_model(messages, modelname)
        print("reponse: ", response)
        answer_board = self._parse_answer(response)
        print("answer_board: ", answer_board)
        return answer_board
        
    except Exception as e:
        print(f"Error getting model response: {e}")
        return None

def _parse_answer(self, response):
    try:
        start = response.find("```") + 3
        end = response.rfind("```")
        if start == -1 or end == -1:
            raise ValueError("No code block found in response")
            
        code_block = response[start:end].strip()
        
        board = []
        for line in code_block.split('\n'):
            if not line.strip():
                continue
            try:
                numbers = [int(n.strip()) for n in line.strip('[]').split(',')]
                if len(numbers) != self.size:
                    raise ValueError(f"Invalid row length: {len(numbers)}")
                board.append(numbers)
            except Exception as e:
                print(f"Error parsing line '{line}': {e}")
                raise
                
        if len(board) != self.size:
            raise ValueError(f"Invalid number of rows: {len(board)}")
            
        return board
        
    except Exception as e:
        print(f"Error parsing answer: {e}")
        return None
    
def run(self, modelname):
    board = self.generate(seed=42)
    print("Generated Sudoku puzzle:")
    for row in board:
        print(row)
        
    answer_board = self.get_answer(modelname)
    score = 0
    if answer_board:
        score = generator.verify(answer_board)
        print(f"score: {score}")
    return score


class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    difficulty : str
    solution : list
    current_board : list
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

# 使用示例
# if __name__ == "__main__":
#     # generator = SudokuGenerator()
#     item = generate(2124)
#     print(f"Generated Sudoku puzzle: {item['current_board']}")
#     print(print_board(item))
#     item['action']='[[1, 5, 8, 3, 4, 7, 6, 9, 2],[6, 9, 2, 1, 5, 8, 3, 4, 7],[3, 4, 7, 6, 9, 2, 1, 5, 8],[5, 8, 3, 4, 7, 6, 9, 2, 1],[4, 7, 6, 9, 2, 1, 5, 8, 3],[9, 2, 1, 5, 8, 3, 4, 7, 6],[8, 3, 4, 7, 6, 9, 2, 1, 5],[2, 1, 5, 8, 3, 4, 7, 6, 9],[7, 6, 9, 2, 1, 5, 8, 3, 4]]'
#     score = verify(item)
#     print(f"score: {score}")
   
