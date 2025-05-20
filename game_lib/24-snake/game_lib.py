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

# 游戏配置
board_size = 8  # 游戏板大小
wall = []       # 墙壁，可以根据需要修改
snake_game_prompt='''
You are a good game player, I'll give you a game board and rules.\nYour task is:\n- First, give your answer according to the game board and rules.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: LEFT'

You are controlling a snake in a Snake game.The board size is 8x8.In the game board. The goal of the game is to control the snake, eat as many apples as possible, and grow the snake in length. Each time the snake eats an apple, the score increases by one. The game ends when the snake collides with itself or the walls of the board.The game board is a grid with walls by '#' around the edges.The snake starts with a length of 1 (just the head). The head is represented by 'H' and the body by 'S'. The game starts with 3 apples placed randomly on the board. Apples are represented by 'A'. The snake starts moving in the 'UP' direction. The snake moves one square at a time in the direction it is facing: 'UP', 'DOWN', 'LEFT', or 'RIGHT'. The player controls the snake’s movement by providing direction inputs. The snake cannot reverse its direction (i.e., it cannot turn 'UP' to 'DOWN' or 'LEFT' to 'RIGHT' directly).The snake loses the game if it collides with itself  or the walls. Each time the snake's head moves to a square occupied by an apple ('A'), the snake eats the apple and grows in length and meanwhile, the score will increase 1 point. The Current direction indicates the direction in which the snake is currently moving forward and the Game board indicates the current map of game.Remember, the game will end after the 100th epoch.

For example,if the board is 
########
#     A#
#S     #
#H     #
#AA    #
#      #
#      #
########

and the direction you give is DOWN,then you will eat an apple ,increase 1 score and the next state of board will be 
########
#  A  A#
#S     #
#S     #
#HA    #
#      #
#      #
########

The Direction you give should be one of 'LEFT', 'RIGHT', 'UP' and 'DOWN'

Game board:

{board}

Current direction: {direction}

Current epoch: {epoch}

Before giving your answer,you need to repeat the current game board first.
'''
# 辅助函数：打印游戏板（用于记录每一步的地图状态）
def print_board(item):
    food = item['food']
    snake = item['snake']
    output = ""
    for i in range(board_size):
        for j in range(board_size):
            if (i, j) == snake[-1]:  # 蛇头
                output += 'H'
            elif (i, j) in snake:
                output += 'S'
            elif (i, j) in food:
                output += 'A'
            elif i == 0 or i == board_size - 1 or j == 0 or j == board_size - 1 or (i, j) in wall:
                output += '#'
            else:
                output += ' '
        output += '\n'
    prompt = snake_game_prompt.format(board=output,direction=item['direction'],epoch=item['epoch'])
    return prompt

# 初始化游戏状态，seed 用于随机数初始化
def generate(seed: int):
    initial_map = [[0 for _ in range(board_size)] for _ in range(board_size)]
    random.seed(seed)
    # 蛇的初始位置（蛇只有一个节点）
    snake = [(board_size // 2, board_size // 2 - 1)]
    
    # 随机生成3个食物，保证它们不与蛇或墙重叠
    food = []
    while len(food) < 3:
        new_food = (random.randint(1, board_size - 2), random.randint(1, board_size - 2))
        if new_food not in snake and new_food not in wall and new_food not in food:
            food.append(new_food)
    
    score = 0
    direction = 'UP'
    item = {
        'initial_map': initial_map,
        'direction': direction,
        'food': food,
        'score': score,
        'snake': snake,
        'is_end': False,
        'response': [],
        'prompt': '',
        'action': '',
        'epoch' : 1,
    }
    # 记录初始地图
    item['map'] = [print_board(item)]
    return item

# 防止逆向移动
def change_direction(direction, new_direction):
    if (direction == 'RIGHT' and new_direction == 'LEFT') or \
       (direction == 'LEFT' and new_direction == 'RIGHT') or \
       (direction == 'UP' and new_direction == 'DOWN') or \
       (direction == 'DOWN' and new_direction == 'UP'):
        return direction
    return new_direction

# 移动蛇，并处理食物和边界碰撞
def move_snake(direction, food, score, snake):
    head_x, head_y = snake[-1]
    if direction == 'LEFT':
        new_head = (head_x, head_y - 1)
    elif direction == 'RIGHT':
        new_head = (head_x, head_y + 1)
    elif direction == 'UP':
        new_head = (head_x - 1, head_y)
    elif direction == 'DOWN':
        new_head = (head_x + 1, head_y)
    else:
        new_head = (head_x, head_y)
    
    # 判断碰撞（蛇撞墙、边界或自身）
    if new_head in snake or new_head in wall or \
       new_head[0] == 0 or new_head[1] == 0 or \
       new_head[0] == board_size - 1 or new_head[1] == board_size - 1:
        return False, food, score, snake

    snake.append(new_head)
    if new_head in food:
        score += 1
        food.remove(new_head)
        # 补充新的食物，确保始终有3个食物
        while len(food) < 3:
            new_food = (random.randint(1, board_size - 2), random.randint(1, board_size - 2))
            if new_food not in snake and new_food not in wall and new_food not in food:
                food.append(new_food)
    else:
        # 如果没有吃到食物，则尾部出队
        snake.pop(0)

    return True, food, score, snake

# 更新游戏状态，根据传入的 action 更新蛇的方向和位置
def verify(item):
    # 根据传入的 action 更新方向（防止逆向移动）
    item['action']=item['action'].strip().upper()
    item['direction'] = change_direction(item['direction'], item['action'])
    
    # 移动蛇并更新游戏状态
    valid_move, food, score, snake = move_snake(item['direction'], item['food'], item['score'], item['snake'])
    item['score'] = score
    # 添加本次移动后的地图状态
    item['map'].append(print_board(item))
    item['food'] = food
    item['snake'] = snake
    item['is_end'] = not valid_move
    item['epoch'] += 1
    
    return item

# --- 定义请求和响应数据模型 ---

class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    initial_map: list
    direction: str
    food: list
    score: int
    snake: list
    is_end: bool
    map: list
    action: str
    response: list
    prompt: str
    epoch: int
# --- API 接口 ---

# 生成初始游戏状态
@app.post("/print_board", response_model=BoardRequest)
def api_print_board(request: GameState):
    state = request.dict()
    # 将 food 和 snake 中的坐标列表转换为元组，保证坐标比较正确
    state['food'] = [tuple(coord) for coord in state['food']]
    state['snake'] = [tuple(coord) for coord in state['snake']]
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
    state['food'] = [tuple(coord) for coord in state['food']]
    state['snake'] = [tuple(coord) for coord in state['snake']]
    updated_state = verify(state)
    return updated_state

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)
