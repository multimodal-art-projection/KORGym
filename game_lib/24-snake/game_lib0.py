import os
import random
import time

# 游戏设置
board_size = 8  # 游戏板大小
wall = []


# 打印游戏板
def print_board(item):
    food=item['food']
    snake=item['snake']
    output=""
    for i in range(board_size):
        for j in range(board_size):
            if (i, j) == snake[-1]:  # 蛇头
                output+='H'
            elif (i, j) in snake:
                output+='S'
            elif (i, j) in food:
                output+='A'
            elif i == 0 or i == board_size - 1 or j == 0 or j == board_size - 1 or (i, j) in wall:
                output+='#'
            else:
                output+=' '
        output+='\n'
    return output

# 初始化函数
def generate(seed):
    initial_map = [[0 for _ in range(board_size)] for _ in range(board_size)]
    random.seed(seed)
    # 初始化游戏状态
    snake = [(board_size // 2, board_size // 2 - 1)]  # 蛇的初始位置
    
    # 生成3个食物，确保它们不和蛇身或墙壁重合
    food = []
    while len(food) < 3:
        new_food = (random.randint(1, board_size - 2), random.randint(1, board_size - 2))
        if new_food not in snake and new_food not in wall and new_food not in food:
            food.append(new_food)
    
    score = 0  # 初始分数
    direction = 'UP'  # 初始方向
    item={}
    item['initial_map']=initial_map
    item['direction']=direction
    item['food']=food
    item['score']=score
    item['snake']=snake
    item['is_end']=False
    # print(print_board(item))
    item['map']=[print_board(item)]
    return item  # 返回初始地图、方向、食物位置和得分

# 改变方向函数
def change_direction(direction, new_direction):
    if (direction == 'RIGHT' and new_direction == 'LEFT') or \
       (direction == 'LEFT' and new_direction == 'RIGHT') or \
       (direction == 'UP' and new_direction == 'DOWN') or \
       (direction == 'DOWN' and new_direction == 'UP'):
        return direction  # 防止逆向移动
    return new_direction

# 移动蛇
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

    # 判断碰撞
    if new_head in snake or new_head in wall or new_head[0] == 0 or new_head[1] == 0 or new_head[0] == board_size - 1 or new_head[1] == board_size - 1:
        return False, food, score, snake  # 碰撞，返回游戏结束状态和无效的地图

    snake.append(new_head)
    if new_head in food:
        score += 1
        food.remove(new_head)  # 移除被吃掉的食物
        # 生成新的食物
        while len(food) < 3:
            new_food = (random.randint(1, board_size - 2), random.randint(1, board_size - 2))
            if new_food not in snake and new_food not in wall and new_food not in food:
                food.append(new_food)
    else:
        snake.pop(0)

    return True, food, score, snake

# 更新函数
def update(item):
    # 处理方向变更
    item['direction'] = change_direction(item['direction'], item['action'])

    # 移动蛇并检查游戏状态
    valid_move, food, score, snake = move_snake(item['direction'], item['food'], item['score'], item['snake'])
    item['score']=score
    item['map'].append(print_board(item))
    
    item['food']=food
    item['snake']=snake
    if not valid_move:
        item['is_end']=True
    else:
        item['is_end']=False

    return item

if __name__ == '__main__':
    
    # 初始化游戏
    item = generate(seed=4822)

    ACTION_LIST = ['UP', 'DOWN', 'RIGHT', 'LEFT']

    while True:
        action = input("Enter direction (UP, DOWN, LEFT, RIGHT): ").strip().upper()
        if action not in ACTION_LIST:
            print("Invalid direction! Please enter one of: UP, DOWN, LEFT, RIGHT.")
            continue
        item['action']=action
        item = update(item)
        print(f"score:{item['score']}")
        if item['is_end']:
            print("Game Over!")
            break
        board=print_board(item)  # 打印游戏板
        print(board)
