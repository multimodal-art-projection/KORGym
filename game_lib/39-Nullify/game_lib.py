import random
from enum import Enum
import math
from copy import deepcopy
import math
from datetime import datetime
import os
import base64
import numpy as np
import random
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid  # 用于生成唯一标识符
from typing import Optional
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
ENV_STORE = {}
app = FastAPI()
game_prompt = """
You are a good game player, I'll give you a game board and rules.\nYour task is:\n- First, give your answer according to the game board and rules.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: LEFT'

In the game, there are several independent units. Among them, units that consist of a sign (+ or -) followed by a number can combine with other units. For example, combining -10 with ×8 results in -80, and combining -7 with ÷3 yields -2.33... If the combination of two units results in 0 (i.e. when a positive and a negative number with equal absolute values are operated on), no new unit is produced; if two units that do not contain a + or - sign…  
Game Objective: Eliminate all units, meaning that the final combined result is 0.  

Current Unit Types:
+number: A positive number, which also represents the addition operation.  
-number: A negative number, which also represents the subtraction operation.  
*number: Multiplication operation.  
/number: Division operation.  
sqrt: Square root operation.  
square: Square operation.  
reciprocal: Reciprocal operation.  
floor: Floor (round down) operation.  
ceil: Ceiling (round up) operation.  

Please output the operation for the current turn by directly providing the two corresponding unit indices, separated by a space (e.g., "Answer: 2 4").  

board:  
{board}
"""
class UnitType(Enum):
    NUMBER = 1  # 正负数字
    ORDINARY_OPERATOR = 2  # 乘除
    OPERATOR = 3  # 其他运算符
    
class Unit:
    def __init__(self, unit_type, symbol, value=None):
        self.type = unit_type
        self.symbol = symbol
        self.value = value
    
    def __repr__(self):
        if self.type == UnitType.NUMBER:
            return f"{self.value}"
        elif self.type == UnitType.ORDINARY_OPERATOR:
            return f"{self.symbol}{self.value}"
        else:
            return f"{self.symbol}"
    
class NullifyQuestionGenerator:
    def __init__(self, seed=None):
        self.question_queue = []
        
        if seed is not None:
            random.seed(seed)
            
            
    #### add numbers and operators ####
    
    def generate_number_unit(self):
        while True:
            value = round(random.randint(-10,10), 4)
            if value != 0:
                sign = "+" if value >= 0 else "-"
                return Unit(UnitType.NUMBER, sign, value)
    
    def generate_ordinary_operator_unit(self):
        while True:
            value = round(random.randint(-10,10), 4)
            if value != 0:
                sign = random.choice(["*","/"])
                return Unit(UnitType.ORDINARY_OPERATOR, sign, value)

    def generate_operator_unit(self):
        operator = random.choice(['floor', 'ceil', 'sqrt', 'square', 'reciprocal'])
        return Unit(UnitType.OPERATOR, operator)
    
    #### process operation ####
    def apply_operator(self, unit, operator_unit):
        if operator_unit.symbol == "floor":
            new_value = math.floor(unit.value)
        elif operator_unit.symbol == "ceil":
            new_value = math.ceil(unit.value)
        elif operator_unit.symbol == 'sqrt':
                new_value = math.sqrt(unit.value)
        elif operator_unit.symbol == 'square':
            new_value = unit.value ** 2
        elif operator_unit.symbol == 'reciprocal':
            new_value = 1 / unit.value
        return Unit(UnitType.NUMBER, unit.symbol, new_value)    
    
    def generate(self):
        current_value = 0
        generation_steps = random.randint(3, 10)

        while True:
            if generation_steps == 0:
                break
            operation_type = random.choice(['add_or_minus', 'multiply', 'apply_operator'])

            if operation_type == "add_or_minus":
                number_unit = self.generate_number_unit()
                current_value -= number_unit.value
                print(f"current_value={current_value}")
                self.question_queue.append(number_unit)
                generation_steps -= 1

            elif operation_type == "multiply":
                if current_value == 0:
                    continue

                ordinary_operator_unit = self.generate_ordinary_operator_unit()
                if ordinary_operator_unit.symbol == "*":
                    if current_value == 0:
                        continue
                    else:
                        current_value /= ordinary_operator_unit.value
                elif ordinary_operator_unit.symbol == "/":
                    current_value *= ordinary_operator_unit.value

                print(f"current_value={current_value}")
                self.question_queue.append(ordinary_operator_unit)
                generation_steps -= 1

            elif operation_type == "apply_operator":
                if current_value == 0:
                    continue

                operator_unit = self.generate_operator_unit()

                if operator_unit.symbol == "sqrt":
                    if current_value < 0:
                        continue
                    current_value = current_value ** 2
                elif operator_unit.symbol == "square":
                    current_value = math.sqrt(abs(current_value))  # 后期优化可以加上正负号
                elif operator_unit.symbol == "reciprocal":
                    current_value = 1 / current_value
                elif operator_unit.symbol in ["floor", "ceil"]:
                    int_part = math.floor(current_value)
                    frac_part = current_value - int_part
                    if frac_part != 0:
                        frac_symbol = "+" if frac_part > 0 else "-"
                        self.question_queue.append(Unit(UnitType.NUMBER, frac_symbol, round(frac_part, 4)))
                    if operator_unit.symbol == "floor":
                        current_value = int_part + random.uniform(0, 1)
                    elif operator_unit.symbol == "ceil":
                        current_value = int_part - random.uniform(0, 1)

                print(f"current value:{current_value}")
                self.question_queue.append(operator_unit)
                generation_steps -= 1

        current_symbol = "+" if current_value > 0 else "-"
        self.question_queue.append(Unit(UnitType.NUMBER, current_symbol, round(current_value, 4)))
        
        # 在返回之前打乱 self.question_queue 的顺序
        random.shuffle(self.question_queue)
        return self.question_queue

    # Note, 现在的board存在self.question_queue中，不用额外输入了
    
    
    # 返回值：
    # 0表示合法操作且游戏继续
    # -1表示操作违法，退出游戏
    # 1表示胜利
    # -2表示只剩最后一个数字但不是0
    def verify(self, action: tuple[int,int]) -> int:
        i, j = action
        if (i < 0 or j < 0 or i >= len(self.question_queue) or j >= len(self.question_queue) or i == j):
            return -1
        temp_queue = deepcopy(self.question_queue)
        unit1 = temp_queue[i]
        unit2 = temp_queue[j]
        print(unit1,unit2)
        
        if unit1.type == UnitType.NUMBER and unit2.type == UnitType.NUMBER:
            sum_val = unit1.value + unit2.value
            print(f"operation_result:{sum_val}")
            if abs(sum_val) < 1e-9:  # 浮点容差
                indices = sorted([i, j], reverse=True)
                for idx in indices:
                    del temp_queue[idx]
                self.question_queue = temp_queue
                return 1 if len(temp_queue) == 0 else 0
            else:
                indices = sorted([i, j], reverse=True)
        
                for idx in indices:
                    del temp_queue[idx]
                    
                new_sign = '+' if sum_val >= 0 else '-'
                new_unit = Unit(UnitType.NUMBER, new_sign, sum_val)
                temp_queue.append(new_unit)

                # 检查是否全部消除
                if len(temp_queue) == 1:
                    if abs(new_unit.value) < 1e-3:
                        return 1
                    else: return -2
                else:
                    self.question_queue = temp_queue
                    return 0
                
        num_unit, op_unit = None, None
        if (unit1.type == UnitType.NUMBER and unit2.type in (UnitType.ORDINARY_OPERATOR, UnitType.OPERATOR)):
            num_unit, op_unit = unit1, unit2
        elif (unit2.type == UnitType.NUMBER and unit1.type in (UnitType.ORDINARY_OPERATOR, UnitType.OPERATOR)):
            num_unit, op_unit = unit2, unit1
        else:
            return -1  # 非法组合
        try:
            if op_unit.type == UnitType.ORDINARY_OPERATOR:
                if op_unit.symbol == '*':
                    new_val = num_unit.value * op_unit.value
                elif op_unit.symbol == '/':
                    if abs(op_unit.value) < 1e-9:
                        return -1
                    new_val = num_unit.value / op_unit.value
                else:
                    return -1
            else:
                # 处理其他运算符
                if op_unit.symbol == 'floor':
                    new_val = math.floor(num_unit.value)
                elif op_unit.symbol == 'ceil':
                    new_val = math.ceil(num_unit.value)
                elif op_unit.symbol == 'sqrt':
                    if num_unit.value < 0:
                        return -1
                    new_val = math.sqrt(num_unit.value)
                elif op_unit.symbol == 'square':
                    new_val = num_unit.value ** 2
                elif op_unit.symbol == 'reciprocal':
                    if abs(num_unit.value) < 1e-9:
                        return -1
                    new_val = 1 / num_unit.value
                else:
                    return -1
            new_sign = '+' if new_val >= 0 else '-'
            new_unit = Unit(UnitType.NUMBER, new_sign, new_val)

        except (ValueError, ZeroDivisionError):
            return -1
        
        indices = sorted([i, j], reverse=True)
        
        for idx in indices:
            del temp_queue[idx]
        temp_queue.append(new_unit)

        # 检查是否全部消除
        if len(temp_queue) == 1:
            if abs(new_unit.value) < 1e-4:
                return 1
            else: return -2
        else:
            self.question_queue = temp_queue
            return 0
        
        return -1
    
    

def generate(seed):
    item = {
        'score': 0,
        'is_end': False,
        'response': [],
        'prompt': '',
        'action': '',
        'epoch': 1,
    }
    uid = str(uuid.uuid4())
    item["uid"] = uid
    game = NullifyQuestionGenerator(seed)
    game.generate()
    ENV_STORE[uid] = game
    board =""
    for idx, unit in enumerate(game.question_queue):
        board = board + f"{idx} {unit}" + '\n'
    item['board'] = board
    return item

def verify(item):
    game = ENV_STORE[item["uid"]]
    action = item['action']
    try:
        i, j = map(int, action.split())
        result = game.verify((i, j))
        board =""
        for idx, unit in enumerate(game.question_queue):
            board = board + f"{idx} {unit}" + '\n'
        item['board'] = board
        if result == -1:
            item['epoch'] += 1
        elif result == 0:
            item['epoch'] += 1
        elif result == 1:
            item['epoch'] += 1
            item['is_end'] = 1
            item['score'] = 1
        elif result == -2:
            item['score'] = 0
            item['epoch'] += 1
            item['is_end'] = 1
    except:
        item['epoch'] += 1
    return item
def print_board(item):
    return game_prompt.format(board=item['board'])
# # 在没有shuffle的情况下，每次输(n-1, n)就可以赢，用的时候可以shuffle一下self.question_queue
# if __name__ ==  "__main__":
#     item=generate(1)
#     while(True):
#         print(print_board(item))
        
#         try:
#             item['action'] = input("请输入要操作的两个单元索引: ").strip()  
            
#         except ValueError:
#             print("Value Error")
#             continue
#         item = verify(item)
        
#         # 处理结果
#         if item['is_end'] == 1:
#             print(f"score: {item['score']}")
#             break
        
class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    board: str
    uid: str
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