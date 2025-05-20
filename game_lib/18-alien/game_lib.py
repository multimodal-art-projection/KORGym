import random
import string
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
game_prompt = """
You are a good game player, I'll give you a game board and rules.\nYour task is:\n- First, give your answer according to the game board and rules.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: 12'
{board}
"""
# 定义外星生物类
class Alien:
    def __init__(self):
        self.name = ''.join(random.choices(string.ascii_letters, k=random.randint(3, 8)))  # 随机生成名称
        self.diet = random.choice(['herbivore', 'carnivore', 'Omnivore', 'Scavenger', 'Parasite', 'Insectivore'])  # 随机生成食性
        self.legs = random.randint(0, 10)  # 随机生成足的个数
        self.horns = random.randint(0, 10)  # 随机生成角的个数
        self.reproduction = random.choice(['mammal', 'oviparous', 'Viviparous', 'Asexual Reproduction', 'Spore Reproduction'])  # 随机生成繁殖方式
        self.color = random.choice(['red', 'orange', 'blue', 'green', 'yellow', 'purple', 'cyan'])  # 随机生成颜色

def print_board(item):
    return game_prompt.format(board=item['board'])
def generate(seed): #随机种子数，外星生物数
    random.seed(seed)
    nums=random.randint(70,200)
    if nums < 1:
        raise ValueError("Number must be larger than 1.")
    num_aliens = random.randint(1, nums) # 外星生物总类别数
    aliens = [Alien() for _ in range(num_aliens)] # 外星生物列表
    parts = [0] * num_aliens

    # 随机给所有生物分配数量
    for i in range(nums):
        index = random.randint(0, num_aliens - 1)
        parts[index] += 1

    query_objects = ['total number', 'horns', 'legs']
    attributes = list(aliens[0].__dict__.keys())
    attributes_features = {}
    for attribute in attributes:
        attributes_features[attribute] = []

    attributes.remove('name')
    selected_attributes = random.sample(attributes, random.randint(2, len(attributes))) #选择提出问题所需的属性
    query_objects = [item for item in query_objects if item not in selected_attributes]
    query_object = random.choice(query_objects)

    #存放所有外星生物的属性
    for alien in aliens:
        attributes_features['name'].append(alien.name)
        attributes_features['diet'].append(alien.diet)
        attributes_features['horns'].append(alien.horns)
        attributes_features['legs'].append(alien.legs)
        attributes_features['reproduction'].append(alien.reproduction)
        attributes_features['color'].append(alien.color)
    #选择插入问题中的属性 （例如，需要满足n条腿，颜色为蓝色/绿色）
    question_attribute = {}
    for selected_attribute in selected_attributes:
        question_attribute[selected_attribute] = list(set(random.sample(attributes_features[selected_attribute], random.randint(1, len(attributes_features[selected_attribute])))))

    #计算答案数量（总数，角数，足数）
    answer = 0
    for i in range(len(aliens)):
        qualified = True #判断当前外星生物是否符合条件
        for attribute in question_attribute.keys():
            if getattr(aliens[i], attribute) not in question_attribute[attribute]:
                qualified = False
        if qualified == True:
            if query_object == 'total number':
                answer += parts[i]
            elif query_object == 'horns':
                answer += parts[i]*getattr(aliens[i], 'horns')
            elif query_object == 'legs':
                answer += parts[i]*getattr(aliens[i], 'legs')

    #组建问题
    question = 'There are several alien beings on a distant planet. Their categories and corresponding features are as follows:\n'
    for i, alien in enumerate(aliens):
        question += f"Alien({i + 1}): Name = {alien.name}, Diet = {alien.diet}, Legs = {alien.legs}, Horns = {alien.horns}, Reproduction = {alien.reproduction}, Color = {alien.color};\n"
    question += 'Now, there are '
    for i in range(len(parts)):
        if parts[i] > 0:
            question += f"{parts[i]} {attributes_features['name'][i]}, "
    question = question.rstrip(", ")
    if query_object == 'total number':
        question += ' in this area. Please help me calculate the total number of alien animals that meet the following criteria, '
    else:
        question += f" in this area, please help me calculate the total number of {query_object} of these alien animals that meet the following criteria, "
    question += "and return the number in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: 16'.\nAll alien animals that: "

    i = 1
    for attribute in question_attribute.keys():
        question = question + str(i) + '. ' + attribute + ' are ' + '/'.join([str(item) if isinstance(item, int) else item for item in question_attribute[attribute]]) + '; '
        i += 1
    question = question.rstrip('; ')
    question += '.'
    item = {
        'score': 0,
        'is_end': False,
        'response': [],
        'prompt': '',
        'action': '',
        'epoch': 1,
    }
    item['board'] = question
    item['answer'] = str(answer)
    return item

def verify(item):
    try:
        action = int(item['action'])
        correct_answer = int(item['answer'])
        if correct_answer == action:
            score = 1
        else:
            score = 0
        item['score'] = score
    except:
        item['score'] = 0
    return item

class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    board: str
    answer: str
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
# # 主函数
# def main():
#     item = generate(44, 100)
#     item['action']=item['answer']
#     score = verify(item)
#     print(score)

# if __name__ == "__main__":
#     main()
