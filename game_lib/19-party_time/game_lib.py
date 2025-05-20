import random
import string
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
# 定义参与人物类
class Participator:
    def __init__(self):
        self.name = ''.join(random.choices(string.ascii_letters, k=random.randint(3, 8)))  # 随机生成名称
        self.gender = [random.choice(['male', 'female'])]  # 随机生成性别
        self.shirt_color = [random.choice(['red', 'orange', 'blue', 'green', 'yellow', 'purple', 'cyan'])]
        self.pants_color = [random.choice(['red', 'orange', 'blue', 'green', 'yellow', 'purple', 'cyan'])]
        self.hair_color = [random.choice(['red', 'orange', 'blue', 'green', 'yellow', 'purple', 'cyan'])]
        self.has_items = random.sample([
            "balloon", "snacks", "camera", "hat", "sunglasses", "lighter",
            "bottle", "phone", "book", "flowers",
            "candy", "guitar", "umbrella", "scarf",
            "perfume", "candle", "wallet", "pencil"
        ], random.randint(1, 6))

def generate(seed: int) -> dict:
    """
    根据给定随机种子生成参与人物信息及对应的问题。
    生成的 item 包含题面 board、答案 answer 以及其他辅助信息。
    """
    random.seed(seed)
    # 保证参与人数至少为2
    nums = random.randint(70, 200)
    participators = [Participator() for _ in range(nums)]
    
    # 随机选择查询类型
    query_objects = ['total number', 'items number']
    query_object = random.choice(query_objects)
    
    # 获取所有属性名，并构建属性值存储字典
    attributes = list(participators[0].__dict__.keys())  # ['name', 'gender', 'shirt_color', 'pants_color', 'hair_color', 'has_items']
    attributes_features = {attribute: [] for attribute in attributes}
    
    # 为构造问题选择部分属性（不包括姓名）
    attributes.remove('name')
    selected_attributes = random.sample(attributes, random.randint(2, len(attributes)))
    
    # 收集所有参与人物的属性值
    for participator in participators:
        attributes_features['name'].append(participator.name)
        attributes_features['gender'].extend(participator.gender)
        attributes_features['shirt_color'].extend(participator.shirt_color)
        attributes_features['pants_color'].extend(participator.pants_color)
        attributes_features['hair_color'].extend(participator.hair_color)
        attributes_features['has_items'].extend(participator.has_items)
    
    # 针对每个选中的属性，从所有可能值中随机挑选一部分作为条件
    question_attribute = {}
    for attr in selected_attributes:
        unique_values = list(set(attributes_features[attr]))
        sample_size = random.randint(1, len(unique_values))
        question_attribute[attr] = random.sample(unique_values, sample_size)
    
    # 计算答案
    answer = 0
    # 当查询类型为 items number 时，从所有物品中随机选取一个进行计数
    sub_query_object = random.choice(list(set(attributes_features['has_items'])))
    for participator in participators:
        qualified = True
        for attr, values in question_attribute.items():
            # 对于列表类型的属性，判断是否至少有一个值符合要求
            if not any(item in values for item in getattr(participator, attr)):
                qualified = False
                break
        if qualified:
            if query_object == 'total number':
                answer += 1
            elif query_object == 'items number':
                if sub_query_object in getattr(participator, 'has_items'):
                    answer += 1
    
    # 构造问题描述
    question = "We invite some students to our party today. Their appearance and their belongings are as follows:\n"
    for i, participator in enumerate(participators):
        question += (
            f"Student({i + 1}): Name = {participator.name}, Gender = {participator.gender[0]}, "
            f"Shirt color = {participator.shirt_color[0]}, Pants color = {participator.pants_color[0]}, "
            f"Hair color = {participator.hair_color[0]}, Has items = {'/'.join(participator.has_items)};\n"
        )
    if query_object == 'total number':
        question += "Please help me calculate the total number of students that meet the following criteria, "
    else:
        question += f"Please help me calculate the total number of {sub_query_object} of these students that meet the following criteria, "
    question += (
        "and return the number in the following format: 'Answer: $YOUR_ANSWER' (without quotes), "
        "where YOUR_ANSWER is your final answer to the question, e.g., 'Answer: 16'.\nAll students that: "
    )
    # 将各条件依次列出
    i = 1
    for attr, values in question_attribute.items():
        question += f"{i}. {attr} belong to {'/'.join(str(v) for v in values)}; "
        i += 1
    question = question.rstrip('; ') + "."
    
    # 构建 item 作为信息传递媒介
    item = {
        'score': 0,
        'is_end': False,
        'response': [],
        'prompt': '',
        'action': '',
        'epoch': 1,
        'board': question,
        'answer': str(answer)
    }
    return item

def verify(item: dict) -> dict:
    """
    根据 item 中的 action 字段判断用户提交答案是否正确，
    若 action 转换为整数与正确答案一致，则 score 为1，否则为0。
    """
    try:
        user_answer = int(item['action'])
        correct_answer = int(item['answer'])
        item['score'] = 1 if user_answer == correct_answer else 0
    except:
        item['score'] = 0
    return item

def print_board(item: dict) -> str:
    """
    返回 item 中保存的题面文本。
    """
    return game_prompt.format(board=item['board'])

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
# 示例主函数，用于测试生成题面和验证答案
# if __name__ == "__main__":
#     # 使用随机种子 44 生成题面
#     item = generate(414)
#     # 打印题面
#     print("题面：")
#     print(print_board(item))
#     print("\n正确答案：", item['answer'])
    
#     # 模拟用户回答正确的情况（这里直接将 action 设置为正确答案）
#     item['action'] = item['answer']
#     item = verify(item)
#     print("\n验证结果（用户答案正确）：score =", item['score'])
    
#     # 模拟用户回答错误的情况
#     item['action'] = str(int(item['answer']) + 1)
#     item = verify(item)
#     print("验证结果（用户答案错误）：score =", item['score'])
