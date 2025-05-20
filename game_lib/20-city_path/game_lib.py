import random
import heapq
import re
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
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
# 定义城市问题提示（用于包装 board 信息）
game_prompt = """
You are an expert in city navigation. Below is the information about a network of cities and roads.
Your task:
- Read the information carefully.
- Calculate the shortest distance from the start city to the target city.
- Provide your answer in the following format: 'Answer: $YOUR_ANSWER' (without quotes).

{board}
"""

# 接口1：打印题面，返回格式化后的提示信息
def print_board(item: dict):
    return game_prompt.format(board=item['board'])

# 接口2：生成城市网络及问题
def generate(seed: int):
    random.seed(seed)
    # 随机生成城市数量，范围在10到15之间
    num = random.randint(70, 200)
    cities = [f"City{i}" for i in range(num)]
    
    # 为确保图连通，先随机生成一棵生成树（构成一条链）
    edges = []
    shuffled_cities = cities[:]
    random.shuffle(shuffled_cities)
    for i in range(len(shuffled_cities) - 1):
        d = random.randint(1, 20)
        edges.append((shuffled_cities[i], shuffled_cities[i+1], d))
    
    # 添加额外的随机边以丰富图的结构
    extra_edges = num  # 额外添加与城市数量相同条数的边
    for _ in range(extra_edges):
        city_a = random.choice(cities)
        city_b = random.choice(cities)
        if city_a == city_b:
            continue
        # 检查该边是否已存在（无向边）
        exists = False
        for (a, b, _) in edges:
            if (a == city_a and b == city_b) or (a == city_b and b == city_a):
                exists = True
                break
        if exists:
            continue
        d = random.randint(1, 20)
        edges.append((city_a, city_b, d))
    
    # 随机选择起始城市和目标城市（确保不相同）
    start_city = random.choice(cities)
    target_city = random.choice(cities)
    while target_city == start_city:
        target_city = random.choice(cities)
    
    # 构造图的邻接字典（如果存在多条边则取较小距离）
    graph = {city: {} for city in cities}
    for (a, b, d) in edges:
        if b not in graph[a] or d < graph[a][b]:
            graph[a][b] = d
        if a not in graph[b] or d < graph[b][a]:
            graph[b][a] = d
    
    # Dijkstra 算法计算最短路径距离
    distances = {city: float('inf') for city in cities}
    distances[start_city] = 0
    visited = set()
    pq = [(0, start_city)]
    
    while pq:
        cur_dist, cur_city = heapq.heappop(pq)
        if cur_city in visited:
            continue
        visited.add(cur_city)
        if cur_city == target_city:
            break
        for neighbor, d in graph[cur_city].items():
            if neighbor in visited:
                continue
            new_dist = cur_dist + d
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))
    
    shortest_distance = distances[target_city] if distances[target_city] != float('inf') else None
    
    # 构造问题描述（board）
    board = "City Network Information:\n"
    board += "Cities: " + ", ".join(cities) + "\n"
    board += "Roads (format: CityA - CityB (distance)):\n"
    for a, b, d in edges:
        board += f"{a} - {b} ({d}), "
    board = board.rstrip(", ") + "\n"
    board += f"Start City: {start_city}\nTarget City: {target_city}\n"
    board += "Question: What is the shortest distance from the start city to the target city?"
    
    # 使用 item 作为信息传递媒介
    item = {
        "board": board,
        "answer": str(shortest_distance) if shortest_distance is not None else "N/A",
        "score": 0,
        "is_end": False,
        "action": "",
        "response": [],
        "prompt": "",
        "epoch": 1,
    }
    return item

# 接口3：验证答案
def verify(item: dict):
    try:
        correct_answer = int(item['answer'])
    except:
        item['score'] = 0
        return item
    # 使用正则表达式从 action 中提取所有数字
    numbers = re.findall(r'\d+', item['action'])
    score = 0
    for num in numbers:
        if int(num) == correct_answer:
            score = 1
            break
    item['score'] = score
    return item

# 以下为 FastAPI 部分

app = FastAPI()

class CityGraphState(BaseModel):
    board: str
    answer: str
    score: int
    is_end: bool
    action: str
    response: List[str]
    prompt: str
    epoch: int

class GenerateRequest(BaseModel):
    seed: int

@app.post("/generate", response_model=CityGraphState)
def api_generate(request: GenerateRequest):
    return generate(request.seed)

@app.post("/verify", response_model=CityGraphState)
def api_verify(state: CityGraphState):
    state_dict = state.dict()
    updated_state = verify(state_dict)
    return updated_state

@app.post("/print_board", response_model=dict)
def api_print_board(state: CityGraphState):
    board_output = print_board(state.dict())
    return {"board": board_output}

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)
