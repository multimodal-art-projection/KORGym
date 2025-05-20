import random
import string
import networkx as nx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import argparse

def parse_init():
    """
    定义并解析命令行参数，用于服务部署地址与端口的配置。
    """
    parser = argparse.ArgumentParser(description="Data creation utility")
    parser.add_argument('-p', '--port', type=int, default=8775, help='服务部署端口')
    parser.add_argument('-H', '--host', type=str, default="0.0.0.0", help='服务部署地址')
    args = parser.parse_args()
    return args
app = FastAPI()

# 定义全局提示模板，用于 print_board 接口
game_prompt = "Cities and Connections Game:\n{board}"

# 定义 generate 接口
def generate(seed: int, n: int = 20, e: int = 50):
    # 边数不能超过 n*(n-1)/2
    n = random.randint(100,200)
    e = random.randint(200,1000)
    max_edges = n * (n - 1) // 2
    if e > max_edges:
        e = max_edges
        print(f"Number of the edges must not be larger than city*(city-1)/2, edge e is set to {max_edges}.")

    random.seed(seed)
    city_list = []
    existing_names = set()

    # 构建城市列表
    for _ in range(n):
        # 确保城市名称唯一
        while True:
            name = ''.join(random.choices(string.ascii_uppercase, k=10))
            if name not in existing_names:
                existing_names.add(name)
                break
        # 随机分配沿海或内陆（1表示沿海，0表示内陆）
        location = random.choice([0, 1])
        city = {
            'name': name,
            'coastal cities': location,
            'inland cities': 1 - location,
            'population': random.randint(100, 10000) * 10000,
            'lumber mills': random.randint(0, 100),
            'hospitals': random.randint(0, 100),
            'churches': random.randint(0, 100),
            'banks': random.randint(0, 100),
            'stadiums': random.randint(0, 100),
            'restaurants': random.randint(0, 100),
            'mines': random.randint(0, 100),
            'factories': random.randint(0, 100),
            'research centers': random.randint(0, 100),
        }
        city_list.append(city)

    # 生成城市图
    G = nx.gnm_random_graph(n, e, seed=seed)
    for u, v in G.edges():
        G[u][v]['distance'] = random.randint(1, 100)

    edge_list = []
    for u, v in G.edges():
        edge_list.append((city_list[u]['name'], city_list[v]['name'], G[u][v]['distance']))

    # 选择至少有边的一个城市作为参考
    nodes_with_edges = [node for node in G.nodes() if G.degree(node) > 0]
    if not nodes_with_edges:
        a_node = 0
    else:
        a_node = random.choice(nodes_with_edges)
    A_name = city_list[a_node]['name']

    # 从指定属性列表中随机选取一个属性作为计算目标
    items_list = ['coastal cities', 'inland cities', 'population', 'lumber mills', 'hospitals', 
                  'churches', 'banks', 'stadiums', 'restaurants', 'mines', 'factories', 'research centers']
    selected_item = random.choice(items_list)

    # 获取参考城市的直接邻居
    neighbors = list(G.neighbors(a_node))
    if not neighbors:
        answer = 0
        k = 0
    else:
        neighbor_distances = [(n, G[a_node][n]['distance']) for n in neighbors]
        sorted_neighbors = sorted(neighbor_distances, key=lambda x: x[1])
        # 随机选取 k 个最近的邻居
        k = random.randint(1, len(sorted_neighbors))
        selected_neighbors = sorted_neighbors[:k]
        answer = sum(city_list[n][selected_item] for (n, d) in selected_neighbors)

    # 生成城市详情描述
    city_details = "\n".join([
        f"City Name={c['name']}: Type={'Coastal' if c['coastal cities'] == 1 else 'Inland'}, Population={c['population']}, "
        f"Lumber Mills={c['lumber mills']}, Hospitals={c['hospitals']}, "
        f"Churches={c['churches']}, Banks={c['banks']}, Stadiums={c['stadiums']}, "
        f"Restaurants={c['restaurants']}, Mines={c['mines']}, "
        f"Factories={c['factories']}, Research Centers={c['research centers']}"
        for c in city_list
    ])

    connection_details = "\n".join([
        f"City {u} is connected to City {v} (Distance: {d}km)"
        for u, v, d in edge_list
    ])

    # 生成问题描述
    question = (
        "Given the following cities and their connections:\n\n"
        "── City Details ──\n"
        f"{city_details}\n\n"
        "── Connections ──\n"
        f"{connection_details}\n\n"
        f"Question: What is the total number of {selected_item} in the {k} nearest cities directly adjacent to {A_name}?\n"
        "Please provide your answer in the following format: 'Answer: $YOUR_ANSWER' (without quotes), "
        "where $YOUR_ANSWER is your final answer."
    )

    # 使用 item 作为信息传递媒介
    item = {
        'board': question,
        'answer': str(answer),
        'score': 0,
        'is_end': False,
        'action': "",
        'response': [],
        'prompt': "",
        'epoch': 1,
    }
    return item

# 定义 verify 接口，比较用户的 action 和正确答案
def verify(item: dict):
    try:
        user_answer = int(item.get('action', '').strip())
        correct_answer = int(item['answer'])
        item['score'] = 1 if user_answer == correct_answer else 0
    except Exception as ex:
        item['score'] = 0
    return item

# 定义 print_board 接口，根据 item 输出游戏版面
def print_board(item: dict):
    return game_prompt.format(board=item['board'])

# --------------------- FastAPI 接口定义 ---------------------

# 请求和状态模型
class GenerateRequest(BaseModel):
    seed: int
    n: int = 20
    e: int = 50

class BoardRequest(BaseModel):
    board: str

class GameState(BaseModel):
    board: str
    answer: str
    score: int
    is_end: bool
    action: str
    response: List[str]
    prompt: str
    epoch: int

# 生成初始游戏状态接口
@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    game_state = generate(request.seed, request.n, request.e)
    return game_state

# 更新并验证用户动作接口
@app.post("/verify", response_model=GameState)
def api_verify(request: GameState):
    state = request.dict()
    updated_state = verify(state)
    return updated_state

# 打印游戏版面接口
@app.post("/print_board", response_model=BoardRequest)
def api_print_board(request: GameState):
    state = request.dict()
    board_output = print_board(state)
    return {"board": board_output}

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)
