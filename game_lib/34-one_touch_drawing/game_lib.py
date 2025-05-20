import random
from collections import Counter, defaultdict
from typing import List, Optional, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
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

game_prompt = '''
You are a good game player, I'll give you a game board and rules.
Your task is:
- First, give your answer according to the game board and rules.
- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question, e.g., 'Answer: node 1,node 3,...'
You are a graph theory expert. Given the following nodes and edges, provide an Eulerian path that traverses each edge exactly once.
Nodes: {nodes}
Edges: {edges}
Your answer should be a comma-separated list of node names. Answer format: "Answer: node X, node Y, ..."
'''

def print_board(item):
    prompt = item['current_problem']
    return prompt

def generate(seed: int, extra_count: Optional[int] = None) -> dict:
    random.seed(seed)
    N = random.randint(10, 40)
    nodes = [f"node {i+1}" for i in range(N)]

    # 构造基本环：保证图连通且所有节点起始度数为偶数
    cycle_edges = []
    for i in range(N):
        cycle_edges.append([nodes[i], nodes[(i+1) % N]])

    # 如果没有指定额外边数，则默认为 N//2 或至少 1
    if extra_count is None:
        extra_count = max(1, N // 2)

    # 候选的额外边，排除已有的环边
    candidate_pool = []
    for i in range(N):
        for j in range(i+1, N):
            if not (j == i+1 or (i == 0 and j == N-1)):
                candidate_pool.append((i, j))

    extra_count = min(extra_count, len(candidate_pool))
    selected = random.sample(candidate_pool, extra_count)
    extra_edges = [[nodes[i], nodes[j]] for i, j in selected]
    edges = cycle_edges + extra_edges

    # 保证图存在欧拉路径：即奇数度节点数为 0 或 2
    while True:
        degrees = defaultdict(int)
        for u, v in edges:
            degrees[u] += 1
            degrees[v] += 1
        odd_nodes = [node for node, d in degrees.items() if d % 2 != 0]
        if len(odd_nodes) in (0, 2):
            break
        # 如果奇数度节点数超过2，则选取任意两个奇节点，添加修正边，使其度数变为偶数
        if len(odd_nodes) >= 2:
            u, v = random.sample(odd_nodes, 2)
            edges.append([u, v])

    # 构造边的字符串表示
    edge_strs = [f"<{u}, {v}>" for u, v in edges]
    problem = game_prompt.format(
        nodes=", ".join(nodes),
        edges=", ".join(edge_strs)
    )

    item = {
        "nodes": nodes,
        "edges": edges,
        "current_problem": problem,
        "score": 0,
        "is_end": False,
        "action": "",
        "response": [],
        "prompt": "",
        "epoch": 1
    }
    return item

def verify(state: dict) -> dict:
    try:
        # 处理 action 字符串，先去掉可能的 "Answer:" 前缀
        action_str = state["action"].strip()


        # 根据逗号分割，移除多余空格及空元素
        action_nodes = [n.strip() for n in action_str.split(",") if n.strip()]
        if not action_nodes:
            state["score"] = 0
            return state

        # 检查路径节点个数是否等于边数加 1
        if len(action_nodes) != len(state["edges"]) + 1:
            state["score"] = 0
            return state

        # 利用边的计数，注意无向图使用frozenset表示无序边
        edge_counter = Counter()
        for edge in state["edges"]:
            key = frozenset(edge)
            edge_counter[key] += 1

        # 按答案顺序逐对验证并消耗对应边
        for i in range(len(action_nodes) - 1):
            key = frozenset((action_nodes[i], action_nodes[i+1]))
            if edge_counter[key] <= 0:
                state["score"] = 0
                return state
            edge_counter[key] -= 1

        # 检查所有边是否均被使用一次
        if sum(edge_counter.values()) == 0:
            state["score"] = 1
        else:
            state["score"] = 0
        return state

    except Exception:
        state["score"] = 0
        return state

# 定义请求和状态数据模型
class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    nodes: list
    edges: list
    current_problem: str
    score: int
    is_end: bool
    action: str
    response: list
    prompt: str
    epoch: int

@app.post("/print_board", response_model=BoardRequest)
def api_print_board(request: GameState):
    state = request.model_dump()
    board_output = print_board(state)
    return {"board": board_output}

@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    return generate(request.seed)

@app.post("/verify", response_model=GameState)
def api_verify(state: GameState):
    # 注意：如果使用的是 Pydantic 1.x，请改为 state.dict()
    return verify(state.model_dump())

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)
