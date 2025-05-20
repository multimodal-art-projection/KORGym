import random
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import ast
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
coloring_prompt = """
You are an expert in graph theory and coloring. Below is the information about a graph.
Your task:
- Read the graph information carefully.
- Provide a valid coloring scheme for the graph using the exact number of colors specified.
- The coloring scheme should be a list of pairs [node, color] for each node.
- Output format: 'Answer: [[0, 1], [1, 0], [2, 1],...]'.

{board}
"""
# 辅助函数：使用回溯法判断图是否 k-可着色
def is_k_colorable(adj, k):
    n = len(adj)
    # 优先处理约束多的节点（按度数降序排序）
    nodes = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    colors = [-1] * n

    def backtrack(index):
        if index == n:
            return True
        node = nodes[index]
        used = set()
        for neighbor in adj[node]:
            if colors[neighbor] != -1:
                used.add(colors[neighbor])
        for color in range(k):
            if color not in used:
                colors[node] = color
                if backtrack(index + 1):
                    return True
                colors[node] = -1
        return False

    return backtrack(0)

# 接口1：生成图着色问题
def generate(seed: int):
    random.seed(seed)
    # 随机生成节点数（5 到 15）
    n = random.randint(10, 50)
    # 随机生成边数，至少 n-1 条（保证连通），最多不超过 min(最大边数, n+5)
    max_edges = n * (n - 1) // 2
    e = random.randint(n - 1, min(max_edges, n + 5))
    
    edges = set()
    nodes = list(range(n))
    while len(edges) < e:
        u = random.choice(nodes)
        v = random.choice(nodes)
        if u == v:
            continue
        edge = tuple(sorted((u, v)))
        if edge in edges:
            continue
        edges.add(edge)
    edges = list(edges)
    
    # 构造邻接表
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # 计算最小着色数 m
    if e == 0:
        m = 1
    else:
        # 先判断是否为二分图
        is_bipartite = True
        color = [-1] * n
        for i in range(n):
            if color[i] == -1:
                queue = [i]
                color[i] = 0
                while queue:
                    node = queue.pop(0)
                    for neighbor in adj[node]:
                        if color[neighbor] == -1:
                            color[neighbor] = color[node] ^ 1
                            queue.append(neighbor)
                        elif color[neighbor] == color[node]:
                            is_bipartite = False
                            break
                    if not is_bipartite:
                        break
            if not is_bipartite:
                break
        if is_bipartite:
            m = 2
        else:
            max_degree = max(len(neighbors) for neighbors in adj) if adj else 0
            m = None
            # 从 3 开始尝试
            for k in range(3, max_degree + 2):
                if is_k_colorable(adj, k):
                    m = k
                    break
            if m is None:
                m = max_degree + 1

    # 构造题面描述文本
    board = "Graph Coloring Problem:\n"
    board += "Nodes: " + ", ".join(str(i) for i in nodes) + "\n"
    board += "Edges (format: NodeA - NodeB):\n"
    for u, v in edges:
        board += f"{u} - {v}, "
    board = board.rstrip(", ") + "\n"
    board += (
        "Question: Provide a valid coloring scheme for the graph using exactly " + str(m) +
        " colors (colors are numbered from 0 to " + str(m-1) + ").\n"
        "The coloring scheme should be a JSON list of pairs [node, color] for each node.\n"
        "Output format: 'Answer: [[0, 1], [1, 0], [2, 1],...]"
    )
    
    # 将信息封装在 item 中
    item = {
        "board": board,
        "answer": str(m),   # 正确的最小着色数（一般不直接告知用户）
        "graph": adj,       # 图的邻接表
        "score": 0,
        "is_end": False,
        "action": "",       # 用户提交的具体涂色方案，格式为列表，每个元素为 [node, color]
        "response": [],
        "prompt": "",
        "epoch": 1,
    }
    return item

# 接口2：验证给出的涂色方案
def verify(item: dict):
    try:
        correct_m = int(item['answer'])
    except:
        item['score'] = 0
        return item

    if 'graph' not in item:
        item['score'] = 0
        return item
    adj = item['graph']
    n = len(adj)
    if isinstance(item['action'], str):
        try:
            action = ast.literal_eval(item['action'])
        except (ValueError, SyntaxError):
            item['score'] = 0
            item['is_end'] = True
            return item
    else:
        action = item['action']
    
    if not isinstance(action, list) or not all(isinstance(row, list) for row in action):
        item['score'] = 0
        item['is_end'] = True
        return item

    # 若可以使用 m-1 种颜色着色，则说明 m 不是最小的
    if correct_m > 1 and is_k_colorable(adj, correct_m - 1):
        item['score'] = 0
        return item

    color_map = {}
    nodes_seen = set()
    for entry in action:
        if not isinstance(entry, list) or len(entry) != 2:
            item['score'] = 0
            return item
        node, color = entry
        if not isinstance(node, int) or not isinstance(color, int):
            item['score'] = 0
            return item
        if node < 0 or node >= n:
            item['score'] = 0
            return item
        if node in nodes_seen:
            item['score'] = 0
            return item
        nodes_seen.add(node)
        if color < 0 or color >= correct_m:
            item['score'] = 0
            return item
        color_map[node] = color

    if len(nodes_seen) != n:
        item['score'] = 0
        return item

    # 检查相邻节点颜色不同
    for u in range(n):
        for v in adj[u]:
            if u < v and color_map.get(u) == color_map.get(v):
                item['score'] = 0
                return item

    item['score'] = 1
    return item

# 接口3：格式化打印题面信息
def print_board(item: dict):
    
    return coloring_prompt.format(board=item['board'])

# FastAPI 部分
app = FastAPI()

class GraphColoringState(BaseModel):
    board: str
    answer: str
    graph: List[List[int]]
    score: int
    is_end: bool
    action: str
    response: List[str]
    prompt: str
    epoch: int

class GenerateRequest(BaseModel):
    seed: int

@app.post("/generate", response_model=GraphColoringState)
def api_generate(request: GenerateRequest):
    return generate(request.seed)

@app.post("/verify", response_model=GraphColoringState)
def api_verify(state: GraphColoringState):
    state_dict = state.dict()
    updated_state = verify(state_dict)
    return updated_state

@app.post("/print_board", response_model=dict)
def api_print_board(state: GraphColoringState):
    board_output = print_board(state.dict())
    return {"board": board_output}

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)