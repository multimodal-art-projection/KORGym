import requests

BASE_URL = "http://localhost:8775"

def test_generate(seed=42):
    url = f"{BASE_URL}/generate"
    payload = {"seed": seed}
    response = requests.post(url, json=payload)
    data = response.json()
    print("=== /generate 返回 ===")
    print(data)
    return data

def test_print_board(state):
    url = f"{BASE_URL}/print_board"
    response = requests.post(url, json=state)
    data = response.json()
    print("\n=== /print_board 返回 ===")
    print(data)
    return data

def find_valid_coloring(adj, m):
    """
    利用回溯法求解图的有效涂色方案。
    adj: 图的邻接表（列表的列表，每个元素为相邻节点列表）
    m: 最小着色数（颜色编号范围为 0 ~ m-1）
    
    返回值: 格式为 [[node, color], ...] 的列表，若不存在解则返回 None
    """
    n = len(adj)
    solution = [-1] * n

    def backtrack(i):
        if i == n:
            return True
        for color in range(m):
            valid = True
            for neighbor in adj[i]:
                if solution[neighbor] == color:
                    valid = False
                    break
            if valid:
                solution[i] = color
                if backtrack(i + 1):
                    return True
                solution[i] = -1
        return False

    if backtrack(0):
        return [[i, solution[i]] for i in range(n)]
    else:
        return None

def test_verify(state, coloring):
    # 将用户提交的涂色方案填入 action 字段
    state["action"] = coloring
    url = f"{BASE_URL}/verify"
    response = requests.post(url, json=state)
    data = response.json()
    print("\n=== /verify 返回 ===")
    print(data)
    return data

def main():
    # 1. 调用 /generate 生成图着色问题（使用固定 seed 保证结果可复现）
    state = test_generate(seed=42)
    
    # 2. 调用 /print_board 展示题面信息
    test_print_board(state)
    
    # 3. 从返回的 state 中提取最小着色数 m 和图的邻接表 graph
    m = int(state["answer"])
    graph = state["graph"]
    print("\n最小着色数 m:", m)
    print("图的邻接表:", graph)
    
    # 4. 计算一个有效的涂色方案
    valid_coloring = find_valid_coloring(graph, m)
    print("\n计算得到的有效涂色方案:", valid_coloring)
    
    # 5. 使用正确的涂色方案进行验证
    print("\n使用正确的涂色方案进行验证：")
    test_verify(state, valid_coloring)
    
    # 6. 构造一个错误的涂色方案进行验证：
    #    例如，将第 0 个节点的颜色设置为 m（超出合法范围 0~m-1）
    if valid_coloring is not None:
        incorrect_coloring = valid_coloring.copy()
        incorrect_coloring[0] = [incorrect_coloring[0][0], m]
        print("\n使用错误的涂色方案进行验证：")
        test_verify(state, incorrect_coloring)

if __name__ == "__main__":
    main()
