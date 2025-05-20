import random
from collections import Counter, defaultdict

def generate(seed, N, extra_count=None):
    """
    生成一笔画题目，并增加额外边以提高难度

    参数：
      seed (int): 随机数种子
      N (int): 节点数量
      extra_count (int, optional): 额外增加的边数（不包括构成环的边），默认取节点数的一半（向下取整，至少为1）

    返回：
      nodes (list of str): 节点名称列表，例如 ["node 1", "node 2", ...]
      full_edges (list of tuple): 完整边信息列表，每条边用 (nodeA, nodeB) 表示，
                                  输出时格式为 <node A, node B>

    说明：
      1. 首先构造一个包含所有节点的环，保证图连通且每个节点初始度为 2（构成欧拉回路）。
      2. 从所有不在环上的节点对中随机增加额外边（数量由 extra_count 控制）。
      3. 如果增加后图的奇度顶点数不为 0 或 2，则尝试自动添加一条连接两个奇度顶点的边，
         使得最终图存在欧拉路径（或欧拉回路）。
    """
    random.seed(seed)
    nodes = ["node " + str(i+1) for i in range(N)]
    
    # 构造环：依次连接所有节点，最后从最后一个节点连接回第一个节点
    cycle_edges = []
    for i in range(N):
        edge = (nodes[i], nodes[(i+1) % N])
        cycle_edges.append(edge)
    
    # 设置额外边数量（如果未指定，默认取节点数的一半，至少为1）
    if extra_count is None:
        extra_count = max(1, N // 2)
    
    # 构造候选池：所有节点对 (i, j)（i<j）中排除环中已有的边（相邻节点，包括首尾相连）
    candidate_pool = []
    for i in range(N):
        for j in range(i+1, N):
            if not (j == i+1 or (i == 0 and j == N-1)):
                candidate_pool.append((i, j))
    
    # 如果候选池中的边数少于要求的额外边，则调整 extra_count
    if extra_count > len(candidate_pool):
        extra_count = len(candidate_pool)
    
    random.shuffle(candidate_pool)
    selected_pairs = candidate_pool[:extra_count]
    extra_edges = [(nodes[i], nodes[j]) for (i, j) in selected_pairs]
    
    # 初步完整边集合
    full_edges = cycle_edges + extra_edges

    def get_odd_nodes(edge_list):
        degree = {node: 0 for node in nodes}
        for u, v in edge_list:
            degree[u] += 1
            degree[v] += 1
        # 返回所有奇度节点
        return [node for node in nodes if degree[node] % 2 == 1]
    
    # 检查欧拉性：允许 0 或 2 个奇度节点
    odd_nodes = get_odd_nodes(full_edges)
    # 如果奇度节点数不为 0 或 2，则尝试添加一条连接两个奇度节点的边（前提是该边未被使用）
    while len(odd_nodes) not in (0, 2):
        found = False
        # 尝试连接任意两个奇度节点
        for i in range(len(odd_nodes)):
            for j in range(i+1, len(odd_nodes)):
                candidate_edge = frozenset((odd_nodes[i], odd_nodes[j]))
                # 检查该边是否已存在于 full_edges 中
                exists = False
                for u, v in full_edges:
                    if frozenset((u, v)) == candidate_edge:
                        exists = True
                        break
                if not exists:
                    full_edges.append((odd_nodes[i], odd_nodes[j]))
                    found = True
                    break
            if found:
                break
        # 重新计算奇度节点
        odd_nodes = get_odd_nodes(full_edges)
        # 如果没有合适的候选边，则退出（一般不会发生）
        if not found:
            break

    return nodes, full_edges

def verify(nodes, edges, candidate):
    """
    验证待测解是否为有效的一笔画解答

    参数：
      nodes (list of str): 节点名称列表
      edges (list of tuple): 边信息列表，每条边用 (nodeA, nodeB) 表示
      candidate (list of str): 待验证的解答，节点序列，例如 ["node 1", "node 5", "node 7", ...]

    返回：
      bool: True 表示解答正确，即每条边恰好使用一次；否则 False。

    验证步骤：
      1. 检查解答长度是否恰好为 (边数 + 1)；
      2. 遍历解答中连续的两个节点，检查对应边是否存在（视为无向边），并将该边“删除”；
      3. 最后所有边必须被使用完毕。
    """
    # Euler 路径中边数应为序列长度 - 1
    if len(candidate) != len(edges) + 1:
        return False
    
    # 使用 Counter 统计边的出现次数（无向边用 frozenset 表示）
    edge_counter = Counter()
    for u, v in edges:
        key = frozenset((u, v))
        edge_counter[key] += 1
    
    # 遍历候选解答，依次“消耗”相应的边
    for i in range(len(candidate) - 1):
        u = candidate[i]
        v = candidate[i+1]
        key = frozenset((u, v))
        if edge_counter[key] <= 0:
            return False
        edge_counter[key] -= 1
    
    # 检查所有边是否恰好使用完毕
    return sum(edge_counter.values()) == 0

def find_eulerian_path(nodes, edges):
    """
    利用 Hierholzer 算法求解欧拉路径/回路

    参数：
      nodes (list of str): 节点名称列表
      edges (list of tuple): 边信息列表，每条边用 (nodeA, nodeB) 表示

    返回：
      list of str: 一笔画序列（欧拉路径或欧拉回路），节点依次排列
    """
    graph = defaultdict(list)
    # 构造无向图（支持多重边）
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    # 如果存在奇度节点，则欧拉路径必须从其中一个奇度节点开始
    start = nodes[0]
    odd_nodes = [node for node in nodes if len(graph[node]) % 2 == 1]
    if odd_nodes:
        start = odd_nodes[0]
    
    stack = [start]
    path = []
    
    # Hierholzer 算法：沿着边前进，直到无可走边，再回退添加到路径中
    while stack:
        cur = stack[-1]
        if graph[cur]:
            next_node = graph[cur].pop()
            graph[next_node].remove(cur)
            stack.append(next_node)
        else:
            path.append(stack.pop())
    
    path.reverse()
    return path

# 示例演示
if __name__ == "__main__":
    seed = 42
    N = 15  # 可以调整节点数量
    # 指定额外边数量（不指定则默认取节点数的一半）
    nodes, edges = generate(seed, N,2*N-2)
    
    # 输出节点信息
    print("节点信息:")
    for node in nodes:
        print(node)
    
    # 输出边信息，格式为 <node A, node B>
    print("\n边信息:")
    for edge in edges:
        print(f"<{edge[0]}, {edge[1]}>")
    
    # 利用 Hierholzer 算法求解欧拉路径（一笔画序列）
    eulerian_path = find_eulerian_path(nodes, edges)
    print("\n一笔画序列:")
    print(", ".join(eulerian_path))
    print(eulerian_path)
    eulerian_path='node 13, node 12, node 11, node 10, node 9, node 15, node 1, node 2, node 3, node 4, node 5, node 6, node 7, node 8, node 4, node 13, node 7, node 3, node 1, node 11, node 2, node 7, node 5, node 9, node 7, node 10, node 6, node 12, node 8, node 9, node 6, node 13, node 3, node 14, node 15, node 7, node 14, node 11, node 4, node 1, node 13, node 8, node 11, node 5, node 13, node 14'.split(',')
    for i in range(len(eulerian_path)):
        eulerian_path[i]=eulerian_path[i].strip()
    # 验证解答是否正确
    is_valid = verify(nodes, edges, eulerian_path)
    print("\n验证结果:", is_valid)
