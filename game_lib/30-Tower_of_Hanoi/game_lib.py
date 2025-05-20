from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
import ast
import uvicorn
from collections import deque
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
game_prompt = '''
You are a good game player, I'll give you a game board and rules.\nYour task is:\n- First, give your answer according to the game board and rules.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: [("a", 2), ("b", 4)]'

{question}
'''

def print_board(item):
    columns_desc = []
    for idx, col in enumerate(item['current_state']):
        column_number = idx + 1
        if col == 'null':
            disks_desc = 'null'
        else:
            disks_desc = col
        columns_desc.append(f"{column_number}: {disks_desc}")
    current_state_str = ", ".join(columns_desc)
    
    board = (
        "The Tower of Hanoi problem consists of four columns and five disks. The objective is to move all the disks to the target column. "
        "The rules are as follows: each move can only move one disk; you can only take off the top disk from a column; "
        "and you can never place a larger disk on top of a smaller one. The disks are labeled as a, b, c, d, e in ascending order of size.\n"
        "The initial state of the Hanoi Tower is similar to: 1: null, 2: a, b, 3: c, d, 4: e. This means column 1 has no disks; "
        "column 2 has disks a and b; column 3 has disks c and d; and column 4 has disk e. Note that column 4 is the target column.\n"
        "Your answer should be a list of moves in the format [(disk, target_column), ...], e.g., [('a', 2), ('b', 4)]. "
        "Here, (a, 2) means moving disk a to column 2.\n\n"
        f"Current state of columns: {current_state_str}"
    )
    prompt = game_prompt.format(question=board)
    return prompt


def generate(seed = None):
    item = {
        'score': 0,
        'is_end': False,
        'response': [],
        'prompt': '',
        'action': '',
        'epoch': 1,
    }
    if seed is not None:
        random.seed(seed)

    level = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    state = _generate(seed, level)
    item['current_state'] = state
    item['difficulty'] = level
    return item

def _generate(seed, level):
    # 定义目标状态：所有盘子按从小到大的顺序在4号柱子上
    target_state = (tuple([]), tuple([]), tuple([]), tuple(['a', 'b', 'c', 'd', 'e']))
    # 使用BFS预计算所有状态及其最短步数
    visited = {}
    queue = deque([(target_state, 0)])
    visited[target_state] = 0

    def can_place(column, disk):
        # 检查是否可以将disk放在柱子顶部
        return not column or disk < column[0]

    def get_prev_states(state):
        prev_states = []
        # 遍历所有可能的源柱子和目标柱子
        for dest in range(4):
            if not state[dest]:
                continue
            # 取出目标柱子顶部盘子
            disk_to_move = state[dest][0]
            # 尝试移动到其他源柱子
            for src in range(4):
                if src == dest:
                    continue
                # 检查是否可以反向移动（即正向移动的逆操作）
                if can_place(state[src], disk_to_move):
                    # 创建新状态
                    new_dest = list(state[dest][1:])
                    new_src = [disk_to_move] + list(state[src])
                    new_state = list(state)
                    new_state = list(map(list, new_state))
                    new_state[dest] = new_dest
                    new_state[src] = new_src
                    new_state = tuple(map(tuple, new_state))
                    prev_states.append((new_state, (disk_to_move, src + 1)))  # src+1是柱号
        return prev_states

    while queue:
        current, steps = queue.popleft()
        # 生成所有可能的下一状态（反向移动）
        for next_state, move in get_prev_states(current):
            if next_state not in visited:
                visited[next_state] = steps + 1
                queue.append((next_state, steps + 1))

    # 按步数分类状态
    steps_dict = {}
    for state, s in visited.items():
        if s not in steps_dict:
            steps_dict[s] = []
        steps_dict[s].append(state)

    # 根据level选择状态
    random.seed(seed)
    if level not in steps_dict:
        raise ValueError(f"Cannot generate state with level {level}, which is larger than 13.")
    chosen_state = random.choice(steps_dict[level])

    # 将元组转换为列表格式
    formatted_state = []
    for col in chosen_state:
        if not col:
            formatted_state.append('null')
        else:
            formatted_state.append(','.join(col))

    return formatted_state

def verify(item):
    try:
        if isinstance(item['action'], str):
            action = ast.literal_eval(item['action'])
        else:
            action = item['action']
    except Exception as e:
        item['score'] = 0
        return item
    item['score'] = _verify(item['current_state'], action)
    return item

def _verify(state, action):
    # 将输入的state转换为内部表示形式
    current_state = []
    for col in state:
        if col == 'null':
            current_state.append([])
        else:
            current_state.append(col.split(','))
    target_column = 3  # 目标柱是第4号柱（索引3）
    # 检查每一步动作的合法性
    for act in action:
        # 解析动作
        if not isinstance(act, tuple) or len(act) != 2:
            return 0
        disk, dest_col = act
        dest_col_idx = dest_col - 1  # 转为0-based索引
        if dest_col_idx < 0 or dest_col_idx >= 4:
            return 0
        # 找到源柱
        src_col_idx = None
        for i in range(4):
            if current_state[i] and current_state[i][0] == disk:
                src_col_idx = i
                break
        if src_col_idx is None:
            return 0  # 盘不在任何柱子的顶部
        # 检查目标柱是否允许放置
        dest_disks = current_state[dest_col_idx]
        if dest_disks:
            dest_top = dest_disks[0]
            if disk >= dest_top:
                return 0  # 不能将较大的盘子放在较小的上面
        # 执行移动
        current_state[src_col_idx].pop(0)
        current_state[dest_col_idx].insert(0, disk)
    # 验证最终状态
    target_disks = current_state[target_column]
    if len(target_disks) != 5:
        return 0
    expected = ['a', 'b', 'c', 'd', 'e']
    for i in range(5):
        if target_disks[i] != expected[i]:
            return 0
    return 1

class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    difficulty: int
    current_state: list
    score: int
    is_end: bool
    action: str
    response: list
    prompt: str
    epoch: int

@app.post("/print_board", response_model=BoardRequest)
def api_print_board(request: GameState):
    state = request.dict()
    board_output = print_board(state)
    return {"board": board_output}

@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    game_state = generate(request.seed)
    return game_state

@app.post("/verify", response_model=GameState)
def api_verify(request: GameState):
    state = request.model_dump()
    updated_state = verify(state)
    return updated_state

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)