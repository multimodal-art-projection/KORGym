import random
import heapq
import string
from collections import defaultdict
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
# 游戏提示模板，用于包装生成的题面信息
game_prompt = """
You are a good game player, I'll give you a game board and rules.
Your task is:
- First, give your answer according to the game board and rules.
- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question, e.g., 'Answer: 12'
{board}
"""

def generate(seed: int) -> dict:
    """
    根据给定随机种子生成建筑公司项目调度游戏的题面、答案及其他辅助信息。
    默认参数：
      - 公司数量：5
      - 每个公司项目数量：5
      - 城市总任务数量：10
      - 同时允许执行的任务数量：3
    返回一个 item 字典，其中包含题面 board、答案 answer 以及其他状态信息。
    """
    random.seed(seed)
    # 默认参数设置
    company_num = random.randint(5,50)
    task_num = random.randint(5,20)
    total_task_num = random.randint(20,100)
    task_allowed = random.randint(2,4)

    # 生成项目池，确保项目数量足够
    project_pool = {}
    total_projects = company_num * task_num * 2
    while len(project_pool) < total_projects:
        proj_name = ''.join(random.choices(string.ascii_lowercase, k=6))
        if proj_name not in project_pool:
            project_pool[proj_name] = 0

    # 为每个公司分配项目，并记录每个项目对应可承接的公司
    companies = []
    project_to_companies = defaultdict(list)
    for company_id in range(1, company_num + 1):
        available_projects = list(project_pool.keys())
        selected_projects = random.sample(available_projects, task_num)
        company = {}
        for proj in selected_projects:
            duration = random.randint(1, 10)  # 项目完成时长（单位：年）
            company[proj] = duration
        companies.append(company)
        for proj in selected_projects:
            project_to_companies[proj].append(company_id)

    # 生成城市规划书，确保选取的项目都有公司承接
    plan = []
    valid_projects = list(project_to_companies.keys())
    do_remove = True
    if len(valid_projects) <= total_task_num:
        do_remove = False
    for _ in range(total_task_num):
        proj = random.choice(valid_projects)
        company_id = random.choice(project_to_companies[proj])
        plan.append((company_id, proj))
        if do_remove:
            valid_projects.remove(proj)

    # 构造题面描述
    companies_info = "\n".join(
        f"Company {idx} can handle:" +
        "".join(f"\n  {proj}: {dur} year{'s' if dur != 1 else ''}"
                for proj, dur in company.items())
        for idx, company in enumerate(companies, 1)
    )
    plan_details = " -> ".join(f"({c}, {p})" for c, p in plan)
    question = f"""[Construction Company Scheduling Game Rules]
1. Game Objective:
Calculate the total time required to complete all projects in the city's plan, considering:
- Projects must be executed in the order listed.
- A maximum of {task_allowed} projects can run simultaneously.

2. Company Capabilities:
{companies_info}

3. City Project Plan (in strict order; data format is (Company ID, Project Name)):
{plan_details}

4. Rules:
- Projects start immediately when a slot is available.
- Time is measured in years.
- If all concurrent slots are occupied, new projects must wait.
- The total duration is from the start of the first project to the completion of the last project.
- Each company can only undertake projects they are capable of.
- When projects are repeated, they must be completed each time.

Please calculate the minimum possible total time to complete all projects.
"""

    def calculate_total_time(plan, companies, task_allowed):
        heap = []
        max_end = 0
        for company_idx, project in plan:
            duration = companies[company_idx - 1][project]
            if len(heap) >= task_allowed:
                start = heapq.heappop(heap)
            else:
                start = 0
            end = start + duration
            heapq.heappush(heap, end)
            max_end = max(max_end, end)
        return max_end

    answer = calculate_total_time(plan, companies, task_allowed)

    # 构造 item 作为信息传递媒介
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
    根据 item 中的 action 字段判断用户提交的答案是否正确，
    若 action 转换为整数与正确答案一致，则 score 为 1，否则为 0。
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
    返回包装后的题面文本，将生成的题面插入到游戏提示模板中。
    """
    return game_prompt.format(board=item['board'])

# 定义 FastAPI 应用
app = FastAPI()

# 定义请求和状态模型
class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    board: str
    answer: str
    score: int
    is_end: bool
    action: str
    response: List[str]
    prompt: str
    epoch: int

class BoardRequest(BaseModel):
    board: str

# 生成初始游戏状态
@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    game_state = generate(request.seed)
    return game_state

# 根据用户动作验证答案
@app.post("/verify", response_model=GameState)
def api_verify(request: GameState):
    state = request.dict()
    updated_state = verify(state)
    return updated_state

# 输出包装后的题面
@app.post("/print_board", response_model=BoardRequest)
def api_print_board(request: GameState):
    state = request.dict()
    board_output = print_board(state)
    return {"board": board_output}

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)