from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
from typing import List, Tuple
import argparse
import uvicorn
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

# ----------------- 游戏提示信息 -----------------
game_prompt = """
You are a good game player, I'll give you a game board and rules.\nYour task is:\n- First, give your answer according to the game board and rules.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: cheat'

When you put in one coin, your opponent will receive three coins, and vice versa. Both of you can choose to "collaborate" (put in one coin) or "cheat" (put in no coins). If you collaborate but your opponent cheats, you will lose one coin, and your opponent gains three (-1 : +3). However, if both choose to cheat, neither gains nor loses anything (0 : 0).

You will play 5 major rounds against an opponent with a fixed strategy, each major round consisting of 8 minor rounds. You have access to the complete records of all previous major rounds and the current major round, while your opponent only has access to the records of the current major round. Your opponent's strategy remains unchanged throughout the game.
{board}
"""

# ----------------- 对手策略函数 -----------------
def opponent_action(strategy: str, round_num: int, history: List[Tuple[str, str]]) -> str:
    # 注意：这里的 round_num 表示当前大轮次内的小轮次编号
    if strategy == "copycat":
        return history[-1][0] if history else "collaborate"
    elif strategy == "little pink":
        return "collaborate"
    elif strategy == "cheater":
        return "cheat"
    elif strategy == "grudger":
        if not history:
            return "collaborate"
        return "cheat" if "cheat" in [h[0] for h in history] else "collaborate"
    elif strategy == "detective":
        if round_num == 1:
            return "collaborate"
        elif round_num == 2:
            return "cheat"
        elif round_num in [3, 4]:
            return "collaborate"
        else:
            if any(h[0] == "cheat" for h in history):
                return history[-1][0]
            else:
                return "cheat"
    elif strategy == "copykitten":
        if len(history) >= 2 and history[-1][0] == history[-2][0] == "cheat":
            return "cheat"
        return history[-1][0] if history else "collaborate"
    elif strategy == "stubborn":
        if not history:
            return "collaborate"
        if history[-1][0] == history[-1][1]:
            return history[-1][1]
        else:
            return "cheat" if history[-1][1] == "collaborate" else "collaborate"
    elif strategy == "random":
        return random.choice(["cheat", "collaborate"])
    else:
        raise ValueError("Unknown strategy")

# ----------------- 游戏状态展示函数 -----------------
def print_board(item: dict) -> str:
    """
    根据当前 item 信息生成游戏状态字符串，
    显示大轮次、小轮次、对手策略、得分以及历史回合记录，
    并提示玩家按照格式输入答案，例如：Answer: cheat
    """
    board_info = (
        f"Major Round: {item['major_round']} / 5\n"
        f"Minor Round: {item['minor_round']} / 8\n"
        f"Score: {item['score']}\n"
        "Completed Major Rounds History:\n"
    )
    if item["history"]:
        for idx, major in enumerate(item["history"], start=1):
            board_info += f"  Major Round {idx}:\n"
            for i, (p, o) in enumerate(major, start=1):
                board_info += f"    Minor {i}: You: {p}, Opponent: {o}\n"
    else:
        board_info += "  None\n"
    board_info += "\nCurrent Major Round History:\n"
    if item["current_history"]:
        for i, (p, o) in enumerate(item["current_history"], start=1):
            board_info += f"  Minor {i}: You: {p}, Opponent: {o}\n"
    else:
        board_info += "  No rounds played yet in this major round.\n"

    board_info += "\nPlease input your action for the next minor round in the format:\nAnswer: cheat   (or)   Answer: collaborate\n"
    return game_prompt.format(board=board_info)

# ----------------- 接口逻辑函数 -----------------
def generate(seed: int) -> dict:
    """
    根据给定的 seed 生成初始游戏状态，
    随机选取对手策略，并初始化大轮次、小轮次和历史记录。
    """
    random.seed(seed)
    opponent_type = random.choice([
        "copycat", "little pink", "cheater", "grudger",
        "detective", "copykitten", "stubborn", "random"
    ])
    item = {
        "seed": seed,
        "opponent_type": opponent_type,
        "score": 0,
        "major_round": 1,
        "minor_round": 1,
        "history": [],           # 已完成的大轮次历史（每个元素为该大轮次内所有小轮次的记录）
        "current_history": [],   # 当前大轮次内的记录
        "is_end": False,
        "prompt": "",
        "action": ""
    }
    return item

def verify(item: dict) -> dict:
    """
    根据 item 中的 action 更新游戏状态：
    - 解析玩家的行动（要求格式为 'Answer: YOUR_ACTION'，如果存在前缀则去除）
    - 依据当前大轮次内的记录计算对手行动
    - 按照博弈规则更新得分，并将当前小轮次加入当前大轮次记录
    - 当当前大轮次进行完 8 个小轮次后，将记录归档，并开始下一大轮次，直至 5 个大轮次全部结束
    - 更新提示信息 prompt
    """
    if item.get("is_end", False):
        return item

    action_line = item.get("action", "").strip()
    # 解析格式，如果包含 "Answer:" 前缀则去掉
    if action_line.lower().startswith("answer:"):
        player_action = action_line.split("Answer:", 1)[1].strip().lower()
    else:
        player_action = action_line.lower()
    if player_action not in ["cheat", "collaborate"]:
        # 无效输入则不更新状态
        return item

    # 当前大轮次内的小轮次编号即为 minor_round
    current_minor = item["minor_round"]
    # 对手仅依据当前大轮次内的记录来判断（item["current_history"]）
    opp_action = opponent_action(item["opponent_type"], current_minor, item["current_history"])

    # 更新得分：依据规则
    if player_action == "collaborate" and opp_action == "collaborate":
        item["score"] += 2
    elif player_action == "collaborate" and opp_action == "cheat":
        item["score"] -= 1
    elif player_action == "cheat" and opp_action == "collaborate":
        item["score"] += 3
    # 双方作弊则得分不变

    # 将本小轮次记录加入当前大轮次历史
    item["current_history"].append((player_action, opp_action))

    # 更新小轮次与大轮次计数：
    if item["minor_round"] < 8:
        item["minor_round"] += 1
    else:
        # 当前大轮次结束，将当前大轮次记录归入总历史，并重置当前记录和小轮次计数
        item["history"].append(item["current_history"])
        item["current_history"] = []
        if item["major_round"] < 5:
            item["major_round"] += 1
            item["minor_round"] = 1
        else:
            # 如果已经完成第五个大轮次，游戏结束
            item["is_end"] = True

    # 清空上次输入，更新提示信息
    item["action"] = ""
    item["prompt"] = print_board(item)
    return item

# ----------------- FastAPI 请求/响应数据模型 -----------------
class GenerateRequest(BaseModel):
    seed: int

class BoardRequest(BaseModel):
    board: str

class GameState(BaseModel):
    seed: int
    opponent_type: str
    score: int
    major_round: int
    minor_round: int
    history: list
    current_history: list
    is_end: bool
    prompt: str
    action: str

# ----------------- FastAPI 接口路由 -----------------
@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    game_state = generate(request.seed)
    return game_state

@app.post("/print_board", response_model=BoardRequest)
def api_print_board(request: GameState):
    board_output = print_board(request.dict())
    return {"board": board_output}

@app.post("/verify", response_model=GameState)
def api_verify(request: GameState):
    state = request.dict()
    updated_state = verify(state)
    return updated_state


if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)