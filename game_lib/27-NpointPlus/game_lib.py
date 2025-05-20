from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
from copy import deepcopy
import uvicorn
from typing import Optional
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

game_prompt = '''
You are a good game problem-solver, I'll give you a game board and rules.
Your task is:
- First, give your answer according to the game board and rules.
- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g. 'Answer: Hit'
At the start of the game, the dealer has one face-up card and one face-down card, while the player has two face-up cards. All cards are drawn from an infinite deck.

Card values are as follows:
- Aces are worth 1 point.
- Number cards (2-10) are worth their numerical value.
- Face cards (J, Q, K) are worth 11, 12, and 13 points, respectively.

In each round, the player can request an additional card (Hit) until they decide to stop (Stand) or exceed N. When both sides have stopped, the settlement occurs.
If one side exceeds N and the other side does not, the side that did not exceed N wins, and the other side loses.
If both sides exceed N, it is a draw.
If neither side exceeds N, the side closest to N wins, and the other side loses.
If both sides have the same total, it is a draw.
Winning earns 1 score, a draw earns 0.5 score, and losing earns 0 score.

You will play 10 rounds against an opponent with a fixed strategy. You have access to the complete records of all previous rounds while your opponent only has access to the records of the current round. Your opponent's strategy remains unchanged throughout the game.

Now the current game situation is as follows:

{board}
'''

# ------------------------ 牌面与点数定义 ------------------------
poker2point = {"A": 1, "J": 11, "Q": 12, "K": 13}
for i in range(2, 11):
    poker2point[str(i)] = i

def get_poker():
    """从无限牌堆中随机抽一张牌"""
    return random.choice(list(poker2point.keys()))

def compute_points(cards):
    """计算一手牌的总点数"""
    return sum(poker2point[card] for card in cards)

# ------------------------ 接口函数定义 ------------------------
def generate(seed: int) -> dict:
    """
    根据给定的 seed 初始化一个游戏状态：
      - 随机确定对手策略（在 "stubborn"、"careful"、"normal"、"repeat" 中随机）
      - 为本回合发牌（玩家和对手各两张牌）
      - 随机生成一个阈值 N（24~50）
      - 初始化本回合状态（如 hit/stand 标识、回合计数 turn、对局日志 moves）
      - 同时初始化累计得分 score、当前回合 current_round 及总回合数 total_rounds（默认 10）
      - 新增 history 字段保存之前所有回合记录
    """
    random.seed(seed)
    strategies = ["stubborn", "careful", "normal", "repeat"]
    opponent_strategy = random.choice(strategies)
    item = {
        "seed": seed,
        "opponent_strategy": opponent_strategy,
        "n": random.randint(24, 50),
        "player_cards": [get_poker(), get_poker()],
        "opponent_cards": [get_poker(), get_poker()],
        "player_stand": False,
        "opponent_stand": False,
        "player_bust": False,
        "opponent_bust": False,
        "turn": 1,
        "moves": [],              # 记录本回合双方的动作
        "last_player_action": None,  # 用于“repeat”策略记录上一次玩家动作
        "opponent_first_move": True, # 用于“repeat”策略判断是否第一回合
        "current_round": 1,       # 当前回合，初始为 1
        "total_rounds": 10,       # 总回合数
        "score": 0,               # 累计得分
        "is_end": False,          # 标识整个游戏是否结束
        "round_result": None,     # 回合结束后存放本回合结果信息
        "history": [],            # 保存之前回合完整记录
        "action": ""              # 玩家下一步操作的输入，格式要求为 "hit" 或 "stand"
    }
    # 初始化提示信息：展示当前局面（对手牌部分隐藏）
    item["prompt"] = print_board(item, reveal=False)
    return item

def verify(item: dict) -> dict:
    """
    根据 item 中的 action（玩家操作）更新当前游戏状态：
      1. 根据玩家操作 ("hit"/"stand") 更新玩家手牌，若玩家因点数超过 N 而爆牌，则标记 player_bust 与 player_stand。
      2. 根据对手策略更新对手操作：
           - "stubborn"：始终 hit
           - "careful"：始终 stand
           - "normal"：当对手总点数 < N-6 时 hit，否则 stand
           - "repeat"：第一回合固定 hit，之后模仿玩家上一回合动作（默认为 stand）
      3. 记录本回合双方动作，更新 turn 计数。
      4. 当本回合双方均已停牌或爆牌时，计算双方得分，并将本回合得分累加到累计 score 中；
         同时将本回合的所有记录（其中包含阈值 N 信息）保存到 history 中；如果当前回合数小于总回合数，则准备新一轮游戏，
         否则将 is_end 置为 True，表示整个游戏结束。
    最后清空 action 字段，并返回更新后的 item。
    """
    # 若整个游戏已经结束，则直接返回状态
    if item.get("is_end", False):
        return item

    action_line = item.get("action", "").strip().lower()
    if action_line not in ["hit", "stand"]:
        # 非法输入，仅更新提示信息后返回
        item["prompt"] = print_board(item, reveal=False)
        return item

    player_action = action_line

    # 玩家操作处理：若尚未停牌/爆牌
    if not item["player_stand"] and not item["player_bust"]:
        if player_action == "hit":
            card = get_poker()
            item["player_cards"].append(card)
            # 若玩家总点数超过 n，则视为爆牌并自动停牌
            if compute_points(item["player_cards"]) > item["n"]:
                item["player_bust"] = True
                item["player_stand"] = True
        elif player_action == "stand":
            item["player_stand"] = True
        # 保存玩家操作，用于“repeat”策略
        item["last_player_action"] = player_action

    # 对手操作处理：若尚未停牌或爆牌
    if not item["opponent_stand"] and not item["opponent_bust"]:
        strat = item["opponent_strategy"]
        if strat == "stubborn":
            opponent_action = "hit"
        elif strat == "careful":
            opponent_action = "stand"
        elif strat == "normal":
            if compute_points(item["opponent_cards"]) < item["n"] - 6:
                opponent_action = "hit"
            else:
                opponent_action = "stand"
        elif strat == "repeat":
            if item["opponent_first_move"]:
                opponent_action = "hit"
                item["opponent_first_move"] = False
            else:
                opponent_action = item["last_player_action"] if item["last_player_action"] in ["hit", "stand"] else "stand"
        else:
            opponent_action = "stand"

        if opponent_action == "hit":
            card = get_poker()
            item["opponent_cards"].append(card)
            if compute_points(item["opponent_cards"]) > item["n"]:
                item["opponent_bust"] = True
                item["opponent_stand"] = True
        elif opponent_action == "stand":
            item["opponent_stand"] = True
    else:
        opponent_action = "stand"

    # 记录本回合双方动作
    move_log = f"Turn {item['turn']}: You: {player_action}; Opponent: {opponent_action}"
    item["moves"].append(move_log)
    item["turn"] += 1

    # 检查本回合是否结束：当双方均已停牌或爆牌时
    if (item["player_stand"] or item["player_bust"]) and (item["opponent_stand"] or item["opponent_bust"]):
        # 本回合结束，计算双方最终点数
        player_total = compute_points(item["player_cards"])
        opponent_total = compute_points(item["opponent_cards"])
        n = item["n"]
        if player_total > n and opponent_total <= n:
            outcome_str = "You lose!"
            round_score = 0
        elif opponent_total > n and player_total <= n:
            outcome_str = "You win!"
            round_score = 1
        elif player_total > n and opponent_total > n:
            outcome_str = "Draw game!"
            round_score = 0.5
        else:
            if player_total > opponent_total:
                outcome_str = "You win!"
                round_score = 1
            elif player_total == opponent_total:
                outcome_str = "Draw game!"
                round_score = 0.5
            else:
                outcome_str = "You lose!"
                round_score = 0

        item["round_result"] = {
            "outcome": outcome_str,
            "round_score": round_score,
            "player_total": player_total,
            "opponent_total": opponent_total
        }
        # 累计更新得分
        item["score"] += round_score

        # 保存本回合的完整记录到 history（包括当前回合的阈值 N）
        round_record = {
            "round": item["current_round"],
            "n": item["n"],
            "moves": deepcopy(item["moves"]),
            "player_cards": item["player_cards"],
            "player_total": player_total,
            "opponent_cards": item["opponent_cards"],
            "opponent_total": opponent_total,
            "round_result": item["round_result"]
        }
        item["history"].append(round_record)

        # 判断是否还有后续回合
        if item["current_round"] < item["total_rounds"]:
            # 开始新一轮游戏：重置本回合状态
            item["current_round"] += 1
            item["n"] = random.randint(24, 50)
            item["player_cards"] = [get_poker(), get_poker()]
            item["opponent_cards"] = [get_poker(), get_poker()]
            item["player_stand"] = False
            item["opponent_stand"] = False
            item["player_bust"] = False
            item["opponent_bust"] = False
            item["turn"] = 1
            item["moves"] = []
            item["last_player_action"] = None
            item["opponent_first_move"] = True
            item["round_result"] = None
            item["action"] = ""
            # 更新提示信息为新一轮局面（当前回合对手牌部分隐藏）
            item["prompt"] = print_board(item, reveal=False)
        else:
            # 所有回合完成，标记整个游戏结束，提示公开所有信息（包括当前回合）
            item["is_end"] = True
            item["prompt"] = print_board(item, reveal=True)
    else:
        # 回合未结束，继续显示当前局面（当前回合对手牌部分隐藏）
        item["prompt"] = print_board(item, reveal=False)

    # 清空本次玩家输入的 action
    item["action"] = ""
    return item

def print_board(item: dict, reveal: bool = False) -> str:
    """
    根据当前 item 状态生成游戏局面文本：
      - 首先显示历史回合记录：其中每一回合都显示完整信息（包括对手牌和当时的阈值 N）。
      - 然后显示当前回合信息：
            若当前回合未结束则对手牌仅显示第一张，其余以 "unknown card" 表示；
            若当前回合结束则公开全部对手牌。
      - 并显示当前回合、总回合数、累计得分及本回合相关信息（阈值 n、双方牌面、回合记录）。
      - 最后提示玩家以 "Answer: hit" 或 "Answer: stand" 的格式输入下一步操作。
    """
    s = ""
    # 显示历史回合记录
    s += "=== History ===\n"
    if item["history"]:
        for record in item["history"]:
            s += f"Round {record['round']}:\n"
            s += f"  Threshold (N): {record['n']}\n"
            s += f"  Moves: {' | '.join(record['moves'])}\n"
            s += f"  Your cards: {record['player_cards']} (Total: {record['player_total']})\n"
            s += f"  Opponent's cards: {record['opponent_cards']} (Total: {record['opponent_total']})\n"
            s += f"  Outcome: {record['round_result']['outcome']}, Round Score: {record['round_result']['round_score']}\n"
            s += "--------------------\n"
    else:
        s += "No previous rounds.\n"
    
    # 显示当前回合信息
    s += "=== Current Round ===\n"
    s += f"Round: {item['current_round']} / {item['total_rounds']}\n"
    s += f"Score: {item['score']}\n"
    s += f"Threshold (N): {item['n']}\n"
    
    # 玩家手牌
    player_cards = item["player_cards"]
    player_total = compute_points(player_cards)
    s += f"Your cards: {player_cards} (Total: {player_total}).\n"
    
    # 对手手牌：若当前回合尚未结束则隐藏部分牌面，否则公开全部
    if not reveal:
        if len(item["opponent_cards"]) > 0:
            opponent_view = [item["opponent_cards"][0]] + ["unknown card"] * (len(item["opponent_cards"]) - 1)
        else:
            opponent_view = []
        s += f"Opponent's cards: {opponent_view}.\n"
    else:
        opponent_cards = item["opponent_cards"]
        opponent_total = compute_points(opponent_cards)
        s += f"Opponent's cards: {opponent_cards} (Total: {opponent_total}).\n"
    
    s += f"Turn: {item['turn']}\n"
    s += "Move history:\n"
    if item["moves"]:
        for move in item["moves"]:
            s += f"  {move}\n"
    else:
        s += "  No moves yet.\n"
    
    return game_prompt.format(board=s)

# ------------------------ FastAPI 请求/响应数据模型 ------------------------
class GenerateRequest(BaseModel):
    seed: int

class BoardRequest(BaseModel):
    board: str

class GameState(BaseModel):
    seed: int
    opponent_strategy: str
    n: int
    player_cards: list
    opponent_cards: list
    player_stand: bool
    opponent_stand: bool
    player_bust: bool
    opponent_bust: bool
    turn: int
    moves: list
    last_player_action: Optional[str] = None
    opponent_first_move: bool
    current_round: int
    total_rounds: int
    score: float
    is_end: bool
    history: list
    round_result: Optional[dict] = None
    prompt: str
    action: str

# ------------------------ FastAPI 接口路由 ------------------------
app = FastAPI()

@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    state = generate(request.seed)
    return state

@app.post("/print_board", response_model=BoardRequest)
def api_print_board(request: GameState):
    board_output = print_board(request.dict(), reveal=request.is_end)
    return {"board": board_output}

@app.post("/verify", response_model=GameState)
def api_verify(request: GameState):
    updated_state = verify(request.dict())
    return updated_state

# ------------------------ 程序入口 ------------------------
if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)
