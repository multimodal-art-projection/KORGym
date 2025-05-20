# game_lib/10-minigrid/game_lib.py
from __future__ import annotations
import gymnasium as gym
from enum import IntEnum
import numpy as np
from minigrid.wrappers import SymbolicObsWrapper
import random
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid  
from typing import Optional
import argparse

def parse_init():
    """
    Parses command-line arguments for FastAPI deployment configuration.

    Returns:
        argparse.Namespace: Parsed arguments containing `host` and `port`.
    """
    parser = argparse.ArgumentParser(description="Data creation utility")
    parser.add_argument('-p', '--port', type=int, default=8775, help='服务部署端口')
    parser.add_argument('-H', '--host', type=str, default="0.0.0.0", help='服务部署地址')
    args = parser.parse_args()
    return args
app = FastAPI()

# 全局字典，用于存储 env 对象，避免通过 JSON 传输
ENV_STORE = {}

game_prompt = """
    You are a good game problem-solver, I'll give you a game board and rules.
    Your task is:
    - First, give your answer according to the game board and rules.
    - Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: move_forward'
         You are an AI agent navigating a grid environment. Your task is:
 
         1. Analyze environmental observations
         2. Choose optimal actions to achieve the mission
 
         **Core Rules**:
         - Strictly use ONLY the provided action list
         - Prioritize shortest paths
         - Avoid dangerous areas (lava)
         - Manage inventory carefully (keys, etc)
         - Closed doors require 'toggle' to open
         - Locked doors need matching key (use pickup first)
 
         **Action Space** (REQUIRED RESPONSE FORMAT):
         turn_left   : Rotate 90° counter-clockwise
         turn_right  : Rotate 90° clockwise
         move_forward: Advance if path clear
         pickup      : Collect keys/objects
         drop        : Drop carried object
         toggle      : Open doors or interact (facing target required)
         done        : ONLY when goal reached
 
         **Observation**:
         You receive the entire grid as an observation, represented as a 3D array of shape (width, height, 3).
         - Coordinates range from (0,0) at top-left to (width-1, height-1) at bottom-right.
         - Each cell contains [object_type, color, state]:
           - object_type: 1=EMPTY, 2=WALL, 3=FLOOR, 4=DOOR, 5=KEY, 6=BALL, 7=BOX, 8=GOAL, 9=LAVA, 10=AGENT
           - color: 0=RED, 1=GREEN, 2=BLUE, 3=PURPLE, 4=YELLOW, 5=GREY
             - For AGENT (10), this is direction: 0=right, 1=down, 2=left, 3=up
           - state: For DOOR, 0=open, 1=closed, 2=locked; otherwise 0
         - Your position is the cell with object_type=10.
         - The mission is a string, e.g., 'get to the green goal square'.
 
         Respond with exactly one lowercase action word,e.g.'Answer: move_forward'
    """
env_names = [
    "MiniGrid-Empty-5x5-v0",
    "MiniGrid-DoorKey-6x6-v0",
    "MiniGrid-LavaGapS5-v0",
    "MiniGrid-BlockedUnlockPickup-v0",
    "MiniGrid-LavaCrossingS9N1-v0",
    "MiniGrid-LavaCrossingS11N5-v0",
    "MiniGrid-DistShift2-v0",
    "MiniGrid-DoorKey-5x5-v0",
    "MiniGrid-DoorKey-8x8-v0",
    "MiniGrid-Dynamic-Obstacles-5x5-v0",
    "MiniGrid-Dynamic-Obstacles-Random-5x5-v0",
    "MiniGrid-Empty-Random-6x6-v0",
    "MiniGrid-Fetch-5x5-N2-v0",
    "MiniGrid-Fetch-8x8-N3-v0",
    "MiniGrid-FourRooms-v0",
    "MiniGrid-GoToDoor-6x6-v0",
    "MiniGrid-GoToObject-6x6-N2-v0",
    "MiniGrid-KeyCorridorS3R2-v0",
    "MiniGrid-LavaGapS5-v0",
    "MiniGrid-LockedRoom-v0",
    "MiniGrid-MultiRoom-N2-S4-v0",
    "MiniGrid-ObstructedMaze-1Dlhb-v0",
    "MiniGrid-ObstructedMaze-Full-v0",
    "MiniGrid-Playground-v0",
    "MiniGrid-PutNear-8x8-N3-v0",
    "MiniGrid-RedBlueDoors-6x6-v0",
    "MiniGrid-Unlock-v0",
    "MiniGrid-Unlock-v0",
    
]

# 定义对象类型和颜色枚举
class ObjectType(IntEnum):
    UNSEEN = 0
    EMPTY = 1
    WALL = 2
    FLOOR = 3
    DOOR = 4
    KEY = 5
    BALL = 6
    BOX = 7
    GOAL = 8
    LAVA = 9
    AGENT = 10
    UNKNOWN = 255

    @classmethod
    def _missing_(cls, value):
        return cls.UNKNOWN

class Color(IntEnum):
    RED = 0
    GREEN = 1
    BLUE = 2
    PURPLE = 3
    YELLOW = 4
    GREY = 5

# 辅助函数：递归转换 NumPy 类型到原生 Python 类型
def convert_numpy_types(item):
    if isinstance(item, dict):
        return {k: convert_numpy_types(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [convert_numpy_types(i) for i in item]
    elif isinstance(item, tuple):
        return tuple(convert_numpy_types(i) for i in item)
    elif isinstance(item, np.integer):
        return int(item)
    elif isinstance(item, np.floating):
        return float(item)
    elif isinstance(item, np.ndarray):
        return item.tolist()
    else:
        return item

# 打印游戏板
def print_board(item):
    return item['prompt']

# 使用函数创建代理状态（功能风格实现）
def create_agent(env_name="MiniGrid-Empty-5x5-v0", max_steps=100):
    """
    Initializes a symbolic MiniGrid environment and wraps it with metadata including:
    - observation/action tracking
    - inventory and agent direction
    - episode metadata

    Returns:
        dict: Agent state dict including env object and config
    """
    if not env_name.startswith("MiniGrid-"):
        raise ValueError(f"Invalid MiniGrid environment: {env_name}")
    
    env = gym.make(
        env_name,
        render_mode="rgb_array",
        max_steps=max_steps
    )
    env = SymbolicObsWrapper(env)
    
    # 使用字典保存状态
    agent = {
        "env": env,
        "env_name": env_name,
        "max_steps": max_steps,
        "current_step": 0,
        "carrying": None,
        "agent_pos": (1, 1),
        "agent_dir": 0,
        "action_meanings": {
            0: 'turn_left',
            1: 'turn_right',
            2: 'move_forward',
            3: 'pickup',
            4: 'drop',
            5: 'toggle',
            6: 'done'
        },
        'score': 0,
        'is_end': False,
        'response': [],
        'prompt': '',
        'action': '',
        'epoch': 1,
    }
    agent["reverse_action_map"] = {v: k for k, v in agent["action_meanings"].items()}

    return agent

# 生成每一步的提示文本
def get_step_prompt(obs_text: str) -> str:
    return (
        f"Current State:\n{obs_text}\n\n"
        "Choose next action based on the state and rules. "
        "Respond ONLY with the action name:"
    )

# 生成可读的观测描述
def get_observation_text(agent: dict, obs: dict) -> str:
    """
    Generates a human-readable summary of the current MiniGrid symbolic state.

    Args:
        agent (dict): Agent state (position, carrying object, direction).
        obs (dict): Symbolic observation with mission and image grid.

    Returns:
        str: Rich description of current state, mission, and inventory.
    """
    mission = obs['mission']
    
    dir_names = {
        0: "right (→)",
        1: "down (↓)",
        2: "left (←)",
        3: "up (↑)"
    }
    
    # 库存状态
    if agent["carrying"]:
        item_color = Color(agent["carrying"].color).name.lower()
        item_type = ObjectType(agent["carrying"].type).name.lower()
        inventory_status = f"carrying a {item_color} {item_type}"
    else:
        inventory_status = "not carrying anything"
    
    # 网格尺寸和代理信息
    grid_width, grid_height = obs['image'].shape[0], obs['image'].shape[1]
    agent_x, agent_y = agent["agent_pos"]
    agent_dir = agent["agent_dir"]
    
    observation_text = (
        f"Mission: {mission}\n"
        f"Grid size: {grid_width}x{grid_height}\n"
        f"Agent at ({agent_x}, {agent_y}), facing {dir_names[agent_dir]}\n"
        f"Status: {inventory_status} | {agent['max_steps'] - agent['current_step']} steps remaining\n"
        f"Observation:\n{np.array2string(obs['image'], separator=', ')}\n"
    )
    return observation_text

def verify(item: dict):
    """
    Executes the agent's action in the symbolic MiniGrid environment and updates the state.

    - Converts action string to index.
    - Performs the step and updates:
        - position
        - direction
        - inventory
        - score
        - is_end flag
    - Computes whether the game should terminate (goal reached or max step).

    Returns:
        dict: Updated game state after applying the action.
    """
    old_pos = item["agent_pos"]
    old_dir = item["agent_dir"]
    item['epoch']+=1
    if item['epoch']==100:
        if item['score']<0:
            item['score']=0
        if item['score']>0:
            item['score']=1
    try:
        action = item["reverse_action_map"][item['action'].strip().lower()]
    except KeyError:
        action = 2  # 默认动作为 move_forward
    
    obs, reward, terminated, truncated, info = item["env"].step(action)
    item["current_step"] += 1
    
    item["agent_dir"] = item["env"].unwrapped.agent_dir
    item["agent_pos"] = item["env"].unwrapped.agent_pos
    item["carrying"] = obs.get('carrying')
    
    if item['action'] == "move_forward" and np.array_equal(old_pos, item["agent_pos"]):
        print(f"Move blocked at {old_pos} facing {old_dir}")
    
    obs_text = get_observation_text(item, obs)
    item['prompt'] = game_prompt + obs_text
    item['score'] += reward
    item['is_end'] = terminated or truncated
    if "reached_goal" in info.get('reason', ''):
        item['is_end'] = True
    if item['is_end'] == True:
        if item['score']<0:
            item['score']=0
        if item['score']>0:
            item['score']=1
    
    return item

def generate(seed):
    """
    Generates a new symbolic MiniGrid episode with a fixed seed.

    - Randomly samples an environment from `env_names`.
    - Extracts symbolic observation and formats it into a reasoning prompt.
    - Stores the env object in global `ENV_STORE`.

    Returns:
        dict: Initial game state including uid, prompt, env_name, and metadata.
    """

    max_steps = 100
    random.seed(seed)
    env_name = random.sample(env_names, 1)[0]
    item = create_agent(env_name, max_steps=max_steps)
    if seed is not None:
        obs, info = item["env"].reset(seed=seed)
    else:
        obs, info = item["env"].reset()
    item["current_step"] = 0
    item["carrying"] = obs.get('carrying')
    item["agent_dir"] = item["env"].unwrapped.agent_dir
    item["agent_pos"] = item["env"].unwrapped.agent_pos
    obs_text = get_observation_text(item, obs)
    item['prompt'] = game_prompt + obs_text

    # 为该状态生成 uid，并存入全局字典，移除不可序列化的 env 字段
    uid = str(uuid.uuid4())
    item["uid"] = uid
    ENV_STORE[uid] = item["env"]
    del item["env"]

    return item



class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    uid: Optional[str] = None  # 用于标识和恢复 env 对象
    env_name: str
    max_steps: int
    current_step: int
    carrying: Optional[dict] = None
    agent_pos: tuple
    agent_dir: int
    action_meanings: dict
    reverse_action_map: dict
    score: float
    is_end: bool
    action: str
    response: list
    prompt: str
    epoch: int

# --- API ---

# 生成游戏板内容
@app.post("/print_board", response_model=BoardRequest)
def api_print_board(request: GameState):
    state = request.dict()
    board_output = print_board(state)
    return {"board": board_output}

# 生成初始游戏状态
@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    game_state = generate(request.seed)
    # 转换 NumPy 数据类型
    game_state = convert_numpy_types(game_state)
    return game_state

# 根据动作更新游戏状态
@app.post("/verify", response_model=GameState)
def api_verify(request: GameState):
    state = request.dict()
    uid = state.get("uid")
    if not uid or uid not in ENV_STORE:
        raise HTTPException(status_code=400, detail="Invalid or expired game id")
    # 恢复 env 对象
    state["env"] = ENV_STORE[uid]
    
    updated_state = verify(state)
    ENV_STORE[uid] = updated_state["env"]
    del updated_state["env"]
    # 转换 NumPy 数据类型后返回
    updated_state = convert_numpy_types(updated_state)
    return updated_state

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)