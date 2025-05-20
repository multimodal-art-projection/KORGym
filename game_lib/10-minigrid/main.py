from __future__ import annotations
import gymnasium as gym
from enum import IntEnum
import numpy as np
from minigrid.wrappers import SymbolicObsWrapper
import random
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

game_prompt="""
    You are a good game problem-solver, I'll give you a game board and rules.\nYour task is:\n- First, give your answer according to the game board and rules.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: move_forward'
         You are an AI agent navigating a grid environment. Your task is:\n 
         1. Analyze environmental observations\n 
         2. Choose optimal actions to achieve the mission\n\n 
        
         **Core Rules**:\n 
         - Strictly use ONLY the provided action list\n 
         - Prioritize shortest paths\n 
         - Avoid dangerous areas (lava)\n 
         - Manage inventory carefully (keys, etc)\n 
         - Closed doors require 'toggle' to open\n 
         - Locked doors need matching key (use pickup first)\n\n 
        
         **Action Space** (REQUIRED RESPONSE FORMAT):\n 
         turn_left   : Rotate 90° counter-clockwise\n 
         turn_right  : Rotate 90° clockwise\n 
         move_forward: Advance if path clear\n 
         pickup      : Collect keys/objects\n 
         drop        : Drop carried object\n 
         toggle      : Open doors or interact (facing target required)\n 
         done        : ONLY when goal reached\n\n 
        
         **Observation**:\n 
         You receive the entire grid as an observation, represented as a 3D array of shape (width, height, 3).\n 
         - Coordinates range from (0,0) at top-left to (width-1, height-1) at bottom-right.\n 
         - Each cell contains [object_type, color, state]:\n 
           - object_type: 1=EMPTY, 2=WALL, 3=FLOOR, 4=DOOR, 5=KEY, 6=BALL, 7=BOX, 8=GOAL, 9=LAVA, 10=AGENT\n 
           - color: 0=RED, 1=GREEN, 2=BLUE, 3=PURPLE, 4=YELLOW, 5=GREY\n 
             - For AGENT (10), this is direction: 0=right, 1=down, 2=left, 3=up\n 
           - state: For DOOR, 0=open, 1=closed, 2=locked; otherwise 0\n 
         - Your position is the cell with object_type=10.\n 
         - The mission is a string, e.g., 'get to the green goal square'.\n\n 
         Respond with exactly one lowercase action word,e.g.'Answer: move_forward'\n
    """
env_names=[
            "MiniGrid-Empty-5x5-v0",
            "MiniGrid-DoorKey-6x6-v0",
            "MiniGrid-LavaGapS5-v0",
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


# 打印游戏板
def print_board(item):
    return item['prompt']
# 使用函数创建代理状态（功能风格实现）
def create_agent(env_name="MiniGrid-Empty-5x5-v0", max_steps=100):
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

# 重置环境，并更新代理状态
def reset_agent(agent: dict, seed=None):
    if seed is not None:
        obs, info = agent["env"].reset(seed=seed)
    else:
        obs, info = agent["env"].reset()
    agent["current_step"] = 0
    agent["carrying"] = obs.get('carrying')
    agent["agent_dir"] = agent["env"].unwrapped.agent_dir
    agent["agent_pos"] = agent["env"].unwrapped.agent_pos
    obs_text = get_observation_text(agent, obs)
    return obs_text, obs, info

def verify(item: dict):
    old_pos = item["agent_pos"]
    old_dir = item["agent_dir"]
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
    item['prompt'] = game_prompt+obs_text
    item['score'] += reward
    item['is_end'] = terminated or truncated
    if "reached_goal" in info.get('reason', ''):
        item['is_end'] = True
    return item

# 执行动作，并更新代理状态
def step_agent(agent: dict, action_str: str):
    old_pos = agent["agent_pos"]
    old_dir = agent["agent_dir"]
    try:
        action = agent["reverse_action_map"][action_str.strip().lower()]
    except KeyError:
        action = 2  # 默认动作为 move_forward
    
    obs, reward, terminated, truncated, info = agent["env"].step(action)
    agent["current_step"] += 1
    
    agent["agent_dir"] = agent["env"].unwrapped.agent_dir
    agent["agent_pos"] = agent["env"].unwrapped.agent_pos
    agent["carrying"] = obs.get('carrying')
    
    if action_str == "move_forward" and np.array_equal(old_pos, agent["agent_pos"]):
        print(f"Move blocked at {old_pos} facing {old_dir}")
    
    obs_text = get_observation_text(agent, obs)
    return obs_text, reward, terminated or truncated, obs, info

def call_llm(conversation, max_retry=3) -> str:
    # 需要自行实现调用LLM的逻辑
    # from call_model import get_response
    # return get_response(conversation, 'gpt-4-turbo-128k')
    pass

def generate(seed):
    env_results = {
        'avg_steps': 0,
        'avg_reward': 0,
        'invalid_actions': 0,
        'conversations': []
    }
    max_steps=100
    env_name = random.sample(env_names,1)[0]
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
    item['prompt'] = game_prompt+obs_text
    return item


def evaluate_llm(
    env_names=["MiniGrid-Empty-5x5-v0", "MiniGrid-DoorKey-8x8-v0"],
    seeds=[42, 123, 999],
    max_steps=100
):
    """多环境多种子评估LLM性能（函数式实现）"""
    results = {}
    
    for env_name in env_names:
        env_results = {
            'avg_steps': 0,
            'avg_reward': 0,
            'invalid_actions': 0,
            'conversations': []
        }
        total_episodes = len(seeds)
        
        print(f"\n=== Evaluating on {env_name} ===")
        
        for seed in seeds:
            item=generate(seed)
            import ipdb
            ipdb.set_trace()
            # conversations = [
            #     {"role": "system", "content": agent["system_prompt"]},
            #     {"role": "user", "content": get_step_prompt(obs_text)}
            # ]
        
            while not item['is_end']:
                # 这里默认动作为 move_forward；实际中可以调用 call_llm 获得LLM的决策
                item['action'] = 'move_forward'
                # obs_text, reward, done, obs, info = step_agent(agent, action)
                item = verify(item)
                
                # conversations.append({"role": "assistant", "content": action})
                # conversations.append({
                #     "role": "user", 
                #     "content": get_step_prompt(obs_text)
                # })

                # 若达到目标则提前终止
                # if "reached_goal" in info.get('reason', ''):
                #     break
            
            env_results['avg_steps'] += item["current_step"]
            env_results['avg_reward'] += item['score']
            
            print(f"Seed {seed}: Steps: {item['current_step']} | Reward: {item['score']:.2f}")
        
        env_results['avg_steps'] /= total_episodes
        env_results['avg_reward'] /= total_episodes
        env_results['conversations'].append(conversations)
        results[env_name] = env_results
    
    print("\n=== Final Evaluation Report ===")
    for env_name, metrics in results.items():
        print(f"\nEnvironment: {env_name}")
        print(f"Average Steps: {metrics['avg_steps']:.1f}")
        print(f"Average Reward: {metrics['avg_reward']:.2f}")
        print(f"Invalid Actions per Episode: {metrics['invalid_actions']/len(seeds):.1f}")
    
    return results

# --- 定义请求和响应数据模型 ---

class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    board: list
    score: int
    is_end: bool
    action: str
    response: list
    prompt: str
    epoch: int
# --- API 接口 ---

# 生成初始游戏状态
@app.post("/print_board", response_model=BoardRequest)
def api_print_board(request: GameState):
    state = request.dict()
    board_output = print_board(state)
    return {"board": board_output}


# 生成初始游戏状态
@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    game_state = generate(request.seed)
    return game_state

# 根据动作更新游戏状态
@app.post("/verify", response_model=GameState)
def api_verify(request: GameState):
    # 从请求中获取游戏状态，并设置新的动作
    state = request.dict()
    state['answer'] = [tuple(coord) for coord in state['agent_pos']]
    updated_state = verify(state)
    return updated_state
    
if __name__ == "__main__":
    # 运行评估流程
    evaluation_results = evaluate_llm(
        env_names=[
            "MiniGrid-Empty-5x5-v0",
            "MiniGrid-DoorKey-6x6-v0",
            "MiniGrid-LavaGapS5-v0"
        ],
        seeds=[42, 123, 456, 789],  # 4个随机种子
        max_steps=100
    )
