import requests

# 定义 FastAPI 服务的基础 URL（请确保服务已启动）
BASE_URL = "http://localhost:8765"

def test_generate(seed: int):
    """
    测试 /generate 接口，根据给定 seed 生成初始游戏状态
    """
    url = f"{BASE_URL}/generate"
    payload = {"seed": seed}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("【/generate】返回结果：")
        print(response.json())
        return response.json()
    else:
        print(f"/generate 接口调用失败，状态码：{response.status_code}")
        return None

def test_print_board(game_state: dict):
    """
    测试 /print_board 接口，根据传入的 game_state 输出当前状态文本
    """
    url = f"{BASE_URL}/print_board"
    response = requests.post(url, json=game_state)
    if response.status_code == 200:
        print("【/print_board】返回结果：")
        print(response.json()['board'])
        return response.json()
    else:
        print(f"/print_board 接口调用失败，状态码：{response.status_code}")
        return None

def test_verify(game_state: dict, action: str):
    """
    测试 /verify 接口，根据 game_state 中的 action 更新游戏状态
    这里 action 需要按照要求填写，例如： "Answer: cheat" 或 "Answer: collaborate"
    """
    url = f"{BASE_URL}/verify"
    # 更新 game_state 中的 action 字段
    game_state["action"] = action
    response = requests.post(url, json=game_state)
    if response.status_code == 200:
        print("【/verify】返回结果：")
        print(response.json())
        return response.json()
    else:
        print(f"/verify 接口调用失败，状态码：{response.status_code}")
        return None

if __name__ == "__main__":
    # 测试示例：使用 seed=42 生成初始状态
    seed = 42
    game_state = test_generate(seed)
    if game_state:
        # 查看当前状态提示信息
        print("\n调用 /print_board 接口查看游戏状态：")
        board_output = test_print_board(game_state)
        
        # 模拟一次玩家操作，例如选择 'cheat'
        # 注意：这里的输入需要按照提示格式，例如 "Answer: cheat"
        player_action = "cheat"
        print(f"\n模拟玩家输入：{player_action}")
        game_state = test_verify(game_state, player_action)
        board_output = test_print_board(game_state)
        game_state = test_verify(game_state, player_action)
        board_output = test_print_board(game_state)
        game_state = test_verify(game_state, player_action)
        board_output = test_print_board(game_state)
        game_state = test_verify(game_state, player_action)
        board_output = test_print_board(game_state)
        game_state = test_verify(game_state, player_action)
        board_output = test_print_board(game_state)
        game_state = test_verify(game_state, player_action)
        board_output = test_print_board(game_state)
        game_state = test_verify(game_state, player_action)
        board_output = test_print_board(game_state)
        game_state = test_verify(game_state, player_action)
        board_output = test_print_board(game_state)
        game_state = test_verify(game_state, player_action)
        board_output = test_print_board(game_state)
        game_state = test_verify(game_state, player_action)
        board_output = test_print_board(game_state)
