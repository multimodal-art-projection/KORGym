import requests
import json

# 服务端 URL 基地址
BASE_URL = "http://0.0.0.0:8775"

def test_generate(seed):
    url = f"{BASE_URL}/generate"
    payload = {"seed": seed}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        game_state = response.json()
        print("Generate 接口返回的状态：")
        print(json.dumps(game_state, indent=4, ensure_ascii=False))
        return game_state
    else:
        print("Generate 接口调用失败", response.text)
        return None

def test_print_board(game_state):
    url = f"{BASE_URL}/print_board"
    response = requests.post(url, json=game_state)
    if response.status_code == 200:
        board_info = response.json()
        print("\n当前棋盘：")
        print(board_info["board"])
    else:
        print("Print_board 接口调用失败", response.text)

def test_verify(game_state, action):
    # 将测试动作赋值到 game_state 中
    game_state["action"] = action
    url = f"{BASE_URL}/verify"
    response = requests.post(url, json=game_state)
    if response.status_code == 200:
        new_state = response.json()
        print("\nVerify 接口返回的状态：")
        print(json.dumps(new_state, indent=4, ensure_ascii=False))
        return new_state
    else:
        print("Verify 接口调用失败", response.text)
        return None

if __name__ == "__main__":
    # 使用指定种子生成初始状态
    seed = 42
    state = test_generate(seed)
    if state:
        # 打印当前生成的棋盘
        test_print_board(state)
        
        # 示例：这里给出空操作序列（通常不正确），实际测试时应根据提示给出合适的操作序列
        sample_actions = "[['line',4],['line',5],['diagonal_black',3],['diagonal_black',4],['diagonal_black',7],['diagonal_black',8]]"
        test_verify(state, sample_actions)
