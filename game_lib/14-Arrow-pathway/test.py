import requests
import json

# API 服务地址
BASE_URL = "http://localhost:8775"

def test_generate(seed):
    url = f"{BASE_URL}/generate"
    payload = {"seed": seed}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("生成状态：")
        state = response.json()
        print(json.dumps(state, indent=2, ensure_ascii=False))
        return state
    else:
        print("生成失败，状态码：", response.status_code)
        return None

def test_print_board(game_state):
    url = f"{BASE_URL}/print_board"
    response = requests.post(url, json=game_state)
    if response.status_code == 200:
        board = response.json().get("board", "")
        print("迷宫文本：")
        print(board)
    else:
        print("打印迷宫失败，状态码：", response.status_code)

def test_verify(game_state):
    # 这里使用生成时的 device_actions 作为正确的动作序列进行验证
    game_state["action"] = "[['D',0,3],['L',6,3],['U',6,0],['R',2,0],['D',2,6]]"
    url = f"{BASE_URL}/verify"
    response = requests.post(url, json=game_state)
    if response.status_code == 200:
        updated_state = response.json()
        print("验证结果：")
        print(json.dumps(updated_state, indent=2, ensure_ascii=False))
    else:
        print("验证失败，状态码：", response.status_code)

if __name__ == "__main__":
    seed = 123  # 可根据需要修改种子值
    # 1. 调用 /generate 接口生成初始游戏状态
    state = test_generate(seed)
    if state is None:
        exit(1)

    # 2. 调用 /print_board 接口打印迷宫
    test_print_board(state)

    # 3. 调用 /verify 接口验证动作序列（此处使用生成的 device_actions 作为测试动作）
    test_verify(state)
