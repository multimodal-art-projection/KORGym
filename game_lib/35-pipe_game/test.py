import requests
import json

# 设定服务 URL，根据实际情况修改 host 和 port
BASE_URL = "http://localhost:8775"

def test_generate(seed):
    """调用 /generate 接口，生成谜题状态"""
    url = f"{BASE_URL}/generate"
    payload = {"seed": seed}
    response = requests.post(url, json=payload)
    if response.ok:
        game_state = response.json()
        print("【生成谜题状态】")
        print(json.dumps(game_state, indent=2, ensure_ascii=False))
        return game_state
    else:
        print("生成谜题失败:", response.text)
        return None

def test_print_board(game_state):
    """调用 /print_board 接口，打印题面展示"""
    url = f"{BASE_URL}/print_board"
    response = requests.post(url, json=game_state)
    if response.ok:
        board_data = response.json()
        print("\n【棋盘展示】")
        print(board_data.get("board"))
    else:
        print("获取棋盘展示失败:", response.text)

def test_verify(game_state):
    """
    模拟用户提交答案，然后调用 /verify 接口进行验证。
    此处使用生成的正确答案作为 action。
    """
    # 此处直接使用服务生成的答案，通常用户在此处会提交经过旋转操作的答案
    game_state["action"] = '[[2,0,0,2,2],[0,0,0,0,0],[0,0,0,0,0],[0,1,1,0,0],[0,0,0,0,0]]'

    url = f"{BASE_URL}/verify"
    response = requests.post(url, json=game_state)
    if response.ok:
        verified_state = response.json()
        print("\n【验证反馈】")
        print(json.dumps(verified_state, indent=2, ensure_ascii=False))
    else:
        print("验证答案失败:", response.text)

def main():
    seed = 0  # 可修改为其它随机种子
    game_state = test_generate(seed)
    if game_state is None:
        return

    test_print_board(game_state)
    test_verify(game_state)

if __name__ == '__main__':
    main()
