import requests

def main():
    base_url = "http://127.0.0.1:8775"
    
    # 1. 调用 /generate 接口生成游戏状态（种子可自行调整）
    seed = 123
    generate_url = f"{base_url}/generate"
    generate_payload = {"seed": seed}
    response = requests.post(generate_url, json=generate_payload)
    if response.status_code == 200:
        game_state = response.json()
        print("=== 生成的游戏状态 ===")
        print(game_state)
    else:
        print("调用 /generate 失败：", response.text)
        return
    
    # 2. 调用 /print_board 接口打印当前地图（同时包含游戏规则和任务提示）
    print_board_url = f"{base_url}/print_board"
    response = requests.post(print_board_url, json=game_state)
    if response.status_code == 200:
        board_output = response.json().get("board")
        print("\n=== 当前地图与提示信息 ===")
        print(board_output)
    else:
        print("调用 /print_board 失败：", response.text)
        return
    
    # 3. 调用 /verify 接口验证答案
    # 此处设置一个示例答案，格式要求为字符串形式的坐标，例如 "(2, 3)"
    # 注意：实际使用时应根据 /print_board 给出的地图和任务，计算出正确的最终坐标。
    game_state["answer"] = "(2, 3)"
    
    verify_url = f"{base_url}/verify"
    response = requests.post(verify_url, json=game_state)
    if response.status_code == 200:
        verified_state = response.json()
        print("\n=== 验证后的游戏状态 ===")
        print(verified_state)
    else:
        print("调用 /verify 失败：", response.text)

if __name__ == "__main__":
    main()
