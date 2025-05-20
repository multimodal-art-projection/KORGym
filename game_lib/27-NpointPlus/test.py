import requests

def main():
    # 假设 FastAPI 服务地址
    base_url = "http://0.0.0.0:8775"

    # 1. 调用 /generate 接口生成初始游戏状态
    generate_url = f"{base_url}/generate"
    # 传入种子和对手策略（例如使用“正常人”策略）
    generate_payload = {
        "seed": 42,
    }
    resp = requests.post(generate_url, json=generate_payload)
    if resp.status_code != 200:
        print("调用 /generate 接口失败！")
        return
    state = resp.json()
    print("初始游戏状态：")
    print(state["prompt"])
    
    # 2. 模拟一次玩家操作
    # 假设玩家当前操作为 "hit"，注意接口要求直接输入 "hit" 或 "stand"
    while(1):
        state["action"] = input("请输入你的操作：")
        verify_url = f"{base_url}/verify"
        resp = requests.post(verify_url, json=state)
        if resp.status_code != 200:
            print("调用 /verify 接口失败！")
            return
        state = resp.json()
        print("\n执行操作 'hit' 后的状态：")
        print(state["prompt"])

    # # 3. 模拟一次玩家操作（例如接下来选择 "stand"）
    # state["action"] = "stand"
    # resp = requests.post(verify_url, json=state)
    # if resp.status_code != 200:
    #     print("调用 /verify 接口失败！")
    #     return
    # state = resp.json()
    # print("\n执行操作 'stand' 后的状态：")
    # print(state["prompt"])

    # # 4. 调用 /print_board 接口，获取最终（或当前）局面展示
    # print_board_url = f"{base_url}/print_board"
    # resp = requests.post(print_board_url, json=state)
    # if resp.status_code != 200:
    #     print("调用 /print_board 接口失败！")
    #     return
    # board_output = resp.json()["board"]
    # print("\n最终局面展示：")
    # print(board_output)

if __name__ == "__main__":
    main()
