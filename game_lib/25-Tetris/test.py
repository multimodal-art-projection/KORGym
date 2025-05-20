import requests

def main():
    # 注意：请确保你的 FastAPI 服务器已在 http://127.0.0.1:8775 启动
    base_url = "http://127.0.0.1:8775"
    
    # 1. 生成初始游戏状态（调用 /generate 接口）
    try:
        seed = int(input("请输入随机种子（整数）："))
    except ValueError:
        print("种子必须为整数！")
        return
    
    generate_url = f"{base_url}/generate"
    response = requests.post(generate_url, json={"seed": seed})
    if response.status_code != 200:
        print("生成游戏状态失败，状态码：", response.status_code)
        return
    
    game_state = response.json()
    print("初始游戏状态已生成。")
    
    round_num = 1
    while True:
        print("\n==================")
        print(f"第 {round_num} 轮：")
        
        # 2. 调用 /print_board 接口，显示当前棋盘和砖块状态
        print_board_url = f"{base_url}/print_board"
        board_resp = requests.post(print_board_url, json=game_state)
        if board_resp.status_code != 200:
            print("打印棋盘失败，状态码：", board_resp.status_code)
            break
        
        board_output = board_resp.json()["board"]
        print("当前棋盘状态：")
        print(board_output)
        
        # 如果游戏已结束，则退出循环
        if game_state.get("is_end", False):
            print("游戏结束！")
            break

        # 3. 用户输入落点与旋转角度（例如： 3 90°）
        action = input("请输入落点和旋转角度（格式如 '3 90°'）：").strip()
        if not action:
            print("未输入有效操作，游戏结束。")
            break
        game_state["action"] = action
        
        # 4. 调用 /verify 接口，验证并更新游戏状态
        verify_url = f"{base_url}/verify"
        verify_resp = requests.post(verify_url, json=game_state)
        if verify_resp.status_code != 200:
            print("验证操作失败，状态码：", verify_resp.status_code)
            break
        game_state = verify_resp.json()
        
        # 显示更新后的分数信息
        print("更新后的分数：", game_state.get("score", 0))
        round_num += 1

    print("\n最终得分：", game_state.get("score", 0))

if __name__ == "__main__":
    main()
