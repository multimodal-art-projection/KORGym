import requests
import json

# 定义服务地址（假设服务运行在本地 8775 端口）
base_url = "http://localhost:8775"

# 1. 调用 /generate 接口生成游戏状态
generate_url = f"{base_url}/generate"
generate_payload = {"seed": 42}

response = requests.post(generate_url, json=generate_payload)
if response.status_code == 200:
    game_state = response.json()
    print("Generate 接口返回结果:")
    print(json.dumps(game_state, indent=4, ensure_ascii=False))
else:
    print("Generate 接口调用失败，状态码：", response.status_code)
    exit(1)

# 2. 调用 /print_board 接口获取当前局面文本描述
print_board_url = f"{base_url}/print_board"
response = requests.post(print_board_url, json=game_state)
if response.status_code == 200:
    board_info = response.json().get("board", "")
    print("\nPrint_board 接口返回结果:")
    print(board_info)
else:
    print("Print_board 接口调用失败，状态码：", response.status_code)
    exit(1)

# 3. 调用 /verify 接口进行验证（假设用户猜测最终位置为 "(0, 0)"）
verify_url = f"{base_url}/verify"
game_state["action"] = "(7, 3)"
response = requests.post(verify_url, json=game_state)
if response.status_code == 200:
    updated_state = response.json()
    print("\nVerify 接口返回结果:")
    print(json.dumps(updated_state, indent=4, ensure_ascii=False))
else:
    print("Verify 接口调用失败，状态码：", response.status_code)
    exit(1)
