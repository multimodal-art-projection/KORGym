import requests
import json

# 基础 URL
base_url = "http://localhost:8775"

# 1. 测试 /generate 接口：生成初始游戏状态
generate_payload = {
    "seed": 123,
    "replacement_ratio": 0.5
}
generate_response = requests.post(f"{base_url}/generate", json=generate_payload)
game_state = generate_response.json()
print("【/generate 接口返回】")
print(json.dumps(game_state, indent=2, ensure_ascii=False))

# 2. 测试 /print_board 接口：根据当前游戏状态生成显示文本
print_board_response = requests.post(f"{base_url}/print_board", json=game_state)
board_output = print_board_response.json()
print("\n【/print_board 接口返回】")
print(json.dumps(board_output, indent=2, ensure_ascii=False))

# 3. 测试 /verify 接口：提交一次猜测更新游戏状态
# 假设我们构造一个示例猜测，格式要求类似 "emoji=letter,emoji=letter"
# 注意：这里的猜测仅为示例，实际结果依赖于 /generate 返回的 answer
game_state["action"] = "😀=a,😂=b"
verify_response = requests.post(f"{base_url}/verify", json=game_state)
updated_game_state = verify_response.json()
print("\n【/verify 接口返回】")
print(json.dumps(updated_game_state, indent=2, ensure_ascii=False))


game_state["action"] = "🤑= g,👻= d,🤔= h,🤭= c,😇= l,🦁= a,🐦= n,🤩= v,🦉= r,🐭= b"
verify_response = requests.post(f"{base_url}/verify", json=game_state)
updated_game_state = verify_response.json()
print("\n【/verify 接口返回】")
print(json.dumps(updated_game_state, indent=2, ensure_ascii=False))
