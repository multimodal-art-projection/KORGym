#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import base64
import os
import json

# 如果你的服务没有跑在本机或端口不同，请修改下面的 URL
BASE_URL = "http://localhost:8775"

def save_image_from_base64(b64str: str, filename: str):
    """把 API 返回的 base64 编码图片保存到文件"""
    data = base64.b64decode(b64str)
    with open(filename, "wb") as f:
        f.write(data)
    print(f"✅ Image saved to {filename}")

def call_generate(seed: int) -> dict:
    """调用 /generate 接口，返回完整的 game state"""
    resp = requests.post(f"{BASE_URL}/generate", json={"seed": seed})
    resp.raise_for_status()
    state = resp.json()
    return state

def call_print_board(state: dict) -> str:
    """调用 /print_board 接口，返回文字版 board prompt"""
    resp = requests.post(f"{BASE_URL}/print_board", json=state)
    resp.raise_for_status()
    return resp.json()["board"]

def call_verify(state: dict, action: list) -> dict:
    """调用 /verify 接口，传入 action 列表，返回更新后的 state"""
    payload = state.copy()
    payload["action"] = str(action)
    resp = requests.post(f"{BASE_URL}/verify", json=payload)
    resp.raise_for_status()
    return resp.json()

def main():
    seed = 42
    print("1️⃣ 生成游戏...")
    state = call_generate(seed)

    # 保存并展示图片（可手动查看）
    os.makedirs("output", exist_ok=True)
    img_path = os.path.join("output", f"board_{seed}.png")
    save_image_from_base64(state["base64_image"], img_path)

    print("\n2️⃣ 获取文字版提示：")
    board_prompt = call_print_board(state)
    print(board_prompt)

    print("\n3️⃣ 验证正确答案：")
    correct = state["col_sums"]
    result = call_verify(state, correct)
    print(f"  action = {correct}\n  score = {result['score']}  (期望 1)")

    print("\n4️⃣ 验证错误答案：")
    wrong = [0] * len(correct)
    result2 = call_verify(state, wrong)
    print(f"  action = {wrong}\n  score = {result2['score']}  (期望 0)")

if __name__ == "__main__":
    main()
