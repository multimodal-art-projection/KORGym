#!/usr/bin/env python3
"""
Spider Solitaire API Test Client

This script interacts with the FastAPI spider solitaire service via HTTP requests.
It generates a new game, then enters a loop where it prints the board, accepts user input for actions,
submits the action to the verify endpoint, and prints the updated board until the game ends or the user quits.
"""
import requests
import argparse
import random
import sys


def print_board(state: dict, base_url: str) -> None:
    """
    Request and print the formatted board from the API.
    """
    response = requests.post(f"{base_url}/print_board", json=state)
    response.raise_for_status()
    data = response.json()
    print(data.get("board", ""))


def generate_game(seed: int, base_url: str) -> dict:
    """
    Generate a new game state using the API.
    """
    payload = {"seed": seed}
    response = requests.post(f"{base_url}/generate", json=payload)
    response.raise_for_status()
    return response.json()


def verify_action(state: dict, action: str, base_url: str) -> dict:
    """
    Send an action to the verify endpoint and return the new state.
    """
    payload = state.copy()
    payload["action"] = action
    response = requests.post(f"{base_url}/verify", json=payload)
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(description="Spider Solitaire API Test Client")
    parser.add_argument("--host", default="http://127.0.0.1", help="API host URL (with http://)")
    parser.add_argument("--port", default=8775, type=int, help="API port")
    parser.add_argument("--seed", type=int, default=0, help="Seed for game generation (0 for random)")
    args = parser.parse_args()

    base_url = f"{args.host}:{args.port}"
    seed = args.seed if args.seed != 0 else random.randint(1, 100000)
    print(f"Generating game with seed {seed}...")

    try:
        state = generate_game(seed, base_url)
    except Exception as e:
        print(f"Failed to generate game: {e}")
        sys.exit(1)

    while True:
        print_board(state, base_url)
        if state.get("is_end", False):
            print("Game has ended. Congratulations!")
            break
        action = input("Enter action [(A,4,B), hit, undo] or 'q' to quit: ").strip()
        if action.lower() in ("q", "quit"):
            print("Quitting test client.")
            break
        try:
            state = verify_action(state, action, base_url)
        except requests.HTTPError as http_err:
            print(f"API error: {http_err.response.text}")
        except Exception as err:
            print(f"Unexpected error: {err}")


if __name__ == "__main__":
    main()
