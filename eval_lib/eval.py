# eval/eval.py

#Standard libraries
import asyncio
import os
import logging
import re
import requests
import json

#Commonly used open-source libraries
import pandas as pd
from tqdm import tqdm

#Project-specific libraries 
from .utils import parse_init
from .eval_lib import predict, save_process

#Configure logging: set the log level and output format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#Judge game playing epoch
game_dict = {
    '1-DateCount':'single','2-GuessWord':'single','3-2048':'multiple','4-SudoKu':'single',
    '5-light_out_game':'single','8-word_puzzle':'single','9-Jigsaw_puzzle':'single',
    '10-minigrid':'multiple','11-maze':'single','12-sokoban':'single','13-play_lines':'single',
    '15-emoji_connect':'single','16-jiafa':'single','17-fill_game':'single','18-alien':'single',
    '19-party_time':'single','20-city_path':'single','21-Anagramania':'single',
    '22-alphabetical_sorting':'single','23-puzzlegame':'single','24-snake':'multiple',
    '25-Tetris':'multiple','26-TrustRovolution':'multiple','27-NpointPlus':'multiple',
    '28-word_encryption':'single','29-Construction_Company':'single','30-Tower_of_Hanoi':'multiple',
    '31-ball_arrange':'multiple','32-numeral_bricks':'single','33-wordle':'multiple',
    '34-one_touch_drawing':'single','35-pipe_game':'single','36-CryptoWord':'multiple',
    '37-SpiderSolitaire':'multiple','38-minesweeper':'multiple','39-Nullify':'multiple',
    '40-CircleTheCat-Text':'multiple','41-PVZ':'multiple','42-diagram_coloring':'single',
    '43-CircleTheCat-Multimodal':'multiple','44-city':'single','45-free_the_key':'multiple',
    '48-map_position_simulation_text':'single','49_map_position_simulation_multimodal':'single',
    '50-SudoKu_MultiModal':'single','51-ball_arrange_multimodal':'multiple',
    '46-wordle_multimodal':'multiple','14-Arrow-pathway':'single','47-jiafa_multimodal':'single','6-LongCat':'single',
    '7-black_white_copy':'single'
}

def normalize_response(response: str) -> str:
    """
    Cleans up the response string by removing LaTeX formatting and special characters
    that may interfere with answer extraction.

    Args:
        response (str): The raw output string from the model.

    Returns:
        str: A simplified and normalized version of the response.
    """
    return (
        response.replace("**", "")
                .replace("$\\boxed{", "")
                .replace("}$", "")
                .replace("\\$", "")
                .replace("$\\text{", "")
                .replace("$", "")
                .replace("\\mathrm{", "")
                .replace("\\{", "")
                .replace("\\text", "")
                .replace("\\(", "")
                .replace("\\mathbf{", "")
                .replace("{", "")
                .replace("\\boxed", "")
    )

def get_prompt0_response(ori_answer):
    """
    Extracts the final answer from a model's response by locating the last occurrence
    of the word 'Answer' and capturing the corresponding value.

    Args:
        ori_answer (str): The original model response string.

    Returns:
        str: The extracted answer string (or empty string if not found).
    """
    if ori_answer is None:
        return ""
    gen = normalize_response(ori_answer)
    pos = gen.lower().rfind("answer")
    if pos == -1:
        return ""
    gen = gen[pos:]
    pattern = r"(?i)Answer\s*:\s*(.*)"
    match = re.findall(pattern, gen)
    return match[-1] if match else ""

def generate(url, seed, level=4):
    """
    Sends a POST request to the game server to generate a new game instance.

    Args:
        url (str): The base URL of the game server.
        seed (int): Random seed to control instance generation.
        level (int): The game difficulty level.

    Returns:
        dict: The generated game instance.
    """
    return requests.post(f"{url}/generate", json={"seed": seed}).json()

def print_board(url, item):
    """
    Requests the visual or textual representation of the current game state.

    Args:
        url (str): The game server URL.
        item (dict): Game state input.

    Returns:
        str: The board state in string format (text-based rendering).
    """
    return requests.post(f"{url}/print_board", json=item).json()['board']

def verify(url, item):
    """
    Verifies the correctness of a model's action using the server's verification API.
    If verification fails, sets the score to 0.

    Args:
        url (str): The game server URL.
        item (dict): The game item containing model's response/action.

    Returns:
        dict: The updated item with verification result and score.
    """
    try:
        resp = requests.post(f"{url}/verify", json=item, timeout=30)
        resp.raise_for_status()
        item.update(resp.json())
    except Exception:
        item['score'] = 0
    return item

async def eval_single_file(output_dir, model_name, address, key, sem, game_name, level, url):
    """
    Evaluates a single-step game (non-interactive) by generating prompts,
    collecting model responses, verifying actions, and saving results.

    Args:
        output_dir (str): Directory to store results.
        model_name (str): The model name to evaluate.
        address (str): The API endpoint.
        key (str): The API key for authentication.
        sem (asyncio.Semaphore): Concurrency limiter.
        game_name (str): The name of the game.
        level (int): The difficulty level.
        url (str): The game server URL.
    """
    checkpoint_dir = os.path.join(output_dir, game_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt = os.path.join(checkpoint_dir, f"{model_name}_{game_name}_level{level}_checkpoint.jsonl")

    processed = []
    seen = set()
    if os.path.exists(ckpt):
        with open(ckpt, 'r', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                processed.append(d)
                seen.add(d['seed'])

    to_run = []
    for seed in range(50):
        if seed in seen:
            continue
        item = generate(url, seed, level)
        item['seed'] = seed
        item['response'] = []
        item['prompt'] = print_board(url, item)
        to_run.append(item)

    if to_run:
        results = await predict(to_run, sem, model_name, address, key)
        for item in results:
            item['action'] = get_prompt0_response(item['response'][-1])
            item = verify(url, item)
            processed.append(item)
            with open(ckpt, 'a', encoding='utf-8') as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    file_name = f"{model_name}_{game_name}_level{level}"
    final_dir = os.path.join(output_dir, game_name)
    save_process(processed, final_dir, file_name)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    logging.info(f"Complete the evaluation of the file: {file_name}")

async def eval_file(output_dir, model_name, address, key, sem, game_name, level, url):
    """
    Evaluates a multi-turn game (interactive) by performing up to 100 rounds of 
    prediction and environment updates, using checkpointing for intermediate saving.

    Args:
        output_dir (str): Directory to store results.
        model_name (str): The model name to evaluate.
        address (str): The API endpoint.
        key (str): The API key for authentication.
        sem (asyncio.Semaphore): Concurrency limiter.
        game_name (str): The name of the game.
        level (int): The difficulty level.
        url (str): The game server URL.
    """
    checkpoint_dir = os.path.join(output_dir, game_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt = os.path.join(checkpoint_dir, f"{model_name}_{game_name}_level{level}_checkpoint.json")

    if os.path.exists(ckpt):
        with open(ckpt, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        count = state['count']+1
        item_list = state['item_list']
        final_list = state['final_list']
        print(f"loading checkpoint:{count}")
    else:
        count = 1
        final_list = []
        item_list = []
        for seed in range(20):
            item = generate(url, seed, level)
            item['seed'] = seed
            item['response'] = []
            item['prompt'] = print_board(url, item)
            item_list.append(item)

    while count <= 100:
        tqdm.write(f'round {count}')
        item_list = await predict(item_list, sem, model_name, address, key)
        i = len(item_list) - 1
        while i >= 0:
            itm = item_list[i]
            itm['action'] = get_prompt0_response(itm['response'][-1])
            itm = verify(url, itm)
            itm['prompt'] = print_board(url, itm)
            if itm.get('is_end'):
                final_list.append(item_list.pop(i))
            i -= 1

        with open(ckpt, 'w', encoding='utf-8') as f:
            json.dump({'count': count, 'item_list': item_list, 'final_list': final_list}, f, ensure_ascii=False)

        if not item_list:
            break
        count += 1

    final_list.extend(item_list)
    file_name = f"{model_name}_{game_name}_level{level}"
    final_dir = os.path.join(output_dir, game_name)
    save_process(final_list, final_dir, file_name)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    logging.info(f"Complete the evaluation of the file: {file_name}")

async def main():
    """
    The main entry point for evaluation. Parses command-line arguments,
    determines whether the game is single-turn or multi-turn, and dispatches
    the appropriate evaluation function.
    """
    sem = asyncio.Semaphore(10)
    args = parse_init()
    if game_dict.get(args.game) == 'single':
        await eval_single_file(args.output, args.model, args.address, args.key, sem, args.game, args.level, args.url)
    else:
        await eval_file(args.output, args.model, args.address, args.key, sem, args.game, args.level, args.url)

if __name__ == "__main__":
    asyncio.run(main())
