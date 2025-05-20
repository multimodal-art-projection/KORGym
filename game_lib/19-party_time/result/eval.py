# eval/eval.py
# python自带的库
import asyncio
import os
import logging
import re
import random
# 常用的开源库
import pandas as pd
from tqdm import tqdm
import tiktoken

# 项目的库
from .utils import parse_init
from .eval_lib import predict, save_process
from .game_lib import generate,verify,print_board
from .prompts import game_prompt
# 配置 logging，设置日志级别和输出格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def normalize_response(response: str) -> str:
    """
    通过删除可能阻止匹配的markdown和LaTeX格式来规范化响应。
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
    获取prompt0的response
    """
    generated_answer = normalize_response(ori_answer)
    pos = generated_answer.lower().rfind("answer")
    if pos == -1:
        return ""
    generated_answer = generated_answer[pos:]
    ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*(.*)"
    match_for_generated_answer = re.findall(ANSWER_PATTERN_MULTICHOICE, generated_answer)
    extracted_answer = match_for_generated_answer[-1] if match_for_generated_answer else ""
    return extracted_answer


def get_token_length(text, model="gpt-3.5-turbo"):
    encoder = tiktoken.encoding_for_model(model)
    tokens = encoder.encode(text)
    return len(tokens)

async def eval_single_file(output_dir, model_name, address, key, sem, k):
    """
    调用单个文件的prompt生成对应的response
    """
    item_list = []
    for i in range(50):
        item_list.append(generate(i,k))
        item_list[i]['response'] = []
        item_list[i]["prompt"] = game_prompt.format(board=print_board(item_list[i]))
    
    item_list = await predict(item_list, sem, model_name, address, key)
        

    for i in range(len(item_list)):
        item_list[i]['action'] = get_prompt0_response(item_list[i]['response'][-1])
        item_list[i] = verify(item_list[i])

    file_name = f'{model_name}_partytime_{k}person_50question'
    save_process(item_list, output_dir, file_name)
    logging.info(f"Complete the evaluation of the file: {file_name}")

async def eval_file(output_dir, model_name, address, key, sem):
    """
    调用单个文件的prompt生成对应的response
    """
    item_list = []
    for i in range(50):
        item_list.append(generate(i))
        item_list[i]['response'] = []
        item_list[i]["prompt"] = game_prompt.format(board=print_board(item_list[i]))
    
    count = 1
    final_list = []
    while count <= 100:
        print(f'round {count}')
        item_list = await predict(item_list, sem, model_name, address, key)
        
        # 使用倒序遍历避免索引越界
        i = len(item_list) - 1
        while i >= 0:
            action = get_prompt0_response(item_list[i]['response'][-1])
            item_list[i]['action'] = action
            item_list[i] = verify(item_list[i])
            print(item_list[i]['is_end'])
            item_list[i]["prompt"] = game_prompt.format(board=print_board(item_list[i]))
            if item_list[i]['is_end']:  # 如果满足条件，删除并加入final_list
                final_list.append(item_list.pop(i))  # 从item_list中删除该项并添加到final_list
            i -= 1  # 递减索引，继续检查下一个元素
        if len(item_list)==0:
            break
        count += 1
    for item in item_list:
        final_list.append(item)
    file_name = f'{model_name}_lightout_game_50epoch'
    save_process(final_list, output_dir, file_name)
    logging.info(f"Complete the evaluation of the file: {file_name}")



async def main():
    """
    主代码块，进行数据的评估，包括调用模型以及对response进行评估
    """
    sem = asyncio.Semaphore(8)  # 将信号量放在这里
    args = parse_init()
    model_name = args.model
    address = args.address
    key = args.key
    await eval_single_file(args.output, model_name, address, key, sem, 100)


if __name__ == "__main__":
    asyncio.run(main())