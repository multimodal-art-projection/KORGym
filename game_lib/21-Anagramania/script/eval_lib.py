# eval/eval_lib.py
# python自带的库
import asyncio
import logging
import time
import os

# 常用的开源库
from tqdm import tqdm
from openai import OpenAI
import torch
import numpy as np
import pandas as pd

# 项目的库
# from .metric_lib import (metric_bbh, metric_math500, metric_mbpp, metric_mmlu, metric_needle, metric_other)

# 配置 logging，设置日志级别和输出格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def llama_process_sync(prompt, model_name, client):
    """
    同步版本的请求函数
    """
    try:
        # 修改后的 prompt 需要是一个消息对象列表
        chat_response = client.chat.completions.create(
            model=model_name, 
            messages=[{"role": "user", "content": prompt}]
        )
        return chat_response.choices[-1].message.content
    except Exception as e:
        logging.error(f"An unexpected error occurred in requesting a model: {e}")
        return ""  # 其他异常时返回空字符串



async def llama_process(prompt, model_name, address, key):
    """
    异步版本的请求函数，使用运行中的事件循环
    """
    client = OpenAI(api_key=key, base_url=address)
    loop = asyncio.get_running_loop()  # 使用 get_running_loop 确保是在同一个事件循环中
    response = await loop.run_in_executor(None, llama_process_sync, prompt, model_name, client)
    return response


async def predict(item_list, sem, model_name, address, key):
    """
    用于请求单个模型并获取到response的代码
    """
    response_list = []

    with tqdm(total=len(item_list), desc="Generating predictions......") as pbar:

        async def run(prompt):
            async with sem:
                try:
                    # 设置超时
                    response = await llama_process(prompt, model_name, address, key)
                except asyncio.TimeoutError:
                    logging.error(f"Predict timeout for prompt: {prompt}")
                    response = ""  # 超时后返回默认值 None
                pbar.update(1)
                return response

        tasks = [run(item["prompt"]) for item in item_list]
        # 使用 asyncio.gather 保持顺序
        response_list = await asyncio.gather(*tasks)

    torch.cuda.empty_cache()
    for i in range(len(item_list)):
        item_list[i]["response"].append(response_list[i])
    return item_list
    
    
# async def metric(item_list, sem):
#     """
#     用于计算回答的正确性，获取分数
#     """
#     score_list = []
#     with tqdm(total=len(item_list), desc="Calculating metrics") as pbar:

#         async def run(item):
#             dataset_type = item["dataset_type"]
#             async with sem:
#                 try:
#                     # 设置超时
#                     question = item["ori_question"]
#                     answer = item["answer"]
#                     response = item["predict0"]
#                     if "bbh" in dataset_type:
#                         score = await metric_bbh.judge_correctness(question, answer, response)
#                     elif "math" in dataset_type:
#                         score = await metric_math500.judge_correctness(question, answer, response)
#                     elif "mbpp" in dataset_type:
#                         test = item["test_list"]
#                         score = await metric_mbpp.multichoice_openai(response, test)
#                     elif "mmlu" in dataset_type:
#                         score = await metric_mmlu.judge_correctness(question, answer, response)
#                     elif "needle" in dataset_type:
#                         score = await metric_needle.judge_correctness(question, answer, response)
#                     else:
#                         score = await metric_other.judge_correctness(question, answer, response)
#                 except asyncio.TimeoutError:
#                     logging.error(
#                         f"Metric calculation timeout for question: {question}"
#                     )
#                     score = 0  # 超时后返回默认分数 0
#                 pbar.update(1)
#                 return score

#         tasks = [run(item) for item in item_list]
#         score_list = await asyncio.gather(*tasks)
#     for i in range(len(item_list)):
#         item_list[i]["score"] = score_list[i]
#     return item_list


def save_process(item_list, output_dir, file_name):
    """
    将评估结果保存到相应的文件目录下
    """
    score_list = [item["score"] for item in item_list]
    avg_score = np.mean(score_list)
    logging.info(f"Avg score is {avg_score} in {file_name}")
    df = pd.DataFrame(item_list)
    output_file_path = os.path.join(output_dir, file_name)
    df.to_json(output_file_path, orient="records", lines=True, force_ascii=False)
    logging.info(f"Data has been saved to {file_name} in {output_dir}")
    # if avg_score < 0:
    #     logging.info(f"{file_name} has been evaluated")
    #     return
    with open(os.path.join(output_dir, "score.txt"), "a") as f:
        f.write(f"{file_name}: {avg_score}\n")