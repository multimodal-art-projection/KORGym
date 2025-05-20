# eval/utils.py
#Standard libraries
import argparse
import logging
import os
import sys


# 配置 logging，设置日志级别和输出格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def check_directory(path, mode):
    """
    Check whether a directory exists and optionally create it.

    Args:
        path (str): Path to the directory.
        mode (str): "i" for input (just check existence), "o" for output (create if not exists).

    Returns:
        bool: True if the directory exists or is successfully created (in output mode), False otherwise.
    """
    if not os.path.isdir(path) and mode == "o":
        os.makedirs(path)
        return True
    elif not os.path.isdir(path) and mode == "i":
        return False
    else:
        return True
    

def check_file(path):
    """
    Check whether a given file exists. If the path is a directory or doesn't exist,
    log an error and terminate the program.

    Args:
        path (str): Path to the input file.

    Side Effects:
        - Logs error if file is missing or if the path is a directory.
        - Terminates the program using sys.exit(1) in case of invalid input.
    """
    if os.path.isdir(path):
        logging.error(f"Please add the -r argument to the command")
        sys.exit(1)
    elif os.path.exists(path):
        logging.info(f"Input file: {path}")
    else:
        logging.error(f"Please enter the correct file address")
        sys.exit(1)


def parse_init():
    """
    Parse command-line arguments for the evaluation script, set up logging, and
    check whether the output directory exists (creating it if necessary).

    Returns:
        argparse.Namespace: Parsed command-line arguments, including:
            - output (str): Output directory for saving evaluation results.
            - model (str): Name of the model to evaluate.
            - address (str): API address of the deployed model.
            - key (str): API key for authentication.
            - game (str): The name of the game to be evaluated.
            - level (int): Difficulty level of the game.
            - url (str): URL for the game environment server.
    """
    parser = argparse.ArgumentParser(description="Data creation utility")

    # 添加命令行参数
    parser.add_argument('-o', '--output', type=str, default="eval/result_data", help='评估结束后的数据文件输出的地址')
    parser.add_argument("-m", "--model", type=str, required=True, default="Qwen2.5-1.5B-Instruct", help="测试的模型名字，需要和设置的一致")
    parser.add_argument("-a", "--address", type=str, required=True, default="http://localhost:9002/v1", help="部署的大模型的地址")
    parser.add_argument("-k", "--key", type=str, required=True, default="EMPTY", help="API的key")
    parser.add_argument("-g", "--game", type=str, required=True, default="EMPTY", help="待测试的游戏")
    parser.add_argument("-l", "--level", type=int, required=True, default="EMPTY", help="保留字段，待测试的游戏难度")
    parser.add_argument("-u", "--url", type=str, required=True, default="http://localhost:8775", help="环境交互的url")
    # 解析命令行参数
    args = parser.parse_args()

    if check_directory(args.output, "o"):
        logging.info(f"Output directory: {os.path.abspath(args.output)}")

    return args