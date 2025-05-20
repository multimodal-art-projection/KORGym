# eval/utils.py
# python自带的库
import argparse
import logging
import os
import sys


# 配置 logging，设置日志级别和输出格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def check_directory(path, mode):
    """
    检查输入、输出目录是否存在
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
    检查输入文件是否存在
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
    定义并解析eval代码的命令行参数，配置日志记录，并检查输入的数据文件目录和输出的目录是否存在。
    """
    parser = argparse.ArgumentParser(description="Data creation utility")

    # 添加命令行参数
    parser.add_argument('-o', '--output', type=str, default="eval/result_data", help='评估结束后的数据文件输出的地址')
    parser.add_argument("-m", "--model", type=str, required=True, default="Qwen2.5-1.5B-Instruct", help="测试的模型名字，需要和设置的一致")
    parser.add_argument("-a", "--address", type=str, required=True, default="http://localhost:9002/v1", help="部署的大模型的地址")
    parser.add_argument("-k", "--key", type=str, required=True, default="EMPTY", help="API的key")

    # 解析命令行参数
    args = parser.parse_args()

    # if args.recursive:
    #     if check_directory(args.input, "i"):
    #         logging.info(f"Input directory: {os.path.abspath(args.input)}")
    #     else:
    #         logging.error(f"Input directory is not exists: {os.path.abspath(args.input)}")
    #         sys.exit(1)
    # else:
    #     check_file(args.input)

    if check_directory(args.output, "o"):
        logging.info(f"Output directory: {os.path.abspath(args.output)}")

    return args