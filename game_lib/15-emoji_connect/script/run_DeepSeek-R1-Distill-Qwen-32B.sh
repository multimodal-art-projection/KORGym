#!/bin/bash
source /map-vepfs/miniconda3/bin/activate
conda activate jiajun

cd /map-vepfs/jiajun/ReasoningGym/experiments/15-emoji_connect/script
nohup bash DeepSeek-R1-Distill-Qwen-32B.sh > DeepSeek-R1-Distill-Qwen-32B.out &

sleep 7m

cd /map-vepfs/jiajun/ReasoningGym/experiments
python -m 15-emoji_connect.eval -o 15-emoji_connect/result -m DeepSeek-R1-Distill-Qwen-32B -a http://localhost:9003/v1 -k None