#!/bin/bash
source /map-vepfs/miniconda3/bin/activate
conda activate jiajun

cd /map-vepfs/jiajun/ReasoningGym/experiments/15-emoji_connect/script
nohup bash Qwen2.5-7B-Instruct.sh > Qwen2.5-7B-Instruct.out &

sleep 2m

cd /map-vepfs/jiajun/ReasoningGym/experiments
python -m 15-emoji_connect.eval -o 15-emoji_connect/result -m Qwen2.5-7B-Instruct -a http://localhost:9003/v1 -k None