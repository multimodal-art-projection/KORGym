#!/bin/bash
source /map-vepfs/miniconda3/bin/activate
conda activate jiajun

cd /map-vepfs/jiajun/ReasoningGym/experiments/19-party_time/script
nohup bash Qwen2.5-7B-Instruct.sh > Qwen2.5-7B-Instruct.out &

sleep 3m

cd /map-vepfs/jiajun/ReasoningGym/experiments
python -m 19-party_time.eval -o 19-party_time/result -m Qwen2.5-7B-Instruct -a http://localhost:9003/v1 -k None