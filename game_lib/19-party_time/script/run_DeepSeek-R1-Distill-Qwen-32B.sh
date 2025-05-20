#!/bin/bash
source /map-vepfs/miniconda3/bin/activate
conda activate jiajun

cd /map-vepfs/jiajun/ReasoningGym/experiments/19-party_time/script
nohup bash DeepSeek-R1-Distill-Qwen-32B.sh > DeepSeek-R1-Distill-Qwen-32B.out &

sleep 5m

cd /map-vepfs/jiajun/ReasoningGym/experiments
python -m 19-party_time.eval -o 19-party_time/result -m DeepSeek-R1-Distill-Qwen-32B -a http://localhost:9003/v1 -k None