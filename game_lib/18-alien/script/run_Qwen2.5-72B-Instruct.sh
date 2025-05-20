#!/bin/bash
source /map-vepfs/miniconda3/bin/activate
conda activate jiajun

cd /map-vepfs/jiajun/ReasoningGym/experiments/18-alien/script
nohup bash Qwen2.5-72B-Instruct.sh > Qwen2.5-72B-Instruct.out &

sleep 5m

cd /map-vepfs/jiajun/ReasoningGym/experiments
python -m 18-alien.eval -o 18-alien/result -m Qwen2.5-72B-Instruct -a http://localhost:9004/v1 -k None