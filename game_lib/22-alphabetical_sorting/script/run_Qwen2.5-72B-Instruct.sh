#!/bin/bash
source /map-vepfs/miniconda3/bin/activate
conda activate jiajun

cd /map-vepfs/jiajun/ReasoningGym/experiments/22-alphabetical_sorting/script
nohup bash Qwen2.5-72B-Instruct.sh > Qwen2.5-72B-Instruct.out &

sleep 7m

cd /map-vepfs/jiajun/ReasoningGym/experiments
python -m 22-alphabetical_sorting.eval -o 22-alphabetical_sorting/result -m Qwen2.5-72B-Instruct -a http://localhost:9004/v1 -k None