#!/bin/bash
source /map-vepfs/miniconda3/bin/activate
conda activate jiajun

cd /map-vepfs/jiajun/ReasoningGym/experiments/22-alphabetical_sorting/script
nohup bash DeepSeek-R1-Distill-Llama-70B.sh > DeepSeek-R1-Distill-Llama-70B.out &

sleep 5m

cd /map-vepfs/jiajun/ReasoningGym/experiments
python -m 22-alphabetical_sorting.eval -o 22-alphabetical_sorting/result -m DeepSeek-R1-Distill-Llama-70B -a http://localhost:9003/v1 -k None