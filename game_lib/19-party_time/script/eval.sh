#!/bin/bash
source /map-vepfs/miniconda3/bin/activate
conda activate wcr

cd /map-vepfs/jiajun/ReasoningGym/experiments

python -m 2048.eval -o 2048/result -m Qwen2.5-72B-Instruct -a http://localhost:9004/v1 -k None