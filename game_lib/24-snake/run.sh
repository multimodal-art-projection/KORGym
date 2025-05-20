#!/bin/bash
source /map-vepfs/miniconda3/bin/activate
conda activate wcr

cd /map-vepfs/jiajun/ReasoningGym/snake_game
nohup bash model.sh > Meta-Llama-3.1-70B.out &

sleep 4m

python eval.py -o result -m Meta-Llama-3.1-70B -a http://localhost:9003/v1 -k None