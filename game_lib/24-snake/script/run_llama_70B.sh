#!/bin/bash
source /map-vepfs/miniconda3/bin/activate
conda activate wcr

cd /map-vepfs/jiajun/ReasoningGym/snake_game/script
nohup bash llama_70B.sh > Meta-Llama-3.1-8B-Instruct.out &

sleep 4m

cd /map-vepfs/jiajun/ReasoningGym
python -m snake_game.eval -o snake_game/result -m Meta-Llama-3.1-8B-Instruct -a http://localhost:9004/v1 -k None