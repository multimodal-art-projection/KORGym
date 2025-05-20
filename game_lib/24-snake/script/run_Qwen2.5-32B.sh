#!/bin/bash
source /map-vepfs/miniconda3/bin/activate
conda activate wcr

cd /map-vepfs/jiajun/ReasoningGym/snake_game/script
nohup bash Qwen2.5-32B.sh > Qwen2.5-32B.out &

sleep 4m

cd /map-vepfs/jiajun/ReasoningGym
python -m snake_game.eval -o snake_game/result -m Qwen2.5-32B -a http://localhost:9003/v1 -k None