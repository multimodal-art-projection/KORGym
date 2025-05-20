#!/bin/bash
source /map-vepfs/miniconda3/bin/activate
conda activate wcr

cd /map-vepfs/jiajun/ReasoningGym
python -m snake_game.eval -o snake_game/result -m ep-20250223170704-j28cs -a https://ark.cn-beijing.volces.com/api/v3 -k c995ddea-161e-4d65-834a-2807068c7bb1 > deepseek_r1.out &