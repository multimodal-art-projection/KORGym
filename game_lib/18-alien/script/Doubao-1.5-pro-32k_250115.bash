#!/bin/bash
source /map-vepfs/miniconda3/bin/activate
conda activate jiajun
cd /map-vepfs/jiajun/ReasoningGym/experiments
python -m 18-alien.eval -o 18-alien/result -m ep-20250226145008-h97kr -a https://ark.cn-beijing.volces.com/api/v3 -k c995ddea-161e-4d65-834a-2807068c7bb1 