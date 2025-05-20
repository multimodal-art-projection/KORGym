source /map-vepfs/miniconda3/bin/activate
conda activate wcr
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server --model /map-vepfs/models/jiajun/Qwen/Qwen2.5-14B-Instruct --served-model-name Qwen2.5-14B-Instruct --max_model_len=15000 --port=9003 --pipeline_parallel_size=1 --tensor_parallel_size=2 --gpu_memory_utilization=0.95
