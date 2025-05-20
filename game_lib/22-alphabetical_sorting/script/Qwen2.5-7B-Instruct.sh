source /map-vepfs/miniconda3/bin/activate
conda activate jiajun
CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server --model /map-vepfs/models/jiajun/Qwen/Qwen2.5-7B-Instruct --served-model-name Qwen2.5-7B-Instruct --max_model_len=15000 --port=9003 --pipeline_parallel_size=1 --tensor_parallel_size=2 --gpu_memory_utilization=0.95
