source /map-vepfs/miniconda3/bin/activate
conda activate wcr
CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server --model /map-vepfs/models/jiajun/Llama3/Meta-Llama-3.1-8B-Instruct --served-model-name Meta-Llama-3.1-8B-Instruct --max_model_len=15000 --port=9003 --pipeline_parallel_size=1 --tensor_parallel_size=2 --gpu_memory_utilization=0.95
