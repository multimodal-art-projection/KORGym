source /map-vepfs/miniconda3/bin/activate
conda activate wcr
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m vllm.entrypoints.openai.api_server --model /map-vepfs/models/jiajun/DeepSeek/DeepSeek-R1-Distill-Llama-70B --served-model-name DeepSeek-R1-Distill-Llama-70B --max_model_len=15000 --port=9003 --pipeline_parallel_size=2 --tensor_parallel_size=4 --gpu_memory_utilization=0.95
