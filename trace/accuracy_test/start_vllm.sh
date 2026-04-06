## CUDA_VISIBLE_DEVICES=2 HIP_VISIBLE_DEVICES=2 \
#    --swap_space 64 \

HF_HUB_OFFLINE=1 \
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct-FP8     \
    --attention_backend TRITON_ATTN     \
    --compilation_config '{"cudagraph_mode": "FULL_AND_PIECEWISE" , "custom_ops": ["-rms_norm", "-silu_and_mul", "-quant_fp8"], "pass_config": { "eliminate_noops": "True"}, "use_inductor_graph_partition": "True" }'     \
    --no_enable_log_requests     \
    --gpu_memory_utilization 0.95     \
    --host 0.0.0.0     \
    --max_model_len 16384     \
    --max_num_seqs 256     \
    --no_async_scheduling \
    --no-enable-prefix-caching \
    --port=8061 \
    --tensor_parallel_size=1
