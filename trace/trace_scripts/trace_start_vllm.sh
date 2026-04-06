vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct-FP8     \
    --attention_backend ROCM_AITER_UNIFIED_ATTN     \
    --compilation_config '{"cudagraph_mode": "FULL_AND_PIECEWISE" , "custom_ops": ["-rms_norm", "-silu_and_mul", "-quant_fp8"]}'     \
    --no_enable_log_requests     \
    --gpu_memory_utilization 0.95     \
    --host 0.0.0.0     \
    --max_model_len 16384     \
    --max_num_seqs 256     \
    --no_async_scheduling     \
    --no_enable_prefix_caching     \
    --port 8027     \
    --tensor_parallel_size 1 \
    --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/tmp/profile/zhuo_traces/"}'
 
