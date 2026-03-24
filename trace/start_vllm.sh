#vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
#    --async_scheduling \
#    --compilation_config {"compile_sizes": [1,2,4,8,16,32,48,64,80,96,112,128,256,512] , "cudagraph_capture_sizes": [1,2,4,8,16,32,48,64,80,96,112,128,256,512] , "cudagraph_mode": "FULL_AND_PIECEWISE" , "custom_ops": ["-rms_norm", "-silu_and_mul"]} \
#    --disable_log_requests \
#    --gpu_memory_utilization 0.95 \
#    --host 0.0.0.0 \
#    --max_model_len 16384 \
#    --no_enable_prefix_caching \
#    --port 8000 \
#    --swap_space 64 \
#    --tensor_parallel_size 1 \
#    --max_num_seqs 256 \
#    --attention_backend TRITON_ATTN


HIP_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
    --async_scheduling \
    --compilation_config '{"compile_sizes": [1,2,4,8,16,32,48,64,80,96,112,128,256,512] , "cudagraph_capture_sizes": [1,2,4,8,16,32,48,64,80,96,112,128,256,512] , "cudagraph_mode": "FULL_AND_PIECEWISE" , "custom_ops": ["-rms_norm", "-silu_and_mul"]}' \
    --disable_log_requests \
    --gpu_memory_utilization 0.95 \
    --host 0.0.0.0 \
    --max_model_len 16384 \
    --no_enable_prefix_caching \
    --port 8000 \
    --swap_space 64 \
    --tensor_parallel_size 1 \
    --max_num_seqs 256 \
    --attention_backend TRITON_ATTN \
    #--enforce-eager \
    #--profiler-config '{"profiler": "torch", "torch_profiler_dir": "/app/projects/vllm/tmp/profile"}'
