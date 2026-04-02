# e2e, looking at total token throughput

vllm bench serve \
    --backend vllm \
    --dataset_name random \
    --ignore_eos \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
    --ready_check_timeout_sec 7200 \
    --percentile_metrics ttft,tpot,itl,e2el \
    --random_input_len 1024 \
    --random_output_len 1024 \
    --max_concurrency 4 \
    --seed 1 \
    --port 8027 \
    --num_warmups 0 \
    --num_prompts 4
