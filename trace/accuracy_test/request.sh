pip install lm-eval[api]
HF_HUB_OFFLINE=1 lm_eval \
    --model local-completions \
    --tasks gsm8k \
    --model_args model=Qwen/Qwen3-Next-80B-A3B-Instruct-FP8,base_url=http://localhost:8022/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False
