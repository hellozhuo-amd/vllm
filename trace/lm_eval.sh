pip install lm-eval[api] transformers

lm_eval --model local-completions --tasks gsm8k --model_args model=Qwen/Qwen3-Next-80B-A3B-Instruct-FP8,base_url=http://localhost:8027/v1/completions,num_concurrent=128,max_retries=3,tokenized_requests=False
