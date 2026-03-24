## send requests with custom data
NAME="fusion_results4.json"
vllm bench serve \
  --backend vllm \
  --dataset_name custom \
  --dataset_path tmp/my_prompts.jsonl \
  --model Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
  --ready_check_timeout_sec 7200 \
  --percentile_metrics ttft,tpot,itl,e2el \
  --max_concurrency 4 \
  --seed 1 \
  --num_warmups 0 \
  --num_prompts 4 \
  --temperature 0 \
  --save-result \
  --save-detailed \
  --result_dir tmp/responses \
  --result_filename $NAME

python tmp/extract_responses.py tmp/responses/$NAME
