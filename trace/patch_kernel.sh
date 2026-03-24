INSTALL_PATH=$1
cp vllm/model_executor/__init__.py $INSTALL_PATH/vllm/model_executor/__init__.py
cp vllm/model_executor/layers/fla/ops/fused_rearrange_recurrent.py $INSTALL_PATH/vllm/model_executor/layers/fla/ops/fused_rearrange_recurrent.py
cp vllm/model_executor/layers/fla/ops/fused_conv1d_rearrange_recurrent.py $INSTALL_PATH/vllm/model_executor/layers/fla/ops/fused_conv1d_rearrange_recurrent.py
cp vllm/model_executor/models/qwen3_next.py $INSTALL_PATH/vllm/model_executor/models/qwen3_next.py
