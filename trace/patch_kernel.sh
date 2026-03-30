INSTALL_PATH=$1
cp -v vllm/model_executor/layers/fla/ops/__init__.py $INSTALL_PATH/vllm/model_executor/layers/fla/ops/__init__.py
cp -v vllm/model_executor/layers/fla/ops/fused_rearrange_recurrent.py $INSTALL_PATH/vllm/model_executor/layers/fla/ops/fused_rearrange_recurrent.py
cp -v vllm/model_executor/layers/mamba/ops/causal_conv1d_fast.py $INSTALL_PATH/vllm/model_executor/layers/mamba/ops/causal_conv1d_fast.py
cp -v vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py $INSTALL_PATH/vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py
cp -v vllm/model_executor/layers/fla/ops/fused_rearrange_sigmoid_gdr.py $INSTALL_PATH/vllm/model_executor/layers/fla/ops/fused_rearrange_sigmoid_gdr.py
cp -v vllm/model_executor/layers/fla/ops/chunk.py $INSTALL_PATH/vllm/model_executor/layers/fla/ops/chunk.py
cp -v vllm/model_executor/layers/fla/ops/chunk_o.py $INSTALL_PATH/vllm/model_executor/layers/fla/ops/chunk_o.py
cp -v vllm/model_executor/layers/fla/ops/fused_recurrent.py $INSTALL_PATH/vllm/model_executor/layers/fla/ops/fused_recurrent.py
cp -v vllm/model_executor/models/qwen3_next.py $INSTALL_PATH/vllm/model_executor/models/qwen3_next.py
