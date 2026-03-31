import os
import torch
import triton

from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.linear import (
    RowParallelLinear,
)

def bench(ipath):

    inputs = torch.load(ipath)
    core_attn_out = inputs['core_attn_out']
    print(f"shape of inputs: {core_attn_out.shape}")

    num_tokens = inputs["num_tokens"]
    z = inputs["z"]
    output = inputs["output"]

    norm = RMSNormGated(
        z.shape[-1],
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        device=output.device,
        dtype=output.dtype,
    )

    out_proj = RowParallelLinear(
        value_dim,
        self.hidden_size,
        bias=False,
        input_is_parallel=True,
        quant_config=quant_config,
        prefix=f"{prefix}.out_proj",
    )

    out_proj = 


    # --- Warmup (needed before graph capture) ---
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fused_reshape_causal_conv1d_update_fast(
                inputs["qkvz"],
                num_actual_tokens,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                inputs["ba"],
                z_fused,
                inputs["conv_state"],
                inputs["weight"],
                inputs["bias"],
                inputs["activation"],
                conv_state_indices=inputs["conv_state_indices"],
                validate_data=True,
            )
    torch.cuda.current_stream().wait_stream(s)

    # --- Capture graph ---
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        fused_reshape_causal_conv1d_update_fast(
            inputs["qkvz"],
            num_actual_tokens,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            inputs["ba"],
            z_fused,
            inputs["conv_state"],
            inputs["weight"],
            inputs["bias"],
            inputs["activation"],
            conv_state_indices=inputs["conv_state_indices"],
            validate_data=True,
        )

    # --- Bench the graph replay ---
    def fn_graph():
        graph.replay()

    ms_graph = triton.testing.do_bench(fn_graph)

    print(f"fused kernel via CUDA graph replay: {ms_graph:.6f} ms")

def main():

    ipath = "/data/gdn_tensors/tensor_1.pt"
    
    #test(ipath, opath)
    bench(ipath)

if __name__ == "__main__":
    main()
