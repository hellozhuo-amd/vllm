import os
import torch
import triton

from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.linear import (
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import get_quantization_config

def bench(ipath1, ipath2):

    inputs = torch.load(ipath1)
    core_attn_out = inputs['core_attn_out']
    print(f"shape of inputs: {core_attn_out.shape}")

    num_tokens = inputs["num_tokens"]
    z = inputs["z"]
    output = inputs["output"]
    head_v_dim = inputs["head_v_dim"]

    norm = RMSNormGated(
        head_v_dim,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        device=output.device,
        dtype=output.dtype,
    )

    out_proj = RowParallelLinear(
        value_dim,
        hidden_size,
        bias=False,
        input_is_parallel=True,
        quant_config=quant_config,
        prefix=f"{prefix}.out_proj",
    )

    # Load the saved data
    data = torch.load(ipath2)

    # Recreate quant_config if it was saved
    quant_config = None
    if data["quant_config"] is not None:
        
        # Use the saved quant_method to get the right class and recreate from config
        quant_config = get_quantization_config(data["quant_config"])

    # Recreate RowParallelLinear with proper quantization
    out_proj = RowParallelLinear(
        data["config"]["input_size"],
        data["config"]["output_size"],
        bias=data["config"]["bias"],
        input_is_parallel=data["config"]["input_is_parallel"],
        quant_config=quant_config,
    )

    # Load the weights
    out_proj.load_state_dict(data["state_dict"])


    # --- Warmup (needed before graph capture) ---
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            ## do something
    torch.cuda.current_stream().wait_stream(s)

    # --- Capture graph ---
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        # do something

    # --- Bench the graph replay ---
    def fn_graph():
        graph.replay()

    ms_graph = triton.testing.do_bench(fn_graph)

    print(f"fused kernel via CUDA graph replay: {ms_graph:.6f} ms")

def main():

    ipath1 = "/data/gdn_tensors/tensor_1.pt"
    ipath2 = "/data/gdn_tensors/out_proj_1.pt"
    
    bench(ipath1, ipath2)

if __name__ == "__main__":
    main()
