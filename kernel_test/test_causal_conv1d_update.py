import os
import torch
import triton

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d_fast import (
    causal_conv1d_update_fast,
)

def compare_accuracy(current, reference):
    """Print quick statistics comparing FP8 and SageAttn tensors."""
    current_f = current.float()
    reference_f = reference.float()
    abs_diff = torch.abs(reference_f - current_f)

    print("Output Tensor Stats:")
    print(
        f"  Reference ({tuple(reference_f.shape)}): min={reference_f.min().item():.6f}, max={reference_f.max().item():.6f}, "
        f"mean={reference_f.mean().item():.6f}, std={reference_f.std().item():.6f}"
    )
    print(
        f"  Test      ({tuple(current_f.shape)}): min={current_f.min().item():.6f}, max={current_f.max().item():.6f}, "
        f"mean={current_f.mean().item():.6f}, std={current_f.std().item():.6f}"
    )

    print("Correctness Comparison:")
    print(f"  Mean Absolute Error: {abs_diff.mean().item():.6e}")
    print(f"  Max Absolute Error: {abs_diff.max().item():.6e}")
    print(f"  Std Absolute Error: {abs_diff.std().item():.6e}")
    ref_flat = reference_f.reshape(-1)
    test_flat = current_f.reshape(-1)
    cos_sim = torch.nn.functional.cosine_similarity(
        ref_flat.unsqueeze(0), test_flat.unsqueeze(0)
    )
    print(f"  Cosine Similarity: {cos_sim.item():.8f}")


def test(ipath, opath):
    inputs = torch.load(ipath)
    print(f"shape of inputs: {inputs['qkv'].shape}")

    out1 = causal_conv1d_update(
        inputs["qkv"],
        inputs["conv_state"],
        inputs["weight"],
        inputs["bias"],
        inputs["activation"],
        conv_state_indices=inputs["conv_state_indices"],
        validate_data=inputs["validate_data"],
    )
    out2 = inputs["conv_state"]

    ## fast kernel
    inputs = torch.load(ipath)
    out1_fast = causal_conv1d_update_fast(
        inputs["qkv"],
        inputs["conv_state"],
        inputs["weight"],
        inputs["bias"],
        inputs["activation"],
        conv_state_indices=inputs["conv_state_indices"],
        validate_data=inputs["validate_data"],
    )
    out2_fast = inputs["conv_state"]

    compare_accuracy(out1, out1_fast)
    compare_accuracy(out2, out2_fast)

def get_func(ipath, fast=False):

    inputs = torch.load(ipath)
    if not fast:
        def fn():
            mixed_qkv_non_spec = causal_conv1d_update(
                inputs["qkv"],
                inputs["conv_state"],
                inputs["weight"],
                inputs["bias"],
                inputs["activation"],
                conv_state_indices=inputs["conv_state_indices"],
                validate_data=inputs["validate_data"],
            )
            return mixed_qkv_non_spec
    else:
        def fn():
            mixed_qkv_non_spec = causal_conv1d_update_fast(
                inputs["qkv"],
                inputs["conv_state"],
                inputs["weight"],
                inputs["bias"],
                inputs["activation"],
                conv_state_indices=inputs["conv_state_indices"],
                validate_data=inputs["validate_data"],
            )
            return mixed_qkv_non_spec

    return fn

def bench(ipath):

    inputs = torch.load(ipath)
    print(f"shape of inputs: {inputs['qkv'].shape}")
    print(f"shape of weights: {inputs['weight'].shape}")

    fn = get_func(ipath)
    fn_fast = get_func(ipath, fast=True)

    ms = triton.testing.do_bench(fn)
    ms_fast = triton.testing.do_bench(fn_fast)

    print(f"before optimization: {ms:.6f} ms")
    print(f"after optimization: {ms_fast:.6f} ms")

def main():

    ipath = "/app/projects/vllm/tmp/debug/4/input_1.pt"
    opath = "/app/projects/vllm/tmp/debug/4/output_1.pt"
    
    test(ipath, opath)
    #bench(ipath)

if __name__ == "__main__":
    main()



