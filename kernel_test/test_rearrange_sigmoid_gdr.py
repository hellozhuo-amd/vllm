import os
import torch
import triton
from einops import rearrange

from vllm.model_executor.layers.fla.ops import (
    fused_sigmoid_gating_delta_rule_update,
    fused_rearrange_sigmoid_gated_delta_rule,
)

#from aiter.ops.triton.fusions.fused_rearrange_recurrent import fused_rearrange_recurrent_gated_delta_rule
#from aiter.ops.triton.fusions.fused_conv1d_rearrange_recurrent import fused_causal_conv1d_update_rearrange_recurrent_gated_delta_rule

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

def rearrange_mixed_qkv(mixed_qkv, key_dim, value_dim, head_k_dim, head_v_dim):
    if mixed_qkv is None:
        return None, None, None
    query, key, value = torch.split(
        mixed_qkv,
        [
            key_dim,
            key_dim,
            value_dim,
        ],
        dim=-1,
    )
    query, key = map(
        lambda x: rearrange(x, "l (h d) -> 1 l h d", d=head_k_dim),
        (query, key),
    )
    value = rearrange(value, "l (h d) -> 1 l h d", d=head_v_dim)
    return query.contiguous(), key.contiguous(), value.contiguous()


def test(ipath, opath):

    inputs = torch.load(ipath)
    print(f"shape of inputs: {inputs['qkv'].shape}")
    query_non_spec, key_non_spec, value_non_spec = rearrange_mixed_qkv(
        inputs["qkv"],
        inputs["key_dim"],
        inputs["value_dim"],
        inputs["head_k_dim"],
        inputs["head_v_dim"],
    )
    out1, out2 = (
        fused_sigmoid_gating_delta_rule_update(
            A_log=inputs["A_log"],
            a=inputs["a"],
            b=inputs["b"],
            dt_bias=inputs["dt_bias"],
            q=query_non_spec,
            k=key_non_spec,
            v=value_non_spec,
            initial_state=inputs["initial_state"],
            inplace_final_state=inputs["inplace_final_state"],
            cu_seqlens=inputs["cu_seqlens"],
            ssm_state_indices=inputs["ssm_state_indices"],
            use_qk_l2norm_in_kernel=inputs["use_qk_l2norm_in_kernel"],
        )
    )

    out3 = inputs["initial_state"]

    ## fused kernel
    inputs = torch.load(ipath)
    out1_fused, out2_fused = (
        fused_rearrange_sigmoid_gated_delta_rule(
            A_log=inputs["A_log"],
            a=inputs["a"],
            b=inputs["b"],
            dt_bias=inputs["dt_bias"],
            qkv=inputs["qkv"],
            key_dim=inputs["key_dim"],
            value_dim=inputs["value_dim"],
            head_k_dim=inputs["head_k_dim"],
            head_v_dim=inputs["head_v_dim"],
            initial_state=inputs["initial_state"],
            inplace_final_state=inputs["inplace_final_state"],
            cu_seqlens=inputs["cu_seqlens"],
            ssm_state_indices=inputs["ssm_state_indices"],
            use_qk_l2norm_in_kernel=inputs["use_qk_l2norm_in_kernel"],
        )
    )

    out3_fused = inputs["initial_state"]

    outputs = torch.load(opath)
    out1_ref = outputs["core_attn_out"]
    out2_ref = outputs["last_recurrent_state"]

    print("vs. old kernels")
    compare_accuracy(out1, out1_fused)
    compare_accuracy(out2, out2_fused)
    compare_accuracy(out3, out3_fused)

    print("vs. ref")
    compare_accuracy(out1_fused, out1_ref)
    compare_accuracy(out2_fused, out2_ref)

def main():

    ipath = "/data/baseline_sigmoid_gdr_tensors/4/input_1.pt"
    opath = "/data/baseline_sigmoid_gdr_tensors/4/output_1.pt"
    
    test(ipath, opath)
    #bench(ipath)

if __name__ == "__main__":
    main()



