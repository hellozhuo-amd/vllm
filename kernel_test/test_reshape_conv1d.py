import os
import torch
import triton

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d_fast import (
    causal_conv1d_update_fast,
    fused_reshape_causal_conv1d_update_fast,
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

@torch.compile(fullgraph=True)
def prepare_gdn_attention_core_inputs(
    mixed_qkvz,
    mixed_ba,
    num_tokens,
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
):
    """
    Derives mixed_qkv, z, b, a, and initializes core_attn_out in a
    single fused kernel launch to minimize launch overhead.
    """
    base_shape_qkvz = mixed_qkvz.size()[:-1]
    base_shape_ba = mixed_ba.size()[:-1]
    ng = num_k_heads

    new_tensor_shape_qkvz = base_shape_qkvz + (
        ng,
        (
            head_k_dim
            + head_k_dim
            + (head_v_dim + head_v_dim)
            * num_v_heads
            // num_k_heads
        ),
    )
    new_tensor_shape_ba = base_shape_ba + (
        ng,
        2 * num_v_heads // num_k_heads,
    )

    mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
    mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

    split_arg_list_qkvz = [
        head_k_dim,
        head_k_dim,
        (num_v_heads // num_k_heads * head_v_dim),
        (num_v_heads // num_k_heads * head_v_dim),
    ]
    split_arg_list_ba = [
        num_v_heads // num_k_heads,
        num_v_heads // num_k_heads,
    ]

    (query, key, value, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=-1)
    (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=-1)

    # 1. Interleave Q, K, V logically.
    # Inside compile, this doesn't allocate memory yet; it just creates an indexing map.
    mixed_qkv_logical = torch.cat([
        query.reshape(num_tokens, -1),
        key.reshape(num_tokens, -1),
        value.reshape(num_tokens, -1)
    ], dim=-1)

    # We flatten everything into a 1D sequence and concatenate. Inductor will launch
    # ONE Triton kernel to populate this single buffer.
    fused = torch.cat([
        mixed_qkv_logical.reshape(-1),
        z.reshape(-1),
        b.reshape(-1),
        a.reshape(-1),
    ], dim=0)

    # 4. Calculate offsets dynamically to slice the buffer back out
    curr = 0
    qkv_numel = mixed_qkv_logical.numel()
    z_numel = z.numel()
    b_numel = b.numel()
    a_numel = a.numel()

    # 5. Slice and reshape (Zero-copy metadata changes)
    mixed_qkv_out = fused[curr : curr + qkv_numel].view(num_tokens, -1)
    curr += qkv_numel

    z_out = fused[curr : curr + z_numel].view(num_tokens, -1, head_v_dim)
    curr += z_numel

    b_out = fused[curr : curr + b_numel].view(num_tokens, num_v_heads)
    curr += b_numel

    a_out = fused[curr : curr + a_numel].view(num_tokens, num_v_heads)
    curr += a_numel

    return mixed_qkv_out, z_out, b_out, a_out


def test(ipath, opath):

    ## fused kernel
    inputs = torch.load(ipath)
    print(f"shape of inputs: {inputs['qkvz'].shape}")

    num_tokens = inputs["qkvz"].shape[0]
    num_actual_tokens = inputs["num_actual_tokens"]
    key_dim = inputs["key_dim"]
    value_dim = inputs["value_dim"]
    head_k_dim = inputs["head_k_dim"]
    head_v_dim = inputs["head_v_dim"]
    num_k_heads = key_dim // head_k_dim
    num_v_heads = value_dim // head_v_dim
    z_fused = torch.zeros(
        (num_tokens, num_v_heads, head_v_dim),
        dtype=inputs["qkvz"].dtype,
        device=inputs["qkvz"].device,
    )
    mixed_qkv_non_spec_fused, b_fused, a_fused = fused_reshape_causal_conv1d_update_fast(
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
    conv_state_fused = inputs["conv_state"]

    ## reference
    inputs = torch.load(ipath)

    mixed_qkv, z, b, a = prepare_gdn_attention_core_inputs(
        inputs["qkvz"], inputs["ba"], num_tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim
    )
    mixed_qkv = mixed_qkv[:num_actual_tokens]
    b = b[:num_actual_tokens]
    a = a[:num_actual_tokens]

    mixed_qkv_non_spec = causal_conv1d_update_fast(
        mixed_qkv,
        inputs["conv_state"],
        inputs["weight"],
        inputs["bias"],
        inputs["activation"],
        conv_state_indices=inputs["conv_state_indices"],
        validate_data=True,
    )
    conv_state = inputs["conv_state"] 

    compare_accuracy(mixed_qkv_non_spec, mixed_qkv_non_spec_fused)
    compare_accuracy(b, b_fused)
    compare_accuracy(a, a_fused)
    compare_accuracy(z, z_fused)
    compare_accuracy(conv_state, conv_state_fused)

def main():

    ipath = "/data/conv1d_tensors/4/input_1.pt"
    opath = "/data/conv1d_tensors/4/output_1.pt"
    
    test(ipath, opath)
    #bench(ipath)

if __name__ == "__main__":
    main()



