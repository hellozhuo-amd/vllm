import os
import torch
import triton

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)

from vllm.model_executor.layers.fla.ops import (
    fused_rearrange_recurrent_gated_delta_rule,
    fused_causal_conv1d_update_rearrange_recurrent_gated_delta_rule,
)
from vllm.model_executor.layers.fla.ops.fused_rearrange_recurrent import (
    fused_rearrange_recurrent_gated_delta_rule_fwd,
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


def test(ipath, opath):

    inputs = torch.load(ipath)
    print(f"shape of inputs: {inputs['qkv'].shape}")
    mixed_qkv_non_spec = causal_conv1d_update(
        inputs["qkv"],
        inputs["conv_state"],
        inputs["weight"],
        inputs["bias"],
        inputs["activation"],
        conv_state_indices=inputs["conv_state_indices"],
        validate_data=inputs["validate_data"],
    )
    out1, out2 = (
        fused_rearrange_recurrent_gated_delta_rule(
            qkv=mixed_qkv_non_spec,
            g=inputs["g"],
            key_dim=inputs["key_dim"],
            value_dim=inputs["value_dim"],
            head_k_dim=inputs["head_k_dim"],
            head_v_dim=inputs["head_v_dim"],
            beta=inputs["beta"],
            initial_state=inputs["initial_state"],
            inplace_final_state=inputs["inplace_final_state"],
            cu_seqlens=inputs["cu_seqlens"],
            ssm_state_indices=inputs["ssm_state_indices"],
            use_qk_l2norm_in_kernel=inputs["use_qk_l2norm_in_kernel"],
        )
    )

    out3 = inputs["conv_state"]
    out4 = inputs["initial_state"]

    ## fused kernel
    inputs = torch.load(ipath)
    out1_fused, out2_fused = (
        fused_causal_conv1d_update_rearrange_recurrent_gated_delta_rule(
            ## for conv1d
            qkv=inputs["qkv"],
            conv_state=inputs["conv_state"],
            weight=inputs["weight"],
            bias=inputs["bias"],
            activation=inputs["activation"],
            conv_state_indices=inputs["conv_state_indices"],
            validate_data=inputs["validate_data"],
            ## now for rearrange and gated delta rule
            g=inputs["g"],
            key_dim=inputs["key_dim"],
            value_dim=inputs["value_dim"],
            head_k_dim=inputs["head_k_dim"],
            head_v_dim=inputs["head_v_dim"],
            beta=inputs["beta"],
            initial_state=inputs["initial_state"],
            inplace_final_state=inputs["inplace_final_state"],
            cu_seqlens=inputs["cu_seqlens"],
            ssm_state_indices=inputs["ssm_state_indices"],
            use_qk_l2norm_in_kernel=inputs["use_qk_l2norm_in_kernel"],
        )
    )

    out3_fused = inputs["conv_state"]
    out4_fused = inputs["initial_state"]

    outputs = torch.load(opath)
    out1_ref = outputs["core_attn_out"]
    out2_ref = outputs["last_recurrent_state"]

    compare_accuracy(out1, out1_fused)
    compare_accuracy(out2, out2_fused)
    compare_accuracy(out3, out3_fused)
    compare_accuracy(out4, out4_fused)

def get_func(ipath, fuse=False, part=0):

    inputs = torch.load(ipath)
    if fuse:
        def fn():
            out1_fused, out2_fused = (
                fused_causal_conv1d_update_rearrange_recurrent_gated_delta_rule(
                    ## for conv1d
                    qkv=inputs["qkv"],
                    conv_state=inputs["conv_state"],
                    weight=inputs["weight"],
                    bias=inputs["bias"],
                    activation=inputs["activation"],
                    conv_state_indices=inputs["conv_state_indices"],
                    validate_data=inputs["validate_data"],
                    ## now for rearrange and gated delta rule
                    g=inputs["g"],
                    key_dim=inputs["key_dim"],
                    value_dim=inputs["value_dim"],
                    head_k_dim=inputs["head_k_dim"],
                    head_v_dim=inputs["head_v_dim"],
                    beta=inputs["beta"],
                    initial_state=inputs["initial_state"],
                    inplace_final_state=inputs["inplace_final_state"],
                    cu_seqlens=inputs["cu_seqlens"],
                    ssm_state_indices=inputs["ssm_state_indices"],
                    use_qk_l2norm_in_kernel=inputs["use_qk_l2norm_in_kernel"],
                )
            )
            return out1_fused, out2_fused
    else:
        if part == 0:
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
                out1, out2 = (
                    fused_rearrange_recurrent_gated_delta_rule(
                        qkv=mixed_qkv_non_spec,
                        g=inputs["g"],
                        key_dim=inputs["key_dim"],
                        value_dim=inputs["value_dim"],
                        head_k_dim=inputs["head_k_dim"],
                        head_v_dim=inputs["head_v_dim"],
                        beta=inputs["beta"],
                        initial_state=inputs["initial_state"],
                        inplace_final_state=inputs["inplace_final_state"],
                        cu_seqlens=inputs["cu_seqlens"],
                        ssm_state_indices=inputs["ssm_state_indices"],
                        use_qk_l2norm_in_kernel=inputs["use_qk_l2norm_in_kernel"],
                    )
                )
                return out1, out2
        elif part == 1:
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
            mixed_qkv_non_spec = causal_conv1d_update(
                inputs["qkv"],
                inputs["conv_state"],
                inputs["weight"],
                inputs["bias"],
                inputs["activation"],
                conv_state_indices=inputs["conv_state_indices"],
                validate_data=inputs["validate_data"],
            )
            def fn():
                out1, out2 = (
                    fused_rearrange_recurrent_gated_delta_rule_fwd(
                        qkv=mixed_qkv_non_spec,
                        g=inputs["g"],
                        key_dim=inputs["key_dim"],
                        value_dim=inputs["value_dim"],
                        head_k_dim=inputs["head_k_dim"],
                        head_v_dim=inputs["head_v_dim"],
                        beta=inputs["beta"],
                        scale=inputs["head_k_dim"] ** -0.5,
                        initial_state=inputs["initial_state"],
                        inplace_final_state=inputs["inplace_final_state"],
                        cu_seqlens=inputs["cu_seqlens"],
                        ssm_state_indices=inputs["ssm_state_indices"],
                        use_qk_l2norm_in_kernel=inputs["use_qk_l2norm_in_kernel"],
                    )
                )
                return out1, out2

    return fn


def bench_cuda_graph(ipath):
    """Bench with CUDA/HIP graph capture to match the trace environment."""
    from vllm.model_executor.layers.fla.ops.fused_rearrange_recurrent import (
        fused_rearrange_recurrent_gated_delta_rule_fwd,
    )

    inputs = torch.load(ipath)
    print(f"shape of inputs: {inputs['qkv'].shape}")

    # --- Prepare inputs for fused_rearrange_recurrent (part=2 equivalent) ---
    mixed_qkv_non_spec = causal_conv1d_update(
        inputs["qkv"],
        inputs["conv_state"],
        inputs["weight"],
        inputs["bias"],
        inputs["activation"],
        conv_state_indices=inputs["conv_state_indices"],
        validate_data=inputs["validate_data"],
    )
    g = inputs["g"].contiguous()
    beta = inputs["beta"].contiguous()

    # Pre-allocate output to avoid allocation during graph capture
    key_dim = inputs["key_dim"]
    value_dim = inputs["value_dim"]
    head_k_dim = inputs["head_k_dim"]
    head_v_dim = inputs["head_v_dim"]

    # --- Warmup (needed before graph capture) ---
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fused_rearrange_recurrent_gated_delta_rule_fwd(
                qkv=mixed_qkv_non_spec,
                g=g,
                key_dim=key_dim,
                value_dim=value_dim,
                head_k_dim=head_k_dim,
                head_v_dim=head_v_dim,
                beta=beta,
                scale=head_k_dim ** -0.5,
                initial_state=inputs["initial_state"],
                inplace_final_state=inputs["inplace_final_state"],
                cu_seqlens=inputs["cu_seqlens"],
                ssm_state_indices=inputs["ssm_state_indices"],
                use_qk_l2norm_in_kernel=inputs["use_qk_l2norm_in_kernel"],
            )
    torch.cuda.current_stream().wait_stream(s)

    # --- Capture graph ---
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        fused_rearrange_recurrent_gated_delta_rule_fwd(
            qkv=mixed_qkv_non_spec,
            g=g,
            key_dim=key_dim,
            value_dim=value_dim,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            beta=beta,
            scale=head_k_dim ** -0.5,
            initial_state=inputs["initial_state"],
            inplace_final_state=inputs["inplace_final_state"],
            cu_seqlens=inputs["cu_seqlens"],
            ssm_state_indices=inputs["ssm_state_indices"],
            use_qk_l2norm_in_kernel=inputs["use_qk_l2norm_in_kernel"],
        )

    # --- Bench the graph replay ---
    def fn_graph():
        graph.replay()

    ms_graph = triton.testing.do_bench(fn_graph)

    # --- Also bench eager (part=2) for comparison ---
    def fn_eager():
        fused_rearrange_recurrent_gated_delta_rule_fwd(
            qkv=mixed_qkv_non_spec,
            g=g,
            key_dim=key_dim,
            value_dim=value_dim,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            beta=beta,
            scale=head_k_dim ** -0.5,
            initial_state=inputs["initial_state"],
            inplace_final_state=inputs["inplace_final_state"],
            cu_seqlens=inputs["cu_seqlens"],
            ssm_state_indices=inputs["ssm_state_indices"],
            use_qk_l2norm_in_kernel=inputs["use_qk_l2norm_in_kernel"],
        )

    ms_eager = triton.testing.do_bench(fn_eager)

    # --- Also bench via the autograd.Function wrapper ---
    fn_k2 = get_func(ipath, fuse=False, part=2)
    ms_wrapper = triton.testing.do_bench(fn_k2)

    print(f"fused_rearrange_recurrent via CUDA graph replay: {ms_graph:.6f} ms")
    print(f"fused_rearrange_recurrent eager (direct _fwd):   {ms_eager:.6f} ms")
    print(f"fused_rearrange_recurrent eager (via wrapper):   {ms_wrapper:.6f} ms")
    print(f"  -> graph vs eager overhead:  {ms_eager - ms_graph:.3f} ms")
    print(f"  -> wrapper vs direct overhead: {ms_wrapper - ms_eager:.3f} ms")


def bench(ipath):

    inputs = torch.load(ipath)
    print(f"shape of inputs: {inputs['qkv'].shape}")

    fn_fused = get_func(ipath, fuse=True)
    #fn = get_func(ipath, fuse=False)
    fn_k1 = get_func(ipath, fuse=False, part=1)
    fn_k2 = get_func(ipath, fuse=False, part=2)

    ms_fused = triton.testing.do_bench(fn_fused)
    #ms = triton.testing.do_bench(fn)
    ms_k1 = triton.testing.do_bench(fn_k1)
    ms_k2 = triton.testing.do_bench(fn_k2)

    #print(f"before fuse: {ms:.6f} ms")
    print(f"kernel 1: {ms_k1:.6f} ms")
    print(f"kernel 2: {ms_k2:.6f} ms")
    print(f"after fuse: {ms_fused:.6f} ms")

def main():

    ipath = "/app/projects/vllm/tmp/debug/4/input_1.pt"
    opath = "/app/projects/vllm/tmp/debug/4/output_1.pt"
    
    #test(ipath, opath)
    bench(ipath)
    #bench_cuda_graph(ipath)

if __name__ == "__main__":
    main()



