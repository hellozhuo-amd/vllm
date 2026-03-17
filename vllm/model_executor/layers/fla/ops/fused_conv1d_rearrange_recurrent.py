# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from .op import exp

@triton.jit()
def _causal_conv1d_update_inner_kernel(
    loaded_x,  # (x_dim,) batch_size and seqlen are both 1, loaded
    w_ptr,  # (dim, width)
    bias_ptr, # (dim,)
    conv_state_ptr, # (num_cache_lines, dim, state_len(>=width -1))
    conv_state_indices_ptr, # (seqlen,)
    num_accepted_tokens_ptr, # None, not used
    query_start_loc_ptr,  # (batch + 1), not used
    block_idx_last_scheduled_token,  # (batch,), not used
    initial_state_idx,  # (batch,), not used
    idx_seq,
    idx_feats, # (x_dim,)
    mask_feats, # (x_dim,)
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
):
    # ruff: noqa: E501
    seqlen = 1

    # IS_APC_ENABLED is False
    conv_state_init = 0
    current_last_index = 0

    # cache_idx
    conv_states_input_coord = tl.load(
        conv_state_indices_ptr + idx_seq * stride_state_indices + conv_state_init
    ).to(tl.int64)

    if USE_PAD_SLOT and conv_states_input_coord == pad_slot_id:  # noqa
        acc = loaded_x.to(tl.float32)

    else:
        original_x_dtype = loaded_x.dtype
        loaded_x = loaded_x.to(conv_state_ptr.type.element_ty)

        # IS_SPEC_DECODING is False
        conv_state_token_offset = 0

        # STEP 1: READ init_state data
        # note: NP2_STATELEN = triton.next_power_of_2(KERNEL_WIDTH - 1)
        idx_cols = tl.arange(0, NP2_STATELEN)
        conv_state_ptrs_cols = (
            conv_state_ptr
            + (conv_states_input_coord * stride_conv_state_seq)
            + conv_state_token_offset * stride_conv_state_tok
            + (idx_feats * stride_conv_state_dim)[:, None]
            + (idx_cols * stride_conv_state_tok)[None, :]
        )  # [x_dim, NP2_STATELEN]
        mask_cols = (
            (conv_states_input_coord < num_cache_lines)
            & mask_feats[:, None]
            & (idx_cols < KERNEL_WIDTH - 1)[None, :]
        )
        cols = tl.load(conv_state_ptrs_cols, mask_cols, other=0.0)

        # STEP 2: assume state_len > seqlen
        idx_tokens = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

        # With speculative decoding, the conv_state updates works in a sliding
        # window manner, at each forward pass, the tokens are shift by 1, so we
        # load since idx_tokens + 1.
        conv_state_ptrs_source = (
            conv_state_ptr
            + (conv_states_input_coord * stride_conv_state_seq)
            + conv_state_token_offset * stride_conv_state_tok
            + (idx_feats * stride_conv_state_dim)[None, :]
            + ((idx_tokens + seqlen) * stride_conv_state_tok)[
                :, None
            ]
        )  # [BLOCK_M, x_dim]
        mask = (
            (conv_states_input_coord < num_cache_lines)
            & ((idx_tokens + seqlen) < state_len)[:, None]
            & mask_feats[None, :]
        )
        conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

        tl.debug_barrier()

        new_conv_state = tl.where(mask, conv_state, loaded_x[None, :])

        # Get the state from the initial_state_idx
        # cache_idx
        conv_states_offset = tl.load(
            conv_state_indices_ptr + idx_seq * stride_state_indices + current_last_index
        ).to(tl.int64)
        conv_state_ptrs_target = (
            conv_state_ptr
            + (conv_states_offset * stride_conv_state_seq)  # Offset from seq
            + (idx_feats * stride_conv_state_dim)
        )[None, :] + (  # [,x_dim]
            idx_tokens * stride_conv_state_tok
        )[:, None]
        mask = (idx_tokens < state_len)[:, None] & mask_feats[None, :]
        tl.store(conv_state_ptrs_target, new_conv_state, mask)

        # STEP 3: init accumulator, not necessary

        # STEP 4:
        # PRE-LOAD WEIGHTS

        w_cols_ptrs = w_ptr + (idx_feats * stride_w_dim)[:, None] + (idx_cols * stride_w_width)[None, :]
        mask_w_cols = mask_feats[:, None] & (idx_cols < KERNEL_WIDTH - 1)[None, :]
        w_cols = tl.load(w_cols_ptrs, mask_w_cols, other=0.0)  # [x_dim, NP2_STATELEN]

        w_last_ptrs = w_ptr + (idx_feats * stride_w_dim) + (KERNEL_WIDTH - 1) * stride_w_width
        w_last = tl.load(w_last_ptrs, mask_feats, other=0.0) # [x_dim]

        acc = tl.sum((w_cols * cols).to(tl.float32), axis=1) + (w_last * loaded_x).to(tl.float32)

        if HAS_BIAS:
            bias = bias_ptr + idx_feats
            acc += tl.load(bias, mask=mask_feats, other=0.0).to(
                tl.float32
            )  # [x_dim]

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))

        acc = acc.to(original_x_dtype).to(tl.float32)

    return acc

@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,
        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def fused_causal_conv1d_update_rearrange_recurrent_gated_delta_rule_kernel(
    qkv,
    weight,  # (dim, width)
    bias,
    conv_state,
    conv_state_indices,
    g,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    scale,
    N: tl.int64,  # num of sequences
    T: tl.int64,  # num of tokens
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    pad_slot_id: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_qkv_l: tl.constexpr,
    stride_qkv_hd: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    INPLACE_FINAL_STATE: tl.constexpr,  # whether to store final state inplace
    IS_BETA_HEADWISE: tl.constexpr,  # whether beta is headwise vector or scalar,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_KDA: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if T == 0:
        # no tokens to process for this sequence
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = qkv + bos * stride_qkv_l + ((i_h * K) + o_k) * stride_qkv_hd
    p_k = qkv + bos * stride_qkv_l + (H * K + (i_h * K) + o_k) * stride_qkv_hd
    p_v = qkv + bos * stride_qkv_l + (2 * H * K + (i_hv * V) + o_v) * stride_qkv_hd

    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv

    if not IS_KDA:
        p_g = g + bos * HV + i_hv
    else:
        p_gk = g + (bos * HV + i_hv) * K + o_k

    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
            # Load state index and check for PAD_SLOT_ID (-1)
            state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(
                tl.int64
            )
            # Skip if state index is invalid (PAD_SLOT_ID = -1)
            if state_idx < 0:
                return
            p_h0 = h0 + state_idx * stride_init_state_token
        else:
            p_h0 = h0 + bos * HV * V * K
        p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in range(0, T):
        # load and do causal_conv1d_update
        idx_seq = bos + i_t
        b_q = tl.load(p_q, mask=mask_k, other=0) # [np2_K,]
        idx_feats = i_h * K + o_k
        b_q = _causal_conv1d_update_inner_kernel(
            b_q,
            weight,
            bias,
            conv_state,
            conv_state_indices,
            None, # num_accepted_tokens
            None, # query_start_loc_ptr
            None, # block_idx_last_scheduled_token
            None, # initial_state_idx
            idx_seq,
            idx_feats,
            mask_k,
            state_len,
            num_cache_lines,
            stride_w_dim,
            stride_w_width,
            stride_conv_state_seq,
            stride_conv_state_dim,
            stride_conv_state_tok,
            stride_state_indices,
            pad_slot_id,
            HAS_BIAS,
            KERNEL_WIDTH,
            SILU_ACTIVATION,
            NP2_STATELEN,
            USE_PAD_SLOT
        )

        b_k = tl.load(p_k, mask=mask_k, other=0) # [np2_K,]
        idx_feats = H * K + (i_h * K) + o_k
        b_k = _causal_conv1d_update_inner_kernel(
            b_k,
            weight,
            bias,
            conv_state,
            conv_state_indices,
            None, # num_accepted_tokens
            None, # query_start_loc_ptr
            None, # block_idx_last_scheduled_token
            None, # initial_state_idx
            idx_seq,
            idx_feats,
            mask_k,
            state_len,
            num_cache_lines,
            stride_w_dim,
            stride_w_width,
            stride_conv_state_seq,
            stride_conv_state_dim,
            stride_conv_state_tok,
            stride_state_indices,
            pad_slot_id,
            HAS_BIAS,
            KERNEL_WIDTH,
            SILU_ACTIVATION,
            NP2_STATELEN,
            USE_PAD_SLOT
        )

        b_v = tl.load(p_v, mask=mask_v, other=0) # [32,]
        idx_feats = 2 * H * K + (i_hv * V) + o_v
        b_v = _causal_conv1d_update_inner_kernel(
            b_v,
            weight,
            bias,
            conv_state,
            conv_state_indices,
            None, # num_accepted_tokens
            None, # query_start_loc_ptr
            None, # block_idx_last_scheduled_token
            None, # initial_state_idx
            idx_seq,
            idx_feats,
            mask_v,
            state_len,
            num_cache_lines,
            stride_w_dim,
            stride_w_width,
            stride_conv_state_seq,
            stride_conv_state_dim,
            stride_conv_state_tok,
            stride_state_indices,
            pad_slot_id,
            HAS_BIAS,
            KERNEL_WIDTH,
            SILU_ACTIVATION,
            NP2_STATELEN,
            USE_PAD_SLOT
        )

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        # [BV, BK]
        if not IS_KDA:
            b_g = tl.load(p_g).to(tl.float32)
            b_h *= exp(b_g)
        else:
            b_gk = tl.load(p_gk).to(tl.float32)
            b_h *= exp(b_gk[None, :])
        # [BV]
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta
        # [BV, BK]
        b_h += b_v[:, None] * b_k[None, :]
        # [BV]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # keep the states for multi-query tokens
        if INPLACE_FINAL_STATE:
            # Load state index and check for PAD_SLOT_ID (-1)
            final_state_idx = tl.load(
                ssm_state_indices + i_n * stride_indices_seq + i_t
            ).to(tl.int64)
            # Only store if state index is valid (not PAD_SLOT_ID)
            if final_state_idx >= 0:
                p_ht = ht + final_state_idx * stride_final_state_token
                p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
        else:
            p_ht = ht + (bos + i_t) * stride_final_state_token
            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        p_q += stride_qkv_l
        p_k += stride_qkv_l
        p_v += stride_qkv_l

        p_o += HV * V
        if not IS_KDA:
            p_g += HV
        else:
            p_gk += HV * K
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)



def fused_causal_conv1d_update_rearrange_recurrent_gated_delta_rule(
    qkv: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    g: torch.Tensor,
    key_dim: int,
    value_dim: int,
    head_k_dim: int,
    head_v_dim: int,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    validate_data=False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused kernel that combines:
    1. causal_conv1d_update: 1D causal convolution on qkv
    2. rearrange + gated delta rule: recurrent attention computation
    
    This fusion reduces memory traffic by avoiding intermediate tensor materialization.
    Uses a single Triton kernel for decode-only path.
    
    Args:
        qkv (torch.Tensor):
            mixed qkv of shape: `[L, H_qk*D_qk + H_qk*D_qk + H_v*D_v]`.
            queries of shape `[L, H_qk*D_qk]`.
            keys of shape    `[L, H_qk*D_qk]`.
            values of shape  `[L, H_v*D_v]`.
        conv_state: (..., dim, state_len), where state_len >= width - 1, for conv1d
        weight: (dim, width), for conv1d
        bias: (dim,), for conv1d
        conv_state_indices: (batch,), dtype int32, for conv1d
            If not None, the conv_state is a larger tensor along the batch dim,
            and we are selecting the batch coords specified by conv_state_indices.
            Useful for a continuous batching scenario.
        initial_state_idx: (batch,), dtype int32, for conv1d
            The pointer into conv_state_indices, where the cache block containing the initial state is located.
        pad_slot_id: int, for conv1d
                if conv_state_indices is passed, lets the kernel identify padded
                entries that will not be processed,
                for example: conv_state_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
                in this case, the kernel will not process entries at
                indices 0 and 3
        g (torch.Tensor):
            g (decays) of shape `[B, T, HV]`.
        key_dim, value_dim: Dimensions for key and value
        head_k_dim, head_v_dim: Per-head dimensions
        beta (torch.Tensor):
            betas of shape `[B, T, HV]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, HV, V, K]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        inplace_final_state: bool:
            Whether to store the final state in-place to save memory.
            Default: `True`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        ssm_state_indices (Optional[torch.Tensor]):
            Indices to map the input sequences to the initial/final states.
        num_accepted_tokens (Optional[torch.Tensor]):
            Number of accepted tokens for each sequence during decoding.
        use_qk_l2norm_in_kernel: Whether to apply L2 normalization to q, k
        
    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, HV, V, K]`.
    """
    ## for conv1d
    if validate_data:
        assert pad_slot_id is not None
        assert qkv.stride(1) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    batch, dim = qkv.shape
    _, width = weight.shape
    num_cache_lines, _, state_len = conv_state.size()
    if validate_data:
        assert dim == weight.size(0)
        assert conv_state.stride(-2) == 1, (
            f"ERROR: expect contiguous along feat-dim of conv_state (currently stride={conv_state.stride()})"
        )
        assert state_len >= width - 1
        # when above happens, we don't shift-left to keep any records in conv_state
        assert dim == conv_state.size(1)
        if conv_state_indices is None:
            assert conv_state.size(0) >= batch
        else:
            assert (batch,) == conv_state_indices.shape

        assert num_cache_lines >= batch
        assert weight.stride(1) == 1  # Need this

    stride_w_dim, stride_w_width = weight.stride()

    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    assert num_accepted_tokens is None, "num_accepted_tokens should be None in the fused kernel"
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    ## for gated delta rule
    expected_shape = (qkv.shape[0], key_dim * 2 + value_dim)
    assert qkv.shape == expected_shape, f"expect qkv to be in shape {expected_shape}, got {qkv.shape}"
    if scale is None:
        scale = head_k_dim ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    if beta is None:
        h_q = key_dim // head_k_dim
        beta = torch.ones([1, qkv.shape[0], h_q], dtype=qkv.dtype, device=qkv.device)

    # decode-only mode (single token per sequence)
    assert (
        cu_seqlens is not None 
        and batch == len(cu_seqlens) - 1
    ), f"cu_seqlens {str(cu_seqlens)} is not compatible for decode mode"
    
    B = 1
    T = qkv.shape[0]
    H = key_dim // head_k_dim
    HV = value_dim // head_v_dim
    K = head_k_dim
    V = head_v_dim
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    o = qkv.new_empty(NK, B, T, HV, V)
    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = qkv.new_empty(T, HV, V, K, dtype=initial_state.dtype)

    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = final_state.stride(0)

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    stride_qkv_l, stride_qkv_hd = qkv.stride()

    grid = (NK, NV, N * HV)
    fused_causal_conv1d_update_rearrange_recurrent_gated_delta_rule_kernel[grid](
        qkv=qkv,
        ## for conv1d
        weight=weight,
        bias=bias,
        conv_state=conv_state,
        conv_state_indices=conv_state_indices,
        state_len=state_len,
        num_cache_lines=num_cache_lines,
        stride_w_dim=stride_w_dim,
        stride_w_width=stride_w_width,
        stride_conv_state_seq=stride_istate_seq,
        stride_conv_state_dim=stride_istate_dim,
        stride_conv_state_tok=stride_istate_token,
        stride_state_indices=stride_state_indices,
        pad_slot_id=pad_slot_id,
        ## for gated delta rule
        g=g,
        beta=beta,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_qkv_l=stride_qkv_l,
        stride_qkv_hd=stride_qkv_hd,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        IS_BETA_HEADWISE=beta.ndim == 4, # v.ndim
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        INPLACE_FINAL_STATE=inplace_final_state,
        IS_KDA=False,
        # for conv1d META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        # others
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o, final_state

