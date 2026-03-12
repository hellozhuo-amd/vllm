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

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.triton_utils import tl, triton

from .fused_rearrange_recurrent import fused_rearrange_recurrent_gated_delta_rule
from .op import exp

@triton.jit()
def _causal_conv1d_update_inner_kernel(
    # Pointers to matrices
    loaded_x,  # (x_dim,) batch_size and seqlen are both 1, loaded
    w_ptr,  # (dim, width)
    bias_ptr, # (dim,)
    conv_state_ptr, # (num_cache_lines, dim, state_len(>=width -1))
    conv_state_indices_ptr, # (1,)
    num_accepted_tokens_ptr, # None, not used
    query_start_loc_ptr,  # (batch + 1), not used
    block_idx_last_scheduled_token,  # (batch,), not used
    initial_state_idx,  # (batch,), not used
    idx_seq,
    idx_feats, # (x_dim,)
    # o_ptr,  # (x_dim,), to be returned
    # Matrix dimensions
    # batch: int, = 1
    dim: tl.constexpr,
    # seqlen: tl.constexpr, = 1
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    # stride_x_seq: tl.constexpr,
    # stride_x_dim: tl.constexpr,
    # stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    # stride_o_seq: tl.constexpr,
    # stride_o_dim: tl.constexpr,
    # stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    # IS_VARLEN: tl.constexpr, False
    # IS_APC_ENABLED: tl.constexpr, False
    # IS_SPEC_DECODING: tl.constexpr, False
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    # BLOCK_N: tl.constexpr,
):
    # ruff: noqa: E501
    #idx_seq = tl.program_id(0)
    #if idx_seq >= batch:
    #    return

    # [BLOCK_N,] elements along the feature-dimension (channel)
    # idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    seqlen = 1

    # IS_APC_ENABLED is False
    conv_state_init = 0
    current_last_index = 0

    # cache_idx
    conv_states_input_coord = tl.load(
        conv_state_indices_ptr + idx_seq * stride_state_indices + conv_state_init
    ).to(tl.int64)

    if USE_PAD_SLOT:  # noqa
        if conv_states_input_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return loaded_x

    # IS_VARLEN is False
    #query_start_index = idx_seq * seqlen
    #query_end_index = query_start_index + seqlen
    #x_offset = idx_seq * stride_x_seq
    #o_offset = idx_seq * stride_o_seq

    #if query_start_index == query_end_index:
    #    return

    # IS_SPEC_DECODING is False
    conv_state_token_offset = 0

    # STEP 1: READ init_state data
    idx_cols = tl.arange(0, KERNEL_WIDTH-1)
    conv_state_ptrs_cols = (
        conv_state_ptr
        + (conv_states_input_coord * stride_conv_state_seq)
        + conv_state_token_offset * stride_conv_state_tok
        + (idx_feats * stride_conv_state_dim)[None, :]
        + (idx_cols * stride_conv_state_tok)[:, None]
    )  # [KERNEL_WIDTH-1, x_dim]
    mask_cols = (
        (conv_states_input_coord < num_cache_lines)
        & (idx_feats < dim)[None, :]
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
        & (idx_feats < dim)[None, :]
    )
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

    #VAL = state_len - seqlen
    #x_base = x_ptr + x_offset + (idx_feats * stride_x_dim)  # [BLOCK_N]

    #x_ptrs = (
    #    x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
    #)  # [BLOCK_M, BLOCK_N]

    #mask_x = (
    #    (idx_tokens - VAL >= 0)[:, None]
    #    & (idx_tokens - VAL < seqlen)[:, None]
    #    & (idx_feats < dim)[None, :]
    #)  # token-index  # token-index  # feature-index
    #loaded_x = tl.load(x_ptrs, mask_x, 0.0)
    tl.debug_barrier()

    new_conv_state = tl.where(mask, conv_state, loaded_x)

    # Get the state from the initial_state_idx
    # cache_idx
    conv_states_offset = tl.load(
        conv_state_indices_ptr + idx_seq * stride_state_indices + current_last_index
    ).to(tl.int64)
    conv_state_ptrs_target = (
        conv_state_ptr
        + (conv_states_offset * stride_conv_state_seq)  # Offset from seq
        + (idx_feats * stride_conv_state_dim)
    )[None, :] + (  # [BLOCK_N,]
        idx_tokens * stride_conv_state_tok
    )[:, None]
    mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask)

    # STEP 3: init accumulator
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
            tl.float32
        )  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # STEP 4:
    # PRE-LOAD WEIGHTS
    # first kernel column, configured for weights to handle BLOCK_N features in range
    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 5:
        w_ptrs = w_base + (4 * stride_w_width)  # [BLOCK_N] tensor
        w_col4 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 6:
        w_ptrs = w_base + (5 * stride_w_width)  # [BLOCK_N] tensor
        w_col5 = tl.load(w_ptrs, mask_w, other=0.0)

    x_base_1d = x_base  # starting of chunk [BLOCK_N]
    mask_x_1d = idx_feats < dim

    # STEP 5: compute each token
    for idx_token in tl.range(seqlen):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 5:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 6:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    matrix_x = col4
                elif j == 5:
                    matrix_w = w_col5
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w  # [BLOCK_N]

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x
        elif KERNEL_WIDTH == 5:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = matrix_x
        elif KERNEL_WIDTH == 6:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = col4
            col4 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < seqlen) & (
            idx_feats < dim
        )  # token-index  # feature-index
        o_ptrs = (
            o_ptr + o_offset + idx_token * stride_o_token + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, acc, mask=mask_1d)



@triton.jit
def _apply_conv1d(
    p_input,
    p_conv_base,
    conv_weight,
    conv_bias,
    feature_offset,
    mask,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_conv_weight_dim: tl.constexpr,
    stride_conv_weight_w: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    BF: tl.constexpr,
    CONV_WIDTH: tl.constexpr,
    STATE_LEN: tl.constexpr,
):
    """
    Apply conv1d with sliding window to a feature dimension.
    Returns the convolved output and updates conv_state in-place.
    """
    p_conv_feat = p_conv_base + feature_offset * stride_conv_state_dim
    
    # Initialize accumulator with bias
    acc = tl.zeros((BF,), dtype=tl.float32)
    if HAS_BIAS:
        p_bias = conv_bias + feature_offset
        acc += tl.load(p_bias, mask=mask, other=0.0)
    
    p_weight = conv_weight + feature_offset * stride_conv_weight_dim
    
    # Compute convolution with sliding window
    for i in tl.static_range(0, CONV_WIDTH):
        if i < CONV_WIDTH - 1:
            # Load from conv_state history
            state_idx = STATE_LEN - (CONV_WIDTH - 1) + i
            p_state_i = p_conv_feat + state_idx * stride_conv_state_tok
            state_val = tl.load(p_state_i, mask=mask, other=0.0)
        else:
            # Load current input
            state_val = tl.load(p_input, mask=mask, other=0.0)
        
        w_i = tl.load(p_weight + i * stride_conv_weight_w, mask=mask, other=0.0)
        acc += state_val * w_i
    
    # Apply activation
    if SILU_ACTIVATION:
        acc = acc * tl.sigmoid(acc)
    
    # Update conv state (shift left and append new value)
    for i in tl.static_range(0, STATE_LEN - 1):
        if i < STATE_LEN - 1:
            p_src = p_conv_feat + (i + 1) * stride_conv_state_tok
            p_dst = p_conv_feat + i * stride_conv_state_tok
            val = tl.load(p_src, mask=mask, other=0.0)
            tl.store(p_dst, val, mask=mask)
    
    # Store new value at end of conv state
    p_new = p_conv_feat + (STATE_LEN - 1) * stride_conv_state_tok
    new_val = tl.load(p_input, mask=mask, other=0.0)
    tl.store(p_new, new_val, mask=mask)
    
    return acc


@triton.jit
def _fused_conv1d_recurrent_decode_kernel(
    # Inputs
    qkv,  # [batch, key_dim*2 + value_dim]
    conv_state,  # [num_cache_lines, dim, state_len]
    conv_weight,  # [dim, width]
    conv_bias,  # [dim] or None
    g,  # [1, batch, HV]
    beta,  # [1, batch, HV]
    conv_state_indices,  # [batch]
    # Output
    output,  # [batch, HV, V]
    ssm_state,  # [num_states, HV, V, K]
    ssm_state_indices,  # [batch]
    # Dimensions
    batch,
    key_dim,
    value_dim,
    HV,
    K,
    V,
    scale,
    # Strides
    stride_qkv_b,
    stride_qkv_d,
    stride_conv_state_seq,
    stride_conv_state_dim,
    stride_conv_state_tok,
    stride_conv_weight_dim,
    stride_conv_weight_w,
    stride_g_b,
    stride_g_h,
    stride_beta_b,
    stride_beta_h,
    stride_o_b,
    stride_o_h,
    stride_o_v,
    stride_ssm_n,
    stride_ssm_h,
    stride_ssm_v,
    stride_ssm_k,
    # Meta params
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    CONV_WIDTH: tl.constexpr,
    STATE_LEN: tl.constexpr,
):
    """
    Fused kernel for decode-only (single token) path that combines:
    1. Causal conv1d update (with width-element sliding window)
    2. Rearrange qkv 
    3. Recurrent gated delta rule update
    
    Each program processes one (batch_idx, head_v_idx) combination.
    """
    pid_b = tl.program_id(0)  # batch index
    pid_hv = tl.program_id(1)  # value head index
    
    if pid_b >= batch or pid_hv >= HV:
        return
    
    # Get head-k index (assuming HV >= H and they're grouped)
    H = key_dim // K
    pid_h = pid_hv * H // HV  # key head index
    
    # Load conv state index
    conv_idx = tl.load(conv_state_indices + pid_b).to(tl.int64)
    if conv_idx < 0:  # PAD_SLOT_ID
        return
    
    # ============================================================
    # Step 1: Causal Conv1D Update
    # ============================================================
    # We process K, K (for q and k), and V dimensions separately
    # This is for single token, so we use sliding window convolution
    
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    
    # Pointers to qkv input for this batch
    p_q_in = qkv + pid_b * stride_qkv_b + (pid_h * K + o_k) * stride_qkv_d
    p_k_in = qkv + pid_b * stride_qkv_b + (key_dim + pid_h * K + o_k) * stride_qkv_d
    p_v_in = qkv + pid_b * stride_qkv_b + (key_dim * 2 + pid_hv * V + o_v) * stride_qkv_d
    
    # Base pointer to conv state for this sequence
    p_conv_base = (
        conv_state + conv_idx * stride_conv_state_seq
    )
    
    # ============================================================
    # Apply Conv1D to Q, K, V using the separate function
    # ============================================================
    b_q = _apply_conv1d(
        p_q_in,
        p_conv_base,
        conv_weight,
        conv_bias,
        pid_h * K + o_k,
        mask_k,
        stride_conv_state_dim,
        stride_conv_state_tok,
        stride_conv_weight_dim,
        stride_conv_weight_w,
        HAS_BIAS,
        SILU_ACTIVATION,
        BK,
        CONV_WIDTH,
        STATE_LEN,
    )
    
    b_k = _apply_conv1d(
        p_k_in,
        p_conv_base,
        conv_weight,
        conv_bias,
        key_dim + pid_h * K + o_k,
        mask_k,
        stride_conv_state_dim,
        stride_conv_state_tok,
        stride_conv_weight_dim,
        stride_conv_weight_w,
        HAS_BIAS,
        SILU_ACTIVATION,
        BK,
        CONV_WIDTH,
        STATE_LEN,
    )
    
    b_v = _apply_conv1d(
        p_v_in,
        p_conv_base,
        conv_weight,
        conv_bias,
        key_dim * 2 + pid_hv * V + o_v,
        mask_v,
        stride_conv_state_dim,
        stride_conv_state_tok,
        stride_conv_weight_dim,
        stride_conv_weight_w,
        HAS_BIAS,
        SILU_ACTIVATION,
        BV,
        CONV_WIDTH,
        STATE_LEN,
    )
    
    # ============================================================
    # Step 2: QK L2 Normalization (optional)
    # ============================================================
    if USE_QK_L2NORM:
        q_norm = tl.sqrt(tl.sum(b_q * b_q, axis=0) + 1e-6)
        k_norm = tl.sqrt(tl.sum(b_k * b_k, axis=0) + 1e-6)
        b_q = b_q / q_norm
        b_k = b_k / k_norm
    
    # Apply scale to query
    b_q = b_q * scale
    
    # ============================================================
    # Step 3: Recurrent Gated Delta Rule Update
    # ============================================================
    # Load SSM state for this head
    ssm_idx = tl.load(ssm_state_indices + pid_b).to(tl.int64)
    if ssm_idx < 0:  # PAD_SLOT_ID
        return
    
    p_h = (
        ssm_state 
        + ssm_idx * stride_ssm_n
        + pid_hv * stride_ssm_h
        + o_v[:, None] * stride_ssm_v
        + o_k[None, :] * stride_ssm_k
    )
    
    mask_h = mask_v[:, None] & mask_k[None, :]
    b_h = tl.load(p_h, mask=mask_h, other=0.0).to(tl.float32)
    
    # Load gating and beta
    b_g = tl.load(g + pid_b * stride_g_b + pid_hv * stride_g_h).to(tl.float32)
    b_beta = tl.load(beta + pid_b * stride_beta_b + pid_hv * stride_beta_h).to(tl.float32)
    
    # Apply gated delta rule update
    # h *= exp(g)
    b_h = b_h * tl.exp(b_g)
    
    # v -= sum(h * k, dim=K)
    b_v_update = b_v - tl.sum(b_h * b_k[None, :], axis=1)
    b_v_update = b_v_update * b_beta
    
    # h += v[:, None] * k[None, :]
    b_h = b_h + b_v_update[:, None] * b_k[None, :]
    
    # o = sum(h * q, dim=K)
    b_o = tl.sum(b_h * b_q[None, :], axis=1)
    
    # ============================================================
    # Step 4: Store outputs
    # ============================================================
    # Store output
    p_o = output + pid_b * stride_o_b + pid_hv * stride_o_h + o_v * stride_o_v
    tl.store(p_o, b_o.to(output.dtype.element_ty), mask=mask_v)
    
    # Store updated SSM state
    tl.store(p_h, b_h.to(ssm_state.dtype.element_ty), mask=mask_h)


def fused_causal_conv1d_update_rearrange_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_actual_tokens: int,
    conv_state: torch.Tensor,  ## for conv1d
    weight: torch.Tensor,  ## for conv1d
    bias: torch.Tensor | None,  ## for conv1d
    activation: bool | str | None,  ## for conv1d
    conv_state_indices: torch.Tensor | None,  ## for conv1d
    validate_data,  ## for conv1d
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused kernel that combines:
    1. causal_conv1d_update: 1D causal convolution on qkv
    2. rearrange + gated delta rule: recurrent attention computation
    
    This fusion reduces memory traffic by avoiding intermediate tensor materialization.
    Uses a single Triton kernel for decode-only path, falls back to sequential for prefill.
    
    Args:
        q, k, v: Query, key, value tensors of shape [num_tokens, key_dim//tp_size], etc.
        num_actual_tokens: Number of valid tokens in the batch
        conv_state: Convolution state cache
        weight: Conv1d weight
        bias: Conv1d bias
        activation: Activation function for conv1d ('silu', 'swish', or None)
        conv_state_indices: Indices for batch gathering in conv_state
        validate_data: Whether to validate inputs
        g: Gating/decay tensor
        key_dim, value_dim: Dimensions for key and value
        head_k_dim, head_v_dim: Per-head dimensions
        beta: Beta scaling tensor
        scale: Attention scale factor
        initial_state: Initial SSM state
        inplace_final_state: Whether to update state in-place
        cu_seqlens: Cumulative sequence lengths
        ssm_state_indices: SSM state indices
        num_accepted_tokens: Number of accepted tokens (for speculative decoding)
        use_qk_l2norm_in_kernel: Whether to apply L2 normalization to q, k
        
    Returns:
        output: Attention output
        final_state: Updated SSM state
    """
    batch_size = num_actual_tokens
    H = key_dim // head_k_dim
    HV = value_dim // head_v_dim
    K = head_k_dim
    V = head_v_dim
    
    if scale is None:
        scale = K ** -0.5
    
    # Determine if we're in decode-only mode (single token per sequence)
    is_decode_only = (
        cu_seqlens is not None 
        and batch_size == len(cu_seqlens) - 1
        and all((cu_seqlens[i+1] - cu_seqlens[i]) == 1 for i in range(batch_size))
    )
    
    # For decode-only path with single tokens, use fused Triton kernel
    if is_decode_only and batch_size > 0:
        # Prepare input (concatenate q, k, v)
        mixed_qkv = torch.cat((q, k, v), dim=-1)
        mixed_qkv = mixed_qkv[:num_actual_tokens]
        
        # Prepare output
        output = torch.empty(
            (batch_size, HV, V),
            dtype=q.dtype, 
            device=q.device
        )
        
        # Setup kernel launch parameters
        conv_width = weight.shape[1]
        state_len = conv_state.shape[2]
        
        BK = triton.next_power_of_2(K)
        BV = min(triton.next_power_of_2(V), 32)
        
        grid = (batch_size, HV)
        
        # Launch fused kernel
        _fused_conv1d_recurrent_decode_kernel[grid](
            # Inputs
            mixed_qkv,
            conv_state,
            weight,
            bias,
            g,
            beta,
            conv_state_indices,
            # Outputs
            output,
            initial_state,  # SSM state (will be updated in-place)
            ssm_state_indices,
            # Dimensions
            batch_size,
            key_dim,
            value_dim,
            HV,
            K,
            V,
            scale,
            # Strides - qkv
            mixed_qkv.stride(0),
            mixed_qkv.stride(1),
            # Strides - conv_state
            conv_state.stride(0),
            conv_state.stride(1),
            conv_state.stride(2),
            # Strides - conv_weight
            weight.stride(0),
            weight.stride(1),
            # Strides - g, beta
            g.stride(1),
            g.stride(2),
            beta.stride(1),
            beta.stride(2),
            # Strides - output
            output.stride(0),
            output.stride(1),
            output.stride(2),
            # Strides - SSM state
            initial_state.stride(0),
            initial_state.stride(1),
            initial_state.stride(2),
            initial_state.stride(3),
            # Meta params
            HAS_BIAS=(bias is not None),
            SILU_ACTIVATION=(activation in ['silu', 'swish']),
            USE_QK_L2NORM=use_qk_l2norm_in_kernel,
            BK=BK,
            BV=BV,
            CONV_WIDTH=conv_width,
            STATE_LEN=state_len,
            num_warps=4,
            num_stages=2,
        )
        
        # Reshape output to match expected format: [1, L, HV, V]
        output = output.view(1, batch_size, HV, V)
        return output, initial_state
    
    else:
        # For prefill or multi-token decode, fall back to sequential kernels
        # This path handles variable-length sequences and multi-token processing
        
        # Step 1: Concatenate q, k, v to create mixed_qkv
        mixed_qkv = torch.cat((q, k, v), dim=-1)
        mixed_qkv_sliced = mixed_qkv[:num_actual_tokens]
        
        # Step 2: Apply causal conv1d update
        mixed_qkv_conv = causal_conv1d_update(
            mixed_qkv_sliced,
            conv_state,
            weight,
            bias,
            activation,
            conv_state_indices=conv_state_indices,
            validate_data=validate_data,
        )
        
        # Step 3: Apply rearrange + gated delta rule
        output, final_state = fused_rearrange_recurrent_gated_delta_rule(
            qkv=mixed_qkv_conv,
            g=g,
            key_dim=key_dim,
            value_dim=value_dim,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        
        return output, final_state

