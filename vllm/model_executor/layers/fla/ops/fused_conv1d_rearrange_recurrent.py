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

from .op import exp

def fused_causal_conv1d_update_rearrange_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_actual_tokens: int,
    conv_state: torch.Tensor, ## for conv1d
    weight: torch.Tensor, ## for conv1d
    bias: torch.Tensor | None, ## for conv1d
    activation: bool | str | None, ## for conv1d
    conv_state_indices: torch.Tensor | None, ## for conv1d
    validate_data, ## for conv1d
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

    ### first, causal conv1d update

    ### second, rearrange + gated delta rule

    raise NotImplementedError("fused_causal_conv1d_update_rearrange_recurrent_gated_delta_rule not implemented yet")
