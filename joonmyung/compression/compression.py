# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
import torch
from typing import Callable, List, Tuple, Union

# A. COMMON   : [use, applied_layer, r_prune, r_merge, r_protected, proportional_attention, trace_source]
# B. MCTF     : [tau_sim, tau_info, tau_size, pooling_type]
# C. VID-TLDR : [mass]
# D. META     : size, attn, source

def token_compression(x, info, layer):
    if not info["use"] or layer not in info["applied_layer"]:
        return x

    T = x.shape[1]
    r_prune = info["r_prune"][layer] if type(info["r_prune"]) == list else info["r_prune"]
    r_prune = int(T * r_prune) if r_prune < 1 else r_prune
    r_prune = max(min(r_prune, T // 2, T - info["r_protected"]), 0)
    T = T - r_prune

    r_merge = info["r_merge"][layer] if type(info["r_merge"]) == list else info["r_merge"]
    r_merge = int(T * r_merge) if r_merge < 1 else r_merge
    r_merge = max(min(r_merge, T // 2, T - info["r_protected"]), 0)

    if not r_prune and not r_merge:
        return x

    if info["size"] is None: info["size"] = torch.ones_like(x[..., 0, None])
    merge = merging(
        x,
        r_merge       = r_merge,
        r_cls         = info["r_cls"],
        tau_sim       = info["tau_sim"],
        tau_info      = info["tau_info"],
        tau_size      = info["tau_size"],
        mass          = info["mass"],
        attn          = info["attn"],
        size          = info["size"],
    )

    if info["trace_source"]:
        info["source"] = merge_source(
            merge, x, info["source"]
        )

    x, info["size"] = merge_wavg(merge, x, info["size"], info["attn"], pooling_type=info["pooling_type"])
    return x

def merging(
        metric : torch.Tensor,
        r_merge:       int,
        r_cls:         int,
        tau_sim:       int,
        tau_info:      int,
        tau_size:      int,
        mass:          int,
        attn:      torch.Tensor,
        size:      torch.Tensor):

    B, T, _ = metric.shape  # (4(B), 197(T), 384(4))
    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)  # (12, 197, 64)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]  # (12, 99, 64), (12, 98, 64)

        if tau_sim:
            W_sim = a @ b.transpose(-1, -2)
            W_sim = ((W_sim + 1) / 2) ** (1 / tau_sim)
        else:
            W_sim = torch.ones((a.shape[0], a.shape[1], b.shape[1]), device=a.device)

        if tau_info > 0 and attn is not None:
            attn_info = attn[:, :, 0].mean(dim=1)  if r_cls else attn.mean(dim=[1,2]) # (3, 1024)
            attn_info = 1 / attn_info # (1(B), 1024(T))
            attn_info = attn_info / attn_info.max(1, keepdim=True)[0] # (192(B), 197(T))
            attn_a, attn_b = attn_info[..., ::2, None], attn_info[..., 1::2, None].transpose(1, 2)

            W_info = (attn_a * attn_b) ** (1 / tau_info)
        else:
            W_info = 1

        if tau_size and size is not None:
            size_info = 1 / size
            size_info = size_info / size_info.max(1, keepdim=True)[0]  # (4(B), 197(T), 1)
            size_a, size_b = size_info[..., ::2, :], size_info[..., 1::2, :].transpose(1, 2)

            W_size = (size_a * size_b) ** (1 / tau_size)
        else:
            W_size = 1

        scores = W_sim * W_info * W_size
        if r_cls:
            scores[..., :r_cls, :] = -math.inf

        n, t1, t2 = scores.shape
        node_max, node_idx = scores.max(dim=-1)  # (12, 99), (12, 99)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]  # (12, 99, 1)
        unm_idx = edge_idx[..., r_merge:, :]  # Unmerged Tokens (12, 83, 1)
        src_idx = edge_idx[..., :r_merge, :]  # Merged Tokens   (12, 16, 1)
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)  # (12, 16, 1)
        unm_idx = unm_idx.sort(dim=1)[0]

        if mass:
            obj_score = get_objective_score(attn, r_cls)
            src_so, dst_so = obj_score[..., ::2, :], obj_score[..., 1::2, :]  # (1, 1176, 1)
            src_so = src_so.gather(dim=-2, index=src_idx)  # (12, 91, 197)

        if r_cls:
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]  # (12, 99, 197), (12, 98, 197)
        n, mid, c = src.shape[0], src.shape[1:-2], src.shape[-1]
        unm = src.gather(dim=-2, index=unm_idx.expand(n, *mid, t1 - r_merge, c))  # (12, 91, 197)
        src = src.gather(dim=-2, index=src_idx.expand(n, *mid, r_merge, c))
        if mass:
            src = src * src_so
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, *mid, r_merge, c), src, reduce=mode)  # (12, 98, 197)
        x = torch.cat([unm, dst], dim=-2)  # (12, 1 + 180, 197)
        return x

    return merge


def merge_wavg(
        merge: Callable, x: torch.Tensor, size: torch.Tensor = None, attn=None, pooling_type = 0,
    ):

    size_max = size.amax(dim=-2, keepdim=True)
    if pooling_type:
        attn_m = attn.mean(dim=[1, 2]).unsqueeze(-1)
        norm = merge(attn_m * size, mode="sum") # (1, 197, 1)

        x = merge(x * attn_m * size, mode="sum")
        size = merge(size, mode="sum")
        x = x / norm
    else:
        x = merge(x * (size / size_max), mode="sum")
        size = merge(size, mode="sum")
        x = x / (size / size_max)

    return x, size

@torch.no_grad()
def merge_source(
        merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)  # (12, 197, 197)

    source = merge(source, mode="amax")
    return source


# PART B. VID-TLDR
def get_objective_score(score_attn, r_cls):
    score_attn = score_attn.mean(dim=1)
    scores = (score_attn * torch.log(score_attn)).sum(dim=2).unsqueeze(-1)

    # BACKGROUND REMOVING
    B, T_R, _ = scores.shape
    scores = scores - scores.amin(dim=1, keepdim=True)
    scores = scores / scores.amax(dim=1, keepdim=True)
    score_mask = scores < scores.mean(dim=1, keepdim=True)

    # FOREGROUND SHARPENING
    scores = scores - scores.mean(dim=1, keepdim=True)
    scores = scores / scores.amax(dim=1, keepdim=True)
    scores[score_mask] = 0.0

    return scores




def parse_r(num_layers: int, r: Union[List[int], Tuple[int, float], int]) -> List[int]:
    inflect = 0
    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return list(r)
    elif isinstance(r, tuple):
        r, inflect = r

    min_val = int(r * (1.0 - inflect))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / (num_layers - 1)

    return [int(min_val + step * i) for i in range(num_layers)]



