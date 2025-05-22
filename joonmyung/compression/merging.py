# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
import torch
from typing import Callable, List, Tuple, Union

# A. COMMON   : [use, r_prune, r_merge, r_protected, proportional_attention, trace_source]
# B. MCTF     : [tau_sim, tau_info, tau_size, pooling_type]
# C. VID-TLDR : [mass]
# D. META     : size, attn, source

def token_compression(x, info, layer, others = None):
    if not info["use"]:
        return x, others

    [x, TD] = [x[None], True] if len(x.shape) == 2 else [x, False]

    B, T1, D = x.shape
    r_prune = info["r_prune"][layer] if type(info["r_prune"]) == list else info["r_prune"]
    r_prune = int(T1 * r_prune) if r_prune < 1 else r_prune
    r_prune = max(min(r_prune, T1 // 2, T1 - info["r_protected"]), 0)
    T2 = T1 - r_prune

    r_merge = info["r_merge"][layer] if type(info["r_merge"]) == list else info["r_merge"]
    r_merge = int(T2 * r_merge) if r_merge < 1 else r_merge
    r_merge = max(min(r_merge, T2 // 2, T2 - info["r_protected"]), 0)

    if (not r_prune and not r_merge) and info["r_half"] != layer:
        return x.squeeze(0) if TD else x, others

    scores = info["importance"][None, :, None]

    if info["source"] is None: info["source"] = info["source"] = torch.zeros((B, T1), dtype=torch.bool, device=x.device)
    if info["size"] is None: info["size"] = torch.ones_like(x[..., 0, None]) # (B, T, 1)

    if r_prune or info["r_half"] == layer:
        x, info["source"], others = pruning(x,
                                            r_prune=r_prune,
                                            r_half=info["r_half"],
                                            scores=scores,
                                            source=info["source"],
                                            others = others)
    if r_merge:
        merge = merging(
            x,
            r_merge       = r_merge,
            scores=scores,
            tau_sim       = info["tau_sim"],
            tau_info      = info["tau_info"],
            tau_size      = info["tau_size"],
            mass          = info["mass"],
            size          = info["size"],
        )

        x, info["size"], info["source"] = merge_wavg(merge, x, info["size"], scores, pooling_type=info["pooling_type"], source=info["source"])

    return x.squeeze(0) if TD else x, others

def merging(
        metric  : torch.Tensor,
        r_merge :      int,
        scores  : torch.Tensor,
        tau_sim :      int,
        tau_info:      int,
        tau_size:      int,
        mass:          int,
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

        if tau_info > 0 and scores is not None:
            attn_info = scores
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

        n, t1, t2 = scores.shape
        node_max, node_idx = scores.max(dim=-1)  # (12, 99), (12, 99)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]  # (12, 99, 1)
        unm_idx = edge_idx[..., r_merge:, :]  # Unmerged Tokens (12, 83, 1)
        src_idx = edge_idx[..., :r_merge, :]  # Merged Tokens   (12, 16, 1)
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)  # (12, 16, 1)
        unm_idx = unm_idx.sort(dim=1)[0]

        if mass:
            src_so, dst_so = scores[..., ::2, :], scores[..., 1::2, :]  # (1, 1176, 1)
            src_so = src_so.gather(dim=-2, index=src_idx)  # (12, 91, 197)


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
        merge: Callable, x: torch.Tensor, size: torch.Tensor = None, scores=None, pooling_type = 0, source = None,
    ):

    size_max = size.amax(dim=-2, keepdim=True)
    if pooling_type:
        norm = merge(scores * size, mode="sum") # (1, 197, 1)

        x = merge(x * scores * size, mode="sum")
        size = merge(size, mode="sum")
        x = x / norm
    else:
        x = merge(x * (size / size_max), mode="sum")
        size = merge(size, mode="sum")
        x = x / (size / size_max)

    if source is not None:
        source = merge(source, mode="amax")
    return x, size, source

def pruning(
    x: torch.Tensor,
    r_prune            : int,
    r_half             : int,
    scores             : torch.Tensor,
    source             : torch.Tensor,
    others             : []):
    b, t, d = x.shape
    if r_half: # REMOVE HALF
        scores_block = scores.reshape(-1, 4).mean(dim=-1)
        mask_block = (scores_block >= scores_block.mean(dim=-1)).reshape(1, -1, 1, 1)
        x_block = x.reshape(b, -1, 4, d)
        x_unprune = x_block.masked_select(mask_block).view(b, -1, d)
        if source is not None:
            source = source.masked_select(mask).view(b, -1, source.shape[-1])

        if others is not None:
            cu_lens, rotary_pos_emb = others
            T_remain = mask.sum()
            cu_lens[1] = T_remain
            rotary_pos_emb = rotary_pos_emb.masked_select(mask).view(T_remain, -1)
            others = [cu_lens, rotary_pos_emb]

        # mean_scores = scores.mean(dim=1, keepdim=True)
        # mask = (scores >= mean_scores)
        # x_unprune = x.masked_select(mask).view(b, -1, d)  # (b, t', d)
        # if source is not None:
        #     source = source.masked_select(mask).view(b, -1, source.shape[-1])
        #     # source = torch.gather(source, dim=1, index=idx_unprune.repeat(1, 1, source.shape[-1]))
        # if others is not None:
        #     cu_lens, rotary_pos_emb = others
        #     T_remain = mask.sum()
        #     cu_lens[1] = T_remain
        #     rotary_pos_emb = rotary_pos_emb.masked_select(mask).view(T_remain, -1)
        #     others = [cu_lens, rotary_pos_emb]
    else:
        idx_unprune = scores.topk(t - r_prune, dim=1, largest=True, sorted=False).indices  # (b, t - r_prune)
        x_unprune = torch.gather(x, dim=1, index=idx_unprune.expand(-1, -1, d))
        if source is not None:
            source = torch.gather(source, dim=1, index=idx_unprune.repeat(1, 1, source.shape[-1]))

    x = x_unprune

    return x, source, others



def ANAsimilarity(x1, x2, info, dim=-1, idx = 1):
    diff = torch.norm(x1 - x2, p=2, dim=dim).detach()
    norm1 = torch.norm(x1, p=1, dim=dim).detach()
    norm2 = torch.norm(x1, p=2, dim=dim).detach()
    if info["analysis"]["use"]:
        info["analysis"]["diff"][idx].append(diff.detach())
        info["analysis"]["norm1"][idx].append(norm1.detach())
        info["analysis"]["norm2"][idx].append(norm2.detach())

    if info[f"compression"]["use"]:
        info[f"compression"]["diff"] = diff.detach()
        info[f"compression"]["norm1"] = norm1.detach()
        info[f"compression"]["norm2"] = norm2.detach()


