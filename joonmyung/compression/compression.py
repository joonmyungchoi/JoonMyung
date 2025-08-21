# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------


from typing import Callable
import torch
import math

def token_compression(x, info, layer_idx, others = []):
    [x, TD] = [x[None], True] if len(x.shape) == 2 else [x, False]
    B, T, D = x.shape
    if not info["use"] or T == 1:
        return x.squeeze(0) if TD else x, others

    T_vis = T if info["img_idx"][0] == None else info["img_idx"][1] - info["img_idx"][0]
    r_use, thr_use, ent_use = (info["prune_r_layer"] == layer_idx and info["prune_r"]), (info["prune_thr_layer"] == layer_idx and info["prune_thr"]), \
                              (info["entroPrune"](T_vis, info["entropy"], layer_idx) if info["entroPrune"] is not None else 0)

    if (r_use or thr_use or ent_use):
        prune_r, prune_thr = None, None
        if r_use: prune_r = int(T_vis * info["prune_r"]) if info["prune_r"] < 1 else info["prune_r"]
        if thr_use: prune_thr = info["prune_thr"]
        if ent_use: prune_r = ent_use

        scores = info["importance"]
        if info["source"] is None: info["source"] = torch.ones((B, (T // info["group_num"]) ), dtype=torch.bool, device=x.device)
        if info["size"] is None: info["size"] = torch.ones_like(x[..., 0, None]) # (B, T, 1)

        x, info["source"], others = pruning(x,
                                            prune_r=prune_r,
                                            prune_thr=prune_thr,
                                            scores=scores,
                                            source=info["source"],
                                            cls=info["cls"],
                                            group_num=info["group_num"],
                                            SE = info["img_idx"],
                                            others = others)

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
    prune_r            : int,
    prune_thr           : float,
    scores             : torch.Tensor,
    source             : torch.Tensor,
    cls                : False,
    group_num          : int = 1,
    others             : [] = None,
    SE                 : [] = None):
    b, t_full, d = x.shape
    scores_block = scores.reshape(b, -1, group_num).mean(dim=-1) # (B, T)
    scores_block = scores_block / scores_block.mean(dim=-1, keepdim=True)
    t_vis = scores_block.shape[1]

    if cls: scores_block[:, 0] = math.inf

    x_block = x.reshape(b, -1, group_num, d)
    if prune_thr: # REMOVE BASED THRESHOLD
        mask_block = (scores_block >= prune_thr)
    else:
        idx_unprune = scores_block.topk(t_vis - int(prune_r // group_num), dim=1, largest=True, sorted=False).indices
        mask_block = torch.zeros_like(scores_block, dtype=torch.bool)
        mask_block = mask_block.scatter(1, idx_unprune, torch.ones_like(idx_unprune, device=idx_unprune.device, dtype=torch.bool))

    if SE[0] is not None:
        start, end, length = SE

        mask_F, mask_L = torch.ones((b, start), device=mask_block.device, dtype=torch.bool), torch.ones(b, t_full - end, device=mask_block.device, dtype=torch.bool)
        mask_block = torch.cat([mask_F, mask_block, mask_L], dim =-1)
        t_num = mask_block.sum().item()
        SE[1], SE[2] = t_num - (t_full - end), t_num

    x_unprune = x_block.masked_select(mask_block.reshape(1, -1, 1, 1)).view(b, -1, d)  # (1, 10032(T), 1280) > (1, 4880(T'), 1280)

    if others is not None:
        T_remain = x_unprune.shape[-2]
        if len(others) == 1:
            cu_lens = others[0]
            if cu_lens is not None: cu_lens[1] = T_remain
            others = [cu_lens]
        elif len(others) == 2:
            cu_lens, rotary_pos_emb = others
            cu_lens[1] = T_remain
            rotary_pos_emb = rotary_pos_emb.reshape(-1, group_num, 40).masked_select(mask_block.reshape(-1, 1, 1)).view(-1, 40)
            others = [cu_lens, rotary_pos_emb]
        elif len(others) == 3: # LLM
            attention_mask, position_ids, cache_position = others
            attention_mask = attention_mask[:, :, :T_remain, :T_remain] if attention_mask is not None else None
            position_ids = position_ids.masked_select(mask_block.reshape(b, 1, -1)).reshape(3, 1, -1)
            cache_position = cache_position.masked_select(mask_block)
            others = [attention_mask, position_ids, cache_position]
        else: # LLM
            attention_mask, position_ids, cache_position, position_embeddings = others
            attention_mask = attention_mask[:, :, :T_remain, :T_remain] if attention_mask is not None else None
            position_ids = position_ids.masked_select(mask_block.reshape(b, 1, -1)).reshape(3, 1, -1)
            cache_position = cache_position.masked_select(mask_block)
            position_embeddings = tuple([v.masked_select(mask_block.reshape(1, 1, -1, 1)).reshape(3, 1, -1, 128) for v in position_embeddings])
            others = [attention_mask, position_ids, cache_position, position_embeddings]

    if source is not None:
        restored_mask = torch.zeros_like(source, device=source.device)
        restored_mask[source] = mask_block
        source = restored_mask


    x = x_unprune

    return x, source, others

def needNaive(info, layer_idx):
    if info["compression"]["use"]:
        if info["compression"]["info_type"] in [1, 2, 3, 4]:
            if (info["compression"]["prune_r"] and info["compression"]["prune_r_layer"] == layer_idx) or \
                    (info["compression"]["prune_thr"] and info["compression"]["prune_thr_layer"] == layer_idx):
                return True
    return False

def needAttn(info, layer_idx):
    if info["compression"]["use"]:
        if info["compression"]["info_type"] in [1, 2, 3, 4]:
            if (info["compression"]["prune_r"] and info["compression"]["prune_r_layer"] == layer_idx) or \
                    (info["compression"]["prune_thr"] and info["compression"]["prune_thr_layer"] == layer_idx):
                return True
    return False


class EntroDropScheduler:
    def __init__(self, start_layer, drop_rate):
        self.drop_rate = torch.as_tensor(drop_rate, dtype=torch.float32)
        self.K = len(drop_rate)
        self.bins = torch.linspace(0, 10, steps=self.K + 1)  # 오름차순 bins
        self.start_layer = start_layer
        self.tokens = []
    def reset(self):
        self.bin_used = torch.zeros(self.K, dtype=torch.bool)
        self.tokens = []

    @torch.no_grad()
    def __call__(self, T, entropy, layer):

        if layer < self.start_layer or entropy == None:
            self.tokens.append(T)
            return 0

        if not hasattr(self, "bin_used"): self.reset()
        T = int(T)

        bid = torch.bucketize(torch.tensor(10.0 - float(entropy)), self.bins[1:-1], right=False).item()  # 0..K-1
        pending = ~self.bin_used[:bid+1]
        if pending.any():
            keep_factor = (1.0 - self.drop_rate[:bid+1][pending]).prod().item()
            self.bin_used[:bid+1] = True
        else:
            keep_factor = 1.0

        keep = max(1, int(torch.ceil(torch.tensor(keep_factor * T)).item()))
        self.tokens.append(keep)
        return T - keep