# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

from timm.models.vision_transformer import Attention, Block, VisionTransformer
from typing import List, Tuple, Union
import torch
from typing import Callable, Tuple
import torch
import math

from models.SA.MHSA import MHSA
from models.modules.blocks import Block_DEIT


class ToMeBlock(Block):
    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x_attn, attn = self.attn(self.norm1(x), attn_size) # (256, 197, 768), (256, 197, 64)
        x = x + self._drop_path1(x_attn)

        r = self._tome_info["rs"].pop(0) # 32
        x = evit(x, attn, r)

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x

def evit(x, attn, r = 0):
    if not r:
        return x
    score_attn = attn.mean(dim=[1])[:, :, None]  # (1, 891, 1)
    attn_val, attn_idx = score_attn.sort(dim=1, descending=True)  # (12(B), 196(T), 1(D))
    x_sort = torch.gather(x, dim=1, index=attn_idx.repeat(1, 1, x.shape[2]))

    x = torch.cat([x_sort[:, :-r], (x_sort[:, -r:] * attn_val[:, -r:] / attn_val[:, -r:].sum(dim=1, keepdim=True)).sum(dim=1, keepdim=True)], dim=1)

    return x
class EViTAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        ) # (3, B, H, T, D)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn_ = attn.detach()
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn_


def make_evit_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            if not self._tome_info["rs"]: self._tome_info["rs"] = parse_r(len(self.blocks), self._tome_info["r"])
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().forward(*args, **kwdargs)

    return ToMeVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True, r = 0
):
    ToMeVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = ToMeVisionTransformer
    model._tome_info = {
        "r"     : r,
        "rs"    : parse_r(len(model.blocks), r),
        "size"  : None,
        "source": None,
        "trace_source" : trace_source,
        "prop_attn"    : prop_attn,
        "class_token"  : hasattr(model, "cls_token"),
        "distill_token": hasattr(model, "dist_token"),
    }

    for module in model.modules():
        if isinstance(module, Block) or isinstance(module, Block_DEIT):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
        elif isinstance(module, Attention) or isinstance(module, MHSA):
            module.__class__ = EViTAttention

def parse_r(num_layers: int, r: Union[List[int], Tuple[int, float], int]) -> List[int]:
    """
    Process a constant r or r schedule into a list for use internally.

    r can take the following forms:
     - int: A constant number of tokens per layer.
     - Tuple[int, float]: A pair of r, inflection.
       Inflection describes there the the reduction / layer should trend
       upward (+1), downward (-1), or stay constant (0). A value of (r, 0)
       is as providing a constant r. (r, -1) is what we describe in the paper
       as "decreasing schedule". Any value between -1 and +1 is accepted.
     - List[int]: A specific number of tokens per layer. For extreme granularity.
    """
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

