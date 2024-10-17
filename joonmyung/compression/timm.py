# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# mae: https://github.com/facebookresearch/mae
# --------------------------------------------------------


import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer

from tome.utils import parse_r

from .timm import ToMeBlock, ToMeAttention

class CompressAttention(Attention):
    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # if size is not None:
        #     attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn_ = attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn_.mean(dim=[1,2]).detach()


def make_tome_class(transformer_class):
    class CompressVisionTransformer(transformer_class):
        def resetCompression(self):
            self.info["compression"]["attn"] = None
            self.info["compression"]["source"] = None

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self.resetCompression()
            return super().forward(*args, **kwdargs)

    return CompressVisionTransformer