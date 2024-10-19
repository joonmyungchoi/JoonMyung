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
from timm.models.vision_transformer import Attention, Block
from torch import nn

from joonmyung.compression.compression import token_compression


class CompressBlock(nn.Module):
    def forward(self, x: torch.Tensor, attn_mask):
        x = x + self.attn(self.ln_1(x))

        if self.info["compression"]["use"]:
            x = token_compression(x, self.info["compression"], self.l)
        x = x + self.mlp(self.ln_2(x))
        return x

class CompressAttention(Attention):
    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4))
        q, k, v = (qkv[0],qkv[1],qkv[2])
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.info["compression"]["use"]:
            self.info["compression"]["attn"] = attn.softmax(dim=-1).detach()
            if self.info["compression"]["size"] is not None:
                attn = attn + self.info["compression"]["size"].log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


def make_compression_class(transformer_class):
    class CompressVisionTransformer(transformer_class):
        def resetCompression(self, compression = None):
            self.info["compression"]["attn"] = None
            self.info["compression"]["size"] = None
            self.info["compression"]["source"] = None
            if compression:
                self.info["compression"]["use"] = True

                # PART A. COMMON   : [r_prune, r_merge, r_protected, proportional_attention, trace_source]
                self.info["compression"]["r_prune"] = compression[0][1]
                self.info["compression"]["r_merge"] = compression[0][2]
                self.info["compression"]["r_protected"] = compression[0][3] + 1
                self.info["compression"]["prop_attn"] = compression[0][4]
                self.info["compression"]["trace_source"] = compression[0][5]

                # PART B. Mcompressionu_sim, tau_info, tau_size, pooling_type]
                self.info["compression"]["tau_sim"] = compression[1][0]
                self.info["compression"]["tau_info"] = compression[1][1]
                self.info["compression"]["tau_size"] = compression[1][2]
                self.info["compression"]["pooling_type"] = compression[1][3]

                # PART C. VID-TLDR : [mass]
                self.info["compression"]["mass"] = compression[2][0]

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self.resetCompression()
            return super().forward(*args, **kwdargs)

        def encode_image(self, *args, **kwdargs) -> torch.Tensor:
            self.resetCompression()
            return super().encode_image(*args, **kwdargs)

    return CompressVisionTransformer

def apply_patch(
    model, compression
):
    model.__class__ = make_compression_class(model.__class__)
    model.info = getattr(model, "info", {})
    depth = len(model.visual.transformer.resblocks)
    model.info["compression"] = {"use": False, "r_cls": 1, "applied_layer": set(range(depth)),
                           "size": None, "attn": None, "source": None}
    model.resetCompression(compression)

    for i, resblock in enumerate(model.visual.transformer.resblocks):
        resblock.__class__ = CompressBlock
        resblock.attn.__class__ = CompressAttention
        resblock.l = i
        resblock.attn.info = resblock.info = model.info



if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    from joonmyung.analysis import JDataset, JModel, ZeroShotInference
    from joonmyung.meta_data import data2path, imnet_label
    from joonmyung.utils import read_classnames
    from joonmyung.log import AverageMeter
    from joonmyung.metric import accuracy
    from tqdm import tqdm
    import torch

    root_path, dataset_name, batch_size, device, debug = "/hub_data1/joonmyung/weights", "imagenet", 100, 'cuda', True
    classnames = read_classnames("/hub_data1/joonmyung/data/imagenet/classnames.txt")
    num_classes, model_name, model_number = len(classnames), "ViT-B/16", 2

    modelMaker = JModel(num_classes, root_path, device=device)
    model = modelMaker.getModel(model_number, model_name)
    # "compression": {"use": False, "applied_layer": set(range(self.depth)) - set(self.return_index), "r_cls": 1,
    #                 "size": None, "attn": None, "source": None}}

    data_path, num_classes, _, _ = data2path(dataset_name)
    dataset = JDataset(data_path, dataset_name, transform_type=2, device=device)
    dataloader = dataset.getAllItems(batch_size)
    model = ZeroShotInference(model, classnames, prompt="a photo of a {}.", device=device)

    compression = [[1, 0, 10, 0, 1, 1], [1, 10, 25, 1], [0]]
    apply_patch(model.model, compression)

    result = {"acc1": AverageMeter(), "acc5": AverageMeter()}
    with torch.no_grad():
        for image, labels in tqdm(dataloader):
            B = image.shape[0]
            image, labels = image.to(device), labels.to(device)
            logits = model(image)
            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
            result["acc1"].update(acc1.item(), n=B)  # 68.2 → 64.6
            result["acc5"].update(acc5.item(), n=B)