from joonmyung.draw import saliency, overlay, drawImgPlot, unNormalize, drawHeatmap
from joonmyung.analysis.model import JModel, ZeroShotInference
from joonmyung.metric import targetPred, accuracy
from joonmyung.analysis.dataset import JDataset
from joonmyung.utils import read_classnames
from joonmyung.meta_data import data2path
from joonmyung.log import AverageMeter
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch
import cv2
from timm.models.vision_transformer import Attention
def anaModel(transformer_class):
    class VisionTransformer(transformer_class):
        #      name_in         name_out  save_name
        #     attn_drop         decoder    attn
        #       qkv             decoder     qkv
        #       head            decoder    head
        #  patch_embed.norm     decoder    feat
        info_key = []
        def dynamic_hook(self, hook_info, module, input, output):
            self.info[hook_info[2]].append(output.detach())
        def resetInfo(self):
            self.info = {n: [] for n in self.info_key}

        def createHook(self, hooks):
            [self.info_key.append(hook[2]) for hook in hooks]
            for name, module in self.named_modules():
                for idx, hook in enumerate(hooks):
                    if hook[0] in name and hook[1] not in name:
                        module.register_forward_hook(lambda mod, inp, out, hook_info=hook:
                                                     self.dynamic_hook(hook_info, mod, inp, out))
    return VisionTransformer

def Analysis(model, hook_info= [["attn_drop", "decoder", "attn"]]):
    model.__class__ = anaModel(model.__class__)
    model.resetInfo()
    model.createHook(hook_info)
    return model

if __name__ == '__main__':
    dataset_name, device, debug = "imagenet", 'cuda', True
    data_path, num_classes, _, _ = data2path(dataset_name)
    activate = [True, False, False, False]   # [ATTN, QKV, HEAD]
    analysis = [0] # [0] : INPUT TYPE, [0 : SAMPLE + POS, 1 : SAMPLE, 2 : POS]

    dataset = JDataset(data_path, dataset_name, device=device)
    data_idxs = [[c, i] for i in range(1000) for c in range(50)]

    modelMaker = JModel(num_classes, device=device)
    model = modelMaker.getModel(2, "ViT-B/16")
    classnames = read_classnames("/hub_data1/joonmyung/data/imagenet/classnames.txt")
    model = ZeroShotInference(model, classnames, prompt="a photo of a {}.", device=device)
    hook_info = [["attn_drop", "decoder", "attn"],
                 ["ln_pre",  "decoder", "feat_1"],
                 ["ln_1",    "decoder", "feat_2"],
                 ["ln_2",    "decoder", "feat_3"],
                 ["ln_post", "decoder", "feat_4"]]
    model.model = Analysis(model.model, hook_info)
    view = [False, True, False, False, True]  # [IMG, SALIENCY:ATTN, SALIENCY:OPENCV, SALIENCY:GRAD, ATTN. MOVEMENT]
    for idx, data_idx in enumerate(data_idxs):
        print(f"------------------------- [{data_idx[0]}]/[{data_idx[1]}] -------------------------")
        model.model.resetInfo()
        sample, target, label_name = dataset[data_idx[0], data_idx[1]]
        output = model(sample)
        info = model.model.info
        drawImgPlot(unNormalize(sample))
        for name, value in info.items():
            if "feat" in name:
                print()

            if "feat" in name:
                print(f"name : {name}")
                image_feat  = (torch.stack(value)[:, :, 1:] @ model.model.visual.proj) # (1, 1, 196, 512)
                L = image_feat.shape[0]
                image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

                text_feat = model.text_features[1][None].t()
                sim = (image_feat @ text_feat).reshape(L, 14, 14)
                drawHeatmap(sim, col = L)
