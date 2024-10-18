from joonmyung.draw import saliency, overlay, drawImgPlot, unNormalize, drawHeatmap
from joonmyung.analysis.model import JModel, ZeroShotInference
from timm.models.vision_transformer import Attention
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

def anaModel(transformer_class):
    class VisionTransformer(transformer_class):
        info_key = []
        def resetInfo(self):
            self.info = {n: [] for n in self.info_key}

        def createHook(self, hooks):
            [self.info_key.append(hook[3]) for hook in hooks]
            for name, module in self.named_modules():
                for idx, hook in enumerate(hooks):
                    if hook[1] in name and hook[2] not in name:
                        if hook[0] == "f":
                            module.register_forward_hook(lambda mod, inp, out, hook_info=hook:
                                                     self.forward_hook(hook_info, mod, inp, out))
                        else:
                            module.register_backward_hook(lambda mod, inp, out, hook_info=hook:
                                                     self.backward_hook(hook_info, mod, inp, out))
        def forward_hook(self, hook_info, module, input, output):
            self.info[hook_info[3]].append(output.detach())

        def backward_hook(self, hook_info, module, input, output):
            self.info[hook_info[3]].append(input[0].detach())

        def forward(self, *args, **kwdargs):
            self.resetInfo()
            return super().forward(*args, **kwdargs)
        def encode_image(self, *args, **kwdargs):
            self.resetInfo()
            return super().encode_image(*args, **kwdargs)

    return VisionTransformer

def Analysis(model, hook_info= [["f", "attn_drop", "decoder", "attn"]]):
    model.__class__ = anaModel(model.__class__)
    model.createHook(hook_info)
    return model

if __name__ == '__main__':
    dataset_name, device, debug = "imagenet", 'cuda', True
    data_path, num_classes, _, _ = data2path(dataset_name)
    analysis = [0] # [0] : INPUT TYPE, [0 : SAMPLE + POS, 1 : SAMPLE, 2 : POS]

    dataset = JDataset(data_path, dataset_name, device=device)
    data_idxs = [[c, i] for i in range(1000) for c in range(50)]

    modelMaker = JModel(num_classes, device=device)
    model = modelMaker.getModel(2, "ViT-B/16")
    classnames = read_classnames("/hub_data1/joonmyung/data/imagenet/classnames.txt")
    model = ZeroShotInference(model, classnames, prompt="a photo of a {}.", device=device)
    hook_info = [["b", "attn_drop", "decoder", "grad"],
                 ["f", "attn_drop", "decoder", "attn"],
                 ["f", "ln_pre",  "decoder", "feat_1"],
                 ["f", "ln_1",    "decoder", "feat_2"],
                 ["f", "ln_2",    "decoder", "feat_3"],
                 ["f", "ln_post", "decoder", "feat_4"]]
    model.model = Analysis(model.model, hook_info)
    view = [False, False, True, True, True, True]  # [IMG, SALIENCY:ATTN, SALIENCY:OPENCV, SALIENCY:GRAD, ATTN. MOVEMENT]
    for idx, data_idx in enumerate(data_idxs):
        print(f"------------------------- [{data_idx[0]}]/[{data_idx[1]}] -------------------------")
        sample, target, label_name = dataset[data_idx[0], data_idx[1]]
        sample.requires_grad = True

        if view[0]:
            drawImgPlot(unNormalize(sample, "imagenet"))

        output = model(sample)
        index = torch.eye(num_classes, device=device)[target]
        (output * index).sum().backward(retain_graph=True)

        attns = model.model.info["attn"]
        grads = model.model.info["grad"]
        if view[1]:
            col, discard_ratios, v_ratio, head_fusion, data_from = 12, [0.0], 0.0, "mean", "patch"
            results = saliency(attns, False, head_fusion=head_fusion, discard_ratios=discard_ratios, data_from=data_from, reshape=True, device=device)

            data_roll = overlay(sample, results["rollout"], dataset_name)
            drawImgPlot(data_roll, col=col)

            data_attn = overlay(sample, results["attentive"], dataset_name)
            drawImgPlot(data_attn, col=col)

            data_vidTLDR = overlay(sample, results["vidTLDR"], dataset_name)
            drawImgPlot(data_vidTLDR, col=col)

            discard_ratios, v_ratio, head_fusion, data_from = [0.0], 0.1, "mean", "cls"
            results = saliency(attns, grads, head_fusion=head_fusion, discard_ratios=discard_ratios, data_from=data_from, reshape=True, device=device)

            data_roll = overlay(sample, results["rollout"], dataset_name)
            drawImgPlot(data_roll, col=col)

            data_attn = overlay(sample, results["attentive"], dataset_name)
            drawImgPlot(data_attn, col=col)

            data_vidTLDR = overlay(sample, results["vidTLDR"], dataset_name)
            drawImgPlot(data_vidTLDR, col=col)

        if view[2]:  # SALIENCY W/ DATA
            img = (dataset.unNormalize(sample)[0].permute(1, 2, 0).detach().cpu().numpy() * 255)
            img_saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            (success, saliencyMap) = img_saliency.computeSaliency(img)
            saliencyMap = (saliencyMap * 255).astype("uint8")

            img_saliency = cv2.saliency.StaticSaliencyFineGrained_create()
            (success, saliencyFineMap) = img_saliency.computeSaliency(img)
            threshMap = cv2.threshold((saliencyFineMap * 255).astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        if view[3]:  # SALIENCY FOR INPUT
            output = model(sample)
            attn = torch.stack(attns, dim=1).mean(dim=[2, 3])[0, -2]
            a = torch.autograd.grad(output[:, 3], sample, retain_graph=True)[0].sum(dim=1)
            b = F.interpolate(a.unsqueeze(0), scale_factor=1.0, mode='nearest')[0]

        if view[4]: # ATTENTION MOVEMENT (FROM / TO)
            attn = torch.stack(attns).mean(dim=2).transpose(0,1) # (8 (B), 12 (L), 197(T_Q), 197(T_K))

            cls2cls     = attn[:, :, :1, 0].mean(dim=2)              # (8(B), 12(L))
            patch2cls   = attn[:, :, :1, 1:].mean(dim=2).sum(dim=-1) # (8(B), 12(L))
            cls2patch   = attn[:, :, 1:, 0].mean(dim=2)
            patch2patch = attn[:, :, 1:, 1:].mean(dim=2).sum(dim=-1)
            # to_np(torch.stack([cls2cls.mean(dim=0), patch2cls.mean(dim=0), cls2patch.mean(dim=0), patch2patch.mean(dim=0)]))
        if view[5]:
            feats = {k: v for k, v in model.model.info if "feat" in k}
            for name, feat in feats.items():
                print(f"Feature Position : {name}")
                image_feat  = (torch.stack(feat)[:, :, 1:] @ model.model.visual.proj) # (1, 1, 196, 512)
                L = image_feat.shape[0]
                image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

                text_feat = model.text_features[1][None].t()
                sim = (image_feat @ text_feat).reshape(L, 14, 14)
                drawHeatmap(sim, col = L)