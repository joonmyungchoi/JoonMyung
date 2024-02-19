import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from joonmyung.analysis.dataset import JDataset
from joonmyung.analysis.model import JModel
from joonmyung.draw import saliency, overlay, drawImgPlot, drawHeatmap, unNormalize
from joonmyung.meta_data import data2path
from joonmyung.metric import targetPred
from joonmyung.log import AverageMeter
from joonmyung.utils import to_leaf, to_np
from tqdm import tqdm
from contextlib import suppress
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch
import cv2


def anaModel(transformer_class):
    class VisionTransformer(transformer_class):
        def forward_features(self, x):
            x = self.patch_embed(x)
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            if self.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)


            if self.analysis[0] == 1:   # PATCH
                x = x # (8, 197, 192)
            elif self.analysis[0] == 2: # POS
                x = self.pos_embed # (1, 197, 192)
            elif self.analysis[0] == 3:  # PATCH (RANDOM I) + POS
                x = torch.rand_like(self.pos_embed, device=x.device) + self.pos_embed
            elif self.analysis[0] == 4:  # PATCH (RANDOM II) + POS
                x = torch.rand_like(self.cls_token, device=x.device).repeat(1, x.shape[1], 1) + self.pos_embed
            else: # PATCH + POS
                x = x + self.pos_embed
            x = self.pos_drop(x)

            x = self.blocks(x)
            x = self.norm(x)
            if self.dist_token is None:
                return self.pre_logits(x[:, 0])
            else:
                return x[:, 0], x[:, 1]

    return VisionTransformer

class Analysis:
    def __init__(self, model, analysis = [0], activate = [True, False, False], detach=True, key_name=None, num_classes = 1000
                 , cls_start=0, cls_end=1, patch_start=1, patch_end=None
                 , ks = 5
                 , amp_autocast=suppress, device="cuda"):
        # Section A. Model
        self.num_classes = num_classes
        self.key_name = key_name

        model_ = anaModel(model.__class__)
        model.__class__ = model_
        model.analysis = analysis
        self.model = model
        self.ks = ks
        self.detach = detach

        # Section B. Attention
        self.kwargs_roll = {"cls_start" : cls_start, "cls_end" : cls_end,
                            "patch_start" : patch_start, "patch_end" : patch_end}


        # Section C. Setting
        hooks = [{"name_i": 'attn_drop', "name_o": 'decoder', "fn_f": self.attn_forward, "fn_b": self.attn_backward},
                 {"name_i": 'qkv', "name_o": 'decoder', "fn_f": self.qkv_forward, "fn_b": self.qkv_backward},
                 {"name_i": 'head', "name_o": 'decoder', "fn_f": self.head_forward, "fn_b": self.head_backward}]
        hooks = [h for h, a in zip(hooks, activate) if a]


        self.amp_autocast = amp_autocast
        self.device       = device

        for name, module in self.model.named_modules():
            for hook in hooks:
                if hook["name_i"] in name and hook["name_o"] not in name:
                    module.register_forward_hook(hook["fn_f"])
                    module.register_backward_hook(hook["fn_b"])

    def attn_forward(self, module, input, output):
        # input  : 1 * (8, 3, 197, 197)
        # output : (8, 3, 197, 197)
        self.info["attn"]["f"].append(output.detach() if self.detach else output)

    def attn_backward(self, module, grad_input, grad_output):
        # input  : 1 * (8, 3, 197, 192)
        # output : (8, 3, 197, 576)
        self.info["attn"]["b"].insert(0, grad_input[0].detach() if self.detach else grad_input[0])

    def qkv_forward(self, module, input, output):
        # input  : 1 * (8, 197, 192)
        # output : (8, 197, 576)
        self.info["qkv"]["f"].append(output.detach())

    def qkv_backward(self, module, grad_input, grad_output):
        # self.info["qkv"]["b"].append(grad_input[0].detach())
       pass

    def head_forward(self, module, input, output):
        # input : 1 * (8(B), 192(D)), output : (8(B), 1000(C))
        # TP = targetPred(output.detach(), self.targets.detach(), topk=self.ks)
        TP = targetPred(to_leaf(output), to_leaf(self.targets), topk=self.ks)
        self.info["head"]["TP"] = torch.cat([self.info["head"]["TP"], TP], dim=0) if "TP" in self.info["head"].keys() else TP

    def head_backward(self, module, grad_input, grad_output):
        pass

    def resetInfo(self):
        self.info = {"attn": {"f": [], "b": []}, "qkv": {"f": [], "b": []},
                     "head": {"acc1" : AverageMeter(),
                              "acc5" : AverageMeter(),
                              "pred" : None

                              }}

    def __call__(self, samples, index=None, **kwargs):
        self.resetInfo()
        self.model.zero_grad()
        self.model.eval()

        if type(samples) == torch.Tensor:
            outputs = self.model(samples, **kwargs)
            return outputs
        else:
            for sample, self.targets in tqdm(samples):
                _ = self.model(sample)
            return False

    def anaSaliency(self, attn=True, grad=False, output=None, index=None,
                    head_fusion="mean", discard_ratios=[0.], data_from="cls",
                    ls_attentive=[], ls_rollout=[],
                    reshape=False, device="cuda"):

        if attn:
            attn = self.info["attn"]["f"]
        if grad:
            self.info["attn"]["b"] = []
            self.model.zero_grad()
            if index == None: index = output.max(dim=1)[1]
            index = torch.eye(self.num_classes, device=self.device)[index]
            loss = (output * index).sum()
            loss.backward(retain_graph=True)
            grad = self.info["attn"]["b"]

        return saliency(attn, grad
                        , head_fusion=head_fusion, discard_ratios=discard_ratios, data_from=data_from
                        , ls_rollout=ls_rollout, ls_attentive=ls_attentive, reshape=reshape, device=device)


if __name__ == '__main__':
    # Section A. Data
    dataset_name, device, amp_autocast = "imagenet", 'cuda', torch.cuda.amp.autocast
    data_path, _, _ = data2path(dataset_name)
    data_num, batch_size, bs = [[0, 0], [1, 0], [2, 0], [3, 0], [0, 1], [1, 1], [2, 1], [3, 1]], 16, []
    view, activate = [False, True, False, False, False], [True, False, False] #
        # VIEW     : IMG, SALIENCY(M), SALIENCY(D), SALIENCY(S), ATTN. MOVEMENT
        # ACTIVATE : ATTN, QKV, HEAD
    analysis = [4]
        # [0] : INPUT TYPE, [0 : SAMPLE + POS, 1 : SAMPLE, 2 : POS]
    dataset = JDataset(data_path, dataset_name, device=device)
    samples, targets, imgs, label_names = dataset.getItems(data_num)
    loader = dataset.getAllItems(batch_size)

    # Section B. Model
    model_number, model_name = 0, "deit_tiny_patch16_224" # deit_tiny_patch16_224, deit_small_patch16_224, deit_base_patch16_224
    # model_number, model_name = 0, "vit_tiny_patch16_224" # vit_tiny_patch16_224,  vit_small_patch16_224,  vit_base_patch16_224
    # model_number, model_name = 1, "deit_tiny_patch16_224"

    modelMaker = JModel(dataset.num_classes, device=device)
    model = modelMaker.getModel(model_number, model_name)
    model = Analysis(model, analysis = analysis, activate = activate, device=device)


    samples_ = samples[bs] if bs else samples
    targets_ = targets[bs] if bs else targets
    output = model(samples_)
    if view[0]:
        drawImgPlot(unNormalize(samples_, "imagenet"))

    if view[1]: # SALIENCY W/ MODEL
        ls_rollout, ls_attentive, col = [0,1,2,3,4,5,6,7,8,9,10,11], [0,1,2,3,4,5,6,7,8,9,10,11], 12
        # discard_ratios, v_ratio, head_fusion, data_from = [0.0, 0.4, 0.8], 0.1, "mean", "cls"  # Attention, Gradient
        discard_ratios, v_ratio, head_fusion, data_from = [0.0], 0.0, "mean", "cls"  # Attention, Gradient
        rollout, attentive = model.anaSaliency(True, False, output, discard_ratios=discard_ratios,
                                         ls_attentive = ls_attentive, ls_rollout=ls_rollout,
                                         head_fusion  = head_fusion, index=targets_, data_from=data_from,
                                         reshape      = True) # (12(L), 8(B), 14(H), 14(W))

        datas_rollout = overlay(samples_, rollout,   dataset_name)
        datas_attn    = overlay(samples_, attentive, dataset_name)
        drawImgPlot(datas_attn, col=col)
        print(1)
        # drawImgPlot(datas_rollout, col=col)
        # drawImgPlot(datas_attn, col=col)
        # drawHeatmap(rollout[:12, 0], col=col, vmin = rollout.reshape(-1, 196)[:12].quantile(v_ratio, dim=1), vmax = rollout.reshape(-1, 196)[:12].quantile(1 - v_ratio, dim=1))
        # drawHeatmap(rollout[12:24, 0], col=col, vmin = rollout.reshape(-1, 196)[12:24].quantile(v_ratio, dim=1), vmax = rollout.reshape(-1, 196)[12:24].quantile(1 - v_ratio, dim=1))
        # drawHeatmap(rollout[24:, 0], col=col, vmin = rollout.reshape(-1, 196)[24:].quantile(v_ratio, dim=1), vmax = rollout.reshape(-1, 196)[24:].quantile(1 - v_ratio, dim=1))
        # drawHeatmap(rollout[:12, 0], col=col)
        # drawHeatmap(rollout[12:24, 0], col=col)
        # drawHeatmap(rollout[24:, 0], col=col)

        # drawHeatmap(attentive[6:, 0], col=col)
        # drawHeatmap(attentive[:, 0], col=col, vmin=attentive.reshape(-1, 196).quantile(v_ratio, dim=1), vmax=attentive.reshape(-1, 196).quantile(1 - v_ratio, dim=1))

    if view[2]:  # SALIENCY W/ DATA
        img = np.array(imgs[0])

        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        (success, saliencyMap) = saliency.computeSaliency(img)
        saliencyMap = (saliencyMap * 255).astype("uint8")

        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        (success, saliencyFineMap) = saliency.computeSaliency(img)
        threshMap = cv2.threshold((saliencyFineMap * 255).astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # plt.imshow(threshMap)
        # plt.show()

    if view[3]:  # SALIENCY FOR INPUT
        samples_.requires_grad, model.detach, k = True, False, 3
        output = model(samples_)
        attn = torch.stack(model.info["attn"]["f"], dim=1).mean(dim=[2,3])[0,-2]
        topK = attn[1:].topk(k, -1, True)[1]
        # a = torch.autograd.grad(attn.sum(), samples, retain_graph=True)[0].sum(dim=1)
        a = torch.autograd.grad(output[:,3], samples_, retain_graph=True)[0].sum(dim=1)
        b = F.interpolate(a.unsqueeze(0), scale_factor=0.05, mode='nearest')[0]
        # drawHeatmap(b)
        print(1)
        # to_np(torch.stack([attn[:, :, 0], attn[:, :, 1:].sum(dim=-1)], -1)[0])

    if view[4]: # ATTENTION MOVEMENT (FROM / TO)
        attn = torch.stack(model.info["attn"]["f"]).mean(dim=2).transpose(0,1) # (8 (B), 12 (L), 197(T_Q), 197(T_K))
        cls2cls     = attn[:, :, :1, 0].mean(dim=2)              # (8(B), 12(L))
        patch2cls   = attn[:, :, :1, 1:].mean(dim=2).sum(dim=-1) # (8(B), 12(L))
        to_np(torch.stack([cls2cls.mean(dim=0), patch2cls.mean(dim=0)]))

        cls2patch   = attn[:, :, 1:, 0].mean(dim=2)
        patch2patch = attn[:, :, 1:, 1:].mean(dim=2).sum(dim=-1)
        to_np(torch.stack([cls2patch.mean(dim=0), patch2patch.mean(dim=0)]))
        print(1)
