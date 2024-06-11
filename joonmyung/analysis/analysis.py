import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from joonmyung.analysis.dataset import JDataset
from joonmyung.analysis.model import JModel
from joonmyung.draw import saliency, overlay, drawImgPlot, drawHeatmap, unNormalize
from joonmyung.meta_data import data2path
from joonmyung.data import getTransform
from joonmyung.metric import targetPred, accuracy
from joonmyung.log import AverageMeter
from joonmyung.utils import to_leaf, to_np
from tqdm import tqdm
from contextlib import suppress
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch
import PIL
import cv2


def anaModel(transformer_class):
    class VisionTransformer(transformer_class):
        def has_parameter(self, parameter_name):
            return parameter_name in self.__init__.__code__.co_varnames

        def forward_features(self, x):
            x = self.patch_embed(x)
            if self.has_parameter("cls_token"):
                cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                x = torch.cat((cls_token, x), dim=1)



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
            if self.has_parameter("cls_token") and self.has_parameter("dist_token"):
                return x[:, 0], x[:, 1]
            elif self.has_parameter("cls_token"):
                return self.pre_logits(x[:, 0])
            else:
                return self.pre_logits(x.mean(dim=1))


    return VisionTransformer

class Analysis:
    def __init__(self, model, analysis = [0], activate = [True, False, False, False], detach=True, key_name=None, num_classes = 1000
                 , cls_start=0, cls_end=1, patch_start=1, patch_end=None
                 , amp_autocast=suppress, device="cuda"):
        # Section A. Model
        self.num_classes = num_classes
        self.key_name = key_name

        model_ = anaModel(model.__class__)
        model.__class__ = model_
        model.analysis = analysis
        self.model = model
        self.detach = detach

        # Section B. Attention
        self.kwargs_roll = {"cls_start" : cls_start, "cls_end" : cls_end,
                            "patch_start" : patch_start, "patch_end" : patch_end}

        # Section C. Setting
        hooks = [{"name_i": 'attn_drop', "name_o": 'decoder', "fn_f": self.attn_forward, "fn_b": self.attn_backward},
                 {"name_i": 'qkv', "name_o": 'decoder', "fn_f": self.qkv_forward, "fn_b": self.qkv_backward},
                 {"name_i": 'head', "name_o": 'decoder', "fn_f": self.head_forward, "fn_b": self.head_backward},
                 {"name_i": 'patch_embed.norm', "name_o": 'decoder', "fn_f": self.input_forward, "fn_b": self.input_backward}]
        hooks = [h for h, a in zip(hooks, activate) if a]


        self.amp_autocast = amp_autocast
        self.device       = device

        for name, module in self.model.named_modules():
            for hook in hooks:
                if hook["name_i"] in name and hook["name_o"] not in name:
                    module.register_forward_hook(hook["fn_f"])
                    module.register_backward_hook(hook["fn_b"])
        self.resetInfo()

    def attn_forward(self, module, input, output):
        # input/output : 1 * (8, 3, 197, 197) / (8, 3, 197, 197)
        self.info["attn"]["f"] = output.detach() if self.detach else output

    def attn_backward(self, module, grad_input, grad_output):
        # input/output : 1 * (8, 3, 197, 192) / (8, 3, 197, 576)
        self.info["attn"]["b"] = grad_input[0].detach() if self.detach else grad_input[0]

    def qkv_forward(self, module, input, output):
        # input/output : 1 * (8, 197, 192) / (8, 197, 576)
        self.info["qkv"]["f"].append(output.detach())

    def qkv_backward(self, module, grad_input, grad_output):
        # self.info["qkv"]["b"].append(grad_input[0].detach())
       pass

    def head_forward(self, module, input, output):
        # input : 1 * (8(B), 192(D)), output : (8(B), 1000(C))
        B = output.shape[0]
        pred = targetPred(output, self.targets, topk=5)
        self.info["head"]["TF"] += (pred[:, 0] == pred[:, 1])

        acc1, acc5 = accuracy(output, self.targets, topk=(1,5))
        self.info["head"]["acc1"].update(acc1.item(), n=B)
        self.info["head"]["acc5"].update(acc5.item(), n=B)

    def head_backward(self, module, grad_input, grad_output):
        pass

    def input_forward(self, module, input, output):
        norm = F.normalize(output, dim=-1)
        self.info["input"]["sim"] += (norm @ norm.transpose(-1, -2)).mean(dim=(-1, -2))

    def input_backward(self, module, grad_input, grad_output):
        pass

    def resetInfo(self):
        self.info = {"attn" : {"f": None, "b": None},
                     "qkv"  : {"f": None, "b": None},
                     "head" : {"acc1" : AverageMeter(),
                               "acc5" : AverageMeter(),
                               "TF"   : [], "pred" : []},
                     "input": {"sim" : []}
                     }

    def __call__(self, samples, targets = None, **kwargs):
        self.model.zero_grad()
        self.model.eval()

        if type(samples) == torch.Tensor:
            self.targets = targets
            outputs = self.model(samples, **kwargs)
            return outputs
        else:
            for sample, targets in tqdm(samples):
                self.targets = targets
                _ = self.model(sample)
            return False

    def anaSaliency(self, attn=True, grad=False, output=None, index=None,
                    head_fusion="mean", discard_ratios=[0.], data_from="cls",
                    reshape=False, activate= [True, True, False], device="cuda"):

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

        return saliency(attn, grad, activate=activate,
                        head_fusion=head_fusion, discard_ratios=discard_ratios, data_from=data_from,
                        reshape=reshape, device=device)


if __name__ == '__main__':
    # Section A. Data
    dataset_name, device, amp_autocast, debug = "imagenet", 'cuda', torch.cuda.amp.autocast, True
    data_path, num_classes, _, _ = data2path(dataset_name)
    view, activate = [False, True, False, False, True], [True, False, False]
        # VIEW     : IMG, SALIENCY:ATTN, SALIENCY:OPENCV, SALIENCY:GRAD, ATTN. MOVEMENT
        # ACTIVATE : ATTN, QKV, HEAD
    analysis = [0] # [0] : INPUT TYPE, [0 : SAMPLE + POS, 1 : SAMPLE, 2 : POS]

    dataset = JDataset(data_path, dataset_name, device=device)
    # data_idxs = [[c, i] for i in range(1000) for c in range(50)]
    data_idxs = [[1, 0]]

    # Section B. Model
    model_number, model_name = 0, "deit_tiny_patch16_224" # deit, vit | tiny, small, base
    # model_number, model_name = 1, "deit_tiny_patch16_224"

    # Section C. Setting
    modelMaker = JModel(num_classes, device=device)
    model = modelMaker.getModel(model_number, model_name)
    model = Analysis(model, analysis = analysis, activate = activate, device=device)
    for idx, data_idx in enumerate(data_idxs):
        print(f"------------------------- [{data_idx[0]}]/[{data_idx[1]}] -------------------------")

        sample, target, label_name = dataset[data_idx[0], data_idx[1]]
        # sample, _, img, _ = dataset.getItemPath('/hub_data1/joonmyung/data/imagenet/train/n01440764/n01440764_39.JPEG')
        output = model(sample)
        if view[0]:
            drawImgPlot(unNormalize(sample, "imagenet"))

        if view[1]: # SALIENCY W/ MODEL
            col, discard_ratios, v_ratio, head_fusion, data_from = 12, [0.0], 0.0, "mean", "patch"  # Attention, Gradient
            results = model.anaSaliency(True, False, output, discard_ratios=discard_ratios,
                                                   head_fusion  = head_fusion, index=target, data_from=data_from,
                                                   reshape      = True, activate=[True, True, True]) # (12(L), 8(B), 14(H), 14(W))
            data_roll = overlay(sample, results["rollout"], dataset_name)
            drawImgPlot(data_roll, col=col)

            data_attn    = overlay(sample, results["attentive"], dataset_name)
            drawImgPlot(data_attn, col=col)

            data_vidTLDR = overlay(sample, results["vidTLDR"], dataset_name)
            drawImgPlot(data_vidTLDR, col=col)

            discard_ratios, v_ratio, head_fusion, data_from = [0.0], 0.1, "mean", "cls"
            results = model.anaSaliency(True, False, output, discard_ratios=discard_ratios,
                                        head_fusion=head_fusion, index=target, data_from=data_from,
                                        reshape=True, activate=[True, True, True])  # (12(L), 8(B), 14(H), 14(W))

            data_roll = overlay(sample, results["rollout"], dataset_name)
            drawImgPlot(data_roll, col=col)

            data_attn = overlay(sample, results["attentive"], dataset_name)
            drawImgPlot(data_attn, col=col)

            data_vidTLDR = overlay(sample, results["vidTLDR"], dataset_name)
            drawImgPlot(data_vidTLDR, col=col)

            print(1)

            # roll = F.normalize(results["rollout"].reshape(12, 196), dim=-1)

            # datas_rollout = overlay(sample, rollout,   dataset_name)
            # drawImgPlot(datas_rollout, col=col)

            # datas_attn = overlay(sample, attentive, dataset_name)
            # drawImgPlot(datas_attn, col=col)

            # a = attentive[5]
            # b = torch.stack([a.clamp(max=a.quantile(1 - v_ratio)) for v_ratio in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]])
            # datas_attn    = overlay(sample, b, dataset_name)
            # drawImgPlot(datas_attn, col=col)
            # print(1)

        if view[2]:  # SALIENCY W/ DATA
            img = np.array(dataset[data_idx[0], data_idx[1], 2][0])

            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            (success, saliencyMap) = saliency.computeSaliency(img)
            saliencyMap = (saliencyMap * 255).astype("uint8")

            saliency = cv2.saliency.StaticSaliencyFineGrained_create()
            (success, saliencyFineMap) = saliency.computeSaliency(img)
            threshMap = cv2.threshold((saliencyFineMap * 255).astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            # plt.imshow(threshMap)
            # plt.show()

        if view[3]:  # SALIENCY FOR INPUT
            sample.requires_grad, model.detach, k = True, False, 3
            output = model(sample)
            attn = torch.stack(model.info["attn"]["f"], dim=1).mean(dim=[2,3])[0,-2]
            topK = attn[1:].topk(k, -1, True)[1]
            # a = torch.autograd.grad(attn.sum(), samples, retain_graph=True)[0].sum(dim=1)
            a = torch.autograd.grad(output[:,3], sample, retain_graph=True)[0].sum(dim=1)
            b = F.interpolate(a.unsqueeze(0), scale_factor=0.05, mode='nearest')[0]
            # drawHeatmap(b)
            print(1)
            # to_np(torch.stack([attn[:, :, 0], attn[:, :, 1:].sum(dim=-1)], -1)[0])

        if view[4]: # ATTENTION MOVEMENT (FROM / TO)
            attn = torch.stack(model.info["attn"]["f"]).mean(dim=2).transpose(0,1) # (8 (B), 12 (L), 197(T_Q), 197(T_K))

            # CLS가 얼마나 참고하는지
            cls2cls     = attn[:, :, :1, 0].mean(dim=2)              # (8(B), 12(L))
            patch2cls   = attn[:, :, :1, 1:].mean(dim=2).sum(dim=-1) # (8(B), 12(L))

            # PATCH가 얼마나 참고하는지
            cls2patch   = attn[:, :, 1:, 0].mean(dim=2)
            patch2patch = attn[:, :, 1:, 1:].mean(dim=2).sum(dim=-1)
            # to_np(torch.stack([cls2cls.mean(dim=0), patch2cls.mean(dim=0), cls2patch.mean(dim=0), patch2patch.mean(dim=0)]))
            print(1)