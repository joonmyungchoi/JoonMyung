from joonmyung.draw import saliency, overlay, drawImgPlot, unNormalize
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
        def attn_forward(self, module, input, output):  # input/output : 1 * (8, 3, 197, 197) / (8, 3, 197, 197)
            self.info["attn"]["f"].append(output.detach())

        def attn_backward(self, module, grad_input, grad_output):  # # input/output : 1 * (8, 3, 197, 192) / (8, 3, 197, 576)
            self.info["attn"]["b"].append(grad_input[0].detach())

        def qkv_forward(self, module, input, output):  # # input/output : 1 * (8, 197, 192) / (8, 197, 576)
            self.info["qkv"]["f"].append(output.detach())

        def head_forward(self, module, input, output):  # input : 1 * (8(B), 192(D)), output : (8(B), 1000(C))
            B = output.shape[0]
            pred = targetPred(output, self.targets, topk=5)
            self.info["head"]["TF"] += (pred[:, 0] == pred[:, 1])

            acc1, acc5 = accuracy(output, self.targets, topk=(1, 5))
            self.info["head"]["acc1"].update(acc1.item(), n=B)
            self.info["head"]["acc5"].update(acc5.item(), n=B)

        def input_forward(self, module, input, output):
            norm = F.normalize(output, dim=-1)
            self.info["input"]["sim"] += (norm @ norm.transpose(-1, -2)).mean(dim=(-1, -2))

        def resetInfo(self):
            self.info = {"attn": {"f": [], "b": []},
                         "qkv": {"f": [], "b": []},
                         "head": {"acc1": AverageMeter(),
                                  "acc5": AverageMeter(),
                                  "TF": [], "pred": []},
                         "input": {"sim": []}
                         }
        def createHook(self, activate):
            hooks = [{"name_i": 'attn_drop', "name_o": 'decoder', "fn_f": self.attn_forward, "fn_b": self.attn_backward},
                     {"name_i": 'qkv', "name_o": 'decoder', "fn_f": self.qkv_forward, "fn_b": None},
                     {"name_i": 'head', "name_o": 'decoder', "fn_f": self.head_forward, "fn_b": None},
                     {"name_i": 'patch_embed.norm', "name_o": 'decoder', "fn_f": self.input_forward, "fn_b": None}]

            for name, module in self.named_modules():
                for idx, hook in enumerate(hooks):
                    if hook["name_i"] in name and hook["name_o"] not in name and activate[idx]:
                        if hook["fn_f"]: module.register_forward_hook(hook["fn_f"])
                        if hook["fn_b"]: module.register_backward_hook(hook["fn_b"])

    return VisionTransformer

def Analysis(model, activate= [True, True, True, True]):
    model.__class__ = anaModel(model.__class__)
    model.resetInfo()
    model.createHook(activate)
    return model


if __name__ == '__main__':
    # 1. ANALYSIS
        # HOOK | FUNCTION | INFO

    # 2. ZeroShotInference
        # ENCODE_IMAGE / ENCODE_TEXT


    dataset_name, device, debug = "imagenet", 'cuda', True
    data_path, num_classes, _, _ = data2path(dataset_name)
    activate = [True, False, False, False]   # [ATTN, QKV, HEAD]
    analysis = [0] # [0] : INPUT TYPE, [0 : SAMPLE + POS, 1 : SAMPLE, 2 : POS]

    dataset = JDataset(data_path, dataset_name, device=device)
    data_idxs = [[c, i] for i in range(1000) for c in range(50)]

    modelMaker = JModel(num_classes, device=device)
    model = modelMaker.getModel(2, "ViT-B/16")


    model = Analysis(model)

    classnames = read_classnames("/hub_data2/joonmyung/data/imagenet/classnames.txt")
    model = ZeroShotInference(model, classnames, prompt="a photo of a {}.", device=device)
    view = [False, True, False, False, True]  # [IMG, SALIENCY:ATTN, SALIENCY:OPENCV, SALIENCY:GRAD, ATTN. MOVEMENT]
    for idx, data_idx in enumerate(data_idxs):
        print(f"------------------------- [{data_idx[0]}]/[{data_idx[1]}] -------------------------")


        sample, target, label_name = dataset[data_idx[0], data_idx[1]]
        output = model(sample)

# model.createHook([True, True, True, False, True])