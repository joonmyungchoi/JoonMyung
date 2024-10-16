import os

from models import tome

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Section A. Data
from joonmyung.analysis import JDataset, JModel, Analysis
from joonmyung.draw import saliency, overlay, drawImgPlot, drawHeatmap, unNormalize
from joonmyung.meta_data import data2path
from joonmyung.data import getTransform
from joonmyung.metric import targetPred
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

dataset_name, device, amp_autocast, debug = "imagenet", 'cuda', torch.cuda.amp.autocast, True
data_path, num_classes, _, _ = data2path(dataset_name)

dataset = JDataset(data_path, dataset_name, device=device)

# data_idxs = [[c, i] for c in range(1000) for i in range(50)]
data_idxs = range(50000)



# Section B. Model
# model_number, model_name = 0, "deit_tiny_patch16_224" # deit, vit | tiny, small, base
# model_number, model_name = 1, "deit_tiny_patch16_224"
model_number, model_name, model_path = 2, "deit_tiny_patch16_224", "/hub_data2/joonmyung/weights/TEMPLATE/deit/tiny_WO_CLS"

modelMaker = JModel(num_classes, model_path=model_path, device=device)


# Section C. Setting
activate, analysis = [True, False, True, True], [0]
model = modelMaker.getModel(model_number, model_name)
tome.apply_patch(model, r = 20)
model = Analysis(model, analysis = analysis, activate = activate, device=device)
results = []

for samples, targets in tqdm(dataset.getAllItems(batch_size=128)):
    with torch.no_grad():
        _ = model(samples, targets)
    model.info["attn"]["f"]

    # print(1)

    # input_sim = torch.stack(model.info["input"]["sim"])
    # idxes = [(i * 0.2 < input_sim) & (input_sim < (i + 1) * 0.2) for i in range(5)]
    # sim_idx = input_sim.argsort(dim=-1, descending=False)
    # input_sim = torch.stack(model.info["head"]["TF"]).gather(dim = -1, index=sim_idx)
    # label = (torch.arange(0, 1000, device="cuda").repeat_interleave(50) == 0)
    # [torch.stack(model.info["head"]["TF"])[idx & label].float().mean() for idx in idxes]

    # model.info["head"]["acc1"].avg, model.info["head"]["acc5"].avg


print(1)


# for idx, data_idx in enumerate(tqdm(data_idxs)):
#     if type(data_idx) == list: data_idx = data_idx[0] * 50 + data_idx[1]
#     sample, target, label_name = dataset[data_idx, 0]
#
#     with torch.no_grad():
#         output = model(sample, target[None])
#     attn = torch.stack(model.info["attn"]["f"]).mean(dim=(2))
#     entropy = (attn * torch.log(attn)).sum(dim=-11)  # (L(12), B(1), T(196))
#     entropy = entropy / entropy.sum(dim=-1, keepdim=True)
#
#     result = {
#         "correct"  : model.info["head"]["TP"][0,0] == model.info["head"]["TP"][0,1],
#         "inputSim" : model.info["input"]["sim"].mean(),
#         "layerSim" : 1,
#         "idx"      : data_idx,
#         "target"   : target
#     }
#     results.append(result)
# [torch.stack([result["correct"] for result in results if i * 0.2 < result["inputSim"] and result["inputSim"] < (i+1) * 0.2]).to(float).mean() for i in range(0, 5)]
