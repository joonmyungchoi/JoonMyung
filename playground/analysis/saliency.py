from joonmyung.analysis import JDataset, JModel, Analysis
from joonmyung.draw import overlay, drawImgPlot, unNormalize
from joonmyung.meta_data import data2path
import torch.nn.functional as F
import numpy as np
import torch
import cv2


dataset_name, device, debug = "imagenet", 'cuda', True
data_path, num_classes, _, _ = data2path(dataset_name)
dataset = JDataset(data_path, dataset_name, device=device)
data_idxs = [[c, i] for i in range(1000) for c in range(50)]

modelMaker = JModel(num_classes, device=device)
model = modelMaker.getModel(2, "ViT-B/16")

activate = [True, False, False, False]  # [ATTN, QKV, HEAD]
analysis = [0]  # [0] : INPUT TYPE, [0 : SAMPLE + POS, 1 : SAMPLE, 2 : POS]
model = Analysis(model, analysis=analysis, activate=activate, device=device)

view = [False, True, False, False, True]  # [IMG, SALIENCY:ATTN, SALIENCY:OPENCV, SALIENCY:GRAD, ATTN. MOVEMENT]
for idx, data_idx in enumerate(data_idxs):
    print(f"------------------------- [{data_idx[0]}]/[{data_idx[1]}] -------------------------")

    sample, target, label_name = dataset[data_idx[0], data_idx[1]]
    output = model(sample)
    if view[0]:
        drawImgPlot(unNormalize(sample, "imagenet"))

    if view[1]:  # SALIENCY W/ MODEL
        col, discard_ratios, v_ratio, head_fusion, data_from = 12, [0.0], 0.0, "mean", "patch"  # Attention, Gradient
        results = model.anaSaliency(True, False, output, discard_ratios=discard_ratios,
                                    head_fusion=head_fusion, index=target, data_from=data_from,
                                    reshape=True, activate=[True, True, True])  # (12(L), 8(B), 14(H), 14(W))
        data_roll = overlay(sample, results["rollout"], dataset_name)
        drawImgPlot(data_roll, col=col)

        data_attn = overlay(sample, results["attentive"], dataset_name)
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
        attn = torch.stack(model.info["attn"]["f"], dim=1).mean(dim=[2, 3])[0, -2]
        topK = attn[1:].topk(k, -1, True)[1]
        a = torch.autograd.grad(output[:, 3], sample, retain_graph=True)[0].sum(dim=1)
        b = F.interpolate(a.unsqueeze(0), scale_factor=0.05, mode='nearest')[0]

    if view[4]:  # ATTENTION MOVEMENT (FROM / TO)
        attn = torch.stack(model.info["attn"]["f"]).mean(dim=2).transpose(0, 1)  # (8 (B), 12 (L), 197(T_Q), 197(T_K))

        # CLS가 얼마나 참고하는지
        cls2cls = attn[:, :, :1, 0].mean(dim=2)  # (8(B), 12(L))
        patch2cls = attn[:, :, :1, 1:].mean(dim=2).sum(dim=-1)  # (8(B), 12(L))

        # PATCH가 얼마나 참고하는지
        cls2patch = attn[:, :, 1:, 0].mean(dim=2)
        patch2patch = attn[:, :, 1:, 1:].mean(dim=2).sum(dim=-1)
        # to_np(torch.stack([cls2cls.mean(dim=0), patch2cls.mean(dim=0), cls2patch.mean(dim=0), patch2patch.mean(dim=0)]))