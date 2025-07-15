from torchvision.transforms.functional import to_pil_image
from joonmyung.utils import to_leaf, to_tensor, to_np
from joonmyung.data import normalization
from scipy.ndimage import binary_erosion
from matplotlib import pyplot as plt
import torch.utils.data.distributed
import torch.nn.functional as F
import matplotlib as mpl
import torch.nn.parallel
import torch.utils.data
from PIL import Image
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import torch.optim
import random
import torch
import copy
import cv2
import PIL
import os

def sortedMatrix(values, layers = None, sort = False, dim = -1, normalize = False, quantile = 0, descending = False, HW = None, dtype=torch.float32, BL=False, cls=False):
    # values : (L, B, T)
    values = values.to(dtype)
    if len(values.shape) == 2: values = values[None]

    if layers is not None: values = values[layers]
    if cls: values = values[:, :, 1:]

    if normalize:
        values = (values - values.min(dim=dim, keepdim=True)[0]) / (values.max(dim=dim, keepdim=True)[0] - values.min(dim=dim, keepdim=True)[0])
        values = values / values.sum(dim=dim, keepdim=True)
    # TODO : BATCH-SIZE
    if quantile: values = values.clamp(values.quantile(quantile, dim=dim, keepdim=True), values.quantile(1-quantile, dim=dim, keepdim=True))
    if sort: values = torch.argsort(values, dim=dim, descending = descending).argsort(dim=dim, descending=descending)

    if BL: values = values.transpose(0, 1)
    if HW:
        values = values.reshape(-1, HW[0], HW[1])
    return  values # LBF

def drawController(data, vis_heatmap=0, vis_overlay = 0, img = None, K = None, use_threshold = None, mask = None,
                   col = 1, save_name=None, save = 1, border = False,  # COMMON
                   fmt=0, fontsize=None, cbar=False,  # DRAW HEATMAP
                   show= True, deactivate=False,
                   **kwargs):
    if deactivate:
        return

    if vis_heatmap:
        drawHeatmap(data, fmt=fmt, col=col, border=border, fontsize=fontsize, cbar=cbar,
                    save_name=save_name if save else None, show=show, **kwargs)
    else:
        if img is not None:
            if vis_overlay: # 이미지와 겹치기
                if K or mask is not None:
                    if mask is None: mask = generate_mask(data, topK=K, use_threshold=use_threshold) # (6, 32, 32)
                    data = mask_to_image(img, mask)
                else:
                    data = overlay(img, data)


        drawImgPlot(data, col=col, border=border,
                    save_name=save_name if save else None, show=show, **kwargs)



def generate_mask(data, topK=10, use_threshold = False, F = 1):
    shape, dtype = data.shape, data.dtype # (L * B, T, T)
    data = data.reshape(-1, F, *shape[-2:]) # (L * B, T, T)
    flattened = data.view(data.shape[0], -1)

    if use_threshold:
        flattened = flattened / flattened.mean(dim=-1, keepdim=True)
        mask = flattened > topK
    else:
        K = topK if type(topK) == int else int(flattened.shape[-1] * ( 1 - topK ))
        sorted_indices = torch.argsort(flattened, dim=1, descending=True)
        top_K_indices = sorted_indices[:, :K]
        mask = torch.zeros_like(flattened, dtype=dtype)
        mask.scatter_(1, top_K_indices, 1)
    return mask.view(shape)

def mask_to_image(image, mask):
    Fr, C, H, W = image.shape # (1, 3, 448, 448)
    if len(mask.shape) == 3:
        mask = mask.reshape(-1, Fr, *mask.shape[-2:])
    L, Fr, M_H, M_W = mask.shape  #

    mask_resized = F.interpolate(mask.float(), size=(H, W), mode='nearest').reshape(L, Fr, 1, H, W) # (L, B, 1, H, W)
    mask_3channel = mask_resized.expand(-1, -1, 3, -1, -1) # (L, B, 3, H, W)
    masked_image = image[None] * mask_3channel

    return masked_image.reshape(-1, C, H, W)


def drawHeatmap(matrixes, col=1, title=[], fmt=1, p=False,
                vmin=None, vmax=None, xticklabels=False, yticklabels=False,
                linecolor=None, linewidths=0.1, fontsize=30, r=[1,1],
                cmap="Greys", cbar=True, border=False,
                output_dir='./result', save_name=None, show=True,
                vis_x= False, vis_y =  False, tqdm_disable=True):
    row = (len(matrixes) - 1) // col + 1
    annot = True if fmt != None else False
    if p:
        print("|- Parameter Information")
        print("  |- Data Info (G, H, W)")
        print("    |- G : Graph Num")
        print("    |- H : height data dimension")
        print("    |- W : weidth data dimension")
        print("  |- AXIS Information")
        print("    |- col        : 컬럼 갯수")
        print("    |- row        : {}, col : {}".format(row, col))
        print("    |- height     : {}, width : {}".format(row * 8, col * 8))
        print("    |- title      : 컬럼별 제목")
        print("    |- p          : 정보 출력")
        print()
        print("  |- Graph Information")
        print("    |- vmin/vmax  : 히트맵 최소/최대 값")
        print("    |- linecolor  : black, ...   ")
        print("    |- linewidths : 1.0...   ")
        print("    |- fmt        : 숫자 출력 소숫점 자릿 수")
        print("    |- cmap        : Grey")
        print("    |- cbar        : 오른쪽 바 On/Off")
        print("    |- xticklabels : x축 간격 (False, 1,2,...)")
        print("    |- yticklabels : y축 간격 (False, 1,2,...)")
    h, w = matrixes[0].shape
    if title:
        title = title + list(range(len(title), len(matrixes) - len(title)))
    fig, axes = plt.subplots(nrows=row, ncols=col, squeeze=False,
                             figsize = (col * w * r[1], row * h * r[0]),
                             gridspec_kw={"wspace": 0.1, 'hspace': 0.1})
    fig.patch.set_facecolor('white')

    for e, matrix in enumerate(tqdm(matrixes, desc="drawHeatmap", disable=tqdm_disable)):
        if type(matrix) == torch.Tensor:
            matrix = matrix.detach().cpu().numpy()
        ax = axes[e // col][e % col]
        sns.heatmap(pd.DataFrame(matrix), annot=annot, fmt=".{}f".format(fmt), cmap=cmap
                          , vmin=vmin, vmax=vmax, yticklabels=yticklabels, xticklabels=xticklabels
                          , linewidths=linewidths, linecolor=linecolor, cbar=cbar, annot_kws={"size": fontsize}
                          , ax=ax)

        if not border:
            ax.set_axis_off()

        if not vis_x: ax.xaxis.set_visible(False)
        if not vis_y: ax.yaxis.set_visible(False)

        if title:
            ax.set(title="{} : {}".format(title, e))
    # plt.tight_layout()
    if output_dir and save_name:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, save_name), transparent = True)
    if show:
        plt.show()


def drawLinePlot(datas, index, col=1, title=[], xlabels=None, ylabels=None, markers=False, columns=None, p=False, ):
    row = (len(datas) - 1) // col + 1
    title = title + list(range(len(title), len(datas) - len(title)))
    fig, axes = plt.subplots(nrows=row, ncols=col, squeeze=False)
    fig.set_size_inches(col * 8, row * 8)

    if p:
        print("|- Parameter Information")
        print("  |- Data Info (G, D, C)")
        print("    |- G : Graph Num")
        print("    |- D : x data Num (Datas)")
        print("    |- C : y data Num (Column)")
        print("  |- Axis Info")
        print("    |- col   : 컬럼 갯수")
        print("    |- row : {}, col : {}".format(row, col))
        print("    |- height : {}, width : {}".format(row * 8, col * 8))
        print("    |- title : 컬럼별 제목")
        print("    |- p     : 정보 출력")
        print("  |- Graph Info")
        print("    |- vmin/row  : 히트맵 최소/최대 값")
        print("    |- linecolor  : black, ...   ")
        print("    |- linewidths : 1.0...   ")
        print("    |- fmt        : 숫자 출력 소숫점 자릿 수")
        print("    |- cmap        : Grey")
        print("    |- cbar        : 오른쪽 바 On/Off")
        print("    |- xticklabels : x축 간격 (False, 1,2,...)")
        print("    |- yticklabels : y축 간격 (False, 1,2,...)")
        print()

    for e, data in enumerate(datas):
        ax = axes[e // col][e % col]
        d = pd.DataFrame(data, index=index, columns=columns).reset_index()
        d = d.melt(id_vars=["index"], value_vars=columns)
        p = sns.lineplot(x="index", y="value", data=d, hue="variable", markers=markers, ax=ax)
        p.set_xlabel(xlabels, fontsize=20)
        p.set_ylabel(ylabels, fontsize=20)

        ax.set(title=title[e])
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    # plt.tight_layout()
    plt.show()

def drawBarChart(df, x, y, splitColName, col=1, title=[], fmt=1, p=False, c=False, c_sites={}, showfliers=True, tqdm_disable=True):
    d2s = df[splitColName].unique()
    d1 = df['d1'].unique()[0]
    d2s = [d2 for d2 in d2s for c_site in c_sites[d1].keys() if c_site in d2]

    row = (len(d2s) - 1) // col + 1

    fig, axes = plt.subplots(nrows=row, ncols=col, squeeze=False)
    fig.set_size_inches(col * 12, row * 12)
    for e, d2 in enumerate(tqdm(d2s, desc="drawBarChart", disable=tqdm_disable)):
        plt.title(d2, fontsize=20)
        ax = axes[e // col][e % col]
        temp = df.loc[df['d2'].isin([d2])]
        if temp["Date_m"].max() != temp["Date_m"].min():
            ind = pd.date_range(temp["Date_m"].min(), temp["Date_m"].max(), freq="M").strftime("%Y-%m")
        else:
            pd.to_datetime(temp["Date_m"].max()).strftime("%Y-%m")
        g = sns.boxplot(x=x, y=y, data=temp, order=ind, ax=ax, showfliers=showfliers)
        g.set(title=d2)
        g.set_xticklabels(ind, rotation=45, fontsize=15)
        g.set(xlabel=None)
        for c_site, c_dates in c_sites[d1].items():
            if c_site in d2:
                for c_date in c_dates:
                    c_ind = (pd.to_datetime(c_date, format='%Y%m%d') - pd.to_datetime(temp["Date_m"].min())).days / 30
                    if c_ind >= 0:
                        g.axvline(c_ind, ls='--', c="red")
    plt.show()

@torch.no_grad()
def saliency(attentions=None, gradients=None, head_fusion="mean",
             discard_ratios = [0.0], data_from="cls", reshape=False,
             activate = [True, True, True], device="cpu", dtype = torch.float32):

    # attentions : L * (B, H, h, w), gradients : L * (B, H, h, w)
    if type(discard_ratios) is not list: discard_ratios = [discard_ratios]
    saliencys = 1.
    if attentions:
        attentions = torch.stack(attentions, dim=0)
        saliencys = saliencys * attentions
    if gradients:
        gradients = torch.stack(gradients, dim=0)
        saliencys = saliencys * gradients

    if head_fusion == "mean":
        saliencys = saliencys.mean(axis=2) #
    elif head_fusion == "max":
        saliencys = saliencys.max(axis=2)[0]
    elif head_fusion == "min":
        saliencys = saliencys.min(axis=2)[0]
    elif head_fusion == "median":
        saliencys = saliencys.median(axis=2)[0]

    saliencys = saliencys.to(device = device, dtype = dtype)

    L, B, _, T = saliencys.shape # (L(12), B(1), T(197), T(197))
    H = W = int(T ** 0.5)

    result = {}
    if activate[0]:
        rollouts, I = [], torch.eye(T, device=device, dtype=dtype)[None].expand(B, -1, -1)  # (B, 197, 197)
        for discard_ratio in discard_ratios:
            for start in range(L):
                rollout = I
                for attn in copy.deepcopy(saliencys[start:]):  # (L, B, w, h)
                    # TODO NEED TO CORRECT
                    if discard_ratio:
                        flat = attn.reshape(B, -1)
                        _, indices = flat.topk(int(flat.shape[-1] * discard_ratio), -1, False)
                        indices = indices * (indices != 0)
                        for b in range(B):
                            flat[b, indices[b]] = 0

                    attn = 0.5 * attn + 0.5 * I
                    attn = attn / attn.sum(dim=-1, keepdim=True)
                    rollout = torch.matmul(attn, rollout)  # (1, 197, 197)

                rollout = rollout[:, 0] if data_from == "cls" else rollout[:, 1:].mean(dim=1) # (L, B, T)
                rollout = rollout / rollout.max(dim=-1, keepdim=True)[0]
                rollouts.append(rollout)
        rollouts = torch.stack(rollouts, dim=0)
        rollouts = rollouts[:, :, 1:]
        rollouts = rollouts / rollouts.sum(dim=-1, keepdim=True)

        if reshape:
            rollouts = rollouts.reshape(-1, B, H, W) # L, B, H, W
        result["rollout"] = rollouts

    if activate[1]:
        # attentive = saliencys[ls_attentive, :, 0] \
        #         if data_from == "cls" else saliencys[ls_attentive, :, 1:].mean(dim=2) # (L, B, T)
        attentive = saliencys[:, :, 0] \
            if data_from == "cls" else saliencys.mean(dim=2)  # (L, B, T)
        attentive = attentive[:, :, 1:]
        attentive = attentive / attentive.sum(dim=-1, keepdim=True)

        if reshape:
            attentive = attentive.reshape(-1, B, H, W)
        result["attentive"] = attentive

    if activate[2]:
        entropy = (saliencys * torch.log(saliencys)).sum(dim=-1)[:, :, 1:] # (L(12), B(1), T(196))
        entropy = entropy - entropy.amin(dim=-1, keepdim=True)
        entropy = entropy / entropy.sum(dim=-1, keepdim=True)
        if reshape:
            entropy = entropy.reshape(L, B, H, W)
        result["vidTLDR"] = entropy


    return result



def data2PIL(datas, RGB = True):
    # RGB : (B, C, H, W)

    if type(datas) == torch.Tensor: # (C, H, W)
        if len(datas.shape) == 2: datas = datas[None]
        datas = datas.detach().cpu()
        pils = datas.permute(1, 2, 0) if RGB else datas # (H, W, C)
    elif type(datas) == np.ndarray:
        datas = cv2.cvtColor(datas, cv2.COLOR_BGR2RGB) if datas.max() <= 1 else datas
        pils = datas.transpose(1, 2, 0) if RGB else datas # (H, W, C)
    elif type(datas) == PIL.Image.Image:
        pils = datas
    else:
        raise ValueError

    return pils # (H, W, C)

def drawImgPlot(datas, col=1, title:str=None, columns=None,
                output_dir='./', save_name=None, show=True,
                RGB = True,
                vis_x = False, vis_y = False, border=False):
    # datas : (B, C, H, W) or (B, H, W)
    if type(datas[0]) != PIL.Image.Image and len(datas.shape) == 3:
        datas = datas[:, None]

    row = (len(datas) - 1) // col + 1
    fig, axes = plt.subplots(nrows=row, ncols=col, squeeze=False)
    fig.set_size_inches(col * 8, row * 8)
    if title: fig.suptitle(title, fontsize=16)
    for i, data in enumerate(datas):
        r_num, c_num   = i // col, i % col
        data = data2PIL(data, RGB) # (H, W, C)
        ax = axes[r_num][c_num]
        if "shape" not in dir(data): # IMAGE
            ax.imshow(data)
        elif data.shape[-1] == 3: #
            ax.imshow(data)
        elif data.shape[-1] == 1: #
            ax.imshow(data, cmap="Greys")
        if not border:
            ax.set_axis_off()
        if columns:
            ax.set_title(columns[c_num] + str(r_num)) if len(columns) == col else ax.set_title(columns[i])
        if not vis_x: ax.xaxis.set_visible(False)
        if not vis_y: ax.yaxis.set_visible(False)
    # plt.tight_layout()

    if output_dir and save_name:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{save_name}.png'), transparent = True)
    if show:
        plt.show()

def overlay_mask(img: Image.Image, mask: Image.Image, colormap: str = "jet", alpha: float = 0.7) -> Image.Image:
    cmap = mpl.cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))

    return overlayed_img



def overlay(imgs, attnsL, dataset=None):
    attnsL = to_leaf(to_tensor(attnsL))
    imgs   = to_leaf(to_tensor(imgs))
    if len(imgs.shape) == 3: imgs = imgs.unsqueeze(0)  # B, C, H, W
    B = imgs.shape[0]
    if dataset:
        imgs = unNormalize(imgs, dataset)

    if type(attnsL) == list:   attnsL = torch.stack(attnsL, 0)
    if len(attnsL.shape) == 2: attnsL = attnsL.unsqueeze(0)
    if len(attnsL.shape) == 3: attnsL = attnsL.unsqueeze(0)
    attnsL = attnsL.reshape(-1, B, *attnsL.shape[-2:]) # L, B, H, W

    results = []
    for attns in attnsL:
        for img, attn in zip(imgs, attns):
            result = overlay_mask(to_pil_image(img), to_pil_image(normalization(attn, type=0), mode='F')) # (3, 224, 224), (1, 14, 14)
            # plt.imshow(overlay_mask(to_pil_image(dataset.unNormalize(samples)[0]), to_pil_image(normalization(a[:, 0]), mode='F'), alpha=0.5))
            results.append(result)
    return results # (L * B) * overlay

def unNormalize(image, dataset="imagenet", reverse=False):
    # images : (B, C, H, W)
    if dataset == "imagenet":
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif dataset == "cifar":
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    else:
        return image

    result = copy.deepcopy(image)

    for c, (m, s) in enumerate(zip(mean, std)):
        if reverse:
            result[:, c].sub_(m).div_(s)
        else:
            result[:, c].mul_(s).add_(m)

    return result


def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def generate_colormap(N: int, seed: int = 0):
    """Generates a equidistant colormap with N elements."""
    random.seed(seed)

    def generate_color():
        return (random.random(), random.random(), random.random())

    return [generate_color() for _ in range(N)]


def make_visualization(
        samples, source: torch.Tensor, patch_size: int = 16, token_nums: int = 1,
        min_merge_nums=0, merge=True, prune=False, unmerge=False
) -> Image:
    imgs = to_np(unNormalize(samples.reshape(-1, 3, 224, 224), "imagenet"))  # (4, 3, 224, 224)

    Fr, C, H, W = imgs.shape  # (4, 3, 224, 224)
    ph = H // patch_size
    pw = W // patch_size

    source = source[:, token_nums:, token_nums:]  # (8(B), 11(T_M), 196(T))
    # SECTION I. MERGING
    if merge:
        source_merge = source[source.sum(dim=2) > min_merge_nums][None]  # (1, 438, 2352)
        source_ummerge = source[source.sum(dim=2) <= min_merge_nums][None]
        vis = source_merge.argmax(dim=1)  # (1(B), 1024(T))
        num_groups = vis.max().item() + 1

        cmap = generate_colormap(num_groups)
        vis_merge = 0

        for i in range(num_groups):
            masks = (vis == i).float().view(Fr, 1, ph, pw)  # (12, 1, 16, 16)
            masks = F.interpolate(masks, size=(H, W), mode="nearest")  # (12, 1, 16, 16)
            masks = to_np(masks)

            color = (masks * imgs).sum(axis=(0, 2, 3)) / np.expand_dims(masks.sum(),
                                                                        -1)  # [0.53578436 0.37368262 0.3504749]
            mask_eroded = np.stack([binary_erosion(mask[0]) for mask in masks])[:, None]  # (8, 1, 224, 224)
            mask_edge = masks - mask_eroded

            if not np.isfinite(color).all():
                color = np.zeros(3)

            vis_merge = vis_merge + mask_eroded * color.reshape(1, 3, 1, 1)  # (12, 3, 224, 224)
            vis_merge = vis_merge + mask_edge * np.array(cmap[i]).reshape(1, 3, 1, 1)  # (12, 3, 224, 224)
            # vis_img = vis_img + mask_edge * np.repeat(np.array(cmap[i]).reshape(1, 3, 1, 1), Fr, axis=0) # (12, 3, 1, 1)

        if source_ummerge.sum() and unmerge:
            masks = source_ummerge.sum(dim=1).float().view(Fr, 1, ph, pw)  # (12, 1, 14, 14)
            masks = F.interpolate(masks, size=(H, W), mode="nearest")  # (12, 1, 14, 14)
            masks = to_np(masks)
            vis_merge = vis_merge * (1 - masks) + masks

        vis_merge = torch.from_numpy(vis_merge.astype(np.float32))
    else:
        vis_merge = None

    # SECTION II. PRUNING
    vis_prune = to_np((1 - source.sum(dim=1)).reshape(12, ph, pw)) if prune else None

    return vis_prune, vis_merge  # (12, 3, 224, 224)

