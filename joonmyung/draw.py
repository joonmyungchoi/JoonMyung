from torchvision.transforms.functional import to_pil_image
from joonmyung.utils import to_leaf, to_tensor
from joonmyung.data import normalization
from matplotlib import pyplot as plt
import torch.utils.data.distributed
import matplotlib as mpl
import torch.nn.parallel
import torch.utils.data
from PIL import Image
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import torch.optim
import torch
import cv2
import PIL
import os

def drawHeatmap(matrixes, col=1, title=[], fmt=1, p=False,
                vmin=None, vmax=None, xticklabels=False, yticklabels=False,
                linecolor=None, linewidths=0.1, fontsize=30, r=[1,1],
                cmap="Greys", cbar=True, border=False,
                output_dir='./result', save_name=None, show=True,
                vis_x= False, vis_y =  False):
    row = (len(matrixes) - 1) // col + 1
    annot = True if fmt > 0 else False
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

    if title:
        title = title + list(range(len(title), len(matrixes) - len(title)))
    fig, axes = plt.subplots(nrows=row, ncols=col, squeeze=False)
    fig.set_size_inches(col * 8 * r[1], row * 8 * r[0])
    # fig.set_size_inches(8, 8)
    fig.patch.set_facecolor('white')
    for e, matrix in enumerate(tqdm(matrixes)):
        if type(matrix) == torch.Tensor:
            matrix = matrix.detach().cpu().numpy()
        ax = axes[e // col][e % col]
        res = sns.heatmap(pd.DataFrame(matrix), annot=annot, fmt=".{}f".format(fmt), cmap=cmap
                          , vmin=vmin, vmax=vmax, yticklabels=yticklabels, xticklabels=xticklabels
                          , linewidths=linewidths, linecolor=linecolor, cbar=cbar, annot_kws={"size": fontsize / np.sqrt(len(matrix))}
                          , ax=ax)

        # if border:
        #     for _, spine in res.spines.items():
        #         spine.set_visible(True)
        if not border:
            ax.set_axis_off()
        # if columns:
        #     ax.set_title(columns[c_num] + str(r_num)) if len(columns) == col else ax.set_title(columns[i])
        if not vis_x: ax.xaxis.set_visible(False)
        if not vis_y: ax.yaxis.set_visible(False)

        if title:
            ax.set(title="{} : {}".format(title, e))
    plt.tight_layout()
    if output_dir and save_name:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, save_name), transparent = True)
    if show:
        plt.show()


def drawLinePlot(datas, index, col=1, title=[], xlabels=None, ylabels=None, markers=False, columns=None, p=False):
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
    plt.tight_layout()
    plt.show()

def drawBarChart(df, x, y, splitColName, col=1, title=[], fmt=1, p=False, c=False, c_sites={}, showfliers=True):
    d2s = df[splitColName].unique()
    d1 = df['d1'].unique()[0]
    d2s = [d2 for d2 in d2s for c_site in c_sites[d1].keys() if c_site in d2]

    row = (len(d2s) - 1) // col + 1

    fig, axes = plt.subplots(nrows=row, ncols=col, squeeze=False)
    fig.set_size_inches(col * 12, row * 12)
    for e, d2 in enumerate(tqdm(d2s)):
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
             activate = [True, True, True], device="cpu"):

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

    saliencys = saliencys.to(device)

    L, B, _, T = saliencys.shape # (L(12), B(1), T(197), T(197))
    H = W = int(T ** 0.5)

    result = {}
    if activate[0]:
        rollouts, I = [], torch.eye(T, device=device).unsqueeze(0).expand(B, -1, -1)  # (B, 197, 197)
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



def data2PIL(datas):
    if type(datas) == torch.Tensor: # MAKE TO (..., H, W, 3)
        if datas.shape[-1] == 3:
            pils = datas
        elif len(datas.shape) == 2:
            pils = datas.unsqueeze(-1).detach().cpu()
        elif len(datas.shape) == 3:
            pils = datas.permute(1, 2, 0).detach().cpu()
        elif len(datas.shape) == 4:
            pils = datas.permute(0, 2, 3, 1).detach().cpu()
    elif type(datas) == np.ndarray:
        if len(datas.shape) == 2:
            datas = np.expand_dims(datas, axis=-1)
        # TODO NEED TO CHECK
        # if datas.max() <= 1:
        #     pils = cv2.cvtColor(datas, cv2.COLOR_BGR2RGB)  # 0.29ms
        # else:
        #     pils = datas

        # image = Image.fromarray(image)                 # 0.32ms
        pils = cv2.cvtColor(datas, cv2.COLOR_BGR2RGB)  # 0.29ms

    elif type(datas) == PIL.JpegImagePlugin.JpegImageFile \
         or type(datas) == PIL.Image.Image:
        pils = datas
    else:
        raise ValueError
    return pils


def drawImgPlot(datas:list, col=1, title:str=None, columns=None, showRows:list=None,
                output_dir='./', save_name=None, show=True, fmt=1,
                vis_x = False, vis_y = False, border=False, p=False):
    if type(datas) != list or 'shape' not in dir(datas[0]) : datas = [datas]

    if showRows is not None:
        for d in range(len(datas)):
            datas[d] = datas[d][showRows]
    data_num = len(datas[0]) * len(datas)
    row = (data_num - 1) // col + 1

    fig, axes = plt.subplots(nrows=row, ncols=col, squeeze=False)
    fig.set_size_inches(col * 8, row * 8)
    if title: fig.suptitle(title, fontsize=16)
    for i in range(data_num):
        r_num, c_num   = i // col, i % col
        dr_num, dc_num = i // len(datas), i % len(datas)
        data = data2PIL(datas[dc_num][dr_num])
        ax = axes[r_num][c_num]
        if "shape" not in dir(data): # IMAGE
            ax.imshow(data)
        elif data.shape[-1] == 3: #
            ax.imshow(data)
        elif data.shape[-1] == 1: #
            ax.imshow(data, cmap="gray")
            # sns.heatmap(data[:,:,0], annot=True, fmt=f".{fmt}f", cmap="Greys"
            #             , yticklabels=False, xticklabels=False, ax=ax, vmax=1.0, vmin=0.0)
        if not border:
            ax.set_axis_off()
        if columns:
            ax.set_title(columns[c_num] + str(r_num)) if len(columns) == col else ax.set_title(columns[i])
        if not vis_x: ax.xaxis.set_visible(False)
        if not vis_y: ax.yaxis.set_visible(False)
    plt.tight_layout()

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



def overlay(imgs, attnsL, dataset="imagenet"):
    attnsL = to_leaf(to_tensor(attnsL))
    imgs   = to_leaf(to_tensor(imgs))
    if type(attnsL) == list:   attnsL = torch.stack(attnsL, 0)
    if len(attnsL.shape) == 2: attnsL = attnsL.unsqueeze(0) #
    if len(attnsL.shape) == 3: attnsL = attnsL.unsqueeze(0) # L, B, h, w
    if len(imgs.shape)   == 3: imgs = imgs.unsqueeze(0)     # B, C, H, W
    if dataset:
        imgs  = unNormalize(imgs, dataset)

    results = []
    for attns in attnsL:
        for img, attn in zip(imgs, attns):
            result = overlay_mask(to_pil_image(img), to_pil_image(normalization(attn, type=0), mode='F')) # (3, 224, 224), (1, 14, 14)
            # plt.imshow(overlay_mask(to_pil_image(dataset.unNormalize(samples)[0]), to_pil_image(normalization(a[:, 0]), mode='F'), alpha=0.5))
            results.append(result)
    return results

import copy
def unNormalize(image, dataset="imagenet"):
    # images : (B, C, H, W)
    if dataset == "imagenet":
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif dataset == "cifar":
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    else:
        return image

    result = copy.deepcopy(image)
    for c, (m, s) in enumerate(zip(mean, std)):
        result[:, c].mul_(s).add_(m)
    return result

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)