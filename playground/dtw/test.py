import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import math
import copy


def draw(matrixes, vmin=None, vmax=None, col=1, p=False, title=[], fmt=1):
    row = (len(matrixes) - 1) // col + 1
    annot = True if fmt > 0 else False
    if p:
        print("row : {}, col : {}".format(row, col))
        print("height : {}, width : {}".format(row * 8, col * 8))

    title = title + list(range(len(title), len(matrixes) - len(title)))
    fig, axes = plt.subplots(nrows=row, ncols=col, squeeze=False)
    fig.set_size_inches(col * 8, row * 8)

    for e, matrix in enumerate(matrixes):
        if type(matrix) == torch.Tensor:
            matrix = matrix.detach().cpu().numpy()
        elif matrix == None:
            continue
        ax = axes[e // col][e % col]
        sns.heatmap(pd.DataFrame(matrix), annot=annot, fmt=".{}f".format(fmt), cmap='Greys'
                    , yticklabels=False, xticklabels=False, vmin=vmin, vmax=vmax
                    , linewidths=.1, linecolor='black'
                    , ax=ax)

        ax.set(title=title[e])
    plt.show()

def getGradient(loss, matrix):
    return torch.autograd.grad(loss, matrix, retain_graph=True)[0].cpu().detach().squeeze(0)

def validation(i, minimum = 0, maximum = 8):
    return True if (minimum <= i and i < maximum) else False

def dtw(cost, r=8, c=8, gamma=0.01):
    tc = torch.ones_like(cost).double()
    for i in range(0, r):
        for j in range(0, c):
            tc[:, i, j] = minGamma(tc, i, j, r=r, c=c, gamma=gamma) + cost[:, i, j]
    return tc


def minGamma(m, i, j, r=8, c=8, gamma=0.1, reverse=False, p=False):
    if reverse:
        if i == 0 and j == 0: return torch.tensor([0], device=torch.device("cpu"))
        b = torch.exp(-m[:, i + 1, j] / gamma) if validation(i + 1, maximum=c) else 0
        a = torch.exp(-m[:, i, j + 1] / gamma) if validation(j + 1, maximum=r) else 0
        c = torch.exp(-m[:, i + 1, j + 1] / gamma) if (validation(i + 1, maximum=r) and validation(j + 1, maximum=c)) else 0
        res = torch.log(a + b + c)
    else:
        if i == 0 and j == 0: return torch.tensor([0], device=torch.device("cpu"))
        a = torch.exp(-m[:, i - 1, j] / gamma) if validation(i - 1, maximum=r) else 0
        b = torch.exp(-m[:, i, j - 1] / gamma) if validation(j - 1, maximum=c) else 0
        c = torch.exp(-m[:, i - 1, j - 1] / gamma) if (validation(i - 1, maximum=r) and validation(j - 1, maximum=c)) else 0
        res = torch.log(a + b + c)

    if p: print("val : {}".format(-gamma * res))
    return -gamma * res


def backward(E, D, R, i, j, r=8, c=8):
    if i == r - 1 and j == c - 1: return torch.tensor([1], device=torch.device("cpu"))
    a = E[:, i + 1, j] * math.exp((R[:, i + 1, j] - R[:, i, j] - D[:, i + 1, j]) / gamma) if validation(i + 1, maximum=r) else 0
    b = E[:, i, j + 1] * math.exp((R[:, i, j + 1] - R[:, i, j] - D[:, i, j + 1]) / gamma) if validation(j + 1, maximum=c) else 0
    c = E[:, i + 1, j + 1] * math.exp((R[:, i + 1, j + 1] - R[:, i, j] - D[:, i + 1, j + 1]) / gamma) if (validation(i + 1, maximum=r) and validation(j + 1, maximum=c)) else 0
    return a + b + c


def backward_smooth(E_S, D_S, D, R_S, i, j, r=8, c=8):
    if i == r - 1 and j == c - 1: return torch.tensor([1], device=torch.device("cpu"))
    a = E_S[:, i + 1, j] * (math.exp((R_S[:, i + 1, j] - R_S[:, i, j] - D_S[:, i + 1, j]) / gamma) + math.exp((D_S[:, i + 1, j] - D[:, i + 1, j] - D[:, i, j]) / gamma)) if validation(i + 1, maximum=r) else 0
    b = E_S[:, i, j + 1] * (math.exp((R_S[:, i, j + 1] - R_S[:, i, j] - D_S[:, i, j + 1]) / gamma) + math.exp((D_S[:, i, j + 1] - D[:, i, j + 1] - D[:, i, j]) / gamma)) if validation(j + 1, maximum=c) else 0
    c = E_S[:, i + 1, j + 1] * (math.exp((R_S[:, i + 1, j + 1] - R_S[:, i, j] - D_S[:, i + 1, j + 1]) / gamma) + math.exp((D_S[:, i + 1, j + 1] - D[:, i + 1, j + 1] - D[:, i, j]) / gamma)) if (validation(i + 1, maximum=r) and validation(j + 1, maximum=c)) else 0
    return a + b + c


def smoothDTW(D, row=8, col=8, gamma=0.05):
    D_S = torch.ones_like(D).double()
    tc = torch.ones_like(D).double()

    for i in range(0, row):
        for j in range(0, col):
            D_S[:, i, j] = minGamma(D, i, j, gamma=gamma) + D[:, i, j]
    for i in range(0, row):
        for j in range(0, col):
            tc[:, i, j] = minGamma(tc, i, j, gamma=gamma) + D_S[:, i, j]
    return tc, D_S




i, gamma = 0, 0.1

# D = dot_product(videoInfo[i]["visual_embd"], videoInfo[i]["text_embd"])
D = torch.tensor(
    [[[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
         , [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
         , [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
         , [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
         , [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
         , [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
         , [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
         , [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]]]
    , device=torch.device("cpu"), requires_grad=True)

_, r, c = D.shape

# # 2. BaseLine
# R = dtw(D, gamma=gamma, r=r, c=c)
# GB = getGradient(R[:, -1, -1], D)
#
# E = torch.zeros_like(D)
# for i in range(7, -1, -1):
#     for j in range(7, -1, -1):
#         E[:, i, j] = backward(E, D, R, i, j, r=8, c=8)

# 3. Local SmoothDTW
# D_S = smooth(D, gamma=gamma, row=r, col=c)
R_S, D_S = smoothDTW(D, gamma=gamma, row=r, col=c)
GB_S = getGradient(R_S[:, -1, -1], D)

# draw([D[0], D_S[0], R_S[0], GB_S], col=2, fmt=2) # GP

E_S = torch.zeros_like(D)
for j in range(7, -1, -1):
    for i in range(7, -1, -1):
        E_S[:, i, j] = backward_smooth(E_S, D_S, D, R_S, i, j, r=8, c=8)

draw([D[0], D_S[0],GB_S, E_S[0]], col=2, fmt=2)  # GP

# draw([D[0], D_S[0], GB, E[0], GB_S, E_S[0]], col=2, fmt=2)  # GP
# videoInfo[i]["analysisInfo"]


