import torch
import numpy

def dot_product(x, y):
    z = torch.matmul(x, y.transpose(1, 2))
    z.requires_grad = True
    return -z


def draw(matrixes, vmin=None, vmax=None, col=1, p=False, title=[], fmt=1):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
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
        ax = axes[e // col][e % col]
        sns.heatmap(pd.DataFrame(matrix), annot=annot, fmt=".{}f".format(fmt), cmap='Greys'
                    , yticklabels=False, xticklabels=False, vmin=vmin, vmax=vmax
                    , linewidths=.1, linecolor='black'
                    , ax=ax)

        ax.set(title=title[e])
    plt.show()


def validation(i, minimum=0, maximum=8):
    return True if (minimum <= i and i < maximum) else False


def minGamma(matrix, i, j, r=8, c=8, gamma=0.1, p=False):
    if i == 0 and j == 0:
        return torch.tensor([0])

    if validation(i - 1, maximum=r) and validation(j - 1, maximum=c):
        a, b, c = -matrix[i - 1, j] / gamma, -matrix[i, j - 1] / gamma, -matrix[i - 1, j - 1] / gamma
        res = torch.log(torch.exp(a) + torch.exp(b) + torch.exp(c))
    elif validation(i - 1, maximum=r):
        res = -matrix[i - 1, j] / gamma
    elif validation(j - 1, maximum=c):
        res = -matrix[i, j - 1] / gamma
    else:
        return False
    if p: print("val : {}".format(-gamma * res))
    return -gamma * res

def softmin(m, dim=0, gamma=1.0):
    return torch.exp(-m/gamma)/torch.exp(-m/gamma).sum(dim=dim).unsqueeze(dim=dim)





costV = torch.tensor(
    [[-1.0000, -0.1984, -0.2729, -0.2018, -0.2683, -0.7421, -0.3576,
          -0.2739],
         [-0.1984, -1.0000, -0.9504, -0.9674, -0.9548, -0.1404, -0.1813,
          -0.2436],
         [-0.2729, -0.9504, -1.0000, -0.9336, -0.9820, -0.2252, -0.1879,
          -0.2557],
         [-0.2018, -0.9674, -0.9336, -1.0000, -0.9366, -0.1817, -0.1706,
          -0.2143],
         [-0.2683, -0.9548, -0.9820, -0.9366, -1.0000, -0.1943, -0.1880,
          -0.2534],
         [-0.7421, -0.1404, -0.2252, -0.1817, -0.1943, -1.0000, -0.5650,
          -0.5052],
         [-0.3576, -0.1813, -0.1879, -0.1706, -0.1880, -0.5650, -1.0000,
          -0.7761],
         [-0.2739, -0.2436, -0.2557, -0.2143, -0.2534, -0.5052, -0.7761,
          -1.0000]])



c = torch.tensor([
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                ])

c.requires_grad = True


def getGradient(loss, matrix):
    return torch.autograd.grad(loss, matrix, retain_graph=True)[0].cpu().detach().squeeze(0)



def dtw_typeC(cost, costV, r=8, c=8, gamma=0.1):
    costV_sm = softmin(costV, gamma=0.1, dim=0)

    tc = torch.zeros_like(cost)
    tc_cluster = torch.zeros_like(cost).double()
    for i in range(0, r):
        for j in range(0, c):
            tc[i, j] = minGamma(tc, i, j, r=r, c=c, gamma=gamma) + cost[i, j]
            tc_cluster[::, j] += tc[i, j] * costV_sm[i]
    return tc, tc_cluster





tc, tcc = dtw_typeC(c, costV)
# tc, tcC = getGradient(l[-1,-1], c)
print()
