from matplotlib import pyplot as plt, axes
from model.model import MTN, MTN_auxi
from utils.utils import makeInput, to_np
import torch.utils.data.distributed
import torch.nn.parallel
import torch.utils.data
import seaborn as sns
import pandas as pd
import numpy as np
import torch.optim
import pprint
import torch



def draw(matrixes, col=1, title=[], fmt=1, p=False,
         vmin=None, vmax=None, xticklabels=1, yticklabels=1,
         linecolor=None, linewidths=0.1,
         cmap="Greys", cbar=True):


    row = (len(matrixes) - 1) // col + 1
    annot = True if fmt > 0 else False
    if p:
        print("|- Parameter Information")
        print("  |- Matrix     : 입력 값")
        print("  |- title      : 컬럼별 제목")
        print("  |- col        : 컬럼 갯수")
        print("  |- p          : 정보 출력")
        print("  |- vmin/vmax  : 히트맵 최소/최대 값")
        print("  |- linecolor  : black, ...   ")
        print("  |- linewidths : 1.0...   ")
        print("  |- fmt        : 숫자 출력 소숫점 자릿 수")
        print("  |- cmap        : Grey")
        print("  |- cbar        : 오른쪽 바 On/Off")
        print("  |- xticklabels : x축 간격 (False, 1,2,...)")
        print("  |- yticklabels : y축 간격 (False, 1,2,...)")
        print()
        print("|- Graph Information")
        print("  |- row : {}, col : {}".format(row, col))
        print("  |- height : {}, width : {}".format(row * 8, col * 8))

    title = title + list(range(len(title), len(matrixes) - len(title)))
    fig, axes = plt.subplots(nrows=row, ncols=col, squeeze=False)
    fig.set_size_inches(col * 8, row * 8)

    for e, matrix in enumerate(matrixes):
        if type(matrix) == torch.Tensor:
            matrix = matrix.detach().cpu().numpy()
        ax = axes[e // col][e % col]
        sns.heatmap(pd.DataFrame(matrix), annot=annot, fmt=".{}f".format(fmt), cmap=cmap
                    , vmin=vmin, vmax=vmax, yticklabels=yticklabels, xticklabels=xticklabels
                    , linewidths=linewidths, linecolor=linecolor, cbar=cbar
                    , ax=ax)

        ax.set(title=title[e])
    plt.show()

def makeSample(shape, min=None,max=None, dataType=int, outputType=np, columns=None):
    if dataType == int:
        d = np.random.randint(min, max, size=shape)
    elif dataType ==float:
        d = np.random.uniform(low=min, high=max, size=shape)

    if outputType == np:
        return d
    elif outputType == pd:
        # df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
        return pd.DataFrame(d, columns=None)




# pp = makeInput((3,3,3), 0, 1, float, np)
# draw(pp, vmin=0, vmax=1, col=3, p=True)




path = "/data/project/rw/joonmyung/EX1.1.2_2/wandb/run-20220225_022246-1ui3c0tl/files"
args = torch.load("{}/args.pt".format(path))["args"]
args.taskNum = (np.array(args.tasks) == 1).sum()
net = MTN_auxi(t_dim=args.taskNum, f_dim=args.transformer_dim[0], i_dim=1, h1_dim=args.transformer_dim[1], h2_dim=args.transformer_dim[2], o_dim=1)
length = 10
for i in range(0, length):
    result = [[] for _ in range(length)]
    with torch.no_grad():
        losses, tasks = makeInput(0, block=500, lossMin=0, lossMax=5)
        net.load_state_dict(torch.load("{}/MTN_{}.pt".format(path, i)))
        temps = []
        for loss in losses:
            temps.append(net(loss.repeat(8,1))[1]) # (1,8,8)
        result[i].append(torch.cat(dim=0))

# losses = to_np(losses)
#
# plt.figure(figsize=(9.16, 4.77))
# #     plt.xticks([i for i in range(0, 1, 20)])
# output = to_np(result)
# plt.plot(losses, output)
# axes = plt.axes()
# axes.set_ylim([0, 1])
#
# plt.title("loss - weight graph {}".format(0))
# plt.xlabel("Loss")
# plt.ylabel("Weight")
#
# plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
# plt.tight_layout()
#
# plt.show()