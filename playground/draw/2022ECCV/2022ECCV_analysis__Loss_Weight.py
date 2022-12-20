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
import pickle


def drawLinePlot(datas, index, col=1, title=[], xlabels=None, ylabels=None, markers=False, columns=None, p=False):  # (G, D, T)

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
        print("    |- vmin/vmax  : 히트맵 최소/최대 값")
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


def makeSample(shape, min=None, max=None, dataType=int, outputType=np, columns=None):
    if dataType == int:
        d = np.random.randint(min, max, size=shape)
    elif dataType == float:
        d = np.random.uniform(low=min, high=max, size=shape)

    if outputType == np:
        return d
    elif outputType == pd:
        return pd.DataFrame(d, columns=None)
    elif outputType == torch:
        return torch.from_numpy(d)


def rangeBlock(block, vmin=0, vmax=5):
    loss = torch.arange(vmin, vmax, (vmax - vmin) / block, requires_grad=False).unsqueeze(dim=1)
    return loss


def information(args):
    task = "RETRIEVAL" if args.eval_task[0] == 1 else "CAPTIONING"
    print("|- TASK INFORMATION")
    print("  |- TASK     : ", task)
    print("  |- DATATYPE : ", args.datatype.upper())


def getInfo(path, epoch, p=True):
    with open("{}/info_{}.pickle".format(path, epoch), 'rb') as f:
        info = pickle.load(f)
    return info


import math
# YOUCOOK RETRIEVAL  : 80 Epochs
# path = "/data/project/rw/joonmyung/ANA:ATTENTION/wandb/run-20220227_205044-uf1v0lgu/files" # 1
# path = "/data/project/rw/joonmyung/ANA:ATTENTION/wandb/run-20220227_233135-23g43ptz/files" # 3
path = "/data/project/rw/joonmyung/ANA:ATTENTION/wandb/run-20220228_003241-1l9gez5w/files"
# path = "/data/project/rw/joonmyung/ANA:SCGEDYKER_YR/wandb/run-20220228_162617-36zqn5i6/files" # lr 고정 : 0.000015

epochs = 5      # 최종 에포크 갯수
term =    1      # 보여줄 간격
view_type =   1  # [0 : 스텝 전체, 1 : 스텝 평균]
vnet_T_type = 3  # [0 : SCALE,   1 : TASK,   3 : SCALE + TASK]
re=True
info = getInfo(path, 0, p=True)
args= info["args"]
information(args)


net = MTN_auxi(t_dim=args.taskNum, f_dim=args.transformer_dim[0], i_dim=1, h1_dim=args.transformer_dim[1], h2_dim=args.transformer_dim[2], o_dim=1)
net.eval()
losses = rangeBlock(10, 0, 1).unsqueeze(-1).repeat(1,8,1)
losses.requires_grad_(True)


length = 1
result = []
for i in np.arange(0, epochs, length):
    net.load_state_dict(torch.load("{}/MTN_{}.pt".format(path, i), map_location=torch.device("cpu")))
    temps = []
    for loss in losses:
#         temps.append(np.log(to_np(torch.autograd.grad(net(loss)[0], loss, retain_graph=True, allow_unused=True)[0].t())))
        temps.append(to_np(torch.autograd.grad(net(loss, vnet_T_type=vnet_T_type)[0], loss, retain_graph=True, allow_unused=True)[0].t()))
    result.append(np.concatenate(temps, axis=0))
result = np.stack(result)

print("data  shape : ", result.shape)                  # (Graph, xTick, Task)
print("xtick shape : ", to_np(losses)[:,0,0].shape)    # (xTick, )

drawLinePlot(result, to_np(losses)[:,0,0], columns=args.taskName, col=3)
