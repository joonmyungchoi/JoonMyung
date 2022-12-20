from matplotlib import pyplot as plt
from model.model import MTN_auxi
from utils.utils import to_np
import torch.utils.data.distributed
import torch.nn.parallel
import torch.utils.data
import seaborn as sns
import pandas as pd
import numpy as np
import torch.optim
import torch
import pickle


def drawHeatmap(matrixes, col=1, title=[], fmt=1, p=False,
                vmin=None, vmax=None, xticklabels=False, yticklabels=False,
                linecolor=None, linewidths=0.1, fontsize=30,
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
    # if title:
    # title = title + list(range(len(title), len(matrixes) - len(title)))
    fig, axes = plt.subplots(nrows=row, ncols=col, squeeze=False)
    # fig.set_size_inches(col * 8, row * 8)
    fig.set_size_inches(col * 8, row * 3)

    for e, matrix in enumerate(matrixes):
        if type(matrix) == torch.Tensor:
            matrix = matrix.detach().cpu().numpy()
        ax = axes[e // col][e % col]
        sns.heatmap(pd.DataFrame(matrix), annot=annot, fmt=".{}f".format(fmt), cmap=cmap
                    , vmin=vmin, vmax=vmax, yticklabels=yticklabels, xticklabels=xticklabels
                    , linewidths=linewidths, linecolor=linecolor, cbar=cbar, annot_kws={"size": fontsize / np.sqrt(len(matrix))}
                    , ax=ax)
        # sns.set(rc={'figure.figsize': (15, 8)})
        if title:
            ax.set(title="{} : {}".format(title, e))
        ax.spines[["bottom", "top", "left", "right"]].set_visible(True)
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


# YOUCOOK Retrieval
# paths = ["/data/project/rw/joonmyung/ANA:ATTENTION/wandb/run-20220227_222133-3jy9gfms/files", # 1
#         "/data/project/rw/joonmyung/ANA:ATTENTION/wandb/run-20220227_205044-uf1v0lgu/files",  # 1
#         "/data/project/rw/joonmyung/ANA:ATTENTION/wandb/run-20220227_233135-23g43ptz/files",  # 3
#         "/data/project/rw/joonmyung/ANA:ATTENTION/wandb/run-20220228_003241-1l9gez5w/files",  # 5
#         "/data/project/rw/joonmyung/ANA:SCGEDYKER_YR/wandb/run-20220228_162617-36zqn5i6/files", # 10
#         "/data/project/rw/joonmyung/ANA:ATTENTION/wandb/run-20220228_012812-fgit8yrf/files",  # 20
#         "/data/project/rw/joonmyung/ANA:ATTENTION/wandb/run-20220228_022012-83khsjj3/files",  # 30
#         "/data/project/rw/joonmyung/ANA:ATTENTION/wandb/run-20220228_031050-1lube4qe/files",  # 50
#         "/data/project/rw/joonmyung/ANA:ATTENTION/wandb/run-20220228_040035-1xupksvd/files"]  # 100

# YOUCOOK Captioning
# paths = ["/data/project/rw/joonmyung/ANA:ATTENTION/wandb/run-20220227_204521-1i1ohy9s/files", # 1
#         "/data/project/rw/joonmyung/ANA:ATTENTION/wandb/run-20220227_221616-zxvio0ox/files", # 3
#         "/data/project/rw/joonmyung/ANA:ATTENTION/wandb/run-20220227_232025-3gawigmq/files", # 5
#         "/data/project/rw/joonmyung/ANA:ATTENTION/wandb/run-20220228_001743-rugur5h7/files", # 10
#         "/data/project/rw/joonmyung/ANA:ATTENTION/wandb/run-20220228_011053-3ay8k1b7/files", # 20
#         "/data/project/rw/joonmyung/ANA:ATTENTION/wandb/run-20220228_020214-547r4gvc/files", # 30
#         "/data/project/rw/joonmyung/ANA:ATTENTION/wandb/run-20220228_025237-2x7xc70j/files"] # 100

# YOUCOOK Retrieval + REG
# paths = ["/data/project/rw/joonmyung/ANA2.1.0/wandb/run-20220305_015435-2csvydyd/files", # 0.1
#         "/data/project/rw/joonmyung/ANA2.1.0/wandb/run-20220305_015432-z8rfsatr/files",  # 0.1
#         "/data/project/rw/joonmyung/ANA3.1.0/wandb/run-20220305_015716-3vgadzxl/files"]  # 0.1


# # YOUCOOK Captioning + REG
# paths = ["/data/project/rw/joonmyung/EX1.6.9/wandb/run-20220304_004732-2xnsdr3j/files", # 1.5
#         "/data/project/rw/joonmyung/EX1.6.9/wandb/run-20220304_004800-2rgdjzyr/files",  # 1.0
#         "/data/project/rw/joonmyung/EX1.6.9/wandb/run-20220304_004744-3vcjt7rv/files",  # 0.5
#         "/data/project/rw/joonmyung/EX1.6.9/wandb/run-20220304_004749-145i8fad/files"]  # 0.3


# ㅇㅇ
# paths = ["/data/project/rw/joonmyung/ANA:ATTENTION/wandb/run-20220228_020214-547r4gvc/files",  # Captioning
#          "/data/project/rw/joonmyung/ANA:ATTENTION/wandb/run-20220228_031050-1lube4qe/files"]  # Retrieval
paths = ["/data/project/rw/joonmyung/ANA:ATTENTION/wandb/run-20220228_020214-547r4gvc/files"]  # YOUCOOK Captioning

import torch.nn as nn
activation = nn.Softmax(dim=0)


epochs = range(0,20) # 최종 에포크 갯수
s_e = 20
loss_fix          =  1  # [0 : range,     1 : 고정]

# epochs = [29] # 에포크
# loss_fix  =  0  # [0 : range,     1 : 고정]

# epochs = [10] # 최종 에포크 갯수
# loss_fix  =  0  # [0 : range,     1 : 고정]

c_ts = [0,1,2,3,4,5,6,7]  # 비교할 테스크
vnet_T_types      = [3]  # [0 : SCALE,   1 : TASK,   3 : SCALE + TASK]
block, vmin, vmax = 100, 0.0000000001, 3
re = True
info = getInfo(paths[0], 0, p=True)
args = info["args"]
information(args)

net = MTN_auxi(t_dim=args.taskNum, f_dim=args.transformer_dim[0], i_dim=1, h1_dim=args.transformer_dim[1], h2_dim=args.transformer_dim[2], o_dim=1)
net.eval()

### SECTION B. SET Inputs, [Loss Block & Mean Losses]
with open("{}/info_{}.pickle".format(paths[0], 0), 'rb') as f:
    info = pickle.load(f)
i_losses = rangeBlock(block, vmin=vmin, vmax=vmax)
o_losses = torch.stack(info["task_losses"]).unsqueeze(-1).mean(0) # dummy
i_losses.requires_grad_(False)

print("i_losses : ", i_losses.shape)  # [100, 8, 1] : [E, T, 1]
print("o_losses : ", o_losses.shape)  # [8, 1]      : [T, 1]

### SECTION C. Get Data
result = []
result_ps = []
for path in paths:
    result_ts = []
    for vnet_T_type in vnet_T_types:
        result_es = []
        for e in epochs:  # 에포크에 대해
            net.load_state_dict(torch.load("{}/MTN_{}.pt".format(path, e), map_location=torch.device("cpu")))
            # net.load_state_dict(torch.load("{}/MELTR_{}.pt".format(path, e), map_location=torch.device("cpu")))
            info = getInfo(path, e)
            result_c_ts = []
            for c_t in c_ts:  # Task에 대해
                # i_loss = torch.stack(info["task_losses"]).unsqueeze(-1).mean(0)
                i_loss = torch.stack(getInfo(path, s_e)["task_losses"]).unsqueeze(-1).mean(0)
                i_loss.requires_grad_(False)
                if loss_fix:
                    i_loss.requires_grad_(True)
                    o_loss = net(i_loss, vnet_T_type=vnet_T_type)[0]
                    result_c_ts.append(to_np(torch.autograd.grad(o_loss, i_loss, retain_graph=True)[0][c_t]))


                    # result_c_ts.append(np.log(to_np(torch.autograd.grad(o_loss, i_loss, retain_graph=True)[0][c_t]) + 1.5))
                    # result_c_ts.append(to_np(torch.autograd.grad(o_loss, i_loss, retain_graph=True)[0][c_t]))
                    # result_c_ts.append(to_np(activation(torch.autograd.grad(o_loss, i_loss, retain_graph=True)[0])[c_t]))
                    # result_c_ts.append(to_np(activation(torch.autograd.grad(o_loss, i_loss, retain_graph=True)[0] * 3)[c_t]))
                    # result_c_ts.append(to_np(activation(torch.autograd.grad(o_loss, i_loss, retain_graph=True)[0] / 20)[c_t]))

                else:
                    result_l = []
                    for loss in i_losses:  # Loss에 대해
                        i_loss.requires_grad_(False)
                        i_loss[c_t] = loss
                        i_loss.requires_grad_(True)
                        o_loss = net(i_loss, vnet_T_type=vnet_T_type)[0]
                        result_l.append(to_np(o_loss))
                        # result_l.append(to_np(torch.autograd.grad(o_loss, i_loss, retain_graph=True)[0][c_t]))
                    result_c_ts.append(np.array(result_l))

            result_es.append(np.stack(result_c_ts).squeeze())  # (L) → (L)
        result_ts.append(np.stack(result_es, axis=0)) # (c_ts, i) → (i, c_ts)
    result_ps.append(np.stack(result_ts, axis=0)) # (E, ITER, C_TS)
result = np.concatenate(result_ps)
# result = (result - result.min())/(result.max()-result.min())
print("min : ", result.min(), "max : ", result.max())
if loss_fix:
    # drawLinePlot(result, epochs, columns=args.taskName[c_ts], col=1)  # (G, D, T)
    drawHeatmap(result.transpose(0,2,1), col=1, fmt=False, p=False, cmap="flare")  # (ALL STEP, T, T) → (SELECT, T, T)
else:
    drawLinePlot(result.squeeze(0).transpose(0,2,1), to_np(i_losses.squeeze()), columns=args.taskName[c_ts], col=1)  # (G, D, T)
    # drawHeatmap(to_np(torch.cat(atten_maps))[index], col=5, fmt=False, p=False)  # (ALL STEP, T, T) → (SELECT, T, T)

with open('./heatmap_epoch_grad_raw_v2.pickle', 'wb') as f:
    pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)





