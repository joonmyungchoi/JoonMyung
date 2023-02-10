import torch

def rangeBlock(block, vmin=0, vmax=5):
    loss = torch.arange(vmin, vmax, (vmax - vmin) / block, requires_grad=False).unsqueeze(dim=1)
    return loss

def columnRename(df, ns):
    for n in ns:
        if n[0] in df.columns:
            df.rename(columns = {n[0]: n[1]}, inplace = True)
#     columnRemove(df, ['c1', 'c2' ... ])


def columnRemove(df, ns):
    delList = []
    for n in ns:
        if n in df.columns:
            delList.append(n)
    df.drop(delList, axis=1, inplace=True)
#     columnRename(df, [['c1_p', 'c1_a'] , ['c2_p', 'c2_a']])


def normalization(t, type = 0):
    if type == 0:
        return t / t.max()
    elif type == 1:
        return t / t.min()