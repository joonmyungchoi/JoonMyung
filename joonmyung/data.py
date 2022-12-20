import torch
import pickle
def rangeBlock(block, vmin=0, vmax=5):
    loss = torch.arange(vmin, vmax, (vmax - vmin) / block, requires_grad=False).unsqueeze(dim=1)
    return loss



# DataFrame
def columnRename(df, ns):
    for n in ns:
        if n[0] in df.columns:
            df.rename(columns = {n[0]: n[1]}, inplace = True)
#     columnRemove(df, ['CON밸브A','CON밸브A연산값','CON밸브B','CON밸브개도','CON밸브연산값'])


def columnRemove(df, ns):
    delList = []
    for n in ns:
        if n in df.columns:
            delList.append(n)
    df.drop(delList, axis=1, inplace=True)
#     columnRename(df, [
#                 ['CON레벨센서', 'CON레벨A']
#                 , ['CON레벨센서2', 'CON레벨B'])
