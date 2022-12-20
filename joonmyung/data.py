import torch
import pickle
def rangeBlock(block, vmin=0, vmax=5):
    loss = torch.arange(vmin, vmax, (vmax - vmin) / block, requires_grad=False).unsqueeze(dim=1)
    return loss


# def getInfo(path, epoch, p=True):
#     with open("{}/info_{}.pickle".format(path, epoch), 'rb') as f:
#         info = pickle.load(f)
#     return info
