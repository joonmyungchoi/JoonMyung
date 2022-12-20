import numpy as np
import torch
import torch.nn.functional as F
from numba import jit
from torch.autograd import Function

def validation(i, minimum = 0, maximum = 8):
    return True if (minimum <= i and i < maximum) else False

def minGamma(matrix, i, j, r=8, c=8, gamma=0.1, device=torch.device("cpu"), p=False):
    if i == 0 and j == 0:
        return torch.tensor([0], device=device)

    if validation(i - 1, maximum=r) and validation(j - 1, maximum=c):
        a, b, c = -matrix[i - 1, j] / gamma, -matrix[i, j - 1] / gamma, -matrix[i - 1, j - 1] / gamma
        res = torch.log(torch.exp(a) + torch.exp(b) + torch.exp(c))
    elif validation(i - 1, maximum=r):
        res = -matrix[i - 1, j] / gamma
    elif validation(j - 1, maximum=c):
        res = -matrix[i, j - 1] / gamma
    else:
        False
    if p: print("val : {}".format(-gamma * res))
    return -gamma * res

def getGradient(loss, matrix):
    return torch.autograd.grad(loss, matrix, retain_graph=True)[0].cpu().detach().squeeze(0)


# @jit(nopython = True)
def compute_loss(cost, gamma=0.1):
    '''
       compute differentiable weak sequence alignment loss
    '''
    r, c = cost.shape
    tc = torch.ones_like(cost).double()
    for i in range(0, r):
        for j in range(0, c):
            tc[i, j] = minGamma(tc, i, j, r=r, c=c, gamma=gamma) + cost[i, j]
    return tc, tc[-1,-1]

# @jit(nopython = True)
def compute_loss_backward(C, D, CV, CT, gamma): # C : Cost, D : Cummulative Matrix

    G = getGradient(D[-1,-1], C)
    ####################################### 수정 #######################################
    # torch.autograd.grad(loss, matrix, retain_graph=True)[0].cpu().detach().squeeze(0)
    return G

class _DWSALoss(Function):
    @staticmethod
    def forward(ctx, C, CV, CT, gamma):
        dev = C.device
        dtype = C.dtype
        gamma = torch.Tensor([gamma]).to(dev).type(dtype)
        g_ = gamma.item()
        D, softmin = compute_loss(C, g_)
        ctx.save_for_backward(C, CV, CT, D, softmin, gamma) # cost, cumulative Matrix, Softmin, gamma

        return softmin

    @staticmethod
    def backward(ctx, grad_output):
        import pdb;
        pdb.set_trace()
        C, CV, CT, D, softmin, gamma = ctx.saved_tensors
        G = compute_loss_backward(D, CV, CT, gamma)
        return grad_output.view(-1, 1).expand_as(G) * G, None

class Loss(torch.nn.Module):
    def __init__(self, beta=0.1, threshold=2):
        super(Loss, self).__init__()
        self.beta = beta
        self.func_apply = _DWSALoss.apply
        self.threshold = threshold

    def forward(self, cost, costV, costT):

        loss = self.func_apply(cost, costV, costT, self.beta)

        return loss
