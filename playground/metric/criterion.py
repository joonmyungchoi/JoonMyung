import torch
import torch.nn.functional as F

x = torch.tensor([0.3, 0.7])
t1 = torch.tensor([0.5, 0.5])
t2 = torch.tensor([0.6, 0.9])


def loss_get(x, t1):
    return torch.sum(-t1 * F.log_softmax(x, dim=-1), dim=-1)


print(1)