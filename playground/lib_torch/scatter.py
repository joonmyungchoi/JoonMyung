import torch

a = torch.zeros(2,8)
a.scatter(1, torch.tensor([[1],[5]]), 1)
print(1)

# >>> torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src)
