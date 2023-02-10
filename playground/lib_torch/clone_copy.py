import torch
import copy

a = torch.tensor([10,10], dtype=torch.float, requires_grad=True)
b = torch.tensor([10,10], dtype=torch.float, requires_grad=True)

c = (a * b).mean()

# (5.,5.)
#  1. clone()
## 1.1 grad_fn (O)
d = c.clone()
d.backward()
print("a.grad : ", a.grad)
#  (5.0, 5.0)

## 1.2 grad (X)
e = a.clone()
print("e.grad : ", e.grad)
#      None

# 2. copy.deepcopy
## 2.1 grad_fn (X)
# f = copy.deepcopy(c) # Error
# f.backward()
# print(a.grad)

## 2.2 grad (O)
g = copy.deepcopy(a)
print("g.grad : ", g.grad)
#  (5.0, 5.0)
