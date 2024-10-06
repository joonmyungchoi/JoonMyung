import torch
import torch.nn as nn
a_1 = torch.rand(3,1,requires_grad=True)
a_2 = nn.Parameter(torch.randn(3, 2))
b = torch.nn.Linear(3, 2)


v = torch.optim.Adam(b.parameters(), lr=0.0001, weight_decay=0.0001)
v.param_groups.append({'params': a_1 })
v.param_groups.append({'params': a_2 })


loss = 10 - b(torch.cat((a_1,a_2), dim=1)).sum()
loss.backward()
