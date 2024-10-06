import torch

# RANDPERM
from timm.data.mixup import one_hot

from joonmyung.meta_data import makeSample


def randperm(T, B, T_mix):
    return torch.stack([torch.randperm(T, device="cpu") for i in range(B)])[:,:T_mix]

def topK(T, B, T_mix):
    return torch.topk(torch.randn(B, T, device="cuda"), T_mix, dim=1)[1]



def one_hot_v1(B, C, smoothing=0.0, device='cuda'):  # 1.0 ✓
    T = torch.randint(low=0, high=C, size=(B, 1), device=device)
    off_value = smoothing / C
    on_value = 1. - smoothing + off_value
    y = one_hot(T, C, on_value = on_value, off_value = off_value, device=device)
    return y

def one_hot_v2(B, C, smoothing=0.0, device="cuda"):  # 1.83
    T = torch.randint(low=0, high=C, size=(B, ), device=device)
    off_value = smoothing / C
    on_value = 1. - smoothing + off_value

    y = torch.eye(C, device=device)[T]
    y = y * on_value + off_value
    return y

def one_hot_v3(B, C, lam=1., smoothing=0.0, device='cuda'):
    T = torch.randint(low=0, high=C, size=(B,), device=device)
    off_value = smoothing / C
    on_value = 1. - smoothing + off_value
    y1 = one_hot(T, C, on_value=on_value, off_value=off_value, device=device)
    y2 = one_hot(T.flip(0), C, on_value=on_value, off_value=off_value, device=device)
    return y1 * lam + y2 * (1. - lam)
# if __name__ == "__main__":
#     one_hot_v2(100, 1000, 0.1)



a = torch.randn((2, 3, 12))
idx = torch.tensor([[0,1], [1,2]])
print(1)
a[range(2), idx]
torch.gather(a, 1, idx.unsqueeze(-1).repeat(1,1,12))





