import torch
def randperm(T, B, T_mix):
    return torch.stack([torch.randperm(T, device="cpu") for i in range(B)])[:,:T_mix]

def topK(T, B, T_mix):
    return torch.topk(torch.randn(B, T, device="cuda"), T_mix, dim=1)[1]


