from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str
from thop import profile
import torch

def thop(model, size):
    input = torch.randn(size)
    macs, params = profile(model, inputs=(input,))
    return macs, params

def numel(model):
    return sum([m.numel() for m in model.parameters()])

def flops(model, inputs, p=1):
    flops = FlopCountAnalysis(model, inputs)
    if p == 1:
        print(flop_count_table(flops))
    elif p == 2:
        print(flop_count_str(flops))
    return flops.total(), flops.by_operator(), flops.by_module(), flops.by_module_and_operator()


