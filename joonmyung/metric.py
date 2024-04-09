from fvcore.nn import FlopCountAnalysis, flop_count_table
from torchprofile import profile_macs
from typing import Tuple
from thop import profile
from tqdm import tqdm
import torch
import time

def thop(model, size, *kwargs,
         round_num=1, eval = True, device="cuda"):
    if eval: model.eval().to(device)
    input = torch.randn(size, device=device)
    macs, params = profile(model, inputs=(input, *kwargs))
    macs, params = macs / 1000000000, params / 1000000

    print(f"thop macs/params : {macs}/{params}")

    return round(macs, round_num), round(params, round_num)

def numel(model,
          round_num=1):
    params = sum([p.numel() for p in model.parameters() if p.requires_grad]) / 1000000
    print(f"numel params : {params}")
    return round(params, round_num)

@torch.no_grad()
def flops(model, size, round_num=1, eval=True, fp16=False, device="cuda", **kwargs):
    if eval: model.eval()
    with torch.cuda.amp.autocast(enabled=fp16):
        inputs = torch.randn(size, device=device, requires_grad=True)
        with torch.no_grad():
            flops = FlopCountAnalysis(model, (inputs, *kwargs))
            flops_num = flops.total() / 1000000000

    print(flop_count_table(flops))
    print(f"fvcore flops : {flops_num}")

    return round(flops_num, round_num)


def get_macs(model, size,
             eval = True, round_num=1, device="cuda"):
    if eval: model.eval()
    inputs = torch.randn(size, device=device, requires_grad=True)
    macs = profile_macs(model, inputs)
    print(f"torchprofile MACS : {macs}")

    return round(macs, round_num)

def benchmark(
    model: torch.nn.Module,
    device: torch.device = 0,
    input_size: Tuple[int] = (3, 224, 224),
    batch_size: int = 1024,
    runs: int = 40,
    throw_out: float = 0.25,
    use_fp16: bool = False,
    verbose: bool = False,
    **kwargs
) -> float:
    """
    Benchmark the given model with random inputs at the given batch size.

    Args:
     - model: the module to benchmark
     - device: the device to use for benchmarking
     - input_size: the input size to pass to the model (channels, h, w)
     - batch_size: the batch size to use for evaluation
     - runs: the number of total runs to do
     - throw_out: the percentage of runs to throw out at the start of testing
     - use_fp16: whether or not to benchmark with float16 and autocast
     - verbose: whether or not to use tqdm to print progress / print throughput at end

    Returns:
     - the throughput measured in images / second
    """

    if not isinstance(device, torch.device):
        device = torch.device(device)
    is_cuda = torch.device(device).type == "cuda"

    model = model.eval().to(device)
    input = torch.rand(batch_size, *input_size, device=device)
    if use_fp16:
        input = input.half()

    warm_up = int(runs * throw_out)
    total = 0
    start = time.time()

    with torch.autocast(device.type, enabled=use_fp16):
        with torch.no_grad():
            for i in tqdm(range(runs), disable=not verbose, desc="Benchmarking"):
                if i == warm_up:
                    if is_cuda:
                        torch.cuda.synchronize()
                    total = 0
                    start = time.time()

                model(input, **kwargs)
                total += batch_size


    if is_cuda:
        torch.cuda.synchronize()

    end = time.time()
    elapsed = end - start

    throughput = total / elapsed

    if verbose:
        print(f"Throughput: {throughput:.2f} im/s")

    return round(throughput)



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    batch_size = target.size(0)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def targetPred(output, target, topk=5):
    _, pred = output.topk(topk, 1, True, True)
    TP = torch.cat([target.unsqueeze(-1), pred], dim=1)
    return TP