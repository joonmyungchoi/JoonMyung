# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------


from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str
from joonmyung.log import MetricLogger
from joonmyung.metric import accuracy
from typing import Tuple
from thop import profile
from tqdm import tqdm
import torch
import time

def thop(model, size, *kwargs, device="cuda"):
    input = torch.randn(size, device=device)
    macs, params = profile(model, inputs=(input, *kwargs))
    return macs / 1000000000, params / 1000000

def numel(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad]) / 1000000

@torch.no_grad()
def flops(model, size, round_num=1, eval=True, device="cuda", fp16=False, **kwargs):
    if eval: model.eval()
    with torch.cuda.amp.autocast(enabled=fp16):
        inputs = torch.randn(size, device=device, requires_grad=True)
        with torch.no_grad():
            flops = FlopCountAnalysis(model, (inputs, *kwargs))
            flops_num = flops.total() / 1000000000

    print(flop_count_table(flops))
    print(f"fvcore flops : {flops_num}")
    return round(flops_num, round_num)

def benchmark(
    model: torch.nn.Module,
    device: torch.device = 0,
    input_size: Tuple[int] = (3, 224, 224),
    batch_size: int = 64,
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
    macs, p = thop(model, (1, 3, 224, 224), device=device)

    n_parameters = numel(model)
    fv_flops = flops(model, (1, 3, 224, 224), p=0, device=device)

    if verbose:
        print(f"Throughput: {throughput:.2f} im/s")
        print(f"numel  :    {n_parameters:.2f} M")
        print(f"fvcore :    {fv_flops:.2f} G")
        print(f"thop   :    {macs:.2f}G, {p:.2f}M")


    return throughput

@torch.no_grad()
def evaluate_benchmark(data_loader, model, device):

    metric_logger = MetricLogger(delimiter="  ")
    header = 'BENCHMARK:'

    # switch to evaluation mode
    model.eval()
    import time
    start = time.time()
    cnt_token, cnt_token_diff, total = None, None, 0
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        total += batch_size

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    end = time.time()
    elapsed = end - start

    throughput = total / elapsed
    print(f"Throughput: {throughput:.2f} im/s")
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



# from benchmark import benchmark, evaluate_benchmark
# evaluate_benchmark(data_loader_val, model, device)
# benchmark(
#     model,
#     device="cuda",
#     verbose=True,
#     runs=50,
#     batch_size=256,
#     input_size=(3, 224, 224)
# )
# return