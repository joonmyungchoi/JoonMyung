from fvcore.nn import FlopCountAnalysis, flop_count_table
from joonmyung.compression import resetInfo
from torchprofile import profile_macs
from thop import profile
from tqdm import tqdm
import tracemalloc
import statistics
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


def get_macs(model, size,
             eval = True, round_num=1, device="cuda"):
    if eval: model.eval()
    inputs = torch.randn(size, device=device, requires_grad=True)
    macs = profile_macs(model, inputs)
    print(f"torchprofile MACS : {macs}")

    return round(macs, round_num)


def dataGenerator(case, batch_size=1, n_page=1, token_enc=10032, token_dec = 2560, layer_len = 28, shape = None, device="cuda", dtype=torch.float32):
    if case == "VISUAL":
        hidden_states = torch.rand((token_enc, 1176), device=device, dtype=dtype)
        grid_thw = torch.tensor([[1, 114, 88]], device=device, dtype=torch.int32).expand(n_page, -1)
        data = [hidden_states, grid_thw]
    elif case == "CACHE":
        attention_mask = torch.ones((1, token_dec), device=device, dtype=torch.bool)
        position_ids = torch.arange(0, token_dec, device=device, dtype=torch.int32).repeat(3, 1)[:, None]
        input_embeds = torch.rand((1, token_dec, 3584), device=device, dtype=dtype)
        cache_position = torch.arange(0, token_dec, device=device, dtype=torch.int32)
        data = [None, attention_mask, position_ids, None, input_embeds, True, False, False, True, cache_position]
    elif case == "GEN":
        # TODO: 계속 self.attn에서 누적되는거 수정필요
        attention_mask = torch.ones((1, token_dec + 1), device=device, dtype=torch.bool)
        position_ids = torch.ones(1, 1, device=device, dtype=torch.int32).repeat(3, 1)[:, None]
        inputs_embeds = torch.rand((1, 1, 3584), device=device, dtype=dtype)
        from transformers import DynamicCache
        past_key_values = DynamicCache()

        for layer_idx in range(layer_len):
            key = torch.rand((1, 4, token_dec, 128), device=device, dtype=dtype)
            value = torch.rand((1, 4, token_dec, 128), device=device, dtype=dtype)
            past_key_values.update(key, value, layer_idx)

        cache_position = torch.tensor([token_dec], device=device, dtype=torch.int64)
        data = [None, attention_mask, position_ids, past_key_values, inputs_embeds, True, False, False, True, cache_position]
    else:
        pixel_values = torch.rand(shape, device=device, dtype=dtype) \
                if shape != None else torch.rand((batch_size, 3, 336, 336), device=device, dtype=dtype)
        data = [pixel_values]

    return data

@torch.no_grad()
def flops(model, batch_size = 1, n_page = 1, drop_rate=0.0, case=None, round_num=1, eval=True, max_depth = 1, dtype=torch.bfloat16, verbose=False, shape=None, device="cuda"):
    if eval: model.eval()

    token_enc, token_dec = n_page * 10032, int(n_page * 2508 * (1 - drop_rate)) + 73
    inputs = dataGenerator(case, batch_size=batch_size, token_enc=token_enc, token_dec = token_dec, layer_len = 28, shape=shape, device=device, dtype = dtype)

    with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
        flops = FlopCountAnalysis(model, (*inputs,))
        flops_num = flops.total() / 1000000000

    if verbose:
        print(flop_count_table(flops, max_depth=max_depth))
        print(f"fvcore flops : {flops_num}")

    return round(flops_num, round_num)

@torch.no_grad()
def benchmark(
    model: torch.nn.Module,
    batch_size:int = 1,
    n_page:int = 1,
    drop_rate:float = 0.0,
    runs: int = 40,
    throw_out: float = 0.25,
    verbose: bool = False,
    case:str = None,
    shape=None,
    device  = "cuda",
    dtype = torch.bfloat16,
) -> float:
    model = model.eval().to(device)
    warm_up = int(runs * throw_out)

    token_enc, token_dec = n_page * 10032, int((n_page * 2508 + (n_page - 1) * 2) * (1 - drop_rate)) + 73
    inputs = dataGenerator(case, batch_size=batch_size, n_page=n_page, token_enc=token_enc, token_dec=token_dec, layer_len=28, shape=shape, device=device)

    total, times, peak_memories = 0, [], []
    for i in tqdm(range(runs), disable=not verbose, desc="Benchmarking"):
        if i == warm_up:
            start, total, times, peak_memories = time.time(), 0, [], []

        tracemalloc.start()
        start_gpt = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
            model(*inputs)
        end_gpt = time.perf_counter()
        total += batch_size

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append(end_gpt - start_gpt)
        peak_memories.append(peak / 1024)  # KB로 변환

    avg_time = statistics.mean(times)
    std_time = statistics.stdev(times)
    avg_memory = statistics.mean(peak_memories)
    std_memory = statistics.stdev(peak_memories)

    end = time.time()
    elapsed = end - start
    throughput = total / elapsed
    if verbose:
        print(f"🔁 Benchmark over {int(runs * (1 - throw_out))} runs:")
        print(f"⏱️ Time:   {avg_time:.6f} sec (± {std_time:.6f})")
        print(f"📦 Memory: {avg_memory:.2f} KB (± {std_memory:.2f})")
        print(f"Throughput: {throughput:.2f} im/s")
        print()

    return throughput


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


