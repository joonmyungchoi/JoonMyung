from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str
from thop import profile
import torch

class AverageMeter:
    ''' Computes and stores the average and current value. '''
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def __str__(self):
        return "\
        end = time.time() \n\
        batch_time = AverageMeter() \n\
        batch_time.update(time.time() - end) \n\
        end = time.time() \n\
        avg_score = AverageMeter()\n\
        accuracy = 0.1\n\
        avg_score.update(accuracy)\n\
        losses = AverageMeter()\n\
        loss = 0\n\
        batch_size = 128\n\
        losses.update(loss.data.item(), batch_size)\n\
        print(f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\n\
              f'loss {losses.val:.4f} ({losses.avg:.4f})\t' \n\
              f'acc {avg_score.val:.4f} ({avg_score.avg:.4f})')"



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


