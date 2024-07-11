import pickle

from joonmyung.utils import is_dist_avail_and_initialized
from collections import defaultdict, deque
from joonmyung.draw import data2PIL
from joonmyung.utils import to_np
import torch.distributed as dist
import datetime
import wandb
import torch
import time
import os

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




class Logger():
    loggers = {}
    def __init__(self, use_wandb=True, wandb_entity=None, wandb_project=None, wandb_name=None, wandb_tags=None
                 , wandb_watch=False, main_process=True, wandb_id=None, wandb_dir='./'
                 , args=None, model=False):
        self.use_wandb = use_wandb
        self.main_process = main_process

        if self.use_wandb and self.main_process:
            wandb.init(entity=wandb_entity, project=wandb_project, name=wandb_name, tags=wandb_tags
                       , save_code=True, resume="allow", id = wandb_id, dir=wandb_dir
                       , config=args, settings=wandb.Settings(code_dir="."))

            if args:
                args.wandb_id = wandb.config.id = wandb.run.id
                torch.save(args, os.path.join(wandb.run.dir, "args.pt"))
            if wandb_watch and model: wandb.watch(model, log='all')


    def getLog(self, k, return_type =None):
        if return_type == "avg":
            return self.loggers[k].avg
        elif return_type == "val":
            return self.loggers[k].val
        else:
            return self.loggers[k]

    def delLog(self, columns: list):
        for column in columns:
            self.loggers.pop(column)

    def resetLog(self):
        self.loggers = {k:AverageMeter() if type(v) == AverageMeter else v for k, v in self.loggers.items()}

    def addLog(self, datas:dict, epoch=None, bs = 1):
        if self.main_process and self.use_wandb:
            for k, v in datas.items():
                data_type = v[0]
                if data_type == 0:  # Values
                    self.loggers[k] = v[1]
                elif data_type == 1: # AverageMeter
                    if k not in self.loggers.keys():
                        self.loggers[k] = AverageMeter()
                    self.loggers[k].update(v[1], bs)
                elif data_type == 2: # Table
                    columns = list(v[1].keys())
                    data_num = len(list(v[1].values())[0])
                    self.loggers[k] = wandb.Table(columns=["epoch"] + columns)
                    for idx in range(data_num):
                        results = []
                        for c in columns:
                            if type(v[1][c]) == torch.Tensor:
                                result = wandb.Image(to_np(data2PIL(v[1][c][idx]))) if len(v[1][c].shape) == 4 else to_np(v[1][c])[idx]
                            elif type(v[1][c]) == list:
                                result = v[1][c][idx]
                            else:
                                raise ValueError
                            results.append(result)
                        self.loggers[k].add_data(str(epoch), *results)
        return True

    def getPath(self):
        return wandb.run.dir

    def logWandb(self):
        if self.main_process and self.use_wandb:
            wandb.log({k: v.avg if type(v) == AverageMeter else v for k, v in self.loggers.items()})
            self.resetLog()

    def finish(self):
        wandb.finish()

    def save(self, model, args, name):
        if self.main_process and self.use_wandb:
            path = os.path.join(wandb.run.dir, f"{name}.pth")
            torch.save({"model" : model, "args"  : args}, path)
            wandb.save(path, wandb.run.dir)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))




from pathlib import Path
def list2name(value):
    return str(value).replace(" ", "").replace("],[", "_").replace("[", "").replace(",", "").replace("]", "")
def list2str(value):
    return str(value).replace(" ", "")


def resultManager(file_name, folder_path = "./result",
                  new_result = None, checkColumns = None, duplicate = False, duplicate_check=False):
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)
    try:
        with open(file_path, 'rb') as picklefile:
            results = pickle.load(picklefile)
    except FileNotFoundError:
        results = []

    if new_result is not None:  # update log file
        if checkColumns is not None and (not duplicate or duplicate_check):  # check exist
            for result in results:
                # 하나도 안 같은경우
                if not sum([list2str(result[checkColumn]) != list2str(new_result[checkColumn]) if checkColumn in result.keys() else True for checkColumn in checkColumns]):
                    return False
            if duplicate_check:
                return True
        results.append(new_result)
    else:
        return results

    with open(file_path, 'wb') as picklefile: # pass
        pickle.dump(results, picklefile)
    print(f"saved result path : {file_path}")
    return True





# if __name__ == "__main__":
#     from playground.analysis.lib_import import *
#     dataset_name, server, device = "imagenet", "148", "cuda"
#     data_path, _ = data2path(server, dataset_name)
#     data_num = [[5, 1],
#                 [5, 2],
#                 [5, 3],
#                 [5, 4]]
#     dataset = JDataset(data_path, dataset_name, device=device)
#     samples, targets, imgs, label_names = dataset.getItems(data_num)
#     model = torch.hub.load('facebookresearch/deit:main', "deit_tiny_patch16_224", pretrained=True)
#
#     logger = Logger(use_wandb=True, wandb_entity="joonmyung", wandb_project="test", wandb_name="AAPP",
#                     wandb_watch=False, wandb_dir="./")
#
#     logger.addLog({ "sample A": [0, 4],
#                     "sample B": [0, 3],
#                     "sample C": [0, 2],
#                     "table  B": [2, {"image" :     samples, "prediction": targets}]})
#
#
#     logger.save(model.state_dict(), "checkpoint_best.pt")
#     logger.logWandb()
#     logger.finish()