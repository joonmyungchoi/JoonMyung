import argparse
import ast, os
import subprocess
import time
import time
import pynvml
from tqdm import tqdm

def on_terminate(proc):
    print("process {} terminated".format(proc))

def str2list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class GPU_Worker():
    def __init__(self, gpus = [], waitTimeInit = 30, waitTime = 60,
                 checkType = 0, reversed=False, p = True):
        self.activate  = False
        self.gpus      = gpus
        self.checkType = checkType
        self.waitTimeInit  = waitTimeInit
        self.waitTime = waitTime
        self.reversed = reversed
        self.p = p

    def setGPU(self):
        time.sleep(self.waitTimeInit) if self.activate else self.activate = True

        availGPUs, count = [], 0
        pynvml.nvmlInit()
        while True:
            count += 1
            for gpu in self.gpus:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)

                # 1. 아무것도 돌지 않는 경우
                if checkType == 0 and len(pynvml.nvmlDeviceGetComputeRunningProcesses(handle)) == 0:
                    availGPUs.append(gpu)

                # 2. 70% 이하를 사용하는 경우
                elif checkType == 1 and getFreeRatio(i) < 70:
                    available.append(i)

            # for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            #     result[gpu] = [proc.pid, proc.usedGpuMemory]

            if len(availGPUs) == 0:
                if self.p: print("{} : Wait for finish".format(count))
                time.sleep(self.waitTime)

            else:
                break
        self.availGPUs = availGPUs
        if self.p: print("Activate GPUS : ", gpus)

    def getGPU(self):
        if len(self.gpus) == 0: self.setGPU()
        return self.gpus.pop() if self.reversed else self.gpus.pop(0)


def time2str(time, type = 0):
    if type == 0:
        return "{:4s}.{:2s}.{:2s} {:2s}:{:2s}:{:2s}".format(time.tm_year, time.tm_mon, time.tm_mday, time.tm_hour, time.tm_min, time.tm_sec)
    else:
        raise ValueError()



def Process_Worker(processes, gpuWorker, p = False):
    start = time.localtime()
    print("------ Start Running!! : {} ------".format(time2str(start)))

    for i, process in enumerate(tqdm(processes)):
        gpu = gpuWorker.get(GPU)
        prefix = f"CUDA_VISIBLE_DEVICES={gpu} nohup "
        suffix = f" > {i+1}:gpu{gpu}.log 2>&1 &"
        if p: print("{}:GPU{}:{}").format(i + 1, process)
        subprocess.call(prefix + process + suffix, shell=True)

    end = time.localtime()
    print("------ End Running!! : {} ------".format(time2str(end)))


parser = argparse.ArgumentParser(description="tuning")
# a. Setting
parser.add_argument('--gpus', default=[0, 1, 2, 3, 4, 5], type=str2list)
parser.add_argument('--server', default='152', type=str)
parser.add_argument('--wandb_table', default='OURS', type=str)
parser.add_argument('--wandb_version', default='1.1.0', type=str, help="version")
parser.add_argument('--wandb_project', default='2023ICCV', type=str, help="version")
parser.add_argument('--wandb_entity',  default='joonmyung', type=str, help="version")

# b. Setting
parser.add_argument('--seed', default=[20170922], type=str2list)
parser.add_argument('--dataset', default=["cifar100"], help="", type=str2list)

# c. Tunning
parser.add_argument('--lr', default=[0.1], help="", type=str2list)
args = parser.parse_args()

processes = []
for seed in args.seed:
    for dataset in args.dataset:
        for lr in args.lr:

            process = f"python train.py \
                  --server {args.server}  --seed {seed} --dataset {dataset} \
                  --use_wandb --wandb_entity {args.wandb_entity} --wandb_project {args.wandb_project} --wandb_table {args.wandb_table} --wandb_version {args.wandb_version} \
                  --lr {lr}"
            processes.append(process)



gpuWorker = GPU_Worker(args.gpus, 30, 120)
Process_Worker(processes, gpuWorker)
# CUDA_VISIBLE_DEVICES=0 python train_baseline.py --lr=0.1 --seed=20170922 --decay=1e-4 --use_wandb --wandb_entity joonmyung --wandb_project 2023ICCV --wandb_table Baseline --wandb_version 1.0.0