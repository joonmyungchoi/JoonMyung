
import numpy as np
import GPUtil
import random
import torch
import os


def setGPU(gpuNum):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuNum
    print('Available devices ', torch.cuda.device_count())
    print('Current cuda device ', torch.cuda.current_device())


def selectGPU(gpuNum, p=True):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuNum)
    if p:
        print('Available devices ', torch.cuda.device_count())
        print('Current cuda device ', torch.cuda.current_device())


def fixSeed(seed, fast=False, p = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if fast:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        # use_deterministic_algorithms : deterministic보다 더 넓은 범위 커버, non-deterministic한 알고리즘이 존재하는 경우 Error 발생
        # pytorch 1.8.0부터 사용
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.__version__ > "1.8.0":
            torch.use_deterministic_algorithms(True)

    if p: print("torch : {} \tcuda : {} \tnumpy : {}".format(torch.initial_seed(), torch.cuda.initial_seed(), np.random.get_state()[1][0]))


def on_terminate(proc):
    print("process {} terminated".format(proc))


def gpuUtil():
    GPUtil.showUtilization()