
import os
import torch

def setGPU(gpuNum):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuNum
    print('Available devices ', torch.cuda.device_count())
    print('Current cuda device ', torch.cuda.current_device())



