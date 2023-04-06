from joonmyung.utils import time2str
from tqdm import tqdm
import subprocess
import time
import pynvml

class GPU_Worker():
    def __init__(self, gpus = [], waitTimeInit = 30, waitTime = 60, count = 0,
                 checkType:int = 0, utilRatio:int = 50, need_gpu=1, reversed=False, p = True):
        self.activate  = False
        self.gpus      = gpus
        self.waitTimeInit = waitTimeInit
        self.waitTime = waitTime
        self.checkType = checkType
        self.need_gpu = int(need_gpu)

        self.reversed  = reversed
        self.utilRatio = utilRatio
        self.p = p
        self.count = count

        self.availGPUs = []

    def getFreeRatio(self, id):
        handle = pynvml.nvmlDeviceGetHandleByIndex(id)
        use = pynvml.nvmlDeviceGetUtilizationRates(handle)
        ratio = 0.5 * (float(use.gpu + float(use.memory)))
        # ratio = float(use.memory)
        # ratio = float(use.gpu)
        return ratio

    def setGPU(self):
        if self.activate: time.sleep(self.waitTimeInit)
        else: self.activate = True

        count = self.count
        pynvml.nvmlInit()
        while True:
            availGPUs = []
            count += 1
            for gpu in self.gpus:
                handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu))

                # 1. 아무것도 돌지 않는 경우
                if self.checkType == 0 and len(pynvml.nvmlDeviceGetComputeRunningProcesses(handle)) == 0:
                    availGPUs.append(str(gpu))

                # 2. n% 이하를 사용하고 있는 경우
                elif self.checkType == 1 and self.getFreeRatio(int(gpu)) < self.utilRatio:
                    availGPUs.append(str(gpu))

            if len(availGPUs) < self.need_gpu:
                if self.p: print("{} : Wait for finish".format(count))
                time.sleep(self.waitTime)
            else:
                break
        self.availGPUs = availGPUs
        if self.p: print("Activate GPUS : ", self.availGPUs)

    def getGPU(self):
        if len(self.availGPUs) < self.need_gpu: self.setGPU()
        if self.reversed:
            gpus, self.availGPUs = self.availGPUs[:self.need_gpu], self.availGPUs[self.need_gpu:]
        else:
            self.availGPUs, gpus = self.availGPUs[:-self.need_gpu], self.availGPUs[-self.need_gpu:]
        return ','.join(gpus)

def Process_Worker(processes, gpuWorker, p = True):
    start = time.localtime()
    print("------ Start Running!! : {} ------".format(time2str(start)))

    for i, process in enumerate(tqdm(processes)):
        gpu = gpuWorker.getGPU()
        prefix = f"CUDA_VISIBLE_DEVICES={gpu} nohup "
        suffix = f" > {i+1}:gpu{gpu}.log 2>&1 &"
        print("------ {}:GPU{}  {} ------".format(i + 1, gpu, prefix + process + suffix))
        subprocess.call(prefix + process + suffix, shell=True)

    end = time.localtime()
    print("------ End Running!! : {} ------".format(time2str(end)))
    print("Training Time :  : {} ------".format(time2str(end - start)))



if __name__ == '__main__':
    # Wokring Sample
    processes = [1,2,3,4,5]
    gpuWorker = GPU_Worker([0,1,2,3], 30, 120, checkType=1, utilRatio=50, need_gpu=4)
    Process_Worker(processes, gpuWorker)