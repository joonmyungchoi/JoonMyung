from joonmyung.utils import time2str
from tqdm import tqdm
import subprocess
import time
import pynvml
import datetime

class GPU_Worker():
    def __init__(self, gpus:list, waitTimeInit = 30, waitTime = 60, count = 0,
                 checkType:int = 0, utilRatio:int = 50, need_gpu=1, max_run_num = 8, p = True):
        self.activate  = False

        self.gpus      = [int(gpu) for gpu in gpus]
        self.waitTimeInit = waitTimeInit
        self.waitTime = waitTime
        self.checkType = checkType
        self.need_gpu = int(need_gpu)
        self.max_run_num = max_run_num

        self.utilRatio = utilRatio
        self.p = p
        self.count = count

        self.availGPUs = []
        self.runGPUs   = {}

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
            self.check_process()
            availGPUs = []
            count += 1
            for gpu in sorted(self.gpus - self.runGPUs.keys()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu))
                # 1. 아무것도 돌지 않는 경우
                if self.checkType == 0 and len(pynvml.nvmlDeviceGetComputeRunningProcesses(handle)) == 0:
                    availGPUs.append(gpu)

                # 2. n% 이하를 사용하고 있는 경우
                elif self.checkType == 1 and self.getFreeRatio(int(gpu)) < self.utilRatio:
                    availGPUs.append(gpu)

                if self.max_run_num <= len(availGPUs):
                    break

            if len(availGPUs) < self.need_gpu:
                if self.p:
                    print(f"{count}  |  CURRENT RUNNING GPUS : {list(self.runGPUs.keys())}")
                time.sleep(self.waitTime)
            else:
                break

        self.availGPUs = availGPUs
        if self.p: print("Activate GPUS : ", self.availGPUs)

    def getGPU(self):
        if len(self.availGPUs) < self.need_gpu: self.setGPU()
        gpus, self.availGPUs = self.availGPUs[:self.need_gpu], self.availGPUs[self.need_gpu:]

        return ','.join(map(str, gpus))


    def check_process(self):
        endGPUs = []
        for gpu, process in self.runGPUs.items():
            if process.poll() is not None:
                endGPUs.append(gpu)

        self.runGPUs = {gpu: p for gpu, p in sorted(self.runGPUs.items()) if gpu not in endGPUs}


    def register_process(self, gpus, p):
        for gpu in gpus.split(','):
            self.runGPUs[int(gpu)] = p


def Process_Worker(processes, gpuWorker, p = True):
    # TODO : 실험이 완전히 끝난 시간 체크할 필요가 존재함
    start = time.localtime()
    print("------ Start Running!! : {} ------".format(time2str(start)))

    for i, process in enumerate(tqdm(processes)):
        gpus = gpuWorker.getGPU()
        prefix = f"CUDA_VISIBLE_DEVICES={gpus} nohup sh -c \'"
        suffix = f"\' > {i+1}:gpu{gpus}.log 2>&1 "
        # print("------ {}:GPU{}  {} ------".format(i + 1, gpus, prefix + process + suffix))
        p = subprocess.Popen(prefix + process + suffix, shell=True)
        gpuWorker.register_process(gpus, p)
    gpuWorker.waitForEnd()

    end = time.localtime()
    print("------ End Running!!   : {} ------".format(time2str(end)))
    training_time = datetime.timedelta(seconds=time.mktime(end) - time.mktime(start))
    print(f"Time 1/all :  {training_time}/{training_time / len(processes)} ------")





if __name__ == '__main__':
    # Wokring Sample
    processes = [1,2,3,4,5]
    gpuWorker = GPU_Worker([0,1,2,3], 30, 120, checkType=1, utilRatio=50, need_gpu=4)
    Process_Worker(processes, gpuWorker)