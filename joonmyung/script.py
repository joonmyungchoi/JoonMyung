from joonmyung.utils import time2str
from tqdm import tqdm
import subprocess
import time
import pynvml
import requests
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
                    print(f"{count} | AVAIL/NEED/RUNNING GPUS : {list(availGPUs)}/{self.need_gpu}/{list(self.runGPUs.keys())}")
                    # print(f"{count} | DONE/REMAIN GPUS : ")
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

    def waitForEnd(self):
        time.sleep(self.waitTimeInit)

        count = 0
        while True:
            self.check_process()
            if self.p:
                count += 1
                print(f"{count}  |  CURRENT RUNNING GPUS : {list(self.runGPUs.keys())}")
            time.sleep(self.waitTime)
            if not len(self.runGPUs):
                break
        return

    def message(self, text, url):
        payload = {"text": text}
        headers = {'Content-type': 'application/json'}

        response = requests.post(url, json=payload, headers=headers)

        return response

def Process_Worker(processes, gpuWorker, s = 0, url = None, server = None):
    start = time.localtime()
    print("------ Start Running!! : {} ------".format(time2str(start)))
    if url:
        gpuWorker.message(f"Experiments Start \n"  
                          f"Server {server} \n",
                          url = url)

    for i, process in enumerate(tqdm(processes)):
        if type(process) == list:
            process = " && ".join(process)
        gpus = gpuWorker.getGPU()
        prefix = f"CUDA_VISIBLE_DEVICES={gpus} nohup bash -c '"
        suffix = f"' > {s+i+1}:gpu{gpus}.log 2>&1 "

        print(f"{i + 1}/{len(processes)} : GPU{gpus} {prefix + process + suffix}")
        session = subprocess.Popen(prefix + process + suffix, shell=True)
        gpuWorker.register_process(gpus, session)
    gpuWorker.waitForEnd()
    end = time.localtime()

    print("------ End Running!!   : {} ------".format(time2str(end)))
    training_time = datetime.timedelta(seconds=time.mktime(end) - time.mktime(start))
    print(f"Time 1/all :  {training_time}/{training_time / len(processes)} ------")
    if url:
        gpuWorker.message(f"Experiments Finished \n"  
                          f"Server {server} \n"
                          f"Time 1/all : {training_time / len(processes)} / {training_time}",
                          url = url)


if __name__ == '__main__':
    # Wokring Sample
    processes = [1,2,3,4,5]
    gpuWorker = GPU_Worker([0,1,2,3], 30, 120, checkType=1, utilRatio=50, need_gpu=4)
    Process_Worker(processes, gpuWorker)



# 1호기 : https://hooks.slack.com/services/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
# 2호기 : https://hooks.slack.com/services/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
# 3호기 : https://hooks.slack.com/services/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

# TEST : curl -X POST -H 'Content-type: application/json' --data '{"text":"Hello, World!"}' https://hooks.slack.com/services/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