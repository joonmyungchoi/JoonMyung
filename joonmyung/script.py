
class GPU_Worker():
    def __init__(self, gpus = [], waitTimeInit = 30, waitTime = 60,
                 checkType = 0, reversed=False, p = True):
        self.activate  = False
        self.gpus      = gpus
        self.checkType = checkType
        self.waitTimeInit  = waitTimeInit
        self.waitTime  = waitTime
        self.reversed  = reversed
        self.availGPUs = []
        self.p = p

    def setGPU(self):
        if self.activate: time.sleep(self.waitTimeInit)
        else: self.activate = True



        availGPUs, count = [], 0
        pynvml.nvmlInit()
        while True:
            count += 1
            for gpu in self.gpus:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)

                # 1. 아무것도 돌지 않는 경우
                if self.checkType == 0 and len(pynvml.nvmlDeviceGetComputeRunningProcesses(handle)) == 0:
                    availGPUs.append(gpu)

                # # 2. 70% 이하를 사용하는 경우
                # elif self.checkType == 1 and self.getFreeRatio(gpu) < 70:
                #     availGPUs.append(gpu)


            # for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            #     result[gpu] = [proc.pid, proc.usedGpuMemory]

            if len(availGPUs) == 0:
                if self.p: print("{} : Wait for finish".format(count))
                time.sleep(self.waitTime)

            else:
                break
        self.availGPUs = availGPUs
        if self.p: print("Activate GPUS : ", self.availGPUs)

    def getGPU(self):
        if len(self.availGPUs) == 0: self.setGPU()
        return self.availGPUs.pop() if self.reversed else self.availGPUs.pop(0)





def Process_Worker(processes, gpuWorker, p = True):
    start = time.localtime()
    print("------ Start Running!! : {} ------".format(time2str(start)))

    for i, process in enumerate(tqdm(processes)):
        gpu = gpuWorker.getGPU()
        prefix = f"CUDA_VISIBLE_DEVICES={gpu} nohup "
        suffix = f" > {i+1}:gpu{gpu}.log 2>&1 &"
        if p:
            print("------ {}:GPU{}  {} ------".format(i + 1, gpu, process))
            # print("------ {}:GPU{} ------".format(i + 1, gpu))
            # print(process)
        subprocess.call(prefix + process + suffix, shell=True)

    end = time.localtime()
    print("------ End Running!! : {} ------".format(time2str(end)))
    print("Training Time :  : {} ------".format(time2str(end - start)))


# Wokring Sample
# gpuWorker = GPU_Worker(args.gpus, 30, 120)
# Process_Worker(processes, gpuWorker)