
# import numpy as np
from joonmyung.draw import showImg


def add(a, b):
    return a+b

def getImage():
    return np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)

def profiler(iter = 10, type=0,  **kwargs):
    final_list = []
    T, B, T_mix = kwargs["T"], kwargs["B"], kwargs["T_mix"]
    for _ in range(iter):
        if type == 0:
            final_list.append(randperm(T, B, T_mix))
        elif type == 1:
            final_list.append(topK(T, B, T_mix))
        else:
            raise ValueError()

    return torch.stack(final_list)

import torch

class Profiler():
    def __init__(self, iter, profiler):
        self.iter = iter
        self.profiler = 0
    def run(self, function):
        result = 0
        for _ in range(self.iter):
            self.
            result.append(function())
        return result

if __name__ == "__main__":
    epoch = 1000

    # 1. Time and
    # time.perf_counter()
    # l = profiler(epoch, kind)
    # time.perf_counter()

    # 2. Profiler / cProfiler
    import profile
    # l = profile.run('profiler(iter, type = 0)')
    l = profile.run('profiler(iter=10, type = 1, T=160000, B=1000, T_mix=5000)')

    # # 3. Yappi
    # import yappi
    #
    # yappi.start()
    # profiler(epoch, kind)
    # yappi.stop()
    #
    # print('\n'.join(yappi.get_stats()))

    print(1)


# 2. cProfile