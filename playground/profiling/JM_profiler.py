import time

from joonmyung.log import AverageMeter
from playground.profiling.test.samples import *
from tqdm import tqdm
import math

class Profiler():
    def __init__(self, iter = 100, warmup_iter= 10):
        self.iter        = iter
        self.warmup_iter = warmup_iter
    def profile(self, function, **kwargs):
        t = AverageMeter()
        for i in tqdm(range(self.iter)):
            start = time.time()
            function(**kwargs)
            end   = time.time()
            if i > self.warmup_iter:
                t.update(end - start)
        return t.avg

    def run(self, functions, sort = False):
        rs, t = {}, math.inf

        for function in functions:
            r = self.profile(function[0], **function[1])
            rs[function[0].__name__] = r
            if r < t:
                t = r

        if sort:
            rs = sorted(rs.items(), key=lambda item: item[1])

        for c, v in rs.items():
            print(f"{c} : {v:.5f} {v / t:.2f}")
        return rs

a = {"T":160000, "B":1000, "T_mix":5000}


if __name__ == "__main__":
    profiler = Profiler(iter=100000, warmup_iter=100)
    # profile = [[topK,     {"T":1600, "B":10, "T_mix":50}], # 1.0 ✓
    #            [randperm, {"T":1600, "B":10, "T_mix":50}]] # 4.0-5.0배
    profile_list = [[one_hot_v1, {"B": 1000, "C": 1000, "smoothing": 0.1}],  # 1.0 ✓
                    [one_hot_v2, {"B": 1000, "C": 1000, "smoothing": 0.1}],  # 1.83
                    [one_hot_v3, {"B": 1000, "C": 1000, "lam":1.0, "smoothing": 0.1}]]  # 2.68

    profiler.run(profile_list)
