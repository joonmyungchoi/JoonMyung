import numpy as np


# 1. Yappi
from joonmyung.draw import showImg


def add(a, b):
    return a+b

def getImage():
    return np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)

def profiler(epoch= 100, kind = "operator"):
    final_list = []
    for i in range(epoch):
        if kind == "operator":
            # out = add(i, i)
            out = i + i
        elif kind == "image":
            out = getImage()
            showImg(out, t=1)
        else:
            raise ValueError()
        final_list.append(out)

    return final_list

import time
if __name__ == "__main__":
    epoch = 1000
    kind = "image"

    # 1. Time and
    # time.perf_counter()
    # l = profiler(epoch, kind)
    # time.perf_counter()

    # 2. Profiler / cProfiler
    # import profile
    # l = profile.run('profiler(epoch, kind)')

    # 3. Yappi
    import yappi

    yappi.start()
    profiler(epoch, kind)
    yappi.stop()

    print('\n'.join(yappi.get_stats()))

    print(1)


# 2. cProfile