# if __name__ == "__main__":
#     epoch = 1000
#
#     # 1. Time and
#     # time.perf_counter()
#     # l = profiler(epoch, kind)
#     # time.perf_counter()
#
#     # 2. Profiler / cProfiler
#     import profile
#     # l = profile.run('profiler(iter, type = 0)')
#     l = profile.run('profiler(iter=10, type = 1, T=160000, B=1000, T_mix=5000)')
#
#     # # 3. Yappi
#     # import yappi
#     #
#     # yappi.start()
#     # profiler(epoch, kind)
#     # yappi.stop()
#     #
#     # print('\n'.join(yappi.get_stats()))
#
#     print(1)
#
#
# # 2. cProfile