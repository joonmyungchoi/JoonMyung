def time2str(time, type = 0):
    if type == 0:
        return "{:4d}.{:2d}.{:2d} {:2d}:{:2d}:{:2d}".format(time.tm_year, time.tm_mon, time.tm_mday, time.tm_hour, time.tm_min, time.tm_sec)
    else:
        raise ValueError()
