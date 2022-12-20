import os
import random

import torch
import numpy as np

def to_np(v):
    return v.detach().cpu().numpy()

def str2list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def time2str(time, type = 0):
    if type == 0:
        return "{:4d}.{:2d}.{:2d} {:2d}:{:2d}:{:2d}".format(time.tm_year, time.tm_mon, time.tm_mday, time.tm_hour, time.tm_min, time.tm_sec)
    else:
        raise ValueError()