import torch.distributed as dist
from pathlib import Path
import numpy as np
import argparse
import zipfile
import torch
import ast
import os

def to_leaf(vs):
    return [v.detach().cpu() for v in vs] if type(vs) == list else vs.detach().cpu()

def to_np(vs):
    if type(vs) == list:
        return [v.detach().cpu().numpy() if type(v) is not np.ndarray else v for v in vs]
    else:
        return vs.detach().cpu().numpy() if type(vs) is not np.ndarray else vs

def to_tensor(vs):
    if type(vs) == list:
        return [torch.Tensor(v) if type(v) is not torch.Tensor else v for v in vs]
    else:
        return torch.Tensor(vs) if type(vs) is not torch.Tensor else vs

def str2list(s):
    v = ast.literal_eval(s.replace(" ", ""))
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

def is_main_process():
    return get_rank() == 0

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def make_zipfile(src_dir, save_path, enclosing_dir="", exclude_dirs=None, exclude_extensions=None,
                 exclude_dirs_substring=None):
    """make a zip file of root_dir, save it to save_path.
    exclude_paths will be excluded if it is a subdir of root_dir.
    An enclosing_dir is added is specified.
    """
    abs_src = os.path.abspath(src_dir)
    with zipfile.ZipFile(save_path, "w") as zf:
        for dirname, subdirs, files in os.walk(src_dir):
            if exclude_dirs is not None:
                for e_p in exclude_dirs:
                    if e_p in subdirs:
                        subdirs.remove(e_p)
            if exclude_dirs_substring is not None:
                to_rm = []
                for d in subdirs:
                    if exclude_dirs_substring in d:
                        to_rm.append(d)
                for e in to_rm:
                    subdirs.remove(e)
            arcname = os.path.join(enclosing_dir, dirname[len(abs_src) + 1:])
            zf.write(dirname, arcname)
            for filename in files:
                if exclude_extensions is not None:
                    if os.path.splitext(filename)[1] in exclude_extensions:
                        continue  # do not zip it
                absname = os.path.join(dirname, filename)
                arcname = os.path.join(enclosing_dir, absname[len(abs_src) + 1:])
                zf.write(absname, arcname)

    #     # save a copy of the codebase. !!!Do not store heavy file in your codebase when using it.
    #     code_dir = dirname(dirname(dirname(os.path.realpath(__file__))))   # '/data/project/rw/joonmyung/conference/2023CVPR/code_alpro'
    #     code_zip_filename = os.path.join(args.output_dir, "code.zip")
    #     LOGGER.info(f"Saving code from {code_dir} to {code_zip_filename}...")
    #     make_zipfile(code_dir, code_zip_filename,
    #                  enclosing_dir="code",
    #                  exclude_dirs_substring="results",
    #                  exclude_dirs=["__pycache__", "output", "data", "ext"],
    #                  exclude_extensions=[".pyc", ".ipynb", ".swap", ".pt"])
    #     LOGGER.info(f"Saving code done.")

def getDir(path):
    # return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return [item.name for item in Path(path).iterdir() if item.is_dir()]

def isDir(path):
    return os.path.exists(path)