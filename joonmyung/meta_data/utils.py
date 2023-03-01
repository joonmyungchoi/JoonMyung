import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch
import os

from joonmyung.meta_data.label import imnet_label, cifar_label


def data2path(server, dataset,
              conference="", wandb_version="", wandb_name=""):
    if "kakao" in server:
        data_path = "/data/opensets/imagenet-pytorch"
        output_dir = "/data/project/rw/joonmyung/conference"
    elif server in ["148", "137", "151", "152", "113", "68", "67", "64"]:
        data_path = "/hub_data/joonmyung/data"
        output_dir = "/hub_data/joonmyung/conference"
    elif server in ["154"]:
        data_path = "/data1/joonmyung/data"
        output_dir = "/data1/joonmyung/conference"
    elif server in ["65"]:
        data_path = "/hub_data2/joonmyung/data"
        output_dir = "/hub_data2/joonmyung/conference"
    elif server in ["kisti"]:
        data_path = "/scratch/x2487a05/data"
        output_dir = "/scratch/x2487a05/data"
    else:
        raise ValueError

    if dataset in ["imagenet", "IMNET"]:
        if "kakao" in server:
            data_path = os.path.join(data_path, "imagenet-pytorch")
        else:
            data_path = os.path.join(data_path, "imagenet")


    output_dir = os.path.join(output_dir, conference, wandb_version, wandb_name) if conference else os.path.join(output_dir, "ETC")


    return data_path, output_dir

def get_label(key, d_name ="imagenet"):
    d_name = d_name.lower()
    if d_name in ["imagenet", "IMNET"] :
        return imnet_label[key]
    elif d_name in ["cifar10", "cifar100"]:
        return cifar_label[key]


def makeSample(shape, min=None, max=None, dataType=int, outputType=np, columns=None):
    if dataType == int:
        d = np.random.randint(min, max, size=shape)
    elif dataType == float:
        d = np.random.uniform(low=min, high=max, size=shape)
    else:
        raise ValueError

    if outputType == np:
        return d
    elif outputType == pd:
        return pd.DataFrame(d, columns=None)
    elif outputType == torch:
        return torch.from_numpy(d)

def makeAttn(shape, dim=1):
    return F.softmax(torch.randn(shape), dim=dim)

def set_dtype(df, dtypes):
    for c_n, d_t in dtypes.items():
        if c_n in df.columns:
            df[c_n] = df[c_n].astype(d_t)
    return df
