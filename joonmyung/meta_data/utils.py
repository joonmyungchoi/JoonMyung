from joonmyung.meta_data.label import imnet_label, cifar_label
import torch.nn.functional as F
import pandas as pd
import numpy as np
import getpass
import socket
import torch
import os

def data2path(dataset,
              conference="", wandb_version="", wandb_name="", hub_num = 1):

    hostname = socket.gethostname()
    server   = hostname if "mlv" in hostname \
                else "kakao" if "dakao" in hostname \
                    else "kisti_"+hostname
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print(server, " : ", hostname, " : ", getpass.getuser())
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    if "kakao" in server:
        data_path  = "/data/opensets"
        output_dir = f"/data/project/rw/{getpass.getuser()}/conference"
    elif "mlv" in server:
        data_path  = f"/hub_data{hub_num}/{getpass.getuser()}/data"
        output_dir = f"/hub_data{hub_num}/{getpass.getuser()}/conference"
    elif "kisti" in server:
        data_path  = f"/scratch/{getpass.getuser()}/data"
        output_dir = f"/scratch/{getpass.getuser()}/result"
    else:
        raise ValueError

    if dataset in ["imagenet", "IMNET"]:
        data_path = os.path.join(data_path, "imagenet") if "kakao" not in server else os.path.join(data_path, "imagenet-pytorch")
        num_classes = 1000
    else:
        raise ValueError

    output_dir    = os.path.join(output_dir, conference, wandb_version, wandb_name)

    return data_path, num_classes, output_dir, server

def get_label(key, d_name ="imagenet"):
    d_name = d_name.lower()
    if d_name in ["imagenet", "IMNET"] :
        return imnet_label[key]
    elif d_name in ["cifar10", "cifar100"]:
        return cifar_label[key]


def makeSample(shape, min=0, max=1, dataType=int, outputType=np):
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