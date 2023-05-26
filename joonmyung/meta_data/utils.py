from joonmyung.meta_data.label import imnet_label, cifar_label
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch
import socket
import os



def data2path(dataset, server = "",
              conference="", wandb_version="", wandb_name="",
              kisti_id=""):

    # server = server if server else
    hostname = socket.gethostname()
    server = server if server is not "" \
                else server if "mlv" in server \
                    else "kakao" if "dakao" in hostname \
                        else "kisti"

    if "kakao" in server:
        data_path = "/data/opensets"
        output_dir = "/data/project/rw/joonmyung/conference"
    elif "mlv" in server:
        data_path = "/hub_data1/joonmyung/data"
        output_dir = "/hub_data1/joonmyung/conference"
    elif server in ["kisti"]:
        data_path = f"/scratch/{kisti_id}/data"
        output_dir = f"/scratch/{kisti_id}/result"
    else:
        raise ValueError

    if dataset in ["imagenet", "IMNET"]:
        data_path = os.path.join(data_path, "imagenet") if "kakao" not in server else os.path.join(data_path, "imagenet-pytorch")
    output_dir = os.path.join(output_dir, conference, wandb_version, wandb_name)

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


