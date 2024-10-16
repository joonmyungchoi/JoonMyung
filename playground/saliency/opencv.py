import sys

from joonmyung.data import normalization

sys.path.append('/home/joonmyung/project/joonmyung')
from joonmyung.utils import str2bool
from joonmyung.draw import drawImgPlot
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import torch
import argparse
import math
import cv2
import os


def makeDirs(path):
    os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--split",       type=int, default=0, help="")
parser.add_argument("--split_n",     type=int, default=4, help="")
parser.add_argument("--debug",       type=str2bool, default=False, help="")

args = parser.parse_args()

train = True
root_path = '/hub_data2/joonmyung/data/imagenet'
if args.debug:
    data_paths = [os.path.join(root_path, "train", "n01440764", "n01440764_39.JPEG")]
else:
    data_paths = sorted(glob(os.path.join(root_path, "train", "**", "*.JPEG")))
    split_range = math.ceil(len(data_paths) / args.split_n)
    data_paths = data_paths[args.split * split_range : (args.split + 1) * split_range]



saliencyCoarse = cv2.saliency.StaticSaliencySpectralResidual_create()
saliencyFine = cv2.saliency.StaticSaliencyFineGrained_create()
for data_path in tqdm(data_paths):
    output_opencv_path = data_path.replace("train", "train_opencv").replace("JPEG", "pt")
    output_opencv_fine_path = data_path.replace("train", "train_opencvFine").replace("JPEG", "pt")
    makeDirs(output_opencv_path), makeDirs(output_opencv_fine_path)

    if os.path.isfile(output_opencv_fine_path) and os.path.isfile(output_opencv_path) and not args.debug:
        continue
    img = cv2.imread(data_path)
    img = cv2.resize(img, (224, 224))


    (success, saliencyMap) = saliencyCoarse.computeSaliency(img)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    if not args.debug:
        torch.save(torch.from_numpy(saliencyMap), output_opencv_path)

    (success, saliencyFineMap) = saliencyFine.computeSaliency(img)
    threshMap = cv2.threshold((saliencyFineMap * 255).astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    if not args.debug:
        torch.save(torch.from_numpy(threshMap), output_opencv_fine_path)

    print(1)

    # drawImgPlot(torch.from_numpy(np.array(img)[:, :, ::-1].copy()).permute(2, 0, 1)[None]) : IMG
    # result = overlay_mask(img,
    #                       to_pil_image(normalization(, type=0), mode='F'))  # (3, 224, 224), (1, 14, 14)

