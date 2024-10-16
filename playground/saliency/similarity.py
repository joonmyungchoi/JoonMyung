import sys
sys.path.append('/home/joonmyung/project/joonmyung')
from joonmyung.utils import str2bool
from playground.models.fastsam.prompt import FastSAMPrompt
from playground.models.fastsam.prompt_ori import FastSAMPrompt as FastSAMPrompt_ORI
from playground.models.fastsam.model import FastSAM
from joonmyung.meta_data import imnet_label
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from tqdm import tqdm
import numpy as np
import argparse
import torch
import math
import os

def makeDirs(path):
    os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--split",       type=int, default=0, help="")
parser.add_argument("--split_n",     type=int, default=4, help="")
parser.add_argument("--debug", type=str2bool, default=False, help="")

args = parser.parse_args()

device, train = "cuda", True
box_prompt, text_prompt, point_prompt, point_label = None, None, None, None
model_path, root_path = "/hub_data2/joonmyung/weights/FastSAM/FastSAM-x.pt", '/hub_data2/joonmyung/data/imagenet'

prompt_process = FastSAMPrompt(device=device)

classes = {c.split("/")[-2] : idx for idx, c in enumerate(sorted(glob(os.path.join(root_path, "train", "*/"))))}
data_paths = sorted(glob(os.path.join(root_path, "train", "**", "*.JPEG")))

split_range = math.ceil(len(data_paths) / args.split_n)
data_paths = data_paths[args.split * split_range : (args.split + 1) * split_range]

for data_path in tqdm(data_paths):
    class_dir = data_path.split("/")[-2]
    qual_path = data_path.replace("train", "train_quality")
    output_path = data_path.replace("train", "train_obj").replace("JPEG", "pt")
    makeDirs(output_path), makeDirs(qual_path)

    text_prompt = label = imnet_label[classes[class_dir]]
    print(f"class_dir : {class_dir}, label : {label}")
    if os.path.isfile(output_path) and not args.debug:
        continue
    img = Image.open(data_path).convert("RGB")
    H, W, _ = np.array(img).shape
    ann = np.zeros((H, W))

    everything_results = model(img, device=device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

    if everything_results:
        prompt_process.resetInfo(img, everything_results)
        if box_prompt:
            result = prompt_process.box_prompt(bboxes=box_prompt)
        elif text_prompt:
            result = prompt_process.text_prompt(text=text_prompt)
        elif point_prompt:
            result = prompt_process.point_prompt(points=point_prompt, pointlabel=point_label)
        else:
            result = prompt_process.everything_prompt()

        if len(result):
            ann = result
            if not args.debug:
                prompt_process.plot(
                    annotations=ann,
                    output_path=qual_path,
                    bboxes=box_prompt,
                    points=point_prompt,
                    point_label=point_label,
                    withContours=False,
                    better_quality=False,
                )
        if not args.debug:
            torch.save(torch.from_numpy(ann), output_path)

# CUDA_VISIBLE_DEVICES=0 python playground/saliency/fastsam.py --split 0 --split_n 4
# CUDA_VISIBLE_DEVICES=0 nohup python playground/saliency/fastsam.py --split 0 --split_n 4 > 0.log 2>&1  &
# CUDA_VISIBLE_DEVICES=1 nohup python playground/saliency/fastsam.py --split 1 --split_n 4 > 1.log 2>&1  &
# CUDA_VISIBLE_DEVICES=2 nohup python playground/saliency/fastsam.py --split 2 --split_n 4 > 2.log 2>&1  &
# CUDA_VISIBLE_DEVICES=3 nohup python playground/saliency/fastsam.py --split 3 --split_n 4 > 3.log 2>&1  &

