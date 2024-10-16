import sys

from playground.models.fastsam.model import FastSAM

sys.path.append('/home/joonmyung/project/joonmyung')
from playground.models.fastsam.prompt_ori import FastSAMPrompt
from joonmyung.meta_data import imnet_label
from PIL import Image
from glob import glob
from tqdm import tqdm
import argparse
import torch
import os

import matplotlib.pyplot as plt
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--split", type=int, default=0, help="")
args = parser.parse_args()

device, train = "cuda", True
box_prompt, text_prompt, point_prompt, point_label = None, None, None, None
model_path, root_path = "/hub_data1/joonmyung/weights/FastSAM/FastSAM-x.pt", '/hub_data1/joonmyung/data/imagenet'
data_len, data_split_num, split_full = 50, 0, 8

split = args.split

model = FastSAM(model_path)
class_dirs = sorted(os.listdir(os.path.join(root_path, "train")))
class_split_len = len(class_dirs) // split_full
class_dirs = class_dirs[split * class_split_len:(split + 1) * class_split_len]
for c_idx, class_name in enumerate(tqdm(class_dirs)):
    class_dir = os.path.join(root_path, "train", class_name)
    qual_path = os.path.join(root_path, "train_quality", class_name)
    obj_path = os.path.join(root_path, "train_obj", class_name)
    os.makedirs(qual_path, exist_ok=True)
    os.makedirs(obj_path, exist_ok=True)

    data_paths = sorted(glob(os.path.join(class_dir, "*.JPEG")))
    text_prompt = label = imnet_label[c_idx]
    print(f"class_dir : {class_dir}, label : {label}")
    for idx, data_path in enumerate(data_paths[data_len * data_split_num:]):
        if idx == data_len:
            break

        try:
            img = np.array(Image.open(data_path).convert("RGB"))
            output_path = data_path.replace("train", "train_obj").replace("JPEG", "pt")

            # if os.path.isfile(output_path):
            #     continue
            everything_results = model(img, device=device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
            if everything_results:
                prompt_process = FastSAMPrompt(img, everything_results, device=device)
                if box_prompt:
                    ann = prompt_process.box_prompt(bboxes=box_prompt)
                elif text_prompt:
                    ann = prompt_process.text_prompt(text=text_prompt)
                elif point_prompt:
                    ann = prompt_process.point_prompt(points=point_prompt, pointlabel=point_label)
                else:
                    ann = prompt_process.everything_prompt()
                if len(ann):
                    prompt_process.plot(
                        annotations=ann,
                        output_path=data_path.replace("train", "train_quality"),
                        bboxes=box_prompt,
                        points=point_prompt,
                        point_label=point_label,
                        withContours=False,
                        better_quality=False,
                    )

                    # torch.save(torch.from_numpy(ann), output_path)
        except Exception as e:
            print(e)
            continue

