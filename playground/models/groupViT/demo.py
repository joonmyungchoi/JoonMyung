import os.path as osp
from collections import namedtuple

import mmcv
import numpy as np
import torch
from datasets import build_text_transform
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.image import tensor2imgs
from mmcv.parallel import collate, scatter
from models import build_model
from omegaconf import read_write
from segmentation.datasets import (COCOObjectDataset, PascalContextDataset,
                                   PascalVOCDataset)
from segmentation.evaluation import (GROUP_PALETTE, build_seg_demo_pipeline,
                                     build_seg_inference)
from utils import get_config, load_checkpoint
import matplotlib.pyplot as plt
from PIL import Image
from google.colab.patches import cv2_imshow




checkpoint_url = 'https://github.com/xvjiarui/GroupViT/releases/download/v1.0.0/group_vit_gcc_yfcc_30e-74d335e6.pth'
cfg_path = 'configs/group_vit_gcc_yfcc_30e.yml'
output_dir = 'demo/output'
device = 'cpu'
vis_modes = ['input_pred_label', 'final_group']

PSEUDO_ARGS = namedtuple('PSEUDO_ARGS',
                         ['cfg', 'opts', 'resume', 'vis', 'local_rank'])

args = PSEUDO_ARGS(
    cfg=cfg_path, opts=[], resume=checkpoint_url, vis=vis_modes, local_rank=0)

cfg = get_config(args)

with read_write(cfg):
    cfg.evaluate.eval_only = True



model = build_model(cfg.model)
model = revert_sync_batchnorm(model)
model.to(device)
model.eval()

load_checkpoint(cfg, model, None, None)

text_transform = build_text_transform(False, cfg.data.text_aug, with_dc=False)
test_pipeline = build_seg_demo_pipeline()

