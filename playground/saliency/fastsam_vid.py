import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np
import torch
import cv2
import os


def show_mask_track(annotation, color_dict):
    num_masks = len(annotation)
    areas = torch.sum(annotation, dim=(1, 2))
    sorted_indices = torch.argsort(areas, descending=False)
    annotation = annotation[sorted_indices]
    colored_masks = annotation[..., None] * color_dict[:num_masks, None, None, :] * 255.0
    result = np.sum(colored_masks.cpu().numpy(), axis=0)
    return result.astype(np.uint8)


max_det = 300
video_path = '../../file/video.mp4'
cap = cv2.VideoCapture(video_path)
model = YOLO("/hub_data2/joonmyung/weights/FastSAM/FastSAM-x.pt")
save_path = './output/' + os.path.split(video_path)[-1][:-4]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(save_path):
    os.makedirs(save_path)

ret, frame = cap.read()
h, w, _ = frame.shape # (240(H), 320(W), 3(C))
video = cv2.VideoWriter(save_path + '/result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
color_dict = torch.rand(max_det, 3, device=device)
while True:
    ret, frame = cap.read()
    try:
        h, w, _ = frame.shape
    except:
        break
    if not ret:
        break

    results = model(frame, device=device, retina_masks=True, iou=0.7, conf=0.25, imgsz=1024, max_det=max_det)

    masks = results[0].masks.data
    mask = show_mask_track(masks, color_dict)
    frame_ = cv2.addWeighted(frame, 1, mask, 0.7, 0)

    video.write(frame_)

video.release()

print(1)