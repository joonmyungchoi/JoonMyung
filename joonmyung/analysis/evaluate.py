import os

from joonmyung.metric import accuracy

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# Section A. Data
from joonmyung.analysis import JDataset, JModel, ZeroShotInference
from joonmyung.meta_data import data2path, imnet_label

root_path, dataset_name, device, debug = "/hub_data1/joonmyung/weights", "imagenet", 'cuda', True
data_path, num_classes, _, _ = data2path(dataset_name)
dataset = JDataset(data_path, dataset_name, device=device)
classnames = list(imnet_label.values())
dataloader = dataset.getAllItems(batch_size = 32)

model_name, model_number = "ViT-B/16", 2
modelMaker = JModel(num_classes, root_path, device=device)
model = modelMaker.getModel(model_number, model_name)

model = ZeroShotInference(model, classnames, prompt = "a photo of a {}.", device = device)

for image, labels in dataloader:
    logits = model(image)
    acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
    print(1)

