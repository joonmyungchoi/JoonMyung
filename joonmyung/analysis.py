from torchvision import transforms
import torch
import copy
import glob
import PIL
import os

from joonmyung.utils import getDir
from playground.data.labels import cifar_label, imnet_label


class JDataset():
    distributions = {"imagenet": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}, "cifar": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010]}}
    transform_cifar    = transforms.Compose([transforms.ToTensor(), transforms.Normalize(distributions["cifar"]["mean"], distributions["cifar"]["std"])])
    transform_imagenet = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(distributions["imagenet"]["mean"], distributions["imagenet"]["std"])])
    transforms = {"imagenet" : transform_imagenet, "cifar" : transform_cifar}

    # CIFAR Setting
    # pip install cifar2png
    # cifar2png cifar100 ./cifar100
    # cifar2png cifar10  ./cifar10
    def validation(self, data):
        return data.lower()

    def unNormalize(self, image):
        result = copy.deepcopy(image)
        for c, (m, s) in enumerate(zip(self.distributions[self.d_kind]["mean"], self.distributions[self.d_kind]["std"])):
            result[:, c].mul_(s).add_(m)
        return result

    def normalize(self, image):
        result = copy.deepcopy(image)
        for c, (m, s) in enumerate(zip(self.distributions[self.d_kind]["mean"], self.distributions[self.d_kind]["std"])):
            result[:, c].sub_(m).div_(s)
        return result

    def __init__(self, root_path, dataset, device="cuda"):
        dataset = dataset.lower()

        self.d      = dataset.lower()
        [self.d_kind, self.d_type] = ["imagenet", "val"] if self.d == "imagenet" else ["cifar", "test"]
        self.device = device

        self.transform = self.transforms[self.d_kind]
        self.data_path = os.path.join(root_path, self.d, self.d_type)
        self.label_name = imnet_label if self.d_kind == "imagenet" else cifar_label
        self.label_paths = sorted(getDir(self.data_path))


    def __getitem__(self, index):
        label_num, img_num= index
        label_path = self.label_paths[label_num]
        img_path = sorted(glob.glob(os.path.join(self.data_path, label_path, "*")))[img_num]
        img = PIL.Image.open(img_path)
        data = self.transform(img)

        return data.unsqueeze(0).to(self.device), torch.tensor(label_num).to(self.device), \
                    img, self.label_name[label_num]
    def getitems(self, indexs):
        ds, ls, ies, lns = [], [], [], []
        for index in indexs:
            d, l, i, ln = self.__getitem__(index)
            ds.append(d)
            ls.append(l)
            ies.append(i)
            lns.append(ln)
        return torch.cat(ds, dim=0), torch.stack(ls, dim=0), ies, lns

if __name__ == "__main__":
    root_path = "/hub_data/joonmyung/data"
    dataset = "cifar100"
    dataset = JDataset(root_path, dataset)
    # sample  = dataset[0, 1]
    samples = dataset.getitems([[0,1], [0,2], [0,3]])
