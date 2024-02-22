from joonmyung.meta_data.label import imnet_label, cifar_label
from timm.data import create_dataset, create_loader
from torchvision import transforms
import torch
import copy
import glob
import PIL
import os

from joonmyung.utils import getDir


class JDataset():
    distributions = {"imagenet": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                        "cifar": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010]}}
    transform_cifar         = transforms.Compose([transforms.ToTensor(), transforms.Normalize(distributions["cifar"]["mean"], distributions["cifar"]["std"])])
    # transform_imagenet_     = transforms.Compose([transforms.ToTensor(), transforms.Normalize(distributions["imagenet"]["mean"], distributions["imagenet"]["std"])])
    # transform_imagenet_vis = transforms.Compose([transforms.Resize(256, interpolation=3), transforms.CenterCrop(224)])
    transform_imagenet_vis  = transforms.Compose([transforms.Resize((224, 224), interpolation=3)])
    transform_imagenet_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(distributions["imagenet"]["mean"], distributions["imagenet"]["std"])])

    # transforms.Resize(int((256 / 224) * input_size), interpolation=InterpolationMode.BICUBIC),
    transforms = {"imagenet" : [transform_imagenet_vis, transform_imagenet_norm], "cifar" : transform_cifar}

    # CIFAR Setting
    # pip install cifar2png
    # cifar2png cifar100 ./cifar100
    # cifar2png cifar10  ./cifar10
    def validation(self, data):
        return data.lower()

    def unNormalize(self, image):
        result = torch.zeros_like(image)
        for c, (m, s) in enumerate(zip(self.distributions[self.d_kind]["mean"], self.distributions[self.d_kind]["std"])):
            result[:, c] = image[:, c].mul(s).add(m)
        return result

    def normalize(self, image):
        result = copy.deepcopy(image)
        for c, (m, s) in enumerate(zip(self.distributions[self.d_kind]["mean"], self.distributions[self.d_kind]["std"])):
            result[:, c].sub_(m).div_(s)
        return result

    def __init__(self, data_path="/hub_data1/joonmyung/data", dataset="imagenet", device="cuda", train = False):
        dataset = dataset.lower()

        self.d      = dataset.lower()
        self.num_classes = 1000 if self.d == "imagenet" else 100
        if train:
            [self.d_kind, self.d_type] = ["imagenet", "val"] if self.d == "imagenet" else ["cifar", "test"]
        else:
            [self.d_kind, self.d_type] = ["imagenet", "train"] if self.d == "imagenet" else ["cifar", "train"]
        self.device = device

        self.transform = self.transforms[self.d_kind]
        self.data_path = data_path
        self.label_name = imnet_label if self.d_kind == "imagenet" else cifar_label
        self.label_paths = sorted(getDir(os.path.join(self.data_path, self.d_type)))
        self.img_paths = [sorted(glob.glob(os.path.join(self.data_path, self.d_type, label_path, "*"))) for label_path in self.label_paths]

    def __getitem__(self, index):
        label_num, img_num = index
        img_path = self.img_paths[label_num][img_num]
        img = PIL.Image.open(img_path)

        img = self.transform[0](img)
        data = self.transform[1](img)
        return data.unsqueeze(0).to(self.device), torch.tensor(label_num).to(self.device), \
                    img, self.label_name[int(label_num)]
    def getItems(self, indexs):
        ds, ls, ies, lns = [], [], [], []
        for index in indexs:
            d, l, i, ln = self.__getitem__(index)
            ds.append(d)
            ls.append(l)
            ies.append(i)
            lns.append(ln)
        return torch.cat(ds, dim=0), torch.stack(ls, dim=0), ies, lns
    def __len__(self):
        return

    def getAllItems(self, batch_size=32):

        dataset = create_dataset(
            root=self.data_path, name="IMNET" if self.d == "imagenet" else self.d.upper()
            , split='validation', is_training=False
            , load_bytes=False, class_map='')

        loader = create_loader(
            dataset,
            input_size=(3, 224, 224),
            batch_size=batch_size,
            use_prefetcher=True,
            interpolation='bicubic',
            mean=self.distributions[self.d_kind]["mean"],
            std=self.distributions[self.d_kind]["std"],
            num_workers=8,
            crop_pct=0.9,
            pin_memory=False,
            tf_preprocessing=False)
        return loader

# if __name__ == "__main__":
#     root_path = "/hub_data1/joonmyung/data"
#     dataset = "cifar100"
#     dataset = JDataset(root_path, dataset)
#     # sample  = dataset[0, 1]
#     samples = dataset.getitems([[0,1], [0,2], [0,3]])