from joonmyung.meta_data.label import imnet_label, cifar_label
from joonmyung.utils import getDir
from torchvision.datasets.folder import default_loader
from torchvision.transforms import InterpolationMode
from timm.data import create_dataset, create_loader
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import copy
import glob
import os


# CIFAR Setting
# pip install cifar2png
# cifar2png cifar100 ./cifar100
# cifar2png cifar10  ./cifar10

class JDataset():
    settings = {"imagenet" : {
                    "num_classes"   : 1000,
                    "data_types"    : ["val", "train"],
                    "label_name"    : imnet_label,
                    "distributions" : {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                    "size": (224, 224)
                    },

                "imagenet_clip": {
                    "num_classes": 1000,
                    "data_types": ["val", "train"],
                    "label_name": imnet_label,
                    "distributions" : {"mean": [0.48145466, 0.4578275, 0.40821073], "std": [0.26862954, 0.26130258, 0.27577711]},
                    "size": (224, 224)
                },


                "cifar100" : {
                    "num_classes" : 100,
                    "data_types": ["test", "train"],
                    "label_name" : cifar_label,
                    "distributions": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010]},
                    "size" : (32, 32)
                    }
    }

    def __init__(self, data_path="/hub_data1/joonmyung/data/imagenet", dataset="imagenet", train=False, transform_type = 0,
                 distribution = None, size = None, device="cuda"):
        self.dataset = dataset.lower()
        setting = self.settings[self.dataset]
        self.label_name     = setting["label_name"]
        self.data_type      = setting["data_types"][train]
        self.transform_type = transform_type
        self.distribution   = distribution if distribution else setting["distributions"]
        size                = size if size else setting["size"]

        self.transform = [
                # DEIT
                transforms.Compose([transforms.Resize((256), interpolation=InterpolationMode.BICUBIC), transforms.CenterCrop(size), transforms.ToTensor(), transforms.Normalize(self.distribution["mean"], self.distribution["std"])]),
                transforms.Compose([transforms.Resize((256), interpolation=InterpolationMode.BICUBIC), transforms.CenterCrop(size), transforms.ToTensor()]),
                # CLIP
                transforms.Compose([transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC), transforms.CenterCrop(size), transforms.ToTensor(), transforms.Normalize(self.distribution["mean"], self.distribution["std"])]),
                transforms.Compose([transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC), transforms.CenterCrop(size), transforms.ToTensor()]),
                transforms.Compose([transforms.Resize(224, interpolation=InterpolationMode.BICUBIC), transforms.CenterCrop(size), transforms.ToTensor(), transforms.Normalize(self.distribution["mean"], self.distribution["std"])]),
                transforms.Compose([transforms.Resize(224, interpolation=InterpolationMode.BICUBIC), transforms.CenterCrop(size), transforms.ToTensor()]),
                transforms.Compose([transforms.ToTensor()])
        ]

        self.device = device
        self.data_path = data_path
        self.label_paths = sorted(getDir(os.path.join(self.data_path, self.data_type)))

        # self.img_paths   = [sorted(glob.glob(os.path.join(self.data_path, self.data_type, "*", "*")))]
        # self.img_paths   = [[path, idx] for idx, label_path in enumerate(self.label_paths) for path in sorted(glob.glob(os.path.join(self.data_path, self.data_type, label_path, "*")))]
        self.img_paths = [sorted(glob.glob(os.path.join(self.data_path, self.data_type, label_path, "*"))) for label_path in self.label_paths]
        self.img_len = [len(labels) for labels in self.img_paths]
        self.img_cum_len = torch.Tensor([0] + [sum(self.img_len[:i+1]) for i in range(len(self.img_len))])

    def __getitem__(self, idx):
        if type(idx) == int:
            label_num = (self.img_cum_len <= idx).sum().item() - 1
            img_num   =  idx - int(self.img_cum_len[label_num].item())
        else:
            label_num, img_num = idx
        img_path = self.img_paths[label_num][img_num]
        sample = default_loader(img_path)
        sample = self.transform[self.transform_type](sample)

        return sample[None].to(self.device), torch.tensor(label_num).to(self.device), self.label_name[int(label_num)], img_path

    def getItems(self, indexs):
        ds, ls, lns, ips = [], [], [], []
        for index in indexs:
            d, l, ln, ip = self.__getitem__(index)
            ds.append(d)
            ls.append(l)
            lns.append(ln)
            ips.append(ip)
        return torch.cat(ds, dim=0), torch.stack(ls, dim=0), lns, ips

    def getAllItems(self, batch_size=32):
        dataset = create_dataset(
            root=self.data_path, name="IMNET"
            , split='validation', is_training=False
            , transform=self.transform[self.transform_type]
            )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=12,
                                             collate_fn=torch.utils.data.dataloader.default_collate,
                                             drop_last=False)
        return loader

    def getIndex(self, c: list = [0, 1000], i: list = [0, 50]):
        [c_s, c_e], [i_s, i_e] = c, i
        c = torch.arange(c_s, c_e).reshape(-1, 1).repeat(1, i_e - i_s).reshape(-1)
        i = torch.arange(i_s, i_e).reshape(1, -1).repeat(c_e - c_s, 1).reshape(-1)
        c_i = torch.stack([c, i], dim=-1)
        return c_i

    def __len__(self):
        return self.img_cum_len[-1]


    def validation(self, data):
        return data.lower()

    def unNormalize(self, image):
        result = torch.zeros_like(image)
        for c, (m, s) in enumerate(zip(self.distribution["mean"], self.distribution["std"])):
            result[:, c] = image[:, c].mul(s).add(m)
        return result

    def normalize(self, image):
        result = copy.deepcopy(image)
        for c, (m, s) in enumerate(zip(self.distribution["mean"], self.distribution["std"])):
            result[:, c].sub_(m).div_(s)
        return result

if __name__ == "__main__":
    root_path = "/hub_data1/joonmyung/data/imagenet"
    dataset = "imagenet"
    dataset = JDataset(root_path, dataset, train=False)
    d, l, l_n  = dataset[[10, 3]]
    # samples = dataset.getitems([[0,1], [0,2], [0,3]])
    print(1)