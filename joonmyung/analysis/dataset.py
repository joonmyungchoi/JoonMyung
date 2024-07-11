from torchvision.transforms import InterpolationMode
from joonmyung.meta_data.label import imnet_label, cifar_label
from torchvision.datasets.folder import default_loader
from timm.data import create_dataset, create_loader
from torchvision import transforms
from joonmyung.utils import getDir
import torch
import copy
import glob
import PIL
import os


# CIFAR Setting
# pip install cifar2png
# cifar2png cifar100 ./cifar100
# cifar2png cifar10  ./cifar10

class JDataset():
    # transforms.Resize(int((256 / 224) * input_size), interpolation=InterpolationMode.BICUBIC),
    settings = {"imagenet" : {
                    "num_classes"   : 1000,
                    "data_types"    : ["val", "train"],
                    "label_name"    : imnet_label,
                    "distributions" : {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
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

    def __init__(self, data_path="/hub_data1/joonmyung/data", dataset="imagenet", train=False, transform_type = 0,
                 distribution = None, size = None, device="cuda"):
        self.dataset = dataset.lower()
        setting = self.settings[self.dataset]
        self.label_name     = setting["label_name"]
        self.data_type      = setting["data_types"][train]
        self.transform_type = transform_type
        self.distribution   = distribution if distribution else setting["distributions"]
        size                = size if size else setting["size"]

        self.transform = [
                transforms.Compose([transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC), transforms.CenterCrop(size), transforms.ToTensor(), transforms.Normalize(self.distribution["mean"], self.distribution["std"])]),
                transforms.Compose([transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC), transforms.CenterCrop(size), transforms.ToTensor()]),
                transforms.Compose([transforms.Resize(size, interpolation=InterpolationMode.BICUBIC), transforms.ToTensor(), transforms.Normalize(self.distribution["mean"], self.distribution["std"])]),
                transforms.Compose([transforms.ToTensor()])
        ]

        self.device = device
        self.data_path = data_path
        self.label_paths = sorted(getDir(os.path.join(self.data_path, self.data_type)))

        # self.img_paths   = [sorted(glob.glob(os.path.join(self.data_path, self.data_type, "*", "*")))]
        self.img_paths   = [[path, idx] for idx, label_path in enumerate(self.label_paths) for path in sorted(glob.glob(os.path.join(self.data_path, self.data_type, label_path, "*")))]


    def __getitem__(self, idx):
        if type(idx) == tuple:
            idx, transform_type = idx
        else:
            transform_type = self.transform_type

        [img_path, targets] = [idx, 0] if type(idx) == str else self.img_paths[idx]

        sample = default_loader(img_path)
        sample = self.transform[transform_type](sample)

        return sample[None].to(self.device), torch.tensor(targets).to(self.device), self.label_name[targets]

    def getItems(self, indexs):
        ds, ls, lns = [], [], []
        for index in indexs:
            d, l, ln = self.__getitem__(index)
            ds.append(d)
            ls.append(l)
            lns.append(ln)
        return torch.cat(ds, dim=0), torch.stack(ls, dim=0), lns

    def getIndex(self, c: list = [0, 1000], i: list = [0, 50]):
        [c_s, c_e], [i_s, i_e] = c, i
        c = torch.arange(c_s, c_e).reshape(-1, 1).repeat(1, i_e - i_s).reshape(-1)
        i = torch.arange(i_s, i_e).reshape(1, -1).repeat(c_e - c_s, 1).reshape(-1)
        c_i = torch.stack([c, i], dim=-1)
        return c_i

    def __len__(self):
        return

    def getAllItems(self, batch_size=32):
        dataset = create_dataset(
            root=self.data_path, name="IMNET"
            , split='validation', is_training=False
            , load_bytes=False, class_map='')

        loader = create_loader(
            dataset,
            input_size=(3, 224, 224),
            batch_size=batch_size,
            use_prefetcher=True,
            interpolation='bicubic',
            mean = self.distribution["mean"],
            std  = self.distribution["std"],
            num_workers=8,
            crop_pct=0.9,
            pin_memory=False,
            tf_preprocessing=False)
        return loader

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
    root_path = "/hub_data1/joonmyung/data"
    dataset = "imagenet"
    dataset = JDataset(root_path, dataset, train=True, )
    d, l, l_n  = dataset[0, 1]
    samples = dataset.getitems([[0,1], [0,2], [0,3]])
    print(1)