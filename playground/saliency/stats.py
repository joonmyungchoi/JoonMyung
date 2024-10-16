import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from torchvision.transforms import InterpolationMode
from torchvision import transforms
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import torch
import os
from joonmyung.draw import drawLinePlot
from joonmyung.utils import to_np

train, resize, data_type = False, True, 1
if data_type == 0: # IMAGENET
    # root_path = '/hub_data2/joonmyung/data/imagenet/train_obj'
    root_path = '/hub_data2/joonmyung/data/imagenet/train_opencvFine'
    data_paths = sorted(glob(os.path.join(root_path, "**", "*.pt")))
    datas = torch.zeros((1, 224, 224), device="cuda")

elif data_type: # MSRVTT
    if data_type == 1: # MSRVTT + OPENCV
        # root_path = f'/hub_data2/joonmyung/data/MSRVTT/videos/saliency_opencv'
        root_path = f'/hub_data2/joonmyung/data/MSRVTT/videos/saliency_opencvFine'
        data_paths = sorted(glob(os.path.join(root_path, "*.pt")))
    elif data_type == 2:  # MSRVTT + FastSAM
        root_path = f'/hub_data2/joonmyung/data/MSRVTT/videos/saliency_sam/' # 152
        data_paths = sorted(glob(os.path.join(root_path, "**", "*.pt")))
    datas = []
if not resize:
    transform = lambda x : x
elif train:
    transform = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC, ), ])
else:
    transform = transforms.Compose([transforms.Resize((224, 224), interpolation=3)])






for idx, data_path in enumerate(tqdm(data_paths)):
    data = torch.load(data_path).to("cuda")
    shape = (1, ) * (3 - data.dim()) + data.shape
    data = transform(data.reshape(shape)).to(dtype=torch.float)
    if data_type == 0:
        datas += data.float() / 255.
    else:
        if data_type == 1:
            data = data.mean(dim=(1, 2))
        else:
            data = data.mean(dim=(1, 2, 3))
        datas.append(data)

# split_list = lambda lst, num_sections: [lst[i::num_sections] for i in range(num_sections)]

if data_type == 0: # IMAGENET
    result = to_np((datas / len(data_paths)).permute(1, 2, 0))
    plt.imshow(result)
    plt.show()
elif data_type: # MSRVTT + OPENCV
    result = torch.stack(datas).float().mean(dim=0)
    print(result)
print(1)
# 152 : 28686
# FALSE : torch.tensor([0.2034, 0.2054, 0.2050, 0.2027, 0.2011, 0.2017, 0.2034, 0.2016, 0.2013, 0.1993, 0.1969, 0.1969], device='cuda:0')
# TRUE  : torch.tensor([0.2139, 0.2161, 0.2157, 0.2134, 0.2116, 0.2123, 0.2140, 0.2121, 0.2119, 0.2097, 0.2073, 0.2072], device='cuda:0')
# 66 : 33000
# FALSE : torch.tensor([0.2071, 0.2089, 0.2110, 0.2077, 0.2031, 0.2037, 0.2064, 0.2051, 0.2065, 0.2046, 0.2035, 0.2024], device='cuda:0')
# TRUE  : torch.tensor([0.2175, 0.2197, 0.2217, 0.2184, 0.2136, 0.2143, 0.2171, 0.2158, 0.2171,0.2152, 0.2140, 0.2128], device='cuda:0')

# 28686 * torch.tensor([0.2034, 0.2054, 0.2050, 0.2027, 0.2011, 0.2017, 0.2034, 0.2016, 0.2013, 0.1993, 0.1969, 0.1969], device='cuda:0') + 33000 * torch.tensor([0.2071, 0.2089, 0.2110, 0.2077, 0.2031, 0.2037, 0.2064, 0.2051, 0.2065, 0.2046, 0.2035, 0.2024], device='cuda:0')
    # tensor([0.2054, 0.2073, 0.2082, 0.2054, 0.2022, 0.2028, 0.2050, 0.2035, 0.2041, 0.2021, 0.2004, 0.1998], device='cuda:0')
# 28686 * torch.tensor([0.2139, 0.2161, 0.2157, 0.2134, 0.2116, 0.2123, 0.2140, 0.2121, 0.2119, 0.2097, 0.2073, 0.2072], device='cuda:0') + 33000 * torch.tensor([0.2175, 0.2197, 0.2217, 0.2184, 0.2136, 0.2143, 0.2171, 0.2158, 0.2171, 0.2152, 0.2140, 0.2128], device='cuda:0')
    # tensor([0.2158, 0.2180, 0.2189, 0.2161, 0.2127, 0.2134, 0.2157, 0.2141, 0.2147, 0.2126, 0.2109, 0.2102], device='cuda:0')